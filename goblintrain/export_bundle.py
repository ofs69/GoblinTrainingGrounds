"""A released checkpoint + the frozen perception -> the GoblinScript ONNX bundle.

GoblinScript (``goblinscript/``, the Rust deploy CLI) runs no PyTorch: every net it
needs is an ONNX graph exported here, and everything that is NOT a net (ffmpeg
decode, tubelet grouping, the int8 round-trip, boundary statistics, composed
styling, funscript emission) is Rust. This script is the ONLY seam between
training and the product.

Five graphs + a manifest land in ``--out`` (default ``goblinscript/bundle/``):

  vjepa_pca.onnx  uint8 frame windows -> whitened latent rows. The frozen
                  encoder with the frozen PCA basis FOLDED IN as a final
                  matmul, so the basis can never drift away from the graph
                  that used it. ImageNet normalization is in-graph: Rust
                  hands over exactly what ffmpeg emitted.
  transnet.onnx   (1,100,27,48,3) uint8 frames -> (1,100) per-frame transition
                  probability. The boundary detector (TransNetV2, 7.6M params);
                  Rust drives the sliding window and thresholds the output.
                  Exported fp32 -- the net is small, and its color-histogram op
                  partitions to the CPU EP on DirectML while the convs run on
                  the GPU.
  head.onnx       (1,W,dim,24,24) int8 latents + (1,W) cut flags -> every
                  track composed styling reads (marginal velocity, level,
                  both rails, the dwell and reversal probabilities), a
                  per-row ``conf`` score (the model's own expected agreement
                  with a human script -- a decode-time signal the review
                  page surfaces, never read by styling), and the trunk
                  features ``h`` the envelope decode needs. The int8 dequant
                  scale and the level decode temperature are baked in. W is
                  dynamic: the deploy chunk is 1024 rows, the tail short.
  mask.onnx       one int8 latent row -> its attention gate. The mask net
                  alone, one row at a time -- the ROI GoblinScript's viewport
                  draws while the encoder runs (it combines the heads the
                  way the pooling does). The int8 dequant scale is baked in.
  env_step.onnx   one step of the generative envelope's autoregressive decode
                  (h_t + the context buffer -> the next envelope value).
                  Exported as a STEP, not an unrolled loop: Rust drives the
                  recurrence, which keeps the graph small and the loop exact.

Precision: the vision graphs export in **fp16** -- production's own encoder
dtype, half the bundle, and several times the DirectML encoder throughput.
Production latents are int8-quantized afterwards, so fp16 noise is largely
invisible to the model by construction -- but fp16 has no RANGE to spare in
this encoder, and a backend that evaluates the graph natively rather than
widening it (DirectML; ORT's CPU provider widens, torch-on-CUDA accumulates
fp32) sees V-JEPA's attention logits reach ~5e5 on real frames -- past fp16's
65504 ceiling in every block, and every latent comes out NaN with nothing
raised. The export therefore emits attention in an overflow-safe,
mathematically identical form (``_fp16_safe_sdpa``: pre-scaled q,
token-centered k). Measured on DirectML over real frames, the fp16 encoder
holds corr 0.9996 against the production int8 cache, and its end-to-end
draft is indistinguishable from an fp32 export's. ``--enc-fp32`` exists as a
debugging axis -- a range-proof graph to attribute numerics against -- and is
never what ships.

    python goblintrain.py export <project> --run v0.6.0
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent   # the repository

from . import common                                    # noqa: E402
from . import encoders                                  # noqa: E402
from . import jepa_infer                                # noqa: E402
from . import perception as P                           # noqa: E402
from .jepa_train import load_model                # noqa: E402
from .jepa_infer import BIAS_FIT_S                # noqa

ENV_GAIN_P = 0.6     # the manifest's hedge exponent for goblinscript's
                     # frequency-loosened amplitude bound; no shipped pack
                     # sets the amp_cap_f0 that would engage it
PERIOD_CAP_S = 2.0   # seconds: the manifest's cap on goblinscript's per-row
                     # refractory; no shipped pack carries the period head
                     # that would drive it (rev_gap_k stays 0)
AMP_CAP_F0 = 0.0     # Hz: the manifest's hedge-curve knee -- the amplitude
                     # bound would loosen with the segment's own frequency
                     # above it on (f/f0)**env_gain_p; 0 keeps the bound
                     # flat everywhere, which is what ships

CLIP_LEN = 32          # frames per encoder window (extract.CLIP_LEN)
TN_WINDOW = 100        # frames per TransNetV2 forward (boundaries.py windowing)
INT8_SCALE = 8.0 / 127.0

MEAN = P.IMAGENET_MEAN
STD = P.IMAGENET_STD


@contextlib.contextmanager
def _fp16_safe_sdpa():
    """Export attention that cannot overflow fp16, computing the same thing.

    PyTorch's SDPA decomposition (which is what the exporter traces) materializes
    the UNSCALED ``q @ k^T`` and divides by sqrt(d) afterwards. Torch itself never
    suffers for it -- its fused kernels accumulate in fp32 -- and neither does ONNX
    Runtime's CPU provider, which has no native fp16 compute and quietly widens.
    DirectML does have native fp16, so it evaluates that intermediate as written:
    the logits reach |25k| by the first block and a later one crosses fp16's 65504
    ceiling, goes Inf, and the whole encoder comes out NaN. Nothing raises -- the
    graph loads, runs, and returns garbage, which is how this was found (the int8
    cache came out all zeros).

    Scaling ``q`` BEFORE the product is algebraically identical and keeps the
    intermediate ~8x smaller. That alone is NOT enough: V-JEPA's q/k carry
    channel outliers (|k| to ~300, near-CONSTANT across tokens), and the true
    scaled logits reach ~5e5 on real frames -- 7.5x past fp16's ceiling, in
    every block. Because the outliers are token-constant they are a per-row
    additive constant ``q @ mean(k)`` in the logits, and softmax is invariant
    to per-row constants: centering k over the token axis removes them while
    changing NOTHING the softmax computes. Measured on block 0, real frames:
    |logit| 494473 -> 28975, inside fp16 with 2.3x headroom. The mean itself
    is accumulated in fp32 (a fp16 sum over 9216 tokens of ~300 overflows on
    its own) and only the centered k returns to the compute dtype.

    The reference forward stays UNPATCHED, so the verification still measures
    this graph against the production computation.
    """
    orig = F.scaled_dot_product_attention

    def safe(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
             scale=None, **kw):
        s = (q.shape[-1] ** -0.5) if scale is None else scale
        k = k - k.float().mean(dim=-2, keepdim=True).to(k.dtype)
        attn = (q * s) @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        return attn.softmax(dim=-1) @ v

    F.scaled_dot_product_attention = safe
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig


def _patch_rope_for_onnx(model):
    """Make the 2.1 RoPE's dtype promotions EXPLICIT, changing no value.

    ``rotate_queries_or_keys`` leans on PyTorch's implicit promotion: ``omega``
    is built in the input dtype (fp16), ``pos`` is fp32, and the einsum between
    them silently promotes to fp32 -- as does every op after it, until the
    result is cast back at the boundary. ONNX has no implicit promotion: a
    mixed-dtype Einsum is simply an invalid graph, and the export dies there.

    So this is the same function with the casts written down. ``omega`` is
    still computed in fp16 (matching what the deployed encoder actually
    evaluates, fp16 rounding and all) and only THEN widened -- which is exactly
    what torch's promotion did to it. Computing omega in fp32 instead would be
    a different encoder, off by ~1e-3 in the RoPE frequencies.

    Mirrors ``encoders._patch_rope_dtype``, whose fp16 fix this replaces during
    export; the module is resolved from the loaded model for the same reason
    (the 2.1 line and V-JEPA 2 keep their copies in different packages).
    """
    import importlib

    def rotate(x, pos, n_registers, has_cls_first):
        B, num_heads, N, D = x.size()
        n_cls = 1 if has_cls_first else 0
        end_ctx = N - n_registers
        x_cls = x[..., :n_cls, :] if n_cls else None
        x_ctx = x[..., n_cls:end_ctx, :]
        x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

        omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
        omega /= D / 2.0
        omega = 1.0 / 10000**omega
        freq = torch.einsum("..., f -> ... f", pos.float(), omega.float())

        emb_sin = freq.sin().repeat_interleave(2, dim=-1)
        emb_cos = freq.cos().repeat_interleave(2, dim=-1)

        y = x_ctx.unflatten(-1, (-1, 2))
        y1, y2 = y.unbind(dim=-1)
        y = torch.stack((-y2, y1), dim=-1).flatten(-2)

        out_ctx = ((x_ctx.float() * emb_cos)
                   + (y.float() * emb_sin)).to(x.dtype)
        parts = ([x_cls] if n_cls else []) + [out_ctx] \
            + ([x_reg] if n_registers else [])
        return torch.cat(parts, dim=-2)

    for blk in model.blocks:
        m = importlib.import_module(type(blk.attn).__module__)
        m.rotate_queries_or_keys = rotate
        return
    raise SystemExit("could not find the RoPE module on the encoder")


# --------------------------------------------------------------------------- #
# graph wrappers -- each one owns the pre/post-processing its Python caller did
# --------------------------------------------------------------------------- #

class VjepaPca(torch.nn.Module):
    """Frozen encoder + frozen PCA basis, as one graph.

    Takes frames exactly as ffmpeg emits them (uint8, HWC, RGB) so the Rust
    side never has to reproduce a normalization convention.

    The encoder and the projection run in the encoder's own dtype -- fp16 in
    production. Normalization stays fp32 and the latents come back fp32: the
    cast happens where PyTorch puts it, so the graph is the deploy computation
    rather than a re-derivation of it. (Converting an fp32 graph to fp16 after
    the fact does NOT reproduce this -- it mangles the uint8 input boundary,
    which is how this was found.)
    """

    def __init__(self, enc, tok_mu, W, grid, res, dtype):
        super().__init__()
        self.enc, self.grid, self.res, self.dtype = enc, grid, res, dtype
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.register_buffer("tok_mu", tok_mu.to(dtype))
        self.register_buffer("W", W.to(dtype))

    def forward(self, frames):                    # (1,32,res,res,3) uint8
        v = frames[0].permute(0, 3, 1, 2).float() / 255.0
        v = ((v - self.mean) / self.std).to(self.dtype)
        y = self.enc(v.permute(1, 0, 2, 3)[None])          # (B,C,T,H,W)
        if isinstance(y, (tuple, list)):
            y = y[-1]
        d = y.shape[-1]
        y = (y.reshape(-1, self.grid, self.grid, d) - self.tok_mu) @ self.W
        return y.permute(0, 3, 1, 2).float()               # (16,dim,g,g)


class TransNetProb(torch.nn.Module):
    """TransNetV2 -> per-frame transition probability, the boundary detector.

    A fixed 100-frame window of the model's own 27x48 uint8 frames in, the
    sigmoid single-frame prediction out. Rust decodes at 27x48 and drives the
    sliding window; the graph is just the forward. The reference forward stays
    UNPATCHED, so verification measures this graph against the model itself.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, frames):                    # (1,100,27,48,3) uint8
        one_hot = self.m(frames)
        if isinstance(one_hot, (tuple, list)):
            one_hot = one_hot[0]
        return torch.sigmoid(one_hot)[..., 0]     # (1,100)


def ckpt_rel(ckpt):
    """The manifest's source-checkpoint pointer: repo-relative when the
    checkpoint lives inside the tree, so a bundle carries no absolute
    path. parity.py resolves it against the repository root."""
    p = Path(ckpt).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


class Head(torch.nn.Module):
    """Mask net + attention pooling + TCN + every head composed styling
    reads.

    Mirrors ``JepaModel.forward``'s non-generative half exactly (the
    marginal velocity is the pre-recombination phase track -- what
    jepa_infer styles from). It also emits the per-row confidence head
    (styling ignores it; the review page renders it) and the trunk
    features, so the envelope's autoregressive decode can run as a
    separate stepped graph.
    """

    def __init__(self, model, pos_temp, scale=INT8_SCALE):
        super().__init__()
        self.m = model
        self.pos_temp = float(pos_temp)
        self.scale = float(scale)

    def forward(self, x_i8, cut):        # (1,W,dim,g,g) int8, (1,W) float
        m = self.m
        x = x_i8.float() * self.scale
        h, _gate = m.features(x, cut)
        out = [m._expect(m.head_v(h))]
        if m.pos_head:
            out.append(m._expect(m.head_p(h) / self.pos_temp,
                                 m.pos_centers))
        if m.ext_head:
            out.append(m._expect(m.head_xlo(h) / self.pos_temp,
                                 m.ext_centers))
            out.append(m._expect(m.head_xhi(h) / self.pos_temp,
                                 m.ext_centers))
        if m.plat_head:
            p = m.head_plat(h).softmax(1)
            out.append(p[:, 1])                   # P(top dwell)
            out.append(p[:, 2])                   # P(bottom dwell)
        if m.rev_head:
            r = m.head_rev(h).softmax(1)
            out.append(r[:, 1])                   # P(peak reversal)
            out.append(r[:, 2])                   # P(valley reversal)
        # per-row confidence, trained so its window mean tracks the phase
        # correlation the model expects against a human script (head_conf).
        out.append(torch.sigmoid(m.head_conf(h)).squeeze(1))   # (1,W)
        return tuple(out) + (h,)


class MaskGate(torch.nn.Module):
    """The mask net alone: ONE latent row -> its eight attention maps.

    The gate is a per-row function and that is what makes a separate graph
    honest rather than an approximation. ``frontend`` reshapes (B,W,dim,g,g)
    to (B*W,dim,g,g) and runs a 2-D conv stack over each grid INDEPENDENTLY --
    no cut flag, no temporal context, no dropout (the docstring there says it:
    the frontend is deterministic) -- so a single-row forward computes exactly
    the gate the draft's own attention pooling used on that row.

    Kept OUT of ``head.onnx``, which already computes this and discards it, for
    a reason that is about wall clock rather than arithmetic. The head does not
    load until a whole stage after the encoder, and the encode stage is the long
    one: a gate emitted from the head would light the viewport for the last few
    seconds of a draft and leave it dark for the minutes before. Emitting it
    from the head is also ~19 MB of gate per 1024-row chunk against 18 KB for
    one row here. This stack is ~40 M multiply-adds (128->48 3x3, 48->32 3x3,
    32->8 1x1 over 576 cells), so `encode.rs` runs it per window on a single CPU
    thread without taking anything from the encoder or from the batch's
    concurrent transcode.
    """

    def __init__(self, model, scale=INT8_SCALE):
        super().__init__()
        self.m = model
        self.scale = float(scale)

    def forward(self, x_i8):             # (1,dim,g,g) int8
        f = x_i8.float() * self.scale
        if getattr(self.m, "coord", False):
            # mirror frontend's coord concat: the mask net's first conv
            # takes dim+2 channels, the extra two being the constant
            # [-1,1] coordinate grids. They fold into the graph as
            # constants, so the exported input stays (1,dim,g,g) int8.
            gh, gw = f.shape[-2], f.shape[-1]
            ys = torch.linspace(-1.0, 1.0, gh, dtype=f.dtype)
            xs = torch.linspace(-1.0, 1.0, gw, dtype=f.dtype)
            cg = torch.stack(torch.meshgrid(ys, xs, indexing="ij"))
            f = torch.cat([f, cg.expand(f.shape[0], 2, gh, gw)], dim=1)
        return torch.sigmoid(self.m.mask(f))


class EnvStep(torch.nn.Module):
    """One step of the envelope's AR decode:
    (h_t, state[, e0]) -> (envelope value, next state).

    The graph OWNS the recurrence: ``state`` is the context buffer, and
    the graph steps it and hands back the value and the rotated buffer.
    Rust carries the state as an opaque vector, reseeded with zeros at
    every chunk, the way the Python loop reseeds its buffer; a stepped
    graph is not an approximation of the unrolled one, it is the same
    computation.

    The categorical head publishes an expectation. The FLOW head SAMPLES,
    integrating its field from a base draw in a fixed number of equal
    Euler steps: fixed is what makes it exportable, since the loop unrolls
    into the traced graph and Rust runs the same straight-line arithmetic.
    ``e0`` is then an INPUT rather than something the graph draws, because
    the draw is the one place the two decoders could diverge:
    ``common.env_base_draw`` is a pure function of the ABSOLUTE row, so
    each side computes it and neither carries RNG state across a chunk
    boundary.
    """

    def __init__(self, model, flow):
        super().__init__()
        self.m = model
        self.flow = bool(flow)
        self.env_ctx = int(model.env_ctx)

    def forward(self, h_t, state, e0=None):
        # (1, C, 1), (1, env_ctx)[, (1, 1)]
        m = self.m
        z = torch.cat([h_t, m.env_emb(state.unsqueeze(-1))], dim=1)
        e = m.env_flow_sample(z, e0)[:, 0] if self.flow \
            else m._expect(m.head_e(z), m.env_centers)[:, 0]     # (1,)
        return e, torch.cat([e.unsqueeze(1), state[:, :-1]], dim=1)


# --------------------------------------------------------------------------- #

def export(mod, args, path, input_names, output_names, dynamic_shapes=None):
    t0 = time.time()
    prog = torch.onnx.export(mod, args, dynamo=True, optimize=True,
                             input_names=input_names,
                             output_names=output_names,
                             dynamic_shapes=dynamic_shapes)
    prog.save(str(path))
    print(f"  {path.name}: {path.stat().st_size / 1e6:.0f} MB "
          f"({time.time() - t0:.0f}s)", flush=True)
    strip_allowzero(path)


def strip_allowzero(path):
    """Make the graph loadable on DirectML, changing no semantics.

    The dynamo exporter stamps ``allowzero=1`` on every Reshape it emits (torch
    reshape treats a 0 in the target shape as a literal zero dimension, where
    ONNX's default is "copy the input's dim"). DirectML's Reshape does not
    implement that attribute at all and refuses the whole graph with
    "the parameter is incorrect" -- which is how the V-JEPA encoder failed to
    load while the head and the envelope loaded fine.

    ``allowzero`` only means anything when the target shape CONTAINS a zero, and
    none of ours do -- a zero-length dimension is not a thing this pipeline can
    produce. So the attribute is dead weight on every node that carries it, and
    dropping it is a rewrite of the graph's encoding, not of its computation.
    Any node whose target genuinely holds a zero is left alone and reported.
    """
    import onnx
    m = onnx.load(str(path))
    const = {i.name for i in m.graph.initializer}
    vals = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    stripped, kept = 0, 0
    for n in m.graph.node:
        if n.op_type != "Reshape":
            continue
        for i, a in enumerate(n.attribute):
            if a.name != "allowzero" or a.i != 1:
                continue
            tgt = vals.get(n.input[1]) if len(n.input) > 1 else None
            if tgt is not None and (tgt == 0).any():
                kept += 1                     # genuinely relies on it
            else:
                del n.attribute[i]
                stripped += 1
            break
    if stripped or kept:
        onnx.save(m, str(path))
        print(f"  {path.name}: dropped allowzero on {stripped} Reshape nodes"
              + (f" ({kept} kept -- real zero targets)" if kept else "")
              + "  [DirectML]", flush=True)
    _ = const


def fuse_attention(path, packed=True):
    """Collapse each exported attention block into one fused-kernel node.

    The exporter emits attention as separate ONNX nodes, which forces the
    (9216 x 9216) score matrix to be MATERIALIZED to memory between the matmul,
    the softmax and the second matmul -- 2.0 GB per block, 12 blocks per
    forward. That traffic, not arithmetic, is what the encoder spends its time
    on: measured on a 4090, attention is 181 ms of a 241 ms DirectML forward
    while the card sustains 165-199 TFLOP/s on plain fp16 GEMM.

    Tiling the score matrix does NOT help (measured: flat from 2 GB tiles down
    to 16 MB ones, bit-identical output) because separate ONNX nodes cannot keep
    a tile in SRAM across the three ops -- only a FUSED kernel can, and
    ``com.microsoft.MultiHeadAttention`` is how ONNX Runtime is asked for one.
    Whole-forward, on DirectML: 241 ms unfused -> 144 ms fused -> **59 ms** fused
    AND packed (see ``packed`` below). Attention is ~39 ms of that 59; the other
    21 ms is every other op in the encoder, measured by ablating these 12 nodes.

    The rewrite is shape bookkeeping around one substitution:

        Mul(q, s) ---.                                q,k,v: (1,12,9216,64)
        Sub(k, mean) -> Transpose -> MatMul -> Softmax -> MatMul(.,v)
                                                            -> Transpose
                                                            -> Reshape (1,9216,768)

    becomes ``MultiHeadAttention(q3, k3, v3)``, whose native (B,S,N*H) output IS
    that final Reshape's layout -- so the node inherits its output name and
    everything downstream is untouched.

    Two invariants this preserves on purpose:

    * **q goes in PRE-SCALED, with scale=1.0.** Handing the node an unscaled q
      and a scale attribute is algebraically identical but computes an 8x larger
      intermediate, and DirectML's MultiHeadAttention materializes it: ~232k,
      past fp16's 65504 ceiling, which is precisely the NaN that
      ``_fp16_safe_sdpa`` exists to prevent. The pre-scaled ordering is the safe
      one and it is kept.
    * **k stays token-centered.** The Sub is left where it is, outside the node,
      so the overflow headroom that centering buys survives on any backend that
      materializes.

    ``packed`` is which of the node's two input layouts to use, and on DirectML
    that choice is worth more than the fusion itself. Handing it q, k and v as
    three separate (B,S,N*H) tensors reaches a generic path that still
    materializes -- 144 ms per forward. Handing it ONE stacked (B,S,N,3,H)
    tensor reaches DirectML's native fused attention operator: **59 ms**, a
    further 2.4x, and CLOSER to production rather than looser (parity corr
    0.999573 / 79.6% of int8 values bit-identical, against 0.999438 / 74.8% for
    the separate form and 0.999446 for the unfused graph -- the fused kernel
    accumulates in fp32 where the materializing path does not). Metacommands are
    doing that work: turning them off costs 2.1x, so this is the vendor kernel.

    ``packed=False`` exists because ONNX Runtime has no CPU kernel for the packed
    layout ("Packed QKV of shape (B, L, N, 3, H) not implemented for CPU"), and
    the CPU provider is the fp32 reference a suspect draft gets bisected against
    (`--cpu`). A packed bundle cannot run there at all, so the escape hatch is a
    re-export with ``--cpu-attn`` rather than a slower graph for everyone.

    EVERY Softmax in the graph must fuse. In this encoder all of them are
    attention softmaxes, so "one left behind" means the exporter's pattern moved
    and this rewrite is reading a graph it no longer understands -- which raises,
    rather than shipping attention that is half fused and half not.
    """
    import onnx
    from onnx import helper, numpy_helper, shape_inference

    m = onnx.load(str(path))
    g = m.graph
    ax3 = "mha_axis3"
    if packed and not any(i.name == ax3 for i in g.initializer):
        g.initializer.append(
            numpy_helper.from_array(np.array([3], np.int64), ax3))
    shapes = {}
    try:
        inf = shape_inference.infer_shapes(m, strict_mode=False)
        for v in list(inf.graph.value_info) + list(g.input) + list(g.output):
            d = v.type.tensor_type.shape.dim
            shapes[v.name] = [x.dim_value for x in d]
    except Exception as e:                                  # pragma: no cover
        raise SystemExit(f"fuse_attention: shape inference failed ({e})")

    producer = {o: n for n in g.node for o in n.output}
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)

    def only_user(t):
        u = users.get(t, [])
        return u[0] if len(u) == 1 else None

    def perm(n):
        return next((list(a.ints) for a in n.attribute if a.name == "perm"), None)

    index = {id(n): i for i, n in enumerate(g.node)}
    drop, inserts, sites = set(), {}, 0
    softmaxes = [n for n in g.node if n.op_type == "Softmax"]

    for sm in softmaxes:
        axis = next((a.i for a in sm.attribute if a.name == "axis"), -1)
        mm1 = producer.get(sm.input[0])
        mm2 = only_user(sm.output[0])
        if axis not in (-1, 3) or mm1 is None or mm1.op_type != "MatMul":
            continue
        if mm2 is None or mm2.op_type != "MatMul" or mm2.input[0] != sm.output[0]:
            continue
        kt = producer.get(mm1.input[1])
        if kt is None or kt.op_type != "Transpose" or perm(kt) != [0, 1, 3, 2]:
            continue
        tp = only_user(mm2.output[0])
        if tp is None or tp.op_type != "Transpose" or perm(tp) != [0, 2, 1, 3]:
            continue
        rs = only_user(tp.output[0])
        if rs is None or rs.op_type != "Reshape":
            continue
        q, k, v = mm1.input[0], kt.input[0], mm2.input[1]
        qs = shapes.get(q)
        if not qs or len(qs) != 4 or 0 in qs:
            continue
        _b, heads, _s, _h = qs
        bsd = rs.input[1]                    # the (1, 9216, 768) target, reused

        nodes, parts = [], []
        for name, t in (("q", q), ("k", k), ("v", v)):
            tag = f"mha{sites}_{name}"
            # (1,N,S,H) -> (1,S,N,H); packed wants a 5th axis, 3D wants a flatten
            nodes.append(helper.make_node("Transpose", [t], [f"{tag}_t"],
                                          name=f"node_{tag}_t", perm=[0, 2, 1, 3]))
            if packed:
                nodes.append(helper.make_node("Unsqueeze", [f"{tag}_t", ax3],
                                              [f"{tag}_u"], name=f"node_{tag}_u"))
                parts.append(f"{tag}_u")
            else:
                nodes.append(helper.make_node("Reshape", [f"{tag}_t", bsd],
                                              [f"{tag}_3"], name=f"node_{tag}_r"))
                parts.append(f"{tag}_3")
        # scale=1.0: q arrives pre-scaled (see above), so the node must not
        # apply the 1/sqrt(d) it would otherwise default to
        if packed:
            nodes.append(helper.make_node("Concat", parts, [f"mha{sites}_qkv"],
                                          name=f"node_mha{sites}_cat", axis=3))
            parts = [f"mha{sites}_qkv"]
        nodes.append(helper.make_node(
            "MultiHeadAttention", parts, [rs.output[0]],
            name=f"node_mha{sites}", domain="com.microsoft",
            num_heads=int(heads), scale=1.0))

        for n in (mm1, sm, mm2, tp, rs):
            drop.add(id(n))
        if len(users.get(kt.output[0], [])) == 1:
            drop.add(id(kt))
        # mm2 is the latest point at which q, k and v are all in scope
        inserts.setdefault(index[id(mm2)], []).extend(nodes)
        sites += 1

    if not sites or sites != len(softmaxes):
        raise SystemExit(
            f"fuse_attention: fused {sites} of {len(softmaxes)} Softmax nodes. "
            f"The exporter's attention pattern changed -- this rewrite must be "
            f"re-read against it, not shipped half-applied")

    out = []
    for i, n in enumerate(g.node):
        out.extend(inserts.get(i, ()))
        if id(n) not in drop:
            out.append(n)
    del g.node[:]
    g.node.extend(out)
    if not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.save(m, str(path))
    print(f"  {path.name}: fused {sites} attention blocks into "
          f"MultiHeadAttention", flush=True)


def verify(path, feeds, refs, label, tol, corr_gate=None, eps=None):
    """ORT-vs-torch on the exported graph. A graph that does not reproduce its
    checkpoint is the one failure mode nothing downstream would catch.

    ``corr_gate`` switches the check from |d| to correlation, which is what a
    SMOKE test on the fp16 encoders can honestly assert: their reference runs
    on CUDA tensor cores and the graph runs on ORT's CPU fp16 emulation, so a
    12-layer ViT's accumulation order differs and |d| drifts -- especially on
    the random-noise input used here, which no encoder ever sees. A structural
    break (wrong weights, dropped op, mangled dtype) destroys correlation; it
    cannot survive at 0.999. The tolerance that actually matters is measured
    against the production latent cache by parity.py, on real frames.

    ``eps`` overrides the provider chain. The CPU provider is the right one to
    check against -- no vendor kernels between the graph and the answer -- but
    a packed-attention encoder has no CPU kernel at all, so that graph gets
    checked on the GPU it is built for.
    """
    import onnxruntime as ort
    t0 = time.time()
    s = ort.InferenceSession(str(path), providers=eps or ["CPUExecutionProvider"])
    got = s.run(None, feeds)
    worst, worst_corr = 0.0, 1.0
    for g, r in zip(got, refs):
        a = np.asarray(g, np.float64).ravel()
        b = r.detach().float().cpu().numpy().astype(np.float64).ravel()
        worst = max(worst, float(np.abs(a - b).max()))
        if b.std() > 1e-9:
            worst_corr = min(worst_corr, float(np.corrcoef(a, b)[0, 1]))
    if corr_gate is not None:
        ok = worst_corr >= corr_gate
        print(f"  verify {label:<22} corr {worst_corr:.6f} "
              f"(gate {corr_gate}) max|d| {worst:.2e} "
              f"[{'OK' if ok else 'FAIL'}] {time.time() - t0:.0f}s", flush=True)
    else:
        ok = worst <= tol
        print(f"  verify {label:<22} max|d| {worst:.2e} (tol {tol:g}) "
              f"[{'OK' if ok else 'FAIL'}] {time.time() - t0:.0f}s", flush=True)
    if not ok:
        raise SystemExit(f"{label}: graph does not reproduce the checkpoint")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=common.DEFAULT_CKPT,
                    help="the checkpoint; its arch drives which heads the "
                         "bundle carries")
    ap.add_argument("--encoder", default=encoders.DEFAULT,
                    choices=sorted(encoders.SPECS))
    ap.add_argument("--basis", default=common.BASIS_PATH,
                    help="the frozen PCA basis npz")
    ap.add_argument("--out", default=str(ROOT / "goblinscript" / "bundle"),
                    help="the bundle directory a full export writes")
    ap.add_argument("--pack", default=None,
                    help="the pack's name: the release that ships the model "
                         "(v0.5.1), because that is what a person compares "
                         "by. A full export names its default pack with it, "
                         "and --into names the pack it adds. Default: the "
                         "checkpoint's run directory name")
    ap.add_argument("--label", default="",
                    help="one line about the pack, shown beside its name")
    ap.add_argument("--into", default=None, metavar="DIR",
                    help="add the checkpoint's pack to the format-5 bundle "
                         "in DIR instead of writing a new bundle. The "
                         "perception stays as it is and is never re-traced, "
                         "after the checkpoint's basis, encoder, grid, dim "
                         "and row clock match it; the pack's three graphs "
                         "are traced and verified, and the manifest gains "
                         "the pack. The default pack stays what it was")
    ap.add_argument("--grid-fps", type=float, default=30.0,
                    help="the perception's decode frame grid; with "
                         "--tubelet-stride/--alignments it must reproduce "
                         "the checkpoint's stamped row rate")
    ap.add_argument("--tubelet-stride", type=int, default=2,
                    help="the perception's tubelet stride")
    ap.add_argument("--alignments", type=int, default=0,
                    help="alignments per tubelet (0 = 2k, one row per "
                         "decoded frame)")
    ap.add_argument("--cpu-attn", action="store_true",
                    help="export attention in the separate-qkv layout instead "
                         "of the packed one. 2.4x slower on DirectML (144 ms "
                         "per forward against 59) and slightly further from "
                         "production -- but ORT has no CPU kernel for packed "
                         "attention, so this is the bundle to build when you "
                         "need `--cpu`, the fp32 reference a suspect draft "
                         "gets bisected against. Budget for it: 4.5 s per CPU "
                         "forward, 74x the shipped GPU path, at a 1.7 GB peak "
                         "working set (down from 10 GB -- ORT's CPU attention "
                         "kernel does not materialize either)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the ORT-vs-torch check (it is the point)")
    ap.add_argument("--device", default="cuda",
                    help="CUDA device the ENCODERS are traced on. There is no "
                         "cpu option: fp16 conv/attention has no CPU kernel "
                         "in torch, so a cpu trace silently produces an fp32 "
                         "bundle -- a different artifact than the one that "
                         "ships, wearing the same manifest. Trace fp32 with "
                         "--enc-fp32, where it is the thing you asked for")
    ap.add_argument("--enc-fp32", action="store_true",
                    help="trace the vision graphs in fp32 instead of the "
                         "shipped fp16 -- twice the bundle, ~6.4x slower on "
                         "DirectML. A debugging axis (numerics attribution "
                         "against a range-proof graph), never what ships")
    return ap


def run(args):
    """Export under a parsed ``build_parser()`` namespace."""
    # torch.onnx's exporter prints a few non-ASCII glyphs (a success tick) as
    # it traces each graph. On Windows a PIPED stdout defaults to the cp1252
    # locale codec, which can't encode them and crashes the export mid-graph
    # (a bare console happens to be UTF-8, so it only bites when the output is
    # piped or redirected). Force UTF-8 so the run is invariant to that.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if not args.device.startswith("cuda"):
        raise SystemExit(
            f"--device {args.device}: the export needs a CUDA device. A cpu "
            f"trace cannot emit the fp16 vision graphs deploy runs, and the "
            f"fp32 bundle it would write instead is indistinguishable from a "
            f"real one until something reads enc_dtype. Use --enc-fp32 on a "
            f"GPU if fp32 graphs are what you want.")
    if not torch.cuda.is_available():
        raise SystemExit("--device cuda but no GPU is visible.")
    dev = args.device
    edtype = torch.float32 if args.enc_fp32 else torch.float16

    out = Path(args.out)
    # a pack added to a bundle that stands: the perception is the
    # bundle's, read from its manifest, and the export traces the pack alone
    into = Path(args.into) if args.into else None
    full = into is None
    prior = None
    if into is not None:
        prior = json.loads((into / "manifest.json").read_text("utf-8"))
        if prior.get("bundle_version") != BUNDLE_VERSION:
            raise SystemExit(
                f"{into} is bundle format v{prior.get('bundle_version')}; "
                f"a pack joins a format v{BUNDLE_VERSION} bundle")
        if prior["encoder"] != args.encoder:
            raise SystemExit(f"{into} carries encoder {prior['encoder']}, "
                             f"not {args.encoder}")
        for k in ("grid_fps", "tubelet_stride", "alignments"):
            setattr(args, k, type(getattr(args, k))(prior[k]))
        out = into
    pack_name = args.pack or Path(args.ckpt).resolve().parent.name
    if prior is not None and pack_name in prior["packs"]:
        raise SystemExit(f"{into} already carries a pack named {pack_name}")
    basis = Path(args.basis)
    _kind, ident, res, patch, _sc, tap = encoders.SPECS[args.encoder]
    grid = res // patch

    # ---- the trained model ------------------------------------------------ #
    model, ck = load_model(args.ckpt, "cpu")
    arch = ck["arch"]
    dim = int(arch["dim"])
    print(f"checkpoint {args.ckpt} (epoch {ck['epoch']}, v_std "
          f"{ck['v_std']:.2f}, basis {ck.get('basis_id')})", flush=True)

    # ---- the frozen perception -------------------------------------------- #
    zb = np.load(basis)
    bid = common.basis_id(zb["mean"], zb["components"], zb["evals"])
    if ck.get("basis_id") and ck["basis_id"] != bid:
        raise SystemExit(
            f"checkpoint was trained on basis {ck['basis_id']} but {basis} is "
            f"{bid} -- the bundle would encode features the head never saw")
    if prior is not None:
        # the pack rides the bundle's perception, so the checkpoint must
        # have been trained on exactly it
        for k, v in (("basis_id", bid), ("grid", grid), ("dim", dim)):
            if prior[k] != v:
                raise SystemExit(
                    f"{into} carries {k} {prior[k]} and the checkpoint "
                    f"{v} -- the pack would read features it never saw")
    # Every refusal that reads only the checkpoint and the flags runs HERE,
    # before the encoder loads and before anything touches the bundle: a
    # refusal after the first graph is written would leave the prior
    # bundle's manifest beside graphs from another checkpoint.
    k = int(args.tubelet_stride)
    row_hz = args.grid_fps * (int(args.alignments) or 2 * k) / (2.0 * k)
    ck_hz = ck.get("row_hz")
    if ck_hz is not None and abs(float(ck_hz) - row_hz) > 0.01 * row_hz:
        raise SystemExit(
            f"checkpoint was trained at {float(ck_hz):g} rows/s but "
            f"--grid-fps/--tubelet-stride/--alignments describe a "
            f"{row_hz:g} rows/s grid -- the bundle would run every row-"
            f"indexed constant at the wrong duration")
    if "env_ctx" not in arch:
        raise SystemExit(
            "checkpoint arch carries no env_ctx -- refusing to guess: the "
            "Rust envelope buffer must match the trained context exactly")
    tok_mu = torch.from_numpy(zb["mean"]).float()
    Wp = torch.from_numpy(
        zb["components"][:, :dim]
        / np.sqrt(np.maximum(zb["evals"][:dim], 1e-6))).float()

    enc = tn = None
    if not full:
        print(f"perception: {into}'s own ({prior['encoder']}, basis "
              f"{prior['basis_id']}, {prior['row_hz']:g} rows/s), not "
              f"re-traced; adding pack {pack_name}", flush=True)
    else:
        enc, tn = load_perception(args, ident, tap, dev, edtype, res, grid)

    # ---- export ----------------------------------------------------------- #
    # Graphs are traced and verified in a sibling staging dir and promoted
    # into --out file by file only once everything passed, so an abort or
    # a failed verify leaves the bundle already in --out untouched. A pack
    # added to a standing bundle stages beside its own directory.
    stage = out.parent / (out.name + ".part") if full \
        else out / "packs" / (pack_name + ".part")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    pstage = stage / "packs" / pack_name if full else stage
    pstage.mkdir(parents=True, exist_ok=True)
    print("exporting:", flush=True)
    torch.manual_seed(0)
    files = {}
    pfiles = {}

    f_enc = torch.randint(0, 255, (1, CLIP_LEN, res, res, 3),
                          dtype=torch.uint8, device=dev)
    f_tn = torch.randint(0, 255, (1, TN_WINDOW, 27, 48, 3), dtype=torch.uint8)
    if full:
        files["encoder"], files["transnet"], r_enc, r_tn = trace_perception(
            enc, tn, tok_mu, Wp, grid, res, edtype, dev, stage, f_enc, f_tn,
            packed=not args.cpu_attn)
    pg = pack_graphs(model, arch, args, dim, grid, pstage, pfiles)
    perc = dict(f_enc=f_enc, f_tn=f_tn, r_enc=r_enc, r_tn=r_tn) if full \
        else None
    finish(args, full, prior, out, stage, pstage, pack_name, files, pfiles,
           pg, perc, model, ck, arch, bid, row_hz, res, grid, dim, edtype)


BUNDLE_VERSION = 5     # bundle.rs BUNDLE_VERSION: one perception, named
                       # packs; an older binary refuses this rather than
                       # reading one pack's manifest as the whole


def load_perception(args, ident, tap, dev, edtype, res, grid):
    """The frozen encoder on the device and TransNetV2 on the CPU, ready to
    trace."""
    hub = torch.hub.load("facebookresearch/vjepa2", ident, trust_repo=True)
    enc = (hub[0] if isinstance(hub, (tuple, list)) else hub)
    if edtype is torch.float16:
        _patch_rope_for_onnx(enc)
    enc = enc.to(dev).to(edtype).eval()
    if tap is not None:
        # A mid-tap encoder is a DIFFERENT feature space, and the checkpoint was
        # trained on it: exporting the full-depth forward would produce a graph
        # that loads, runs, and feeds the head features it has never seen. The
        # tap is selected the way `encoders.Encoder` selects it, off the
        # encoder's own trained tap list.
        taps = list(getattr(enc, "hierarchical_layers", []))
        if tap not in taps:
            raise SystemExit(
                f"{args.encoder}: block index {tap} is not one of the "
                f"encoder's trained taps {taps}")
        enc.out_layers = [tap]
    # TransNetV2 (boundary detector) exports fp32 on CPU regardless of --device:
    # the net is 7.6M params, and its conv3d has no CUDA slow-path kernel under
    # the constructor's deterministic default, which the load helper undoes.
    import transnetv2_pytorch as _tnp
    from transnetv2_pytorch import TransNetV2
    tn = TransNetV2(device="cpu")
    tn.load_state_dict(torch.load(
        Path(_tnp.__file__).parent / "transnetv2-pytorch-weights.pth",
        map_location="cpu"))
    tn.eval()
    torch.use_deterministic_algorithms(False)
    print(f"encoder {args.encoder} ({res}px, grid {grid}) traced on {dev} in "
          f"{str(edtype).split('.')[-1]} + TransNetV2 (7.6M) fp32 on cpu",
          flush=True)
    return enc, tn


def trace_perception(enc, tn, tok_mu, Wp, grid, res, edtype, dev, stage,
                     f_enc, f_tn, packed=True):
    """The two perception graphs into ``stage``: (p_enc, p_tn, r_enc,
    r_tn), the torch references beside the files for the verify legs."""
    m_enc = VjepaPca(enc, tok_mu, Wp, grid, res, edtype).to(dev).eval()
    p_enc = stage / "vjepa_pca.onnx"
    with torch.no_grad():
        r_enc = m_enc(f_enc)
    # The frame axis is NOT dynamic and cannot be: V-JEPA 2.1's rotary embedding
    # guards on its token count (576 * T/2 == 9216), so torch.export specializes
    # T=32 whatever Dim you hand it. GoblinScript covers a clip's short tail by
    # sliding the window back over the last 32 real frames instead.
    with _fp16_safe_sdpa():
        export(m_enc, (f_enc,), p_enc, ["frames"], ["lat"])
    # ...then hand the attention back to a fused kernel. Runs BEFORE verify(),
    # so the export's own torch check measures the graph that actually ships.
    fuse_attention(p_enc, packed=packed)

    # TransNetV2 exports via the TorchScript path (its color-histogram scatter
    # trips the dynamo exporter on a console-encoding glyph, not a graph issue);
    # a fixed 100-frame window, fp32.
    m_tn = TransNetProb(tn).eval()
    p_tn = stage / "transnet.onnx"
    with torch.no_grad():
        r_tn = m_tn(f_tn)
    torch.onnx.export(m_tn, (f_tn,), str(p_tn), dynamo=False,
                      input_names=["frames"], output_names=["prob"],
                      opset_version=17)
    strip_allowzero(p_tn)
    print(f"  transnet.onnx: {p_tn.stat().st_size / 1e6:.0f} MB", flush=True)
    return p_enc, p_tn, r_enc, r_tn


def pack_graphs(model, arch, args, dim, grid, pstage, pfiles):
    """The pack's three graphs into ``pstage`` (``pfiles`` names them):
    the head, the mask net's gate and the envelope step. Returns the
    fixtures and torch references the verify legs and the manifest read."""
    Wdim = torch.export.Dim("W", min=32, max=4096)
    # Latents are WHITENED (~unit variance, |x| < 8), so the int8 cache holds
    # roughly N(0, 1/scale). Verifying on uniform int8 instead would be an 8
    # sigma input: the mask net saturates, the TCN amplifies, and the graph
    # reads as broken (max|d| ~ 4) when it is exact on anything real.
    f_x = (torch.randn(1, 256, dim, grid, grid) / INT8_SCALE) \
        .round().clamp(-127, 127).to(torch.int8)
    f_cut = torch.zeros(1, 256)
    f_cut[0, 100] = 1.0
    m_head = Head(model, jepa_infer.POS_TEMP).eval()
    p_head = pstage / "head.onnx"
    with torch.no_grad():
        r_head = m_head(f_x, f_cut)
    names = ["vmarg"]
    if model.pos_head:
        names.append("level")
    if model.ext_head:
        names += ["ext_lo", "ext_hi"]
    if model.plat_head:
        names += ["plat_top", "plat_bot"]
    if model.rev_head:
        names += ["rev_top", "rev_bot"]
    names.append("conf")
    names.append("h")
    export(m_head, (f_x, f_cut), p_head, ["x_i8", "cut"], names,
           dynamic_shapes={"x_i8": {1: Wdim}, "cut": {1: Wdim}})
    pfiles["head"] = p_head

    # the viewport's graph: one row of the head's OWN int8 fixture, so the two
    # are checked on the same distribution
    f_m = f_x[0, :1]
    m_mask = MaskGate(model).eval()
    p_mask = pstage / "mask.onnx"
    export(m_mask, (f_m,), p_mask, ["x_i8"], ["gate"])
    pfiles["mask"] = p_mask

    p_env = None
    env_state = int(arch["env_ctx"])
    if model.gen_env:
        tcn = int(arch["tcn_ch"])
        f_h = torch.randn(1, tcn, 1)
        f_buf = torch.zeros(1, env_state)
        flow = bool(getattr(model, "env_flow", False))
        # the flow head SAMPLES from a base draw, so its graph takes one
        # more input; the categorical head publishes an expectation and
        # takes the two it always did
        f_e0 = torch.full(
            (1, 1), float(common.env_base_draw(0, model.env_seed)))
        m_env = EnvStep(model, flow).eval()
        env_args = (f_h, f_buf, f_e0) if flow else (f_h, f_buf)
        env_ins = ["h_t", "state", "e0"] if flow else ["h_t", "state"]
        p_env = pstage / "env_step.onnx"
        with torch.no_grad():
            r_env = m_env(*env_args)
        export(m_env, env_args, p_env, env_ins, ["env", "state_next"])
        pfiles["env_step"] = p_env
    env_feed = None
    if p_env is not None:
        env_feed = {"h_t": f_h.numpy(), "state": f_buf.numpy()}
        if flow:
            env_feed["e0"] = f_e0.numpy()
    return dict(p_head=p_head, f_x=f_x, f_cut=f_cut, r_head=r_head,
                p_mask=p_mask, f_m=f_m, p_env=p_env, env_feed=env_feed,
                r_env=(list(r_env) if p_env is not None else None),
                env_state=env_state)


def finish(args, full, prior, out, stage, pstage, pack_name, files, pfiles,
           pg, perc, model, ck, arch, bid, row_hz, res, grid, dim, edtype):
    """Verify every traced graph, write the format-5 manifest and promote
    the staged files: a full bundle whole, or one pack into a bundle that
    stands."""
    # ---- verify ----------------------------------------------------------- #
    if not args.no_verify:
        print("verifying (ORT vs torch):", flush=True)
        if full:
            # The encoder tolerance that matters is the int8 step its
            # latents are quantized to downstream (8/127 = 0.063): anything
            # under that is invisible to the model by construction. fp16
            # accumulation order also differs between torch-on-CUDA and
            # ORT-on-CPU, so this is a correctness check, not a bitwise one.
            # A packed-attention encoder has no CPU kernel, so it is checked
            # on the GPU it targets; the separate-qkv bundle keeps the CPU
            # reference.
            verify(files["encoder"], {"frames": perc["f_enc"].cpu().numpy()},
                   [perc["r_enc"]], "vjepa_pca", 0.0, corr_gate=0.999,
                   eps=None if args.cpu_attn
                   else ["DmlExecutionProvider", "CPUExecutionProvider"])
            verify(files["transnet"], {"frames": perc["f_tn"].numpy()},
                   [perc["r_tn"]], "transnet", 1e-3)
        f_x, f_cut, f_m = pg["f_x"], pg["f_cut"], pg["f_m"]
        verify(pg["p_head"], {"x_i8": f_x.numpy(), "cut": f_cut.numpy()},
               pg["r_head"], "head", 1e-3)
        # this graph must agree with the gate the HEAD pools with, so it is
        # checked against the frontend's own gate rather than against itself
        with torch.no_grad():
            _h0, g_ref = model.frontend(f_x[:, :1].float() * INT8_SCALE,
                                        f_cut[:, :1])
        verify(pg["p_mask"], {"x_i8": f_m.numpy()}, [g_ref[0]], "mask", 1e-4)
        if pg["p_env"] is not None:
            verify(pg["p_env"], pg["env_feed"], pg["r_env"], "env_step", 1e-4)
    env_state = pg["env_state"]

    # ---- manifest --------------------------------------------------------- #
    # One perception at the top level, and every model as a named pack
    # under it: a person picks a pack, drafts, picks another, and pays the
    # encode once, because the latent cache keys on the perception alone.
    if full:
        top = {
            # 5: one perception, named packs. Must match bundle.rs
            # BUNDLE_VERSION -- an older binary refuses this rather than
            # reading a pack's fields off the top level.
            "bundle_version": BUNDLE_VERSION,
            "basis_id": bid,
            "encoder": args.encoder,
            "enc_dtype": "fp16" if edtype is torch.float16 else "fp32",
            # which MultiHeadAttention input layout the encoder was fused
            # with. "packed" is 2.4x faster on DirectML and has NO CPU
            # kernel, so Rust reads this to refuse `--cpu` with an
            # explanation instead of letting the run die on an opaque
            # kernel error eight minutes in.
            "attn": "separate" if args.cpu_attn else "packed",
            "graphs": {k: v.name for k, v in files.items()},
            # perception config -- Rust reproduces these exactly. grid_fps
            # is the DECODE frame rate; row_hz is the latent ROW rate the
            # styling and every row->ms conversion ride
            # (grid_fps * alignments / 2k).
            "transcode": {"height": 480, "fps": 30.0, "crf": 23,
                          "preset": "medium"},
            "grid_fps": args.grid_fps, "enc_res": res, "grid": grid,
            "dim": dim,
            "clip_len": CLIP_LEN,
            "tubelet_stride": int(args.tubelet_stride),
            "alignments": (int(args.alignments)
                           or 2 * int(args.tubelet_stride)),
            "row_hz": row_hz,
            "int8_scale": INT8_SCALE,
            # boundary detector (boundaries.py transnet constants)
            "transnet": {"input_h": 27, "input_w": 48, "window": TN_WINDOW,
                         "step": 50, "thr": 0.5, "min_gap_s": 0.5},
            "packs": {},
            "default": pack_name,
        }
    else:
        top = dict(prior)
    pack = {
        "label": args.label,
        # repo-relative when the checkpoint lives inside the tree: a
        # bundle carries no absolute path, and parity resolves it
        # against the repository root
        "checkpoint": ckpt_rel(args.ckpt),
        "epoch": int(ck["epoch"]),
        "graphs": {k: v.name for k, v in pfiles.items()},
        # head config -- chunk/min_chunk are the jepa_infer forward windows
        # (common.DECODE_CHUNK_S / 2 s) restated on the bundle's row grid,
        # so both decoders chunk at the same wall-clock length
        "v_std": float(ck["v_std"]),
        "chunk": common.rows_at(common.DECODE_CHUNK_S, row_hz),
        "min_chunk": common.rows_at(2.0, row_hz),
        # rows of real context forwarded each side of a chunk and cut from
        # the output (common.DECODE_CTX_S): kept rows carry a warm receptive
        # field and a warm env AR history across chunk boundaries
        "ctx": common.rows_at(common.DECODE_CTX_S, row_hz),
        "tcn_ch": int(arch["tcn_ch"]),
        # the envelope graph's recurrent state, carried opaque by Rust.
        # members and joint are format-5 fields the reader checks;
        # nothing ships a vote, so they are constants here
        "members": 1,
        "env_state": env_state,
        "joint": False,
        # the flow envelope's contract: heads.rs feeds env_step a base
        # draw per row, so it needs the seed the draw is keyed by, the
        # support it spans and the fact that the graph wants the input at
        # all. A categorical bundle carries env_flow false and no e0
        "env_flow": bool(getattr(model, "env_flow", False)),
        "env_flow_steps": int(getattr(model, "env_flow_steps", 0)),
        "env_seed": int(getattr(model, "env_seed", common.ENV_DRAW_SEED)),
        "env_base_hi": float(common.ENV_BASE_HI),
        # attention heads mask.onnx emits. Recorded, not read: the viewport
        # takes the count off the graph's own output shape and averages
        # whatever it finds, so no head count is a special case.
        "mask_heads": int(model.heads),
        "heads": {"pos": bool(model.pos_head), "env": bool(model.gen_env),
                  "ext": bool(model.ext_head), "plat": bool(model.plat_head),
                  "rev": bool(model.rev_head), "period": False},
        # styling knobs, read from jepa_infer and common rather than
        # restated: the bundle and the Python decoder are one decode,
        # and a literal here goes stale silently the first time the
        # constant moves
        "pos_temp": jepa_infer.POS_TEMP, "still_eps": jepa_infer.STILL_EPS,
        "ext_snap": jepa_infer.EXT_SNAP,
        # the amplitude bound and the hedge exponent it loosens on
        "amp_cap_x": float(jepa_infer.AMP_CAP_X),
        "amp_cap_f0": AMP_CAP_F0,
        "env_gain_p": ENV_GAIN_P,
        "plat_thr": jepa_infer.PLAT_THR, "plat_lo": jepa_infer.PLAT_LO,
        "plat_peak": jepa_infer.PLAT_PEAK,
        "plat_veto": jepa_infer.PLAT_VETO,
        "plat_rail_track": True,
        "plat_soft": [float(x) for x in jepa_infer.PLAT_SOFT],
        "plat_shift_cap": jepa_infer.PLAT_SHIFT_CAP,
        "rev_snap_s": float(common.REV_SNAP_S),
        "subframe": jepa_infer.SUBFRAME,
        # The reversal SEGMENTATION contract. 'viterbi' replaces the vmarg
        # crossings with the alternating event decode over the rev head and
        # is what fast capture rides on; the carrier is untouched either
        # way. Every duration here is SECONDS -- Rust turns them into rows
        # once, against the row clock it is running on. The
        # emission prior is NOT a constant: it is fitted per clip so the
        # decoded event count matches the head's own probability mass over
        # the first bias_fit_s seconds -- deploy has no script to fit
        # against, and a bias of 0 under-emits badly.
        "rev_source": (jepa_infer.REV_SOURCE if bool(model.rev_head)
                       else "cross"),
        "rev_smooth_s": float(common.REV_SMOOTH_S),
        "rev_gap_s": float(common.EVENT_GAP_S),
        "rev_gap_prior": bool(jepa_infer.REV_GAP_PRIOR and model.rev_head),
        "rev_gap_k": 0.0,
        "period_cap_s": PERIOD_CAP_S,
        "speed_ref_s": float(jepa_infer.SPEED_REF_S),
        "bias_fit_s": float(BIAS_FIT_S),
    }
    man = top
    man["packs"] = dict(man["packs"], **{pack_name: pack})
    man_tmp = stage / "manifest.json"
    man_tmp.write_text(json.dumps(man, indent=2), "utf-8")

    # ---- promote ---------------------------------------------------------- #
    # graphs first, the manifest that names them last
    if full:
        out.mkdir(parents=True, exist_ok=True)
        (out / "packs" / pack_name).mkdir(parents=True, exist_ok=True)
        for p in files.values():
            os.replace(p, out / p.name)
        for p in pfiles.values():
            os.replace(p, out / "packs" / pack_name / p.name)
    else:
        dst = out / "packs" / pack_name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for p in pfiles.values():
            os.replace(p, dst / p.name)
    os.replace(man_tmp, out / "manifest.json")
    shutil.rmtree(stage)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\nbundle -> {out}  ({total / 1e6:.0f} MB; packs "
          f"{sorted(man['packs'])}, default {man['default']})", flush=True)
    print(json.dumps(pack["heads"]), flush=True)


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
