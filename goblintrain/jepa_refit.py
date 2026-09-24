"""Head refits on a FROZEN trunk -> band probes + stillness envelope + dwells.

Grafts every post-hoc head onto a copy of the source checkpoint in one
pass. The trunk never updates, so the phase gate is the source
checkpoint's by construction -- gradients into the trunk and even the
shared global grad-norm clip were each measured to jitter gate phase,
so the refits get no path into the trunk of any kind. The two refits
are independent given the frozen features: each batch's feature
forward (the expensive part) is computed once and serves both.

* Band-extent probes (``arch.ext_head``): two 1x1-conv CE heads on the
  local band floor / ceiling (rolling 31-frame min/max of the script
  position, ~2 s). jepa_infer's composed styling uses them as pose
  rails: a stroke endpoint within EXT_SNAP of the predicted edge
  stretches to it (scripters tap the extremes at reversals; level +-
  env/2 stops short). Scripted rows only. The expectation collapse the
  decode runs (softmax(logits/pos_temp)@centers) is reach-calibrated at
  the deploy temperature -- measured 2026-07-29; the compression that
  motivated a collapsed-value term exists only at temp 1.0.
* Stillness envelope: ``env_emb`` + ``head_e`` warm-started from the
  source checkpoint (the refit perturbs a trained head, it does not
  relearn amplitude) and retrained with an extra row weight on
  quiet-target rows (``--hold-w``; row weighting, not class balance --
  the envelope's high bins are ~0% of rows and saturate an
  inverse-frequency weight cap, crushing the quiet bin), reinstated
  flat-gap holds (``--hold-gaps``; envelope-only -- trunk-level
  reinstatement is REJECTED, it breaks the gate) and self-conditioning
  (``--self-p``: condition on the head's own decode to visit the
  loud-history/quiet-target regime free-run needs at hold onsets,
  which teacher forcing almost never does).
* Trapezoid-dwell head (``arch.plat_head``, ``--plat-head``): 3-class CE
  (none/top/bottom) over ``common.dwell_labels``. jepa_infer's level lock
  reads it to pin a predicted dwell's local mean to the band rail, ripple
  intact. It is a DECODE-time signal and nothing else consumes it, so the
  trunk has no business carrying it -- and carrying it costs: co-trained
  through the shared trunk its off-phase loss still buys its dwells with
  phase (measured; a small screen with capacity to spare showed no such
  trade, so a screen pass cannot clear a head that touches the trunk).
  Refit here, it cannot.
* Reversal-event head (``arch.rev_head``, ``--rev-head``): 3-class CE
  (none/peak/valley) over ``common.reversal_labels``. jepa_infer's
  ``--rev-snap`` reads it to re-localize composed-styling crossings --
  a fast/mid reversal-TIMING lever (dt 34->26 / 30->22 ms at snap 2;
  the slow band is feature-limited and does not move). Decode-time
  signal, same frozen-trunk reasoning as the dwell head.

Supervision replicates training exactly: seg_split val segments (plus
margins) and unscripted gap rows supervise nothing, so no head ever
sees a val row. Defaults reproduce the released head stack
(common.DEFAULT_CKPT, the checkpoint the goblinscript bundle is
exported from) from its shipped bare trunk
(weights/checkpoints/<release>-trunk.pt). jepa_train calls
``train_heads`` on its own frozen best checkpoint after training,
leaving the pre-graft trunk as jepa_trunk.pt, so a fresh run comes out
deploy-ready; ``goblintrain train --from <release>`` refits a shipped
trunk.

Usage:
    python goblintrain.py train <project> --from v0.6.0
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common
from . import h0_store
from .jepa_infer import SPEED_REF_S
from .jepa_train import (IDS_DEFAULT, READ_POOL, JepaClip, envelope,
                        load_model, masked_ce, prefetched, seg_split)


# Iterations of the vectorized fixed-point decode that stands in for the
# free-run AR envelope: iterating e <- decode(h, prev(e)) from the
# teacher seed makes every row's history self-consistent out to this
# many rows (~1 s; the 0.6 s envelope's AR memory is shorter) without
# the sequential per-row loop, and each pass is two 1x1 convs over the
# window -- trunk-free, so the loss stays io-bound. The gain head trains
# and reads out under this conditioning because deploy's buffer carries
# the base decode's own values, never the teacher's.
ENV_FREE_ITERS = 32

# The pair term's reach: the longest spacing that still counts as CLOSE
# for a slow passage. A slow-band half-stroke runs a few tenths of a
# second, so every lag out to here is a spacing the author could have
# written and the head can be asked not to crowd. The reach starts at
# the refractory's own g + 1 rows, the closest pair the decode can
# emit, so the head's temporal blur is not read as a crowded pair. Refit-side only --
# nothing downstream reads it, so it is converted per grid here rather
# than shared through common.py.
def band_targets(clip, k=None, cut_aware=False):
    """Attach the probes' targets: local band floor/ceiling, the rolling
    ~2 s min/max of the script position -- the same wall-clock window as
    the level target, on this cache's row grid (31 rows at 15 rows/s,
    63 at 30).

    ``cut_aware`` takes the rolling window INSIDE each shot. Run across a
    cut, a 2 s window mixes two scenes for a second either side, so the
    target ramps where the scene steps and the head learns the ramp."""
    k = common.level_pool_rows(clip.times_ms) if k is None else k
    if cut_aware:
        clip.band_hi = common.segment_pool1d(clip.pos, clip.shot_edges, k,
                                             "max")
        clip.band_lo = common.segment_pool1d(clip.pos, clip.shot_edges, k,
                                             "min")
        return
    clip.band_hi = F.max_pool1d(clip.pos[None], k, stride=1,
                                padding=k // 2)[0]
    clip.band_lo = -F.max_pool1d(-clip.pos[None], k, stride=1,
                                 padding=k // 2)[0]


def _env_publish(model, head_e, head_ef, steps, z, rows=None, seed=None):
    """The envelope this refit's checkpoint will PUBLISH for conditioning
    ``z`` -- the flow sample when a flow head is being fit, the bin
    expectation otherwise. Every val column reads deploy's own quantity
    through here rather than whichever head happens to exist."""
    if head_ef is None:
        return model._expect(head_e(z), model.env_centers)
    w = z.shape[-1]
    r = np.arange(w) if rows is None else rows
    e0 = torch.as_tensor(common.env_base_draw(
        r, common.ENV_DRAW_SEED if seed is None else seed),
        dtype=z.dtype, device=z.device).expand(z.shape[0], w)
    e = e0
    n = max(1, int(steps))
    for i in range(n):
        tau = torch.full_like(e, i / n)
        e = e + head_ef(torch.cat(
            [z, e.unsqueeze(1), tau.unsqueeze(1)], dim=1))[:, 0] / n
    return e.clamp(0.0, common.ENV_BASE_HI)


def train_heads(model, clips, sups_ext, sups_env, items, gen, *, v_std,
                win, batch, epochs, lr, bins, hold_w, quiet_thr, self_p,
                env_noise, env_drop, device, weight_decay=1e-4,
                ext_init=None, env_init=None,
                sups_plat=None, plat_w=0.3, plat_weight=None,
                plat_mlp=True, plat_mlp_ch=64, sups_rev=None, rev_w=0.3,
                rev_weight=None, rev_cnt_w=0.0,
                label_smooth_s=common.LABEL_SMOOTH_S,
                env_flow=False, env_flow_steps=2,
                resume_path=None,
                val_items=None,
                vsups_ext=None, vsups_env=None, vsups_plat=None,
                vsups_rev=None):
    """Train the refits on the FROZEN ``model`` (eval mode, no grad into
    it): probe CE on ``sups_ext`` rows, quiet-weighted self-conditioned
    envelope CE on ``sups_env`` rows, and -- when ``sups_plat`` is given --
    the trapezoid-dwell CE on those rows. One feature forward per batch;
    host batch staging (memmap reads + stacking + pinning) runs through
    jepa_train's ``prefetched()`` pipeline so it overlaps the GPU step --
    at corpus scale each refit epoch re-streams the whole latent cache.
    ``ext_init``/``env_init`` are (state_dict, state_dict) warm starts;
    env defaults to the model's own envelope modules. Returns the trained
    modules (``head_plat`` is None when the dwell head is off).

    The dwell head is a DECODE-time signal -- jepa_infer's level lock
    reads it and nothing else does -- so it belongs here rather than in
    the trunk: co-trained, its off-phase loss still sends gradient
    through the shared trunk and buys its dwells with phase. Refit on
    the frozen trunk it CANNOT: the gate is the source checkpoint's."""
    ch, emb, hid = model.inp.out_channels, 32, 64
    head_xlo = nn.Conv1d(ch, bins, 1).to(device)
    head_xhi = nn.Conv1d(ch, bins, 1).to(device)
    if ext_init is not None:
        head_xlo.load_state_dict(ext_init[0])
        head_xhi.load_state_dict(ext_init[1])
    # env_ctx override: a resized AR context cannot warm-start across the
    # shape change, so the envelope pair cold-starts and the 8 epochs
    # relearn amplitude from the frozen features (the per-epoch val line
    # is the judge). None = the checkpoint's own context, warm-started --
    # the released recipe.
    env_ctx = model.env_ctx
    env_emb = nn.Conv1d(env_ctx, emb, 1).to(device)
    head_e = nn.Sequential(
        nn.Conv1d(ch + emb, hid, 1), nn.GELU(),
        nn.Conv1d(hid, model.env_bins, 1)).to(device)
    env_emb.load_state_dict(model.env_emb.state_dict()
                            if env_init is None else env_init[0])
    head_e.load_state_dict(model.head_e.state_dict()
                           if env_init is None else env_init[1])
    head_ef = None
    if env_flow:
        head_ef = nn.Sequential(
            nn.Conv1d(ch + emb + 2, hid, 1), nn.GELU(),
            nn.Conv1d(hid, 1, 1)).to(device)
    params = (list(head_xlo.parameters()) + list(head_xhi.parameters())
              + list(env_emb.parameters())
              + list((head_ef if env_flow else head_e).parameters()))
    head_plat = None
    plat_logprior = torch.zeros(3, device=device)
    if sups_plat is not None:
        head_plat = (nn.Sequential(nn.Conv1d(ch, plat_mlp_ch, 1),
                                   nn.GELU(),
                                   nn.Conv1d(plat_mlp_ch, 3, 1))
                     if plat_mlp else nn.Conv1d(ch, 3, 1)).to(device)
        params += list(head_plat.parameters())
        if plat_weight is not None:
            plat_weight = plat_weight.to(device)
            # same mechanism as the rev head's fold below: rebalanced CE
            # learns a rebalanced SCORE, and the per-class offset that
            # undoes it folds into the output bias once, after training,
            # so the saved head publishes a calibrated posterior and
            # --plat-thr/--plat-lo/--plat-peak stop moving with corpus
            # dwell frequency
            plat_logprior = plat_weight.log()
    head_rev = None
    rev_logprior = torch.zeros(3, device=device)
    if sups_rev is not None:
        head_rev = nn.Sequential(nn.Conv1d(ch, hid, 1), nn.GELU(),
                                 nn.Conv1d(hid, 3, 1)).to(device)
        params += list(head_rev.parameters())
        if rev_weight is not None:
            rev_weight = rev_weight.to(device)
            # Class-rebalanced CE learns p proportional to w_c * p_true(c|x),
            # so the raw
            # softmax is a rebalanced SCORE and not a posterior. The offset
            # that undoes it is a constant per class, which is why it can be
            # folded into the output bias once, after training.
            rev_logprior = rev_weight.log()
    for c in clips:
        k = common.rows_at(SPEED_REF_S, c.row_hz, odd=True)
        et = np.convolve(
            np.pad(np.abs(c.vel.numpy().astype(np.float64)),
                   (k // 2,) * 2, mode="edge"),
            np.ones(k) / k, mode="valid")
        sc = c.scripted.numpy().astype(bool)
        q1, q2 = np.percentile(et[sc], [33.3, 66.7])
        c.env_band = torch.from_numpy(
            ((et > q1).astype(np.float32) + (et > q2).astype(np.float32)))
        if sups_rev is not None:
            c.rev_slow = torch.from_numpy(
                ((et <= q1) & sc).astype(np.float32))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # A refit epoch re-streams the whole corpus, so the pass is long enough
    # to be worth resuming rather than restarting: state is written after
    # every epoch and a kill costs at most the one in flight. The heads are
    # the ONLY trainable state here (the trunk is frozen), so an epoch
    # boundary is a complete checkpoint.
    named = [("head_xlo", head_xlo), ("head_xhi", head_xhi),
             ("env_emb", env_emb), ("head_e", head_e),
             ("head_ef", head_ef),
             ("head_plat", head_plat), ("head_rev", head_rev)]
    # only what a rerun can change: the constant knobs (weight decay,
    # loss weights, noise) cannot mismatch between a save and a resume
    fingerprint = {
        "ids": [c.id for c in clips], "win": win, "batch": batch,
        "epochs": epochs, "lr": lr, "quiet_thr": quiet_thr,
        "rev_cnt_w": rev_cnt_w, "env_ctx": env_ctx,
        "items": len(items),
        "heads": [n for n, m in named if m is not None],
        "h0": getattr(clips[0], "h0", None) is not None}
    start_epoch = 1
    resume_path = Path(resume_path) if resume_path else None
    if resume_path is not None and resume_path.exists():
        st = torch.load(resume_path, weights_only=False)
        if st.get("fingerprint") != fingerprint:
            raise SystemExit(
                f"refit --resume: config fingerprint mismatch, refusing.\n"
                f"  saved: {st.get('fingerprint')}\n  now:   {fingerprint}")
        for name, m in named:
            if m is not None and st["mods"].get(name) is not None:
                m.load_state_dict(st["mods"][name])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"])
        torch.set_rng_state(st["cpu_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        start_epoch = st["epoch"] + 1
        print(f"refit RESUMED from epoch {st['epoch']}; continuing at "
              f"epoch {start_epoch}", flush=True)
    f_scale = clips[0].feat_scale
    if any(c.feat_scale != f_scale for c in clips):
        raise SystemExit("mixed latent cache formats across the refit clips")

    # h0 store (h0_store.py): when the clips carry pre-computed frontend
    # rows, a batch reads those (bf16 bits in an int16 stream) and the
    # forward is the TCN alone -- ~1/150th the read volume and none of
    # the mask-net compute, behaviorally identical to streaming the
    # latents (see h0_store's equivalence scope).
    use_h0 = getattr(clips[0], "h0", None) is not None
    if use_h0:
        win_shape = (win, clips[0].h0_ch)
    else:
        win_shape = (win, clips[0].feat_dim) \
            + tuple(clips[0].feats.shape[2:])


    def prep(bt, sx=sups_ext, se=sups_env, sp=sups_plat, sr=sups_rev):
        s_of = lambda ci, i: clips[ci].starts[i]
        grab = lambda attr: torch.stack(
            [getattr(clips[ci], attr)[s_of(ci, i):s_of(ci, i) + win]
             for ci, i in bt])
        rows = lambda sups: torch.stack(
            [sups[ci][s_of(ci, i):s_of(ci, i) + win]
             for ci, i in bt]).float()
        # each window is read from disk straight into its slot of the ONE
        # pinned batch, and the reads run CONCURRENTLY: an epoch re-streams
        # the whole latent cache, and at queue depth 1 that is loader-bound
        # (io 294 s of a 448 s epoch) while the drive sits idle between
        # reads. The allocation is always full-batch sized and sliced down,
        # so the pinned cache holds one block size however the last batch
        # falls. The clip's accepted banner-mask rects are then zeroed
        # in place on it, exactly like the training windows.
        arena = torch.empty((batch,) + win_shape,
                            dtype=clips[0].h0.dtype if use_h0
                            else clips[0].feats.dtype,
                            pin_memory=device == "cuda")
        x = arena[:len(bt)]

        def _win(j_ci_i):
            j, (ci, i) = j_ci_i
            s = s_of(ci, i)
            if use_h0:      # masks and dequant are baked into the store
                clips[ci].h0.read_into(s, s + win, x[j])
            else:
                clips[ci].fwin_into(s, s + win, x[j])
        list(READ_POOL.map(_win, enumerate(bt)))
        if not use_h0:
            for j, (ci, i) in enumerate(bt):
                if clips[ci].masked:
                    clips[ci].mask_feats(x[j])
        return (x, grab("cut"), grab("band_lo"), grab("band_hi"),
                grab("vel"), rows(sx), rows(se), grab("env_band"),
                grab("plat") if head_plat is not None else None,
                rows(sp) if head_plat is not None else None,
                grab("rev") if head_rev is not None else None,
                rows(sr) if head_rev is not None else None,
                grab("rev_slow") if head_rev is not None else None)

    vprep = (lambda bt: prep(bt, vsups_ext, vsups_env, vsups_plat,
                             vsups_rev)) \
        if val_items else None

    def run_val():
        """Held-out heads, each in the unit DEPLOY consumes it in.

        The training CEs are not readable as quality: ``ext`` is the SUM of
        two rail heads (so it sits near a single head's chance line while
        being far below it), and ``rev``'s is class-rebalanced and label
        smoothed, which makes its scale arbitrary. These are the same heads
        measured as position units, position units per second, precision /
        recall and milliseconds -- the quantities composed styling actually
        spends.

        Teacher-forced and noise-free, so it measures the HEADS rather than
        jepa_infer's free-run decode, and epochs stay comparable. It draws
        no RNG and takes no gradient, so adding the pass cannot move a run's
        trajectory -- a refit with and without it trains identically.
        """
        pos_per_bin = 100.0 / (bins - 1)
        ms_per_row = 1000.0 / clips[0].row_hz
        snap_r = common.rows_at(common.REV_SNAP_S, clips[0].row_hz)
        rail = {"lo": [], "hi": []}
        env_all, env_quiet, snap = [], [], []
        s_num, s_den = np.zeros(3), np.zeros(3)
        dw = np.zeros((3, 3))          # class -> tp, fp, fn
        cnt_p = cnt_t = cnt_ps = cnt_ts = 0.0
        vb = [val_items[b0:b0 + batch]
              for b0 in range(0, len(val_items), batch)]
        for (x, cut, bl, bh, v, mx, me, eband, plb, mp, rvb, mr,
             rslow) in prefetched(vb, vprep, depth=3):
            if not use_h0:
                x = x.to(device, non_blocking=True).float()
                if f_scale != 1.0:
                    x.mul_(f_scale)
            cut, mx, me = cut.to(device), mx.to(device), me.to(device)
            bl, bh, v = bl.to(device), bh.to(device), v.to(device)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device == "cuda"):
                    if use_h0:
                        x0v = x.to(device, non_blocking=True) \
                            .view(torch.bfloat16).transpose(1, 2)
                        h = model.trunk_final(x0v)
                    else:
                        h, _g = model.features(x, cut)
                h = h.float()
                m = mx > 0
                for key, head, tgt in (("lo", head_xlo, bl),
                                       ("hi", head_xhi, bh)):
                    tb = (tgt / 100.0 * (bins - 1)).clamp(0, bins - 1)
                    e = (head(h).argmax(1).float() - tb).abs() * pos_per_bin
                    rail[key].append(e[m].cpu().numpy())

                env_t = envelope((v / v_std).clamp(-6, 6), model.env_k)
                prev = model._prev_stack(env_t, env_ctx, win)
                ev = _env_publish(model, head_e, head_ef, env_flow_steps,
                                  torch.cat([h, env_emb(prev)], 1))
                # the envelope is a SPEED -- jepa_infer integrates it over a
                # stroke's duration to get that stroke's travel -- so its
                # natural unit is position units per second, not bins.
                # BASE head only: the gain's divergence from the target is
                # intentional overshoot, measured by the slope line below
                e = (ev - env_t).abs() * v_std
                mm = me > 0
                env_all.append(e[mm].cpu().numpy())
                q = mm & (env_t < quiet_thr)
                if bool(q.any()):
                    env_quiet.append(e[q].cpu().numpy())
                # per-band through-origin slope of the PUBLISHED envelope
                # on the target under the base decode's free-run stand-in
                # (deploy's buffer carries the base value; the gain rides
                # outside the recurrence), plus the gain head's own
                # per-band mean -- trained right, the hi band's slope
                # rises by its gain while lo/mid hold their calibration
                ef = env_t
                for _ in range(ENV_FREE_ITERS):
                    zf = torch.cat([h, env_emb(model._prev_stack(
                        ef, env_ctx, win))], 1)
                    ef = _env_publish(model, head_e, head_ef,
                                      env_flow_steps, zf)
                gg = torch.ones_like(ef)
                ebd = eband.to(device)
                for bi in range(3):
                    mb = mm.float() * (ebd == bi).float()
                    s_num[bi] += float((ef * gg * env_t * mb).sum())
                    s_den[bi] += float((env_t ** 2 * mb).sum())

                if head_plat is not None:
                    ok = mp.to(device) > 0
                    # the SAVED head publishes the prior-corrected posterior
                    # (the bias fold below), so score what deploy will read
                    pr = (head_plat(h)
                          - plat_logprior.view(1, -1, 1)).argmax(1)
                    tg = plb.to(device).long()
                    for c in (1, 2):
                        dw[c] += (float(((pr == c) & (tg == c) & ok).sum()),
                                  float(((pr == c) & (tg != c) & ok).sum()),
                                  float(((pr != c) & (tg == c) & ok).sum()))

                if head_rev is not None:
                    ok = mr.to(device) > 0
                    # the SAVED head publishes the prior-corrected posterior
                    # (the bias fold below), so score what deploy will read
                    lg = head_rev(h) - rev_logprior.view(1, -1, 1)
                    tg = rvb.to(device).long()
                    p = lg.softmax(1)
                    cnt_p += float(((p[:, 1] + p[:, 2]) * ok.float()).sum())
                    cnt_t += float(((tg != 0).float() * ok.float()).sum())
                    # the slow band separately: deploy force-emits the
                    # head's sum, and a global surplus lands here
                    sl = ok.float() * rslow.to(device)
                    cnt_ps += float(((p[:, 1] + p[:, 2]) * sl).sum())
                    cnt_ts += float(((tg != 0).float() * sl).sum())
                    # --rev-snap moves a crossing to the head's LOCAL ARGMAX
                    # within its radius, so that is what to measure: an
                    # argmax over the 3 classes never fires at all (the
                    # corrected posterior puts ~90% on 'none' by base rate)
                    # and would score every epoch at zero.
                    pn = p.cpu().numpy()
                    tgn, okn = tg.cpu().numpy(), ok.cpu().numpy()
                    W = pn.shape[2]
                    for b in range(pn.shape[0]):
                        for c in (1, 2):
                            for t in np.where((tgn[b] == c) & okn[b])[0]:
                                a, z = max(0, t - snap_r), min(W, t + snap_r + 1)
                                d = int(np.argmax(pn[b, c, a:z])) + a - t
                                snap.append(abs(d) * ms_per_row)
        def pr_str(acc, c):
            tp, fp, fn = acc[c]
            p = tp / max(tp + fp, 1.0)
            r = tp / max(tp + fn, 1.0)
            return f"P{p:.2f} R{r:.2f}"

        lo_e = np.concatenate(rail["lo"]) if rail["lo"] else np.zeros(1)
        hi_e = np.concatenate(rail["hi"]) if rail["hi"] else np.zeros(1)
        ea = np.concatenate(env_all) if env_all else np.zeros(1)
        eq = np.concatenate(env_quiet) if env_quiet else np.zeros(1)
        sl = s_num / np.maximum(s_den, 1e-9)
        out = [f"  val  rail lo {lo_e.mean():4.1f} / hi {hi_e.mean():4.1f} "
               f"pos (p90 {np.percentile(np.concatenate([lo_e, hi_e]), 90):4.1f})"
               f"   env {ea.mean():5.1f} pos/s (quiet {eq.mean():5.1f})"
               f" free slope lo/mid/hi {sl[0]:.2f}/{sl[1]:.2f}/{sl[2]:.2f}"
               ]
        tail = []
        if head_plat is not None:
            tail.append(f"dwell top {pr_str(dw, 1)} bot {pr_str(dw, 2)}")
        if head_rev is not None:
            sn = np.asarray(snap) if snap else np.zeros(1)
            tail.append(f"rev |snap| {np.median(sn):3.0f} ms "
                        f"(<=1f {float((sn <= ms_per_row).mean()):.2f})"
                        f"  cnt {cnt_p / max(cnt_t, 1.0):.2f}x"
                        f" (slow {cnt_ps / max(cnt_ts, 1.0):.2f}x)")
        if tail:
            out.append("        " + "   ".join(tail))
        return "\n".join(out)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        perm = torch.randperm(len(items), generator=gen).tolist()
        batches = [[items[j] for j in perm[b0:b0 + batch]]
                   for b0 in range(0, len(perm), batch)]
        tot_x, tot_e, tot_p, tot_r, nb = 0.0, 0.0, 0.0, 0.0, 0
        cnt_pred = cnt_true = 0.0
        io_wait = [0.0]
        # depth 3, matching the train loop: each staged batch holds PINNED
        # (non-evictable) host RAM, and at corpus scale the epoch already
        # re-streams the whole int8 cache -- 8 batches racing ahead is an
        # OOM risk on this box, not extra overlap
        for (x, cut, bl, bh, v, mx, me, eband, plb, mp, rvb, mr,
             rslow) in prefetched(batches, prep, depth=3, wait=io_wait):
            if not use_h0:
                # .float() makes the writable device copy; dequant in place
                x = x.to(device, non_blocking=True).float()
                if f_scale != 1.0:
                    x.mul_(f_scale)
            cut = cut.to(device)
            mx, me = mx.to(device), me.to(device)
            bl = bl.to(device)
            bh = bh.to(device)
            v = v.to(device)
            env_t = envelope((v / v_std).clamp(-6, 6), model.env_k)
            teacher = env_t + env_noise * torch.randn(
                env_t.shape, generator=gen).to(device)
            keep = torch.rand(env_t.shape[0], generator=gen) >= env_drop
            teacher = teacher * keep.to(device).float()[:, None]
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=device == "cuda"):
                if use_h0:
                    x0 = x.to(device, non_blocking=True) \
                        .view(torch.bfloat16).transpose(1, 2)
                    h = model.trunk_final(x0)
                else:
                    h, _gate = model.features(x, cut)
            h = h.float()
            lb = (bl / 100.0 * (bins - 1)).clamp(0, bins - 1)
            hb = (bh / 100.0 * (bins - 1)).clamp(0, bins - 1)
            # a row whose script band is narrower than narrow_units is a
            # smooth passage; counted 1+narrow_w times, the rails learn
            # to sit at the level there, where a rail past the level is
            # the amplitude the snap lends a spike
            wx = mx
            l_ext = masked_ce(head_xlo(h), lb, wx) \
                + masked_ce(head_xhi(h), hb, wx)
            prev = model._prev_stack(teacher, env_ctx, win)
            if self_p > 0:
                # the self history: the head's own decode from the
                # teacher's, iterated ``self_iters`` times so the buffer
                # holds what a free run of that depth would have written
                # (one step is the standing recipe; ENV_FREE_ITERS is the
                # gain head's free-run stand-in)
                with torch.no_grad():
                    e1 = teacher
                    e1 = _env_publish(
                        model, head_e, head_ef, env_flow_steps,
                        torch.cat([h, env_emb(model._prev_stack(
                            e1, env_ctx, win))], dim=1))
                prev_self = model._prev_stack(e1, env_ctx, win)
                use = (torch.rand(env_t.shape[0], generator=gen)
                       < self_p).to(device).float()[:, None, None]
                prev = use * prev_self + (1 - use) * prev
            ez = torch.cat([h, env_emb(prev)], dim=1)
            w = me * (1.0 + hold_w * (env_t < quiet_thr).float())
            if head_ef is not None:
                # flow matching, which is plain regression: draw a base
                # point and a position on the straight path to the true
                # envelope, and regress the field onto that path's own
                # constant velocity. No noise schedule and no score
                # identity -- the only stochastic part is WHERE on the
                # path each row is supervised
                e0 = common.ENV_BASE_HI * torch.rand(
                    env_t.shape, generator=gen).to(device)
                tau = torch.rand(env_t.shape, generator=gen).to(device)
                ep = (1.0 - tau) * e0 + tau * env_t
                vf = head_ef(torch.cat(
                    [ez, ep.unsqueeze(1), tau.unsqueeze(1)], dim=1))[:, 0]
                l_env = (((vf - (env_t - e0)) ** 2) * w).sum()                     / w.sum().clamp_min(1.0)
            else:
                logits = head_e(ez)
                eb = (env_t / 6.0 * (model.env_bins - 1))                     .clamp(0, model.env_bins - 1)
                l_env = masked_ce(logits, eb, w)
            loss = l_ext + l_env
            l_plat = None
            if head_plat is not None:
                mp = mp.to(device)
                plb = plb.to(device).float()
                plat_logits = head_plat(h)
                l_plat = masked_ce(plat_logits, plb, mp, weight=plat_weight)
                loss = loss + plat_w * l_plat
            l_rev = None
            if head_rev is not None:
                mr = mr.to(device)
                rvb = rvb.to(device).float()
                rev_logits = head_rev(h)
                l_rev = masked_ce(rev_logits, rvb, mr,
                                  weight=rev_weight)
                loss = loss + rev_w * l_rev
                if rev_cnt_w > 0:
                    # COUNT calibration. The CE above supervises WHICH row
                    # is a reversal and is indifferent to how many the head
                    # believes in overall -- but `reversal_labels` marks one
                    # row per reversal, so the summed event probability IS
                    # an event count, and deploy (no script to read a rate
                    # from) fits its emission prior to exactly that sum.
                    # Three things in this recipe inflate it and none of
                    # them is visible to a per-row loss: the class
                    # rebalancing below, the label smoothing inside
                    # masked_ce, and the head's own temporal blur. This term
                    # supervises the total directly, on the PRIOR-CORRECTED
                    # posterior that the saved head will publish.
                    q = (rev_logits - rev_logprior.view(1, -1, 1)).softmax(1)
                    ev_q = q[:, 1] + q[:, 2]
                if rev_cnt_w > 0:
                    c_pred = (ev_q * mr).sum(1)
                    c_true = ((rvb != 0).float() * mr).sum(1)
                    l_cnt = (((c_pred - c_true)
                              / c_true.clamp_min(1.0)) ** 2).mean()
                    loss = loss + rev_cnt_w * l_cnt
                    cnt_pred += float(c_pred.detach().sum())
                    cnt_true += float(c_true.detach().sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_x += l_ext.item()
            tot_e += l_env.item()
            tot_p += l_plat.item() if l_plat is not None else 0.0
            tot_r += l_rev.item() if l_rev is not None else 0.0
            nb += 1
        plat_s = f" dwell CE {tot_p/max(nb,1):.4f}" \
            if head_plat is not None else ""
        rev_s = f" rev CE {tot_r/max(nb,1):.4f}" \
            if head_rev is not None else ""
        if cnt_true > 0:
            rev_s += f" cnt {cnt_pred/cnt_true:.2f}x"
        # the pair term's own handle: close-pair mass the head writes in
        # slow rows against what the script wrote there. 1.00x = spaced
        # like the author, above = crowded
        print(f"epoch {epoch}/{epochs} ext CE {tot_x/max(nb,1):.4f} "
              f"env {'MSE' if head_ef is not None else 'CE'} {tot_e/max(nb,1):.4f}{plat_s}{rev_s} "
              f"({time.time()-t0:.0f}s, io {io_wait[0]:.0f})", flush=True)
        if val_items:
            print(run_val(), flush=True)
        if resume_path is not None:
            tmp = resume_path.with_suffix(resume_path.suffix + ".tmp")
            torch.save({
                "epoch": epoch, "fingerprint": fingerprint,
                "mods": {n: (m.state_dict() if m is not None else None)
                         for n, m in named},
                # the selection so far, beside the weights it resumes
                "opt": opt.state_dict(), "gen": gen.get_state(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
            }, tmp)
            tmp.replace(resume_path)      # atomic: a kill mid-write cannot
            #                               leave a torn resume file
    if resume_path is not None and resume_path.exists():
        resume_path.unlink()      # the pass is complete; the next one is new
    if head_rev is not None and float(rev_logprior.abs().max()) > 0:
        # Fold the prior correction into the output bias, ONCE, so the saved
        # head publishes a calibrated posterior. Every consumer just takes a
        # softmax, so nothing downstream -- jepa_infer, the ONNX
        # export, goblinscript -- needs to know this happened.
        with torch.no_grad():
            head_rev[-1].bias -= rev_logprior
    if head_plat is not None and float(plat_logprior.abs().max()) > 0:
        with torch.no_grad():
            (head_plat[-1] if isinstance(head_plat, nn.Sequential)
             else head_plat).bias -= plat_logprior
    return (head_xlo, head_xhi, env_emb, head_e, head_plat, head_rev,
            head_ef)


def graft(ck, head_xlo, head_xhi, env_emb, head_e, head_plat=None,
          head_rev=None, head_ef=None, *, env_flow_steps=2, bins, recipe,
          plat_mlp=True, plat_mlp_ch=64):
    """Copy of ``ck`` with the refit heads written in (arch.ext_head on,
    arch.plat_head / arch.rev_head on when those heads were refit). The
    trunk + marginal bytes are untouched: the gate is inherited, not
    re-measured."""
    sd = dict(ck.get("model", {}))
    mods = [("head_xlo", head_xlo), ("head_xhi", head_xhi),
            ("env_emb", env_emb), ("head_e", head_e)]
    if head_plat is not None:
        # a co-trained linear head's keys don't overlap the MLP's --
        # drop them or the graft leaves stale weights behind
        for k in [k for k in sd if k.startswith("head_plat.")]:
            del sd[k]
        mods.append(("head_plat", head_plat))
    if head_rev is not None:
        for k in [k for k in sd if k.startswith("head_rev.")]:
            del sd[k]
        mods.append(("head_rev", head_rev))
    if head_ef is not None:
        for k in [k for k in sd if k.startswith("head_ef.")]:
            del sd[k]
        mods.append(("head_ef", head_ef))
    for name, mod in mods:
        for k, t in mod.state_dict().items():
            sd[f"{name}.{k}"] = t.detach().cpu()
    sd["ext_centers"] = torch.linspace(0.0, 100.0, bins)
    ck2 = dict(ck)
    ck2["model"] = sd
    arch = dict(ck["arch"], ext_head=True)
    if head_ef is not None:
        arch["env_flow"] = True
        arch["env_flow_steps"] = int(env_flow_steps)
    if head_plat is not None:
        arch["plat_head"] = True
        arch["plat_mlp"] = plat_mlp
        arch["plat_mlp_ch"] = plat_mlp_ch
    if head_rev is not None:
        arch["rev_head"] = True
    ck2["arch"] = arch
    ck2["head_refit"] = recipe
    return ck2


# The deploy heads' fixed configuration: what every shipped head set was
# refit with. A recipe sets the schedule (epochs, window, batch, learning
# rate, seed), the envelope flow, the quiet threshold and the reversal
# count weight; these it does not.
WEIGHT_DECAY = 1e-4
EXT_BINS = 21
HOLD_W = 10.0
HOLD_GAPS_S = 60.0
LABEL_SMOOTH_S = 0.2
ENV_FLOW_STEPS = 2
SELF_P = 0.5
ENV_NOISE = 0.25
ENV_DROP = 0.1
PLAT_W = 0.3
PLAT_MLP_CH = 64
REV_W = 1.0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=None, help="the project directory")
    ap.add_argument("--ids", nargs="*", default=IDS_DEFAULT)
    ap.add_argument("--ckpt", default=None,
                    help="refit SOURCE: a bare trunk checkpoint (no ext "
                         "heads); the defaults reproduce the shipped head "
                         "stack from it")
    ap.add_argument("--out", default=None,
                    help="output DIRECTORY; the grafted checkpoint is "
                         "written as <out>/model.pt")
    ap.add_argument("--win", type=int, default=None,
                    help="window rows; defaults to the WINDOW GEOMETRY "
                         "stamped on --ckpt, which is what the trunk saw")
    ap.add_argument("--stride", type=int, default=None,
                    help="window hop; defaults to the stamp on --ckpt. "
                         "Refitting at another stride changes the gradient "
                         "steps per epoch and silently fits a different "
                         "recipe")
    ap.add_argument("--val-frac", type=float, default=None,
                    help="held-out fraction. Default follows the trunk "
                         "CHECKPOINT's stamp, so the refit never supervises "
                         "rows the trunk held out")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=888)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--env-flow", action="store_true",
                    help="fit the envelope as a FLOW FIELD instead of "
                         "21 bin logits: plain regression onto the "
                         "rectified path, sampled at decode in "
                         "--env-flow-steps Euler steps. The head becomes "
                         "a distribution the decode DRAWS from rather "
                         "than one collapsed to its mean, which is the "
                         "one-to-many authorship axis the envelope "
                         "exists for. The base draw is "
                         "common.env_base_draw, a pure function of the "
                         "absolute row and a seed, so jepa_infer and "
                         "goblinscript integrate the same ODE. It "
                         "REPLACES head_e in the optimizer -- fitting "
                         "both would spend the refit on a head the "
                         "checkpoint will not read")
    ap.add_argument("--quiet-thr", type=float, default=0.3,
                    help="normalized envelope units: a row quieter than "
                         "this counts as a hold row for --hold-w")
    ap.add_argument("--rev-cnt-w", type=float, default=0.6,
                    help="weight of the reversal COUNT-calibration term: "
                         "squared relative error between the summed event "
                         "probability of a window and the number of reversals "
                         "actually in it. common.reversal_labels marks one "
                         "row per reversal, so that sum is an event count -- "
                         "and it is the only rate DEPLOY has, since with no "
                         "script there is nothing to fit the emission prior "
                         "against. Per-row CE does not constrain it: label "
                         "smoothing puts a floor on every row and the head's "
                         "temporal blur spreads each reversal over ~2.3 rows "
                         "at 30 rows/s, both invisible to a per-row loss. "
                         "The shipped recipes set 0.3, a product call "
                         "on a measured-monotone dial: more dose buys "
                         "slow-band ms timing at slow-band travel.")
    return ap


def run(args, ap=None):
    """Refit under a parsed ``build_parser()`` namespace."""
    ap = ap or build_parser()
    for name in ("dataset", "ckpt", "out"):
        if getattr(args, name) is None:
            ap.error(f"--{name} is required")
    args.ids = common.resolve_ids(args.ids, args.dataset)   # rosters by NAME
    # --out is a DIRECTORY (the graft lands at <out>/model.pt); a file
    # path here would only fail at the write, AFTER the head training
    if Path(args.out).exists() and not Path(args.out).is_dir():
        ap.error(f"--out {args.out} is a file; pass the output DIRECTORY "
                 "(the graft is written as <out>/model.pt)")
    common.cap_working_set()    # memmap streaming must not eat the box

    torch.manual_seed(args.seed)
    model, ck = load_model(args.ckpt, args.device)
    if ck["arch"].get("ext_head", False):
        raise SystemExit(f"{args.ckpt} already carries ext heads")
    if not ck["arch"].get("gen_env", False):
        raise SystemExit(f"{args.ckpt} carries no envelope head")
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"frozen trunk: {args.ckpt} (epoch {ck['epoch']})", flush=True)

    # Window geometry comes from the checkpoint: the heads are refit on the
    # windows the TRUNK saw, and a different stride changes the gradient
    # steps per epoch while still reporting a clean run. Checkpoints from
    # before the stamp fall back to the old defaults and say so, since for
    # those the caller has to know.
    for name, legacy in (("win", 384), ("stride", 192)):
        if getattr(args, name) is None:
            if name in ck:
                setattr(args, name, int(ck[name]))
            else:
                setattr(args, name, legacy)
                print(f"warning: {args.ckpt} carries no {name} stamp; "
                      f"using {legacy}. Pass --{name} if the trunk was "
                      f"trained on another geometry", flush=True)
    print(f"windows: win {args.win} stride {args.stride}", flush=True)

    # The held-out geometry comes from the checkpoint too. Refitting on rows
    # the TRUNK held out would train the deploy heads on the very material
    # the composed metrics score, so the heads -- not the trunk -- become the
    # contamination path. Matching the stamp closes it by construction.
    if args.val_frac is None:
        args.val_frac = float(ck.get("val_frac", 0.15))
        if "val_frac" not in ck:
            print(f"warning: {args.ckpt} carries no val_frac stamp; using "
                  f"{args.val_frac}. Pass --val-frac if the trunk held out "
                  f"another fraction", flush=True)
    print(f"held out: val_frac {args.val_frac}, split "
          f"'{ck.get('split', 'tail')}' -> refit supervises the rest",
          flush=True)
    # seg_split always rebuilds the CURRENT geometry (common.val_regions),
    # so a trunk trained under another one would have its heads fitted on
    # rows every downstream scorer then measures -- exactly the
    # contamination the stamp exists to close. audit and infer refuse
    # loudly; the refit must too.
    if ck.get("split", "tail") != common.VAL_SPLIT:
        raise SystemExit(
            f"{args.ckpt} was trained under the '{ck.get('split', 'tail')}' "
            f"held-out geometry but this refit would supervise the "
            f"complement of '{common.VAL_SPLIT}' -- the heads would train "
            f"on rows the scorers measure. Retrain the trunk under "
            f"'{common.VAL_SPLIT}' first.")

    clips, sups_ext, sups_env, items = [], [], [], []
    sups_plat = []
    sups_rev = []
    # the mirror image of the same split: the rows the refit does NOT
    # supervise are the ones its per-epoch readout is measured on, so the
    # heads are never scored on material they were fitted to
    val_items, vsups_ext, vsups_env = [], [], []
    vsups_plat = []
    vsups_rev = []
    for vid in args.ids:
        c = JepaClip(args.dataset, vid, args.win, args.stride,
                     feat_dir=ck.get("feat_dir", common.LATENTS_DIR),
                     gap_mask=True, masks_dir=ck.get("masks_dir"),
                     row_hz=ck.get("row_hz"))
        band_targets(c)
        tr, _segs, sup = seg_split(c, args.val_frac, args.win)
        clips.append(c)
        sup_ext = sup & c.scripted
        sups_ext.append(sup_ext)
        sups_env.append(sup & (c.scripted
                               | c.hold_rows(HOLD_GAPS_S)))
        # the dwell head sees exactly the ext rows
        c.plat = torch.from_numpy(
            common.dwell_labels(c.pos.numpy(), c.scripted.numpy(),
                                row_hz=c.row_hz))
        sups_plat.append(sup_ext)
        # the rev head too
        c.rev = torch.from_numpy(
            common.reversal_labels(c.pos.numpy(), c.scripted.numpy(),
                                   row_hz=c.row_hz,
                                   smooth_s=LABEL_SMOOTH_S))
        sups_rev.append(sup_ext)
        items += [(len(clips) - 1, i) for i in tr]

        vsup = ~sup
        vsup_ext = vsup & c.scripted
        vsups_ext.append(vsup_ext)
        vsups_env.append(vsup & (c.scripted | c.hold_rows(HOLD_GAPS_S)))
        vsups_plat.append(vsup_ext)
        vsups_rev.append(vsup_ext)
        veff = vsup & c.trainable
        val_items += [(len(clips) - 1, i) for i, s in enumerate(c.starts)
                      if int(veff[s:s + args.win].sum()) > 1
                      and c.vel[s:s + args.win][veff[s:s + args.win]]
                      .std() > 1e-3]
    print(f"{len(clips)} clips, {len(items)} train windows, "
          f"{len(val_items)} val windows", flush=True)
    lo = torch.cat([c.band_lo[sups_ext[ci]]
                    for ci, c in enumerate(clips)]).numpy()
    hi = torch.cat([c.band_hi[sups_ext[ci]]
                    for ci, c in enumerate(clips)]).numpy()
    print(f"rail targets: lo p0/p1/p5 {lo.min():.1f}"
          f"/{np.percentile(lo, 1):.1f}/{np.percentile(lo, 5):.1f}"
          f"  hi p0/p1/p5 {hi.min():.1f}/{np.percentile(hi, 1):.1f}"
          f"/{np.percentile(hi, 5):.1f} (supervised rows)", flush=True)
    assert 0.0 <= lo.min() and hi.max() <= 100.0, "rail targets off-scale"

    cnt = torch.zeros(3)
    for ci, c in enumerate(clips):
        cnt += torch.bincount(c.plat[sups_plat[ci]],
                              minlength=3).float()
    freq = cnt / cnt.sum().clamp_min(1)
    plat_weight = 1.0 / (freq + 1e-4).sqrt()
    plat_weight = plat_weight / plat_weight.mean()
    print(f"dwell labels: {freq[1]:.1%} top / {freq[2]:.1%} bottom "
          f"of supervised rows", flush=True)
    cnt = torch.zeros(3)
    for ci, c in enumerate(clips):
        cnt += torch.bincount(c.rev[sups_rev[ci]],
                              minlength=3).float()
    freq = cnt / cnt.sum().clamp_min(1)
    rev_weight = 1.0 / (freq + 1e-4).sqrt()
    rev_weight = rev_weight / rev_weight.mean()
    print(f"reversal labels: {freq[1]:.1%} peak / {freq[2]:.1%} "
          f"valley of supervised rows",
          flush=True)

    h0_store.attach(model, ck, clips, args.dataset, args.device)

    warm = {}
    gen = torch.Generator().manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)   # the per-epoch resume state
    #                                          lands here before the graft
    mods = train_heads(model, clips, sups_ext, sups_env, items, gen,
                       v_std=ck["v_std"], win=args.win, batch=args.batch,
                       epochs=args.epochs, lr=args.lr,
                       weight_decay=WEIGHT_DECAY, bins=EXT_BINS,
                       hold_w=HOLD_W, quiet_thr=args.quiet_thr,
                       self_p=SELF_P,
                       env_noise=ENV_NOISE,
                       env_drop=ENV_DROP, device=args.device,
                       sups_plat=sups_plat, plat_w=PLAT_W,
                       plat_weight=plat_weight, plat_mlp=True,
                       plat_mlp_ch=PLAT_MLP_CH,
                       sups_rev=sups_rev, rev_w=REV_W,
                       rev_weight=rev_weight, rev_cnt_w=args.rev_cnt_w,
                       label_smooth_s=LABEL_SMOOTH_S,
                       env_flow=args.env_flow,
                       env_flow_steps=ENV_FLOW_STEPS,
                       resume_path=out / "refit_resume.pt",
                       val_items=val_items, vsups_ext=vsups_ext,
                       vsups_env=vsups_env, vsups_plat=vsups_plat,
                       vsups_rev=vsups_rev,
                       ext_init=warm.get("ext"), env_init=warm.get("env"))
    recipe = {"src": Path(args.ckpt).name, "hold_w": HOLD_W,
              "quiet_thr": args.quiet_thr, "self_p": SELF_P,
              "hold_gaps": HOLD_GAPS_S, "epochs": args.epochs,
              "win": args.win, "batch": args.batch,
              "seed": args.seed,
              "label_smooth_s": LABEL_SMOOTH_S,
              "plat_head": True, "plat_w": PLAT_W,
              "plat_mlp": True,
              "rev_head": True, "rev_w": REV_W,
              "rev_class_weight": True,
              "rev_cnt_w": args.rev_cnt_w,
              "env_flow": bool(args.env_flow),
              "env_flow_steps": ENV_FLOW_STEPS}
    torch.save(graft(ck, *mods, env_flow_steps=ENV_FLOW_STEPS,
                     bins=EXT_BINS, recipe=recipe,
                     plat_mlp=True, plat_mlp_ch=PLAT_MLP_CH),
               out / "model.pt.tmp")
    (out / "model.pt.tmp").replace(out / "model.pt")   # whole or absent
    print(f"wrote {out / 'model.pt'} (frozen trunk + refit heads)",
          flush=True)


def main():
    ap = build_parser()
    run(ap.parse_args(), ap)


if __name__ == "__main__":
    main()
