"""Per-clip global script<->video lag fit (one offset per clip).

The frozen model predicts stroke velocity from video; cross-correlating that
prediction against the script velocity (both signed, same semantics) finds
the global time offset. It is measured OUT OF SAMPLE: the released model
never trained on a user's clip, so it is the reference for every clip a
project imports. (A model that trained on a clip memorized any offset and
reports ~0.)

Writes ``<project>/lag/<id>.json`` per clip. Fields:
  lag_ms      raw sub-frame fit (parabolic peak)          -- always recorded
  peak,corr0,dcorr  correlation at the peak / at 0 / gain -- always recorded
  shot_med_ms,shot_std_ms,n_shots  per-shot cluster (single author -> tight
                                   spread corroborates a real offset)
  applied_ms  the PEAK-GATED value training uses: lag_ms when the offset is
              confident + material, else 0 (protects the well-synced majority
              from the ~10ms estimate noise). ``gated`` flags a zeroed clip.
  drift_gap   median per-window peak minus the global peak. Local windows
              aligning while the clip as a whole does not means no ONE
              offset describes it, so ``lag_ms`` is a mean and
              ``applied_ms`` would be a fiction.
  drift_spread_ms  IQR of the per-window lags -- how far the offset actually
              MOVES. The DRIFT alarm needs both this and the gap.
  drift_ms_per_min,drift_total_ms,drift_t,drift_win_lags_ms  the slope and
              its significance, DESCRIBING a flagged clip; they never fire
              the alarm.

Both alarms are ALARMS: polarity and drift are recorded and reported, never
acted on. Inverting a script (``invert`` in the sidecar) or removing a clip
is a human decision; the fitter preserves ``invert`` across re-fits and
never sets it. Training reads ``applied_ms`` only.
"""
import json

import numpy as np

from . import common, forward
from .jepa_train import JepaClip
from .project import atomic_write_text

MAXF_S = 133.333333   # per-forward chunk
POLARITY_THR = -0.2   # corr at lag 0 below this => positions look inverted

# The fit and gate configuration. ``max_lag_s`` bounds the search to
# plausible authoring offsets so an inverted-polarity clip cannot alias onto
# a half-stroke-period "lag". The gate applies a lag only when the offset is
# confident (dcorr, or corroborating per-window lags) and material.
CONFIG = {
    "max_lag_s": 0.15,
    "drift_windows": 10, "drift_wide_lag_s": 2.0, "drift_min_peak": 0.3,
    "drift_min_gap": 0.10, "drift_min_spread_ms": 15.0,
    "min_peak": 0.4, "min_dcorr": 0.03, "min_abs_ms": 15.0,
    "win_agree": 0.8, "win_tol_ms": 15.0,
}


def lag_spread_ms(fit):
    """How far a clip's offset MOVES: the IQR of its per-window lags, in ms.
    None when too few windows carry usable lags."""
    w = [v for v in (fit.get("drift_win_lags_ms") or []) if v is not None]
    if len(w) < 4:
        return None
    q25, q75 = np.percentile(np.asarray(w, dtype=np.float64), [25, 75])
    return round(float(q75 - q25), 1)


def drift_alarm(fit, min_gap, min_spread_ms):
    """Does the offset move, and does it move by enough to matter?

    ``drift_gap`` answers the first question and cannot answer the second:
    a short window is easier to align than a whole clip whatever the offset
    does, so the gap is monotone in drift size AND in plain difficulty.
    ``lag_spread_ms`` answers the second in the units the defect lives in. A
    clip with too few usable windows to measure a spread alarms on the gap
    alone -- unresolvable is not the same as clean.
    """
    gap = fit.get("drift_gap")
    if gap is None or gap < min_gap:
        return False
    spread = lag_spread_ms(fit)
    return spread is None or spread >= min_spread_ms


def fit_drift(pred, script, mask, row_hz, n_win, max_lag_s, wide_lag_s,
              min_peak, global_peak):
    """Is the offset a CONSTANT or a function of time? ``drift_gap`` (median
    per-window peak minus the global peak, both at the narrow cap) is the
    trigger; the per-window lags at the wide cap describe the slope. None
    when too few windows carry usable signal."""
    T = len(pred)
    if T < n_win * 2 or n_win < 4:
        return None
    edges = np.linspace(0, T, n_win + 1).astype(int)
    need = 4 * int(round(wide_lag_s * row_hz))
    ts, lags, narrow_pks = [], [], []
    for i in range(n_win):
        lo, hi = edges[i], edges[i + 1]
        if int(mask[lo:hi].sum()) < need:
            continue
        _, npk, _ = common.xcorr_lag_subframe(pred[lo:hi], script[lo:hi],
                                              row_hz, max_lag_s,
                                              mask=mask[lo:hi])
        narrow_pks.append(float(npk))
        wl, wp, _ = common.xcorr_lag_subframe(pred[lo:hi], script[lo:hi],
                                              row_hz, wide_lag_s,
                                              mask=mask[lo:hi])
        if wp >= min_peak:
            ts.append((lo + hi) / 2.0 / row_hz)   # window centre, seconds
            lags.append(float(wl))
    if len(narrow_pks) < 4:
        return None
    out = {"drift_gap": round(float(np.median(narrow_pks)) - float(global_peak), 3),
           "drift_windows": len(narrow_pks),
           "drift_win_lags_ms": [round(v, 1) for v in lags]}
    out["drift_spread_ms"] = lag_spread_ms(out)
    if len(ts) < 4:
        return out
    t = np.asarray(ts)
    y = np.asarray(lags)
    tm, ym = float(t.mean()), float(y.mean())
    var = float(((t - tm) ** 2).mean())
    if var <= 0:
        return out
    slope = float(((t - tm) * (y - ym)).mean()) / var      # ms per second
    n = len(ts)
    resid = y - (ym + slope * (t - tm))
    s2 = float((resid ** 2).mean()) * n / max(1, n - 2)
    se = (s2 / (n * var)) ** 0.5
    out.update({
        "drift_ms_per_min": round(slope * 60.0, 2),
        "drift_total_ms": round(slope * float(t[-1] - t[0]), 1),
        "drift_t": round(abs(slope) / se, 2) if se > 0 else 0.0,
    })
    return out


def fit_clip(model, clip, device, cfg):
    """Out-of-sample global + per-shot sub-frame lag for one clip. Timing is
    fit over SCRIPTED rows (unscripted gaps carry no timing evidence)."""
    max_lag_s = cfg["max_lag_s"]
    T = len(clip.vel)
    row_hz = clip.row_hz
    _, tr = forward.predict_shots(model, clip, [(0, T)], device,
                                     common.rows_at(MAXF_S, row_hz),
                                     collect_tracks=True)
    pred = np.where(np.isfinite(tr["vmarg"]), tr["vmarg"], 0.0)
    script = clip.vel.numpy()
    mask = clip.scripted.numpy()
    glag, gpk, c0 = common.xcorr_lag_subframe(pred, script, row_hz, max_lag_s,
                                              mask=mask)
    edges = list(clip.shot_edges)
    slags = []
    need = 4 * int(round(max_lag_s * row_hz))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if int(mask[lo:hi].sum()) < need:
            continue
        pl, pk, _ = common.xcorr_lag_subframe(pred[lo:hi], script[lo:hi],
                                              row_hz, max_lag_s,
                                              mask=mask[lo:hi])
        if pk > 0.3:
            slags.append(pl)
    dr = None
    if cfg["drift_windows"]:
        dr = fit_drift(pred, script, mask, row_hz, cfg["drift_windows"],
                       max_lag_s, cfg["drift_wide_lag_s"],
                       cfg["drift_min_peak"], gpk)
    return {
        **(dr or {}),
        "lag_ms": round(float(glag), 1),
        "peak": round(float(gpk), 3),
        "corr0": round(float(c0), 3),
        "dcorr": round(float(gpk - c0), 3),
        "shot_med_ms": round(float(np.median(slags)), 1) if slags else None,
        "shot_std_ms": round(float(np.std(slags)), 1) if len(slags) >= 2 else None,
        "n_shots": len(slags),
        # a strongly NEGATIVE correlation at lag 0: the prediction
        # ANTI-tracks the script, so the positions are authored inverted
        "polarity_suspect": bool(c0 < POLARITY_THR),
        # a lag that is a FUNCTION OF TIME is not something one number can
        # express; alarm only, never gate or retime
        "drift_suspect": bool(
            dr is not None
            and drift_alarm(dr, cfg["drift_min_gap"], cfg["drift_min_spread_ms"])),
    }


def gate(fit, cfg):
    """Peak gate: apply the fit only when the offset is confident + material.
    Confidence is ``dcorr`` (the correlation gained by shifting to the
    peak), or corroboration: enough of the drift check's per-window lags
    agreeing with the global fit in sign and magnitude."""
    lag = fit["lag_ms"]
    confident = fit["dcorr"] >= cfg["min_dcorr"]
    if not confident:
        wl = [w for w in (fit.get("drift_win_lags_ms") or [])]
        if len(wl) >= 6 and lag:
            agree = sum(1 for w in wl if w * lag > 0) / len(wl)
            confident = (agree >= cfg["win_agree"]
                         and abs(float(np.median(wl)) - lag) <= cfg["win_tol_ms"])
    material = (fit["peak"] >= cfg["min_peak"] and confident
                and abs(lag) >= cfg["min_abs_ms"])
    return (lag if material else 0.0), (not material)


def path(project, clip_id):
    return project.root / common.LAG_DIR / f"{clip_id}.json"


def load(project, clip_id):
    p = path(project, clip_id)
    return json.loads(p.read_text("utf-8")) if p.is_file() else None


def ensure(project, clip_id, model, ck, ckpt_name, device, log=print,
           cfg=CONFIG):
    """Fit and write the clip's lag sidecar unless it exists. Returns the
    sidecar and whether it was written."""
    prev = load(project, clip_id)
    if prev is not None:
        return prev, False
    fd = ck.get("feat_dir", common.LATENTS_DIR)
    clip = JepaClip(project.root, clip_id, 384, 192, feat_dir=fd, gap_mask=True,
                    apply_lag=False,   # fit the RAW script, not the residual
                    masks_dir=ck.get("masks_dir"),
                    row_hz=ck.get("row_hz"))
    fit = fit_clip(model, clip, device, cfg)
    applied, gated = gate(fit, cfg)
    trained = common.trained_ids(ck, project.root, [clip_id])
    side = {"id": clip_id, **fit, "applied_ms": round(float(applied), 1),
            "gated": bool(gated), "invert": False,
            "gate": {k: cfg[k] for k in ("min_peak", "min_dcorr", "min_abs_ms",
                                         "win_agree", "win_tol_ms")},
            "method": "in-sample" if clip_id in trained else "out-of-sample",
            "ckpt": ckpt_name, "max_lag_s": cfg["max_lag_s"]}
    atomic_write_text(path(project, clip_id), json.dumps(side, indent=1))
    log(f"  [{clip_id}] lag {fit['lag_ms']:+.0f} ms, peak {fit['peak']:.2f}"
        + (f", applied {applied:+.0f} ms" if not gated else ", not applied")
        + (", POLARITY ALARM" if fit["polarity_suspect"] else "")
        + (", DRIFT ALARM" if fit["drift_suspect"] else ""))
    return side, True
