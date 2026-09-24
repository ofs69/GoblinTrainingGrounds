"""Funscript reading, sanitization and the content signature.

Times stay absolute and aligned to the video clock; nothing here re-bases
an action. A sanitized script is a list of ``(at_ms, pos)`` tuples.
"""
import hashlib
import json

# Fuzzy-dedup signature grid. The coarseness is the fuzziness: larger bins
# and bigger pos steps collapse more near-duplicates.
DEDUP_BIN_MS = 250
DEDUP_POS_STEP = 10

# Peak playback speed of the device, pos units per second. A transition
# steeper than this is not something a funscript can perform, so strokes
# are held to it by depth.
MAX_POS_RATE = 600.0


def actions_from_raw(raw):
    """The action list out of parsed funscript JSON, else None. Some
    funscripts are a bare array; most wrap it as ``{"actions": [...]}``."""
    if isinstance(raw, dict) and "actions" in raw:
        raw = raw["actions"]
    return raw if isinstance(raw, list) else None


def load_actions(raw_bytes):
    """Parsed actions from funscript bytes, or None when unusable."""
    try:
        raw = json.loads(raw_bytes)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return None
    return actions_from_raw(raw)


def sanitize_aligned(raw, duration_ms=None, max_speed=MAX_POS_RATE):
    """Sanitize a funscript while preserving absolute time. Clamps pos to
    0..100, drops negative or NaN times, drops actions past ``duration_ms``
    when given, keeps strictly increasing times, and holds strokes to
    ``max_speed`` by depth. Accepts dicts or ``(at, pos)`` pairs."""
    cleaned = []
    for a in raw:
        if isinstance(a, dict):
            at, pos = a.get("at"), a.get("pos")
        elif isinstance(a, (tuple, list)) and len(a) == 2:
            at, pos = a
        else:
            continue
        if isinstance(at, bool) or isinstance(pos, bool):
            continue
        if not isinstance(at, (int, float)) or not isinstance(pos, (int, float)):
            continue
        if at != at or pos != pos:
            continue
        if at < 0:
            continue
        if duration_ms is not None and at > duration_ms:
            continue
        p = int(round(pos))
        cleaned.append((float(at), min(100, max(0, p))))
    if not cleaned:
        return []
    cleaned.sort(key=lambda x: x[0])
    out = []
    last_at = None
    for at, pos in cleaned:
        if last_at is None or at > last_at:
            out.append((at, pos))
            last_at = at
    return clamp_speed(out, max_speed)


def clamp_speed(clean, max_speed=MAX_POS_RATE):
    """Hold every stroke to the device's peak speed by reducing its depth.
    Timing never moves: the endpoint is pulled toward its predecessor until
    the transition is playable. Causal single pass."""
    if max_speed is None or max_speed <= 0 or len(clean) < 2:
        return clean
    out = [clean[0]]
    for at, pos in clean[1:]:
        p_at, p_pos = out[-1]
        lim = max_speed * (at - p_at) / 1000.0
        d = pos - p_pos
        if abs(d) > lim:
            step = int(lim)          # truncate: rounding up could pass the limit
            pos = p_pos + (step if d > 0 else -step)
            pos = min(100, max(0, pos))
        out.append((at, pos))
    return out


def resample_envelope(clean, bin_ms=DEDUP_BIN_MS):
    """Time-weighted mean pos per bin over [0, last_at], treating the script
    as a step signal. Averaging makes the fingerprint robust to ms-level
    jitter. Empty for fewer than two actions."""
    if len(clean) < 2 or bin_ms <= 0:
        return []
    last_at = clean[-1][0]
    n_bins = int(last_at // bin_ms) + 1
    out = []
    j = 0
    cur = clean[0][1]
    while j < len(clean) and clean[j][0] <= 0:
        cur = clean[j][1]
        j += 1
    for i in range(n_bins):
        lo = i * bin_ms
        hi = lo + bin_ms
        t = lo
        acc = 0.0
        while j < len(clean) and clean[j][0] < hi:
            nt = clean[j][0]
            if nt > t:
                acc += cur * (nt - t)
                t = nt
            cur = clean[j][1]
            j += 1
        acc += cur * (hi - t)
        out.append(acc / bin_ms)
    return out


def script_signature(clean, bin_ms=DEDUP_BIN_MS, pos_step=DEDUP_POS_STEP):
    """Fuzzy content fingerprint of a sanitized script: the envelope on a
    ``bin_ms`` grid, quantized to ``pos_step`` buckets, hashed. Two scripts
    describing the same motion collapse to one signature across whitespace,
    key order, ms jitter and a few added or dropped actions. Empty when the
    script is too short to fingerprint."""
    env = resample_envelope(clean, bin_ms)
    if not env:
        return ""
    step = max(1, int(pos_step))
    quant = bytes(min(255, int((p + step / 2) // step)) for p in env)
    payload = f"{int(bin_ms)}:{step}:".encode() + quant
    return hashlib.md5(payload).hexdigest()


def signature_from_bytes(raw_bytes):
    """The signature of a script from its bytes alone, with no duration
    bound, so the same value is recomputed on every import for duplicate
    detection. Empty when unusable."""
    acts = load_actions(raw_bytes)
    if acts is None:
        return ""
    return script_signature(sanitize_aligned(acts))


def content_key(raw_bytes, video_bytes):
    """The exact key: script hash plus source video size."""
    return hashlib.md5(raw_bytes).hexdigest() + f":{video_bytes}"
