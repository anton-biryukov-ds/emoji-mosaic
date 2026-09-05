"""Emoji mosaic maker.

Upload an image, tile it into a grid, and replace each tile with the emoji
whose color best matches the tile (nearest neighbor in CIE Lab space, with an
optional 3D color-histogram mode). Glyphs come from the Twemoji 15.1 color
emoji image set (72x72 PNGs) bundled under assets/emoji/.

A "variety" control samples among the top-k nearest emoji per tile
(temperature softmax over the distances) instead of always taking the
argmin, so mosaics avoid repeating one emoji in near-tie regions. A
"discourage repeats" control adds a soft usage penalty so over-used emojis
lose ties to fresher ones. Both are seeded for reproducibility.

Run:  streamlit run app.py
"""

import gc
import io
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOJI_DIR = os.path.join(BASE_DIR, "assets", "emoji")
EMOJI_TEST = os.path.join(BASE_DIR, "assets", "emoji-test.txt")
EMOJI_UNAVAILABLE = os.path.join(BASE_DIR, "assets", "emoji-unavailable.txt")
TWEMOJI_URL = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{}.png"
EMOJI_TEST_URL = "https://unicode.org/Public/emoji/15.1/emoji-test.txt"

# ---------------------------------------------------------------------------
# Color math (sRGB <-> CIE Lab, D65)
# ---------------------------------------------------------------------------

def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _rgb_to_lab_rows(rgb):
    """Row-wise core of rgb_to_lab. rgb: (N, 3) float array in [0, 1]."""
    lin = srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb):
    """rgb: (..., 3) float array in [0, 1] -> Lab (..., 3).

    Processes rows in chunks so large images don't allocate full-size
    temporaries for every intermediate step (OOM guard for small hosts).
    The math is identical to running _rgb_to_lab_rows on the whole array.
    """
    rgb = np.asarray(rgb)
    shape = rgb.shape
    flat = rgb.reshape(-1, 3)
    chunk = 65536
    if flat.shape[0] <= chunk:
        return _rgb_to_lab_rows(flat).reshape(shape)
    out = None
    for lo in range(0, flat.shape[0], chunk):
        res = _rgb_to_lab_rows(flat[lo:lo + chunk])
        if out is None:
            out = np.empty((flat.shape[0], 3), dtype=res.dtype)
        out[lo:lo + chunk] = res
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Emoji metadata + color profiles
# ---------------------------------------------------------------------------

def twemoji_name(codepoints):
    """emoji-test.txt codepoint list -> twemoji filename stem (FE0F stripped)."""
    return "-".join(c.lower() for c in codepoints.split() if c.lower() != "fe0f")


def ensure_assets(progress=None):
    """Download the Twemoji glyph set + Unicode emoji metadata on first run."""
    os.makedirs(EMOJI_DIR, exist_ok=True)
    if not os.path.exists(EMOJI_TEST):
        urllib.request.urlretrieve(EMOJI_TEST_URL, EMOJI_TEST)
    catalog = load_emoji_catalog()
    have = {f[:-4] for f in os.listdir(EMOJI_DIR) if f.endswith(".png")}
    unavailable = set()
    if os.path.exists(EMOJI_UNAVAILABLE):
        with open(EMOJI_UNAVAILABLE, encoding="utf-8") as fh:
            unavailable = {ln.strip() for ln in fh if ln.strip()}
    missing = sorted(set(catalog) - have - unavailable)
    if not missing:
        return 0
    def _get(stem):
        try:
            urllib.request.urlretrieve(TWEMOJI_URL.format(stem),
                                       os.path.join(EMOJI_DIR, stem + ".png"))
            return True
        except Exception:
            return False
    done = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        for ok in ex.map(_get, missing):
            done += ok
            if progress is not None:
                progress.progress(min(1.0, done / len(missing)))
    return done


@st.cache_data(show_spinner="Indexing emoji set...")
def load_emoji_catalog():
    """Map twemoji filename stem -> (emoji char, name, group)."""
    catalog = {}
    group = "Other"
    if os.path.exists(EMOJI_TEST):
        with open(EMOJI_TEST, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("# group:"):
                    group = line.split(":", 1)[1].strip()
                elif line and not line.startswith("#") and "; fully-qualified" in line:
                    cps, rest = line.split(";", 1)
                    name = rest.split("#", 1)[1].split(" E", 1)[0].split(" ", 1)[1]
                    char = rest.split("#", 1)[1].strip().split(" ", 1)[0]
                    catalog[twemoji_name(cps)] = (char, name, group)
    return catalog


@st.cache_data(show_spinner="Computing emoji color profiles...")
def load_profiles():
    """Per-emoji alpha-weighted mean Lab color and 4x4x4 Lab histogram.

    Returns (stems, chars, names, groups, mean_lab (N,3), hists (N,64)).
    """
    catalog = load_emoji_catalog()
    stems, chars, names, groups, means, hists = [], [], [], [], [], []
    edges = [np.linspace(0, 100, 5), np.linspace(-128, 128, 5), np.linspace(-128, 128, 5)]
    for fn in sorted(os.listdir(EMOJI_DIR)):
        if not fn.endswith(".png"):
            continue
        stem = fn[:-4]
        im = Image.open(os.path.join(EMOJI_DIR, fn)).convert("RGBA")
        a = np.asarray(im).astype(np.float32)
        alpha = a[..., 3:4] / 255.0
        if alpha.sum() < 20:  # skip nearly-empty glyphs
            continue
        # Profile the glyph the way it actually renders: composited over white.
        flat = a[..., :3] * alpha + 255.0 * (1.0 - alpha)
        means.append(rgb_to_lab(flat.mean(axis=(0, 1)) / 255.0))
        lab = rgb_to_lab(flat / 255.0)
        h, _ = np.histogramdd(lab.reshape(-1, 3), bins=edges)
        h = h.ravel()
        hists.append(h / max(h.sum(), 1e-9))
        stems.append(stem)
        char, name, grp = catalog.get(stem, ("?", stem, "Other"))
        chars.append(char)
        names.append(name)
        groups.append(grp)
    return (stems, chars, names, groups,
            np.array(means, dtype=np.float32), np.array(hists, dtype=np.float32))


@st.cache_data(show_spinner="Rendering emoji thumbnails...")
def load_thumbs(stems_key, thumb=16):
    """Each emoji rendered over white at thumb x thumb px, Lab, flattened."""
    mats = []
    for stem in stems_key:
        im = Image.open(os.path.join(EMOJI_DIR, stem + ".png")).convert("RGBA")
        im = im.resize((thumb, thumb), Image.LANCZOS)
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        flat = Image.alpha_composite(bg, im).convert("RGB")
        mats.append(rgb_to_lab(np.asarray(flat).astype(np.float32) / 255.0).ravel())
    return np.array(mats, dtype=np.float32)


@st.cache_data(show_spinner="Ranking emoji by color coverage...")
def diversity_order(mean_lab):
    """Greedy k-center ordering: first emojis span the color space best."""
    n = len(mean_lab)
    order = [int(np.argmax(mean_lab[:, 0]))]
    dmin = np.full(n, np.inf, dtype=np.float32)
    for _ in range(n - 1):
        last = mean_lab[order[-1]]
        d = ((mean_lab - last) ** 2).sum(axis=1)
        dmin = np.minimum(dmin, d)
        nxt = int(np.argmax(dmin))
        if nxt in order:  # numerical tie; fall back to first unused
            used = np.zeros(n, dtype=bool)
            used[order] = True
            nxt = int(np.argmax(~used))
        order.append(nxt)
        if len(order) >= n:
            break
    return order


# ---------------------------------------------------------------------------
# Variety-aware matching
# ---------------------------------------------------------------------------

def topk_candidates(dist_chunks, k):
    """From an iterable of (n_tiles_chunk, pool) distance chunks, keep the
    k nearest pool candidates per tile. Returns (top_idx, top_dist), each
    (n_tiles, k), sorted by ascending distance."""
    idxs, vals = [], []
    for Dc in dist_chunks:
        kk = min(k, Dc.shape[1])
        part = np.argpartition(Dc, kk - 1, axis=1)[:, :kk]
        v = np.take_along_axis(Dc, part, axis=1)
        s = np.argsort(v, axis=1, kind="stable")
        idxs.append(np.take_along_axis(part, s, axis=1))
        vals.append(np.take_along_axis(v, s, axis=1))
    return np.concatenate(idxs), np.concatenate(vals)


def sample_matches(top_idx, top_dist, n_pool, variety=0.0, repel=0.0, seed=0):
    """Pick one pool index per tile from its top-k nearest candidates.

    variety=0 -> always the nearest candidate (argmin, original behavior).
    variety>0 -> softmax sampling over the top-k with temperature scaled per
    tile by its own top-k distance spread, so the knob means the same thing
    across matching modes. Distances are normalized to gaps in [0, 1]
    (0 = nearest, 1 = k-th nearest), so far matches stay unreachable.
    repel>0   -> subtract repel * log1p(usage count) from each candidate's
    logit as tiles are assigned, so heavily used emojis lose near-ties.
    Deterministic for a given (top_idx, top_dist, seed).
    """
    n_tiles, k = top_idx.shape
    dmin = top_dist[:, :1]
    spread = np.maximum(top_dist[:, -1:] - dmin, 1e-9)
    gap = (top_dist - dmin) / spread  # 0 = best candidate, 1 = k-th
    if variety <= 0 and repel <= 0:
        return top_idx[:, 0].astype(np.int64)
    rng = np.random.default_rng(seed)
    inv_temp = 1.0 / max(variety, 0.02)  # variety -> 0 clamps to argmin
    pick = np.empty(n_tiles, dtype=np.int64)
    counts = np.zeros(n_pool, dtype=np.float64)
    for t in range(n_tiles):
        cand = top_idx[t]
        logits = -gap[t] * inv_temp - repel * np.log1p(counts[cand])
        if variety > 0:
            logits -= logits.max()
            w = np.exp(logits)
            w /= w.sum()
            j = int(rng.choice(k, p=w))
        else:
            j = int(np.argmax(logits))  # repel-only: deterministic re-rank
        p = int(cand[j])
        pick[t] = p
        counts[p] += 1.0
    return pick


# ---------------------------------------------------------------------------
# Mosaic core (pure functions, callable without Streamlit)
# ---------------------------------------------------------------------------

TOP_K = 12  # candidates considered per tile when variety/repel are active


def build_mosaic(img, stems, mean_lab, hists, grid_w, tile_px,
                 mode="appearance", emoji_indices=None, thumbs=None,
                 color_boost=1.3, variety=0.0, repel=0.0, seed=0):
    """Tile img into a grid_w-wide grid, match each tile to an emoji.

    variety/repel/seed control variety-aware sampling among the top-k
    nearest emoji (see sample_matches). Returns (PIL RGB image,
    matched stems per tile row-major, (grid_w, grid_h)).
    """
    img = img.convert("RGB")
    w, h = img.size
    grid_h = max(1, round(h / w * grid_w))
    idx_pool = list(emoji_indices) if emoji_indices is not None else range(len(stems))
    n_tiles = grid_h * grid_w
    n_pool = len(idx_pool)
    k = min(TOP_K, n_pool)
    exact = variety <= 0 and repel <= 0

    if mode == "appearance" and thumbs is not None:
        thumb = int(round(thumbs.shape[1] / 3))  # thumb*thumb
        thumb = int(thumb ** 0.5)
        small = img.resize((grid_w * thumb, grid_h * thumb), Image.LANCZOS)
        pool = thumbs[list(idx_pool)]
        e2 = (pool ** 2).sum(axis=1)
        tile_vec = thumb * thumb * 3
        # Process the resized image in bands of tile rows so the float32 Lab
        # working set stays bounded (~12MB per band; band + its Lab conversion
        # + the tile-vector copy coexist transiently) on small hosts.
        band_rows = max(1, int(12e6 / (grid_w * tile_vec * 4)))
        pick = np.empty(n_tiles, dtype=np.int64)
        top_idx_all, top_dist_all = [], []
        for r0 in range(0, grid_h, band_rows):
            r1 = min(grid_h, r0 + band_rows)
            band = np.asarray(small.crop((0, r0 * thumb,
                                          grid_w * thumb, r1 * thumb))
                              ).astype(np.float32) / 255.0
            lab = rgb_to_lab(band)
            lab[..., 1:] *= color_boost
            del band
            n_band = (r1 - r0) * grid_w
            tiles = lab.reshape(r1 - r0, thumb, grid_w, thumb, 3)
            tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(n_band, -1)
            del lab

            def chunks():
                for lo in range(0, n_band, 512):
                    t = tiles[lo:lo + 512]
                    yield ((t ** 2).sum(axis=1, keepdims=True) + e2[None, :]
                           - 2.0 * (t @ pool.T))

            if exact:
                for lo, Dc in zip(range(0, n_band, 512), chunks()):
                    pick[r0 * grid_w + lo:
                         r0 * grid_w + lo + Dc.shape[0]] = np.argmin(Dc, axis=1)
            else:
                ti, td = topk_candidates(chunks(), k)
                top_idx_all.append(ti)
                top_dist_all.append(td)
            del tiles
        del small
        gc.collect()
        if not exact:
            top_idx = np.concatenate(top_idx_all)
            top_dist = np.concatenate(top_dist_all)
            pick = sample_matches(top_idx, top_dist, n_pool, variety, repel, seed)
        matched = [stems[idx_pool[p]] for p in pick]
    elif mode == "histogram":
        block = 8  # sub-samples per tile axis for the tile histogram
        small = img.resize((grid_w * block, grid_h * block), Image.LANCZOS)
        pool_h = hists[list(idx_pool)]
        # Banded like appearance mode: bound the float32 Lab working set.
        band_rows = max(1, int(12e6 / (grid_w * block * block * 3 * 4)))
        tile_hists = np.empty((n_tiles, 64), dtype=np.float32)
        for r0 in range(0, grid_h, band_rows):
            r1 = min(grid_h, r0 + band_rows)
            band = np.asarray(small.crop((0, r0 * block,
                                          grid_w * block, r1 * block))
                              ).astype(np.float32) / 255.0
            lab = rgb_to_lab(band)
            lab[..., 1:] *= color_boost
            del band
            # Quantize to the 4x4x4 Lab bins, then count per tile vectorized
            # (one bincount-style pass instead of a per-tile histogramdd).
            lb = np.clip((lab[..., 0] / 25).astype(np.int64), 0, 3)
            ab = np.clip(((lab[..., 1] + 128) / 64).astype(np.int64), 0, 3)
            bb = np.clip(((lab[..., 2] + 128) / 64).astype(np.int64), 0, 3)
            del lab
            binned = (lb * 16 + ab * 4 + bb).reshape(-1, block * block)
            del lb, ab, bb
            n_band = binned.shape[0]
            counts = np.zeros((n_band, 64), dtype=np.float32)
            np.add.at(counts, (np.arange(n_band)[:, None], binned), 1.0)
            del binned
            counts /= np.maximum(counts.sum(axis=1, keepdims=True), 1e-9)
            tile_hists[r0 * grid_w:r1 * grid_w] = counts
            del counts
        del small
        gc.collect()

        # chi-square distance, chunked over tiles to bound memory
        def chunks():
            for lo in range(0, n_tiles, 256):
                th = tile_hists[lo:lo + 256]
                num = (th[:, None, :] - pool_h[None, :, :]) ** 2
                den = th[:, None, :] + pool_h[None, :, :] + 1e-9
                yield 0.5 * (num / den).sum(axis=2)

        if exact:
            pick = np.empty(n_tiles, dtype=np.int64)
            for lo, Dc in zip(range(0, n_tiles, 256), chunks()):
                pick[lo:lo + 256] = np.argmin(Dc, axis=1)
        else:
            top_idx, top_dist = topk_candidates(chunks(), k)
            pick = sample_matches(top_idx, top_dist, n_pool, variety, repel, seed)
        matched = [stems[idx_pool[p]] for p in pick]
    else:
        small = img.resize((grid_w, grid_h), Image.LANCZOS)
        lab = rgb_to_lab(np.asarray(small).astype(np.float32) / 255.0)
        lab[..., 1:] *= color_boost
        q = lab.reshape(-1, 3)
        pool_lab = mean_lab[list(idx_pool)]
        if exact:
            e2 = (pool_lab ** 2).sum(axis=1)
            pick = np.empty(n_tiles, dtype=np.int64)
            for lo in range(0, n_tiles, 1024):
                t = q[lo:lo + 1024]
                D = ((t ** 2).sum(axis=1, keepdims=True) + e2[None, :]
                     - 2.0 * (t @ pool_lab.T))
                pick[lo:lo + t.shape[0]] = np.argmin(D, axis=1)
        else:
            e2 = (pool_lab ** 2).sum(axis=1)

            def chunks():
                for lo in range(0, n_tiles, 1024):
                    t = q[lo:lo + 1024]
                    yield ((t ** 2).sum(axis=1, keepdims=True) + e2[None, :]
                           - 2.0 * (t @ pool_lab.T))

            top_idx, top_dist = topk_candidates(chunks(), k)
            pick = sample_matches(top_idx, top_dist, n_pool, variety, repel, seed)
        matched = [stems[idx_pool[p]] for p in pick]

    # RGB canvas directly: an RGBA canvas plus .convert("RGB") would hold two
    # full-size frames at once (OOM guard for small hosts).
    out = Image.new("RGB", (grid_w * tile_px, grid_h * tile_px), (255, 255, 255))
    cache = {}
    for t, stem in enumerate(matched):
        gy, gx = divmod(t, grid_w)
        glyph = cache.get(stem)
        if glyph is None:
            glyph = Image.open(os.path.join(EMOJI_DIR, stem + ".png")).convert("RGBA")
            glyph = glyph.resize((tile_px, tile_px), Image.LANCZOS)
            cache[stem] = glyph
        out.paste(glyph, (gx * tile_px, gy * tile_px), glyph)
    return out, matched, (grid_w, grid_h)


def load_uploaded(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Emoji Mosaic", page_icon="🧩", layout="wide")
    st.title("🧩 Emoji Mosaic")
    st.caption("Turn any image into a mosaic of color-matched emoji (Twemoji glyphs, Lab-space nearest neighbor).")

    have = [f for f in os.listdir(EMOJI_DIR)] if os.path.isdir(EMOJI_DIR) else []
    if len(have) < 2500:
        st.info("First run: downloading the Twemoji emoji set (~3,800 small PNGs, one minute once).")
        bar = st.progress(0.0)
        ensure_assets(progress=bar)
        bar.empty()
    else:
        ensure_assets()
    stems, chars, names, groups, mean_lab, hists = load_profiles()
    order = diversity_order(mean_lab)
    all_groups = sorted(set(groups))

    with st.sidebar:
        st.header("Controls")
        grid_w = st.slider("Grid resolution (tiles wide)", 8, 400, 96)
        tile_px = st.slider("Output tile size (px)", 8, 72, 32,
                            help="Output PNG is grid × tile size on each axis.")
        pick_groups = st.multiselect("Emoji groups to use", all_groups, default=all_groups)
        group_idx = [i for i in order if groups[i] in pick_groups]
        if not group_idx:
            st.warning("Select at least one emoji group.")
            return
        max_n = len(group_idx)
        n_emoji = st.slider("Emoji palette size (most color-diverse first)",
                            min(10, max_n), max_n, min(500, max_n))
        boost = st.slider("Color boost", 1.0, 2.0, 1.3, 0.05,
                          help="Scales tile chroma before matching so saturated emoji win over grayish ones.")
        mode = st.selectbox("Matching mode",
                            ["appearance", "mean color", "histogram"],
                            help="appearance: compares a mini-render of each emoji with each tile in Lab space (best fidelity). mean color: classic average-color match (fastest). histogram: 3D Lab color-histogram match (bonus).")
        variety = st.slider("Variety", 0.0, 1.0, 0.0, 0.05,
                            help="0 = always pick the single closest emoji. Higher values let close "
                                 "runners-up win sometimes (softmax over the top-12 nearest, temperature "
                                 "scaled per tile). Far matches are never picked, so colors stay right.")
        repel = st.slider("Discourage repeats", 0.0, 1.0, 0.0, 0.05,
                          help="Soft-penalizes emojis already used many times in this mosaic, "
                               "spreading usage across the palette. Works with or without Variety.")
        seed = st.number_input("Random seed", 0, 999999, 42,
                               help="Same image + settings + seed = identical mosaic.")

    sel = group_idx[:n_emoji]
    st.sidebar.write(f"Using **{len(sel)}** emoji from **{len(pick_groups)}** groups.")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if uploaded is None:
        st.info("Upload an image to build your mosaic.")
        return

    img = load_uploaded(uploaded.getvalue())

    # Cap the output canvas so it fits in small-host memory (a full-size RGB
    # frame is 3 bytes/px; the free host kills the process over 512MB).
    MAX_OUT_PIXELS = 36_000_000
    grid_h_est = max(1, round(img.height / img.width * grid_w))
    out_px = grid_w * tile_px * grid_h_est * tile_px
    if out_px > MAX_OUT_PIXELS:
        orig_mp = out_px / 1e6
        tile_px = max(8, int(tile_px * math.sqrt(MAX_OUT_PIXELS / out_px)))
        st.warning(f"Output would be {orig_mp:.0f}MP at the chosen tile size - "
                   f"capped tile size to {tile_px}px (~{MAX_OUT_PIXELS / 1e6:.0f}MP) "
                   f"to stay within host memory. "
                   f"Run locally for the full-resolution render.")

    # Everything that affects the output, in one signature, so we can tell
    # when the mosaic on screen no longer matches the current controls.
    signature = (uploaded.name, uploaded.size, grid_w, tile_px, tuple(sel),
                 boost, mode, variety, repel, int(seed))
    last = st.session_state.get("last_render")

    render = st.button("🖼️ Render mosaic", type="primary",
                       help="Rebuild the mosaic with the current settings.")
    if last is None:
        render = True  # first visit: render once automatically

    if render:
        # Free the previous mosaic before building the new one so both
        # full-size frames are never held at once.
        st.session_state.pop("last_render", None)
        last = None
        gc.collect()
        t0 = time.perf_counter()
        with st.spinner("Rendering mosaic... (large grids can take a few seconds)"):
            thumbs = load_thumbs(tuple(stems))
            mosaic, matched, (gw, gh) = build_mosaic(
                img, stems, mean_lab, hists, grid_w, tile_px,
                mode=mode.replace("mean color", "mean"), emoji_indices=sel, thumbs=thumbs,
                color_boost=boost, variety=variety, repel=repel, seed=int(seed))
        gc.collect()
        last = {"signature": signature, "mosaic": mosaic, "matched": matched,
                "gw": gw, "gh": gh, "elapsed": time.perf_counter() - t0}
        st.session_state.last_render = last

    if last["signature"] != signature:
        st.warning("Settings changed - the mosaic below is from the previous settings. "
                   "Press **Render mosaic** to update it.")
    else:
        st.caption(f"✅ Up to date - rendered in {last['elapsed']:.1f}s.")

    mosaic, matched, gw, gh = last["mosaic"], last["matched"], last["gw"], last["gh"]

    c1, c2 = st.columns(2)
    c1.image(img, caption=f"Original ({img.width}×{img.height})", width="stretch")
    c2.image(mosaic, caption=f"Mosaic ({gw}×{gh} tiles, {mosaic.width}×{mosaic.height}px)",
             width="stretch")

    used = sorted(set(matched))
    with st.expander(f"{len(used)} distinct emoji used"):
        st.write(" ".join(chars[stems.index(s)] for s in used))

    buf = io.BytesIO()
    mosaic.save(buf, format="PNG")
    st.download_button("⬇️ Download PNG", buf.getvalue(),
                       file_name="emoji_mosaic.png", mime="image/png")


if __name__ == "__main__":
    main()
