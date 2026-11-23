#!/usr/bin/env python3
"""
Imports the ShadowNigth/models asset bank (CS2 / CS:GO extracted weapons) into this
Godot project, replacing every first-person / world / player weapon model and weapon
sound with the ones stored in that repository.

What it does
------------
  1. Viewmodels  (v_*.glb) : skin weights are compacted, meshes welded + LOD decimated,
     every animation channel is curve-reduced (constant bones collapse to one key) and
     re-serialised as standard glTF, textures are repacked at 1024 (weapon) / 256 (arms).
  2. World + player models (w_*.glb, p_*.glb) : welded, decimated ~35 %, 256px textures.
  3. Sounds      : source WAVs are down-mixed / resampled to mono 22.05 kHz 16-bit PCM
     and laid out as  assets/sounds/weapons/fire/<weapon>/<n>.wav  (shots) and
     assets/sounds/weapons/foley/<event>.wav  (draw, clip in/out, bolt, silencer, ...).
  4. data/weapons.json + data/models_meta.json are regenerated: new weapons, new sound
     paths, animation names and per-clip foley event tables used by WeaponController.

Usage:  python3 tools/import_models_repo.py /path/to/models-checkout
"""
import io
import json
import math
import os
import re
import shutil
import struct
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import fast_simplification as fsimp
except ImportError:  # pragma: no cover
    fsimp = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'assets', 'models', 'weapons')
SND = os.path.join(ROOT, 'assets', 'sounds', 'weapons')

# --------------------------------------------------------------------------- glTF io
COMP = {5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2), 5123: ('H', 2), 5125: ('I', 4), 5126: ('f', 4)}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
NP_DT = {5120: '<i1', 5121: '<u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}


def load_glb(path):
    raw = open(path, 'rb').read()
    magic, _ver, _length = struct.unpack('<III', raw[:12])
    if magic != 0x46546C67:
        raise ValueError('not a GLB: ' + path)
    off, js, binc = 12, None, b''
    while off < len(raw):
        clen, ctype = struct.unpack('<II', raw[off:off + 8])
        data = raw[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(data.decode('utf-8'))
        elif ctype == 0x004E4942:
            binc = data
        off += 8 + clen
    return js, binc


def save_glb(j, binc, path):
    js = dict(j)
    js['buffers'] = [{'byteLength': len(binc)}]
    jtxt = json.dumps(js, separators=(',', ':')).encode('utf-8')
    while len(jtxt) % 4:
        jtxt += b' '
    bb = bytearray(binc)
    while len(bb) % 4:
        bb += b'\x00'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(jtxt) + 8 + len(bb)))
        f.write(struct.pack('<II', len(jtxt), 0x4E4F534A))
        f.write(jtxt)
        f.write(struct.pack('<II', len(bb), 0x004E4942))
        f.write(bytes(bb))


def acc_np(j, b, ai):
    a = j['accessors'][ai]
    bv = j['bufferViews'][a['bufferView']]
    base = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    comps = NCOMP[a['type']]
    dt = np.dtype(NP_DT[a['componentType']])
    n = a['count']
    elem = dt.itemsize * comps
    stride = bv.get('byteStride') or elem
    kind = np.float64 if dt.kind == 'f' else np.int64
    if stride == elem:
        return np.frombuffer(b, dtype=dt, count=n * comps, offset=base).reshape(n, comps).astype(kind)
    out = np.empty((n, comps), dtype=kind)
    for i in range(n):
        out[i] = np.frombuffer(b, dtype=dt, count=comps, offset=base + i * stride)
    return out


def acc_img(j, b, idx):
    im = j['images'][idx]
    if 'bufferView' not in im:
        return None
    bv = j['bufferViews'][im['bufferView']]
    off = bv.get('byteOffset', 0)
    return b[bv.get('byteOffset', 0): off + bv['byteLength']]


class Writer:
    """Accumulates the output binary chunk and the buffers/accessors tables."""

    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors = [], []

    def _align(self):
        while len(self.bin) % 4:
            self.bin += b'\x00'

    def acc(self, arr, ctype, type_str, target=None, minmax=True):
        arr = np.ascontiguousarray(arr)
        self._align()
        off = len(self.bin)
        data = arr.astype(np.dtype(NP_DT[ctype])).tobytes()
        self.bin += data
        view = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
        if target is not None:
            view['target'] = target
        self.views.append(view)
        a = {'bufferView': len(self.views) - 1, 'componentType': ctype,
             'count': int(arr.shape[0]), 'type': type_str}
        if minmax:
            if arr.ndim == 1:
                a['min'] = [float(arr.min())]
                a['max'] = [float(arr.max())]
            else:
                a['min'] = [float(x) for x in arr.min(axis=0)]
                a['max'] = [float(x) for x in arr.max(axis=0)]
        self.accessors.append(a)
        return len(self.accessors) - 1

    def blob(self, data):
        self._align()
        off = len(self.bin)
        self.bin += data
        self.views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(data)})
        return len(self.views) - 1


# --------------------------------------------------------------------------- transforms
def quat_mat(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
                     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


def node_local(n):
    if 'matrix' in n:
        m = n['matrix']
        return np.array([[m[0], m[4], m[8], m[12]], [m[1], m[5], m[9], m[13]],
                         [m[2], m[6], m[10], m[14]], [m[3], m[7], m[11], m[15]]], dtype=np.float64)
    M = quat_mat(n.get('rotation', [0, 0, 0, 1]))
    M[:3, :3] *= np.array(n.get('scale', [1, 1, 1]), dtype=np.float64)[None, :]
    M[:3, 3] = n.get('translation', [0, 0, 0])
    return M


def world_mats(j):
    nodes = j['nodes']
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get('children', []):
            parent[c] = i
    W = [None] * len(nodes)

    def get(i):
        if W[i] is None:
            M = node_local(nodes[i])
            if i in parent:
                M = get(parent[i]) @ M
            W[i] = M
        return W[i]

    for i in range(len(nodes)):
        get(i)
    return W


def skin_ibm(j, b, si):
    sk = j['skins'][si]
    if 'inverseBindMatrices' not in sk:
        return None, sk['joints']
    raw = acc_np(j, b, sk['inverseBindMatrices'])
    mats = np.array([[[r[0], r[4], r[8], r[12]], [r[1], r[5], r[9], r[13]],
                      [r[2], r[6], r[10], r[14]], [r[3], r[7], r[11], r[15]]] for r in raw])
    return mats, sk['joints']


# --------------------------------------------------------------------------- textures
def uv_mask(j, b, img_idx, size):
    tex = {ti for ti, t in enumerate(j.get('textures', [])) if t.get('source') == img_idx}
    mats = {mi for mi, m in enumerate(j.get('materials', []))
            if (m.get('pbrMetallicRoughness', {}).get('baseColorTexture') or {}).get('index') in tex}
    mask = Image.new('L', size, 0)
    dr = ImageDraw.Draw(mask)
    for mesh in j.get('meshes', []):
        for pr in mesh.get('primitives', []):
            if pr.get('material') not in mats or 'TEXCOORD_0' not in pr.get('attributes', {}):
                continue
            uv = acc_np(j, b, pr['attributes']['TEXCOORD_0'])
            idx = acc_np(j, b, pr['indices']).astype(np.int64).reshape(-1) if 'indices' in pr else np.arange(len(uv))
            for tri in idx.reshape(-1, 3):
                dr.polygon([(float(uv[i][0]) * size[0], float(uv[i][1]) * size[1]) for i in tri], fill=255)
    return mask


def repack(img, mask, factor):
    """Downscale by `factor` while bleeding valid colour over the black gutters so that
    mip filtering never fringes the weapon silhouette."""
    if factor <= 1.0:
        return img
    filled = img.copy()
    m = mask
    for _ in range(2):
        grown = m.filter(ImageFilter.MaxFilter(5))
        filled = Image.composite(filled, filled.filter(ImageFilter.GaussianBlur(2)), grown)
        m = grown
    w, h = img.size
    return filled.resize((max(1, int(round(w / factor))), max(1, int(round(h / factor)))), Image.LANCZOS)


def matrix_to_trs(m):
    """Decompose a column-major glTF 4x4 node matrix into (translation, quaternion WXYZ, scale).
    glTF forbids animating the TRS of a node that also declares `matrix` and Godot's importer
    drops any clip that does it -- the CS:GO knife viewmodels rely on exactly that pattern."""
    M = np.array(m, dtype=np.float64).reshape(4, 4).T
    t = M[:3, 3].copy()
    R = M[:3, :3].copy()
    scale = np.linalg.norm(R, axis=0)
    neg = float(np.linalg.det(R)) < 0.0
    if neg:
        scale[0] = -scale[0]
    scale[scale == 0.0] = 1.0
    R = R / np.abs(scale)[None, :]
    if neg:
        R[:, 0] = -R[:, 0]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        sq = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * sq, (R[2, 1] - R[1, 2]) / sq, (R[0, 2] - R[2, 0]) / sq, (R[1, 0] - R[0, 1]) / sq
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        sq = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / sq, 0.25 * sq, (R[0, 1] + R[1, 0]) / sq, (R[0, 2] + R[2, 0]) / sq
    elif R[1, 1] > R[2, 2]:
        sq = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / sq, (R[0, 1] + R[1, 0]) / sq, 0.25 * sq, (R[1, 2] + R[2, 1]) / sq
    else:
        sq = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / sq, (R[0, 2] + R[2, 0]) / sq, (R[1, 2] + R[2, 1]) / sq, 0.25 * sq
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= max(1e-12, float(np.linalg.norm(q)))
    return [float(v) for v in t], [float(v) for v in q], [float(v) for v in scale]


def to_pot(img):
    """Round a texture down to the nearest power-of-two size (never below 16 px)."""
    w, h = img.size
    if (w & (w - 1)) == 0 and (h & (h - 1)) == 0:
        return img
    def pot(v):
        p = 16
        while p * 2 <= v:
            p *= 2
        return p
    nw, nh = pot(w), pot(h)
    if (nw, nh) == (w, h):
        return img
    return img.resize((nw, nh), Image.LANCZOS)


def encode_jpeg(img, quality):
    out = io.BytesIO()
    img.convert('RGB').save(out, 'JPEG', quality=quality, optimize=True, subsampling=2)
    return out.getvalue()


# --------------------------------------------------------------------------- mesh LOD
def decimate(V, TR, target_ratio, agg=5.0, preserve_border=False):
    """Collapse mesh; returns (V, TR) with triangle winding kept consistent."""
    nt = len(TR)
    if fsimp is None or target_ratio >= 1.0 or nt < 200:
        return V, TR
    target = max(16, int(nt * target_ratio))
    simpl = fsimp.simplify(V.astype(np.float32), TR.astype(np.int32),
                           target_reduction=1.0 - target / nt, agg=agg,
                           preserve_border=preserve_border)
    Vs = simpl[0].astype(np.float64)
    Ts = simpl[1].astype(np.int64).reshape(-1, 3)
    if len(Ts) == 0:
        return V, TR
    return Vs, Ts


def nearest_src_map(src, dst):
    """index of the closest source vertex for every destination vertex."""
    out = np.empty(len(dst), dtype=np.int64)
    scale = 1.0 / max(1e-6, float(np.abs(src).max()) * 1e-6)
    key = np.round(src / max(1e-9, 1.0 / scale) * 1e-6).astype(np.int64)
    table = {}
    for i, k in enumerate(map(tuple, key)):
        table.setdefault(k, i)
    miss = []
    for i, p in enumerate(dst):
        k = tuple(np.round(p / max(1e-9, 1.0 / scale) * 1e-6).astype(np.int64))
        got = table.get(k)
        if got is None:
            miss.append(i)
            out[i] = 0
        else:
            out[i] = got
    if miss:
        idx = np.array(miss)
        # chunked brute force nearest neighbour
        for s in range(0, len(idx), 256):
            sel = idx[s:s + 256]
            d = ((dst[sel][:, None, :] - src[None, :, :]) ** 2).sum(-1)
            out[sel] = np.argmin(d, axis=1)
    return out


def kd_lookup(src, dst, k=1):
    """Indices of the k nearest `src` vertices for every `dst` vertex."""
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(src)
        _d, idx = tree.query(dst, k=k)
        return np.atleast_2d(idx) if k > 1 else idx.reshape(-1)
    except Exception:
        out = np.empty(len(dst), dtype=np.int64)
        for s in range(0, len(dst), 256):
            chunk = dst[s:s + 256]
            d = ((chunk[:, None, :] - src[None, :, :]) ** 2).sum(-1)
            out[s:s + 256] = np.argmin(d, axis=1)
        return out


def fix_winding(V, TR, N):
    tri = V[TR]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ref = N[TR].mean(axis=1)
    flip = np.einsum('ij,ij->i', n, ref) < 0
    if flip.any():
        TR = TR.copy()
        TR[flip] = TR[flip][:, [0, 2, 1]]
    return TR


# --------------------------------------------------------------------------- animation
def curve_reduce(times, values, eps):
    """Greedy keyframe reduction: keep keys needed to stay within `eps` of the curve."""
    n = len(times)
    if n <= 2:
        return np.arange(n)
    keep = [0]
    last = 0
    i = 1
    while i < n - 1:
        # can we drop everything between `last` and `i+1`?
        t0, t1 = times[last], times[i + 1]
        span = t1 - t0
        if span <= 0:
            keep.append(i)
            last = i
            i += 1
            continue
        f = (times[last:i + 1] - t0)[:, None] / span
        interp = values[last][None, :] * (1 - f) + values[i + 1][None, :] * f
        err = np.abs(interp - values[last:i + 1]).max()
        if err > eps:
            keep.append(i)
            last = i
        i += 1
    keep.append(n - 1)
    return np.array(sorted(set(keep)))


def sample_clip(j, b, anim, fps):
    dur = 0.0
    for s in anim['samplers']:
        mx = j['accessors'][s['input']].get('max')
        if mx:
            dur = max(dur, float(mx[0]))
    n = max(2, int(round(dur * fps)) + 1)
    times = np.linspace(0.0, dur, n)
    chans = []
    for ch in anim['channels']:
        s = anim['samplers'][ch['sampler']]
        src_t = acc_np(j, b, s['input']).reshape(-1).astype(np.float64)
        vals = acc_np(j, b, s['output']).astype(np.float64)
        if len(src_t) == 0 or len(vals) == 0:
            continue
        idx = np.clip(np.searchsorted(src_t, times, side='right') - 1, 0, len(src_t) - 1)
        nxt = np.clip(idx + 1, 0, len(src_t) - 1)
        span = src_t[nxt] - src_t[idx]
        frac = np.zeros(len(times))
        ok = (span > 1e-9) & (nxt != idx)
        frac[ok] = (times[ok] - src_t[idx][ok]) / span[ok]
        v0, v1 = vals[idx], vals[nxt]
        if vals.shape[1] == 4:  # quaternion slerp
            d = np.sum(v0 * v1, axis=1)
            v1 = np.where(d[:, None] < 0, -v1, v1)
            cos = np.clip(np.abs(d), 0.0, 1.0)
            th = np.arccos(cos)
            st = np.sin(th)
            f = frac[:, None]
            sl = np.where(st[:, None] > 1e-5,
                          (np.sin((1 - f) * th[:, None]) / np.maximum(st[:, None], 1e-9)) * v0
                          + (np.sin(f * th[:, None]) / np.maximum(st[:, None], 1e-9)) * v1, v0)
            out = np.where(ok[:, None], sl, v0)
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        else:
            out = np.where(ok[:, None], v0 + (v1 - v0) * frac[:, None], v0)
        # drop channels that never move (single key)
        if np.abs(out - out[0]).max() < 1e-6 and ch['target']['path'] != 'rotation':
            out = out[:1]
        elif vals.shape[1] == 4 and np.abs(out - out[0]).max() < 1e-6:
            out = out[:1]
        chans.append((ch['target']['node'], ch['target']['path'], out))
    return dur, times, chans


def write_animations(w, j, b, keep_names=None, fps=30.0, rot_eps=0.008, loc_eps=5e-4,
                     rest_override=None):
    # default TRS of every node: constant channels identical to it are not exported at all
    rest = {}
    for i, nd in enumerate(j['nodes']):
        rest[i] = {
            'translation': np.array(nd.get('translation', [0.0, 0.0, 0.0]), dtype=np.float64),
            'rotation': np.array(nd.get('rotation', [0.0, 0.0, 0.0, 1.0]), dtype=np.float64),
            'scale': np.array(nd.get('scale', [1.0, 1.0, 1.0]), dtype=np.float64),
        }
    anims_out = []
    for anim in j.get('animations', []):
        name = anim.get('name', '')
        if keep_names is not None and name not in keep_names:
            continue
        dur, times, chans = sample_clip(j, b, anim, fps)
        if not chans:
            continue
        samplers, channels = [], []
        cache = {}
        time_cache = {}
        for node, path, vals in chans:
            base_v = rest.get(node, {}).get(path)
            if base_v is not None and np.abs(vals - base_v[None, :]).max() < 1e-6:
                continue  # bone keeps its bind transform for this whole clip
            key = (path, vals.tobytes()[:64], len(vals))
            if len(vals) == 1:
                t_sub = np.array([0.0], dtype=np.float64)
            else:
                eps = rot_eps if path == 'rotation' else loc_eps
                kk = curve_reduce(times, vals, eps)
                if len(kk) < 2:
                    kk = np.array([0, len(times) - 1])
                t_sub = times[kk]
                vals = vals[kk]
            tkey = t_sub.tobytes()
            ck = (tkey, path, vals.tobytes())
            if ck in cache:
                si = cache[ck]
            else:
                if tkey in time_cache:
                    ti = time_cache[tkey]
                else:
                    # glTF requires min/max on animation input accessors; Godot derives the
                    # Animation node's length from them (without them every clip is 0 s long,
                    # which silently disables all viewmodel animation playback).
                    ti = w.acc(t_sub.astype(np.float32), 5126, 'SCALAR', minmax=True)
                    time_cache[tkey] = ti
                vi = w.acc(vals.astype(np.float32), 5126, 'VEC4' if path == 'rotation' else 'VEC3', minmax=False)
                samplers.append({'input': ti, 'output': vi, 'interpolation': 'LINEAR'})
                si = len(samplers) - 1
                cache[ck] = si
            channels.append({'sampler': si, 'target': {'node': node, 'path': path}})
        if not channels or not samplers:
            # e.g. 'idle': the whole pose already matches the bind pose, so every channel was
            # dropped. Emitting it would produce an empty 0 s clip that breaks playback.
            continue
        anims_out.append({'name': name, 'channels': channels, 'samplers': samplers})
    return anims_out


# --------------------------------------------------------------------------- sound bank
TARGET_RATE = 22050


def read_wav(path):
    with wave.open(path, 'rb') as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        a = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648.0
    else:
        return None, rate
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a, rate


def resample(a, src_rate, dst_rate):
    if src_rate == dst_rate:
        return a
    n_out = int(round(len(a) * dst_rate / float(src_rate)))
    if n_out <= 1:
        return a[:1]
    x_old = np.linspace(0.0, 1.0, num=len(a), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    try:
        from scipy.signal import resample_poly  # type: ignore
        g = math.gcd(int(src_rate), int(dst_rate))
        return resample_poly(a, dst_rate // g, src_rate // g).astype(np.float32)
    except Exception:
        return np.interp(x_new, x_old, a).astype(np.float32)


def write_wav_mono(audio, path, peak_db=0.0):
    a = np.asarray(audio, dtype=np.float32)
    peak = float(np.abs(a).max()) if a.size else 0.0
    if peak > 0.0:
        target = 10 ** (peak_db / 20.0)
        if peak > target:  # only tame clipping, never boost (keeps the original punch)
            a = a * (target / peak)
    data = np.clip(a * 32767.0, -32768, 32767).astype('<i2').tobytes()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(data)


def convert_sound(src, dst):
    audio, rate = read_wav(src)
    if audio is None or len(audio) == 0:
        return None
    audio = resample(audio, rate, TARGET_RATE)
    write_wav_mono(audio, dst)
    return os.path.getsize(dst)


# --------------------------------------------------------------------------- viewmodel
def convert_viewmodel(src, dst, ratio=0.32, keep_names=None, fps=30.0, tex=640, arms_tex=160,
                      arms_ratio=0.34):
    j, b = load_glb(src)
    w = Writer()

    # ---- images: weapon map @1024, arm skins @256
    images = []
    new_images = []
    for ii, im in enumerate(j.get('images', [])):
        raw = acc_img(j, b, ii)
        img = None
        if raw:
            try:
                img = Image.open(io.BytesIO(raw)).convert('RGB')
            except Exception:
                img = None
        if img is None and im.get('uri'):
            # The CS:GO knife viewmodels reference some maps as sibling files. The bank does
            # not ship the '../../textures/cs2_catalog/*' paths they point at, so fall back to
            # the textures the exporter wrote next to the GLB (<model>_3/_4/_5.png = albedo /
            # normal / orm, in material slot order).
            src_dir = os.path.dirname(os.path.abspath(src))
            stem = os.path.basename(src)[:-4]
            cand = os.path.normpath(os.path.join(src_dir, im['uri']))
            if not os.path.exists(cand):
                side = os.path.join(src_dir, '%s_%d.png' % (stem, ii))
                cand = side if os.path.exists(side) else cand
            if os.path.exists(cand):
                try:
                    img = Image.open(cand).convert('RGB')
                except Exception:
                    img = None
        placeholder = False
        if img is None:
            # Never emit an image entry without a bufferView (the validator rejects it and
            # Godot ends up with a null texture). Six of the knife viewmodels point at maps the
            # bank does not ship, so they get a mid-grey metal instead of blinding white.
            img = Image.new('RGB', (16, 16), (150, 150, 158))
            placeholder = True
        name = (im.get('name') or '').lower()
        is_arms = ('glove' in name) or ('arms' in name)
        target = arms_tex if is_arms else tex
        if img.size[0] > target:
            img = repack(img, uv_mask(j, b, ii, img.size), img.size[0] / float(target))
        # power-of-two dimensions: Godot's texture importer mipmaps/filters POT textures
        # correctly and the Khronos validator warns on NPOT base colour maps.
        img = to_pot(img)
        data = encode_jpeg(img, 82 if not is_arms else 82)
        vi = w.blob(data)
        _nm = im.get('name')
        if not _nm:
            _uri = im.get('uri') or ''
            _nm = os.path.splitext(os.path.basename(_uri))[0] if _uri else ('map%d' % ii)
            if placeholder:
                _nm += '_fallback'
        new_images.append({'bufferView': vi, 'mimeType': 'image/jpeg', 'name': _nm})
        images.append(img)
    # textures still index the ORIGINAL image list: keep everything aligned
    missing = [ii for ii in range(len(j.get('images', []))) if ii >= len(new_images)]
    for ii in missing:
        vi = w.blob(encode_jpeg(Image.new('RGB', (16, 16), (255, 255, 255)), 82))
        new_images.append({'bufferView': vi, 'mimeType': 'image/jpeg', 'name': 'fallback%d' % ii})
        images.append(None)

    # ---- mesh primitives
    parent_node = {}
    for _i, _nd in enumerate(j['nodes']):
        for _c in _nd.get('children', []):
            parent_node[_c] = _i
    skins = j.get('skins', [])
    ibm, joints = (skin_ibm(j, b, 0) if skins else (None, []))
    prims = []
    for ni, n in enumerate(j['nodes']):
        if 'mesh' not in n:
            continue
        for pr in j['meshes'][n['mesh']]['primitives']:
            pos = acc_np(j, b, pr['attributes']['POSITION']).astype(np.float64)
            uv = acc_np(j, b, pr['attributes']['TEXCOORD_0']).astype(np.float64)
            nrm = acc_np(j, b, pr['attributes']['NORMAL']).astype(np.float64)
            tri = (acc_np(j, b, pr['indices']).astype(np.int64).reshape(-1)
                   if 'indices' in pr else np.arange(len(pos)))
            jj = acc_np(j, b, pr['attributes']['JOINTS_0']).astype(np.int64)
            ww = acc_np(j, b, pr['attributes']['WEIGHTS_0']).astype(np.float64)
            mat = pr.get('material', 0)
            vtx, tr, uv_o, n_o = [], [], [], []
            base = 0
            # weld within the primitive (position+uv+joints) to help the decimator
            key = np.concatenate([np.round(pos, 5), np.round(uv, 5), jj.astype(np.float64)], axis=1)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            V = pos[np.unique(inv, return_index=True)[1]][np.argsort(np.unique(inv, return_index=True)[1])] \
                if False else None
            # build representative arrays
            first = np.zeros(len(uniq), dtype=np.int64)
            seen = np.zeros(len(uniq), dtype=bool)
            for i, u in enumerate(inv):
                if not seen[u]:
                    seen[u] = True
                    first[u] = i
            V = pos[first]
            UVs = uv[first]
            NRM = nrm[first]
            JJ = jj[first]
            WW = ww[first]
            T = inv[tri].reshape(-1, 3)
            is_arms = any((j['materials'][mat].get('name', '').lower().find(k) >= 0)
                          for k in ('glove', 'arms'))
            tr_ratio = min(0.97, arms_ratio) if is_arms else ratio
            V2, T2 = decimate(V, T, tr_ratio, agg=5.0)
            srcidx = kd_lookup(V, V2) if len(V2) != len(V) else np.arange(len(V)) if len(V2) != len(V) else np.arange(len(V))
            UV2 = UVs[srcidx]
            NRM2 = NRM[srcidx]
            JJ2 = JJ[srcidx].copy()
            WW2 = WW[srcidx].copy()
            # rigid parts (gun body, casings) keep a single dominant bone so collapsing an
            # arm holding them can never drag the weapon to a different skeleton branch
            wsum = WW2.sum(axis=1, keepdims=True)
            wsum[wsum == 0] = 1.0
            dom = np.argmax(WW2, axis=1)
            has = wsum[:, 0] > 0
            if (not is_arms) and has.any():
                WW2[:] = 0.0
                JJ2[:, 0] = JJ2[np.arange(len(JJ2)), dom]
                WW2[:, 0] = 1.0
            T2 = fix_winding(V2, T2, NRM2)
            # compact joint indices (only bones actually used by this weapon's mesh)
            prims.append(dict(V=V2, UV=UV2, N=NRM2, T=T2, JJ=JJ2, WW=WW2, mat=mat))

    # ---- joint list: keep animated joints + every joint a vertex is bound to, plus their
    #      joint-space ancestors, then remap JOINTS_0 onto the compacted list.
    node_to_joint = {nd: ji for ji, nd in enumerate(joints)}
    anim_joints = set()
    for anim in j.get('animations', []):
        for ch in anim['channels']:
            ji = node_to_joint.get(int(ch['target']['node']))
            if ji is not None:
                anim_joints.add(ji)
    bound_joints = set(int(x) for p in prims for x in np.unique(p['JJ'][p['WW'] > 0]))
    keep = set(bound_joints) | anim_joints
    # joint parent chain (in joint index space)
    joint_parent = {}
    for ji, nd in enumerate(joints):
        pnd = parent_node.get(nd)
        seen = set()
        while pnd is not None and pnd not in seen:
            seen.add(pnd)
            if pnd in node_to_joint:
                joint_parent[ji] = node_to_joint[pnd]
                break
            pnd = parent_node.get(pnd)
    for ji in list(keep):
        k = joint_parent.get(ji)
        while k is not None:
            keep.add(k)
            k = joint_parent.get(k)
    used = sorted(keep)
    remap = {old: new for new, old in enumerate(used)}
    for p in prims:
        newj = np.zeros_like(p['JJ'])
        neww = np.zeros_like(p['WW'])
        for k in range(4):
            for old_i, new_i in remap.items():
                sel = (p['JJ'][:, k] == old_i) & (p['WW'][:, k] > 0)
                if sel.any():
                    newj[sel, k] = new_i
                    neww[sel, k] = p['WW'][sel, k]
        ssum = neww.sum(axis=1, keepdims=True)
        zero = (ssum[:, 0] <= 1e-8)
        if zero.any():
            newj[zero, 0] = 0
            neww[zero, 0] = 1.0
            ssum = neww.sum(axis=1, keepdims=True)
        p['JJ'] = newj
        p['WW'] = neww / ssum

    # ---- write accessors
    out_prims = []
    for p in prims:
        a_pos = w.acc(p['V'], 5126, 'VEC3', target=34962)
        a_nrm = w.acc(p['N'], 5126, 'VEC3', target=34962, minmax=False)
        a_uv = w.acc(p['UV'], 5126, 'VEC2', target=34962, minmax=False)
        a_j = w.acc(p['JJ'].astype(np.uint8), 5121, 'VEC4', target=34962, minmax=False)
        # 8-bit normalised weights: every vertex is a single-bone bind (weapon/casing bones
        # are rigid), so 255/0 quantisation is loss-less and halves the vertex footprint.
        wq = np.clip(np.round(p['WW'] * 255.0), 0, 255).astype(np.uint8)
        a_w = w.acc(wq, 5121, 'VEC4', target=34962, minmax=False)
        w.accessors[a_w]['normalized'] = True
        a_i = w.acc(p['T'].reshape(-1).astype(np.uint16 if len(p['V']) < 65536 else np.uint32),
                    5123 if len(p['V']) < 65536 else 5125, 'SCALAR', target=34963, minmax=False)
        out_prims.append({'attributes': {'POSITION': a_pos, 'NORMAL': a_nrm, 'TEXCOORD_0': a_uv,
                                        'JOINTS_0': a_j, 'WEIGHTS_0': a_w},
                          'indices': a_i, 'material': p['mat'], 'mode': 4})

    # ---- materials / textures / samplers
    new_mats = json.loads(json.dumps(j.get('materials', [])))
    placeholder_mat = set()
    for ti, t in enumerate(j.get('textures', [])):
        src_i = t.get('source')
        if src_i is not None and src_i < len(new_images) and \
                str((new_images[src_i] or {}).get('name', '')).endswith('_fallback'):
            placeholder_mat.add(ti)
    for m in new_mats:
        pbr = m.get('pbrMetallicRoughness', {})
        pbr.setdefault('metallicFactor', 0.0)
        pbr['roughnessFactor'] = 0.62
        # drop normal / orm / occlusion lookups that resolve to a missing map
        for k in ('normalTexture', 'occlusionTexture'):
            if m.get(k, {}).get('index') in placeholder_mat:
                m.pop(k, None)
        if pbr.get('metallicRoughnessTexture', {}).get('index') in placeholder_mat:
            pbr.pop('metallicRoughnessTexture', None)
    samplers = [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}]
    textures = []
    for t in j.get('textures', []):
        textures.append({'sampler': 0, 'source': t.get('source', 0)})

    # ---- nodes: keep the scene/skeleton chain of the retained bones
    used_nodes = [joints[i] for i in used]
    keep_nodes = set(used_nodes) | set(j['scenes'][0]['nodes'])
    for ni in list(keep_nodes):
        k = ni
        while k in parent_node:
            k = parent_node[k]
            keep_nodes.add(k)
    order = sorted(keep_nodes)
    new_index = {old: i for i, old in enumerate(order)}
    out_nodes = []
    for old in order:
        src_nd = j['nodes'][old]
        nd = {'name': src_nd.get('name', 'node%d' % old)}
        if 'matrix' in src_nd:
            # always emit TRS: the animations target translation/rotation/scale, which is
            # illegal on a node that also declares `matrix` (Godot drops those clips). Keeping
            # the whole tree in TRS also avoids the validator's ONE_OF_MISMATCH errors.
            t, r, sc = matrix_to_trs(src_nd['matrix'])
            if not (abs(t[0]) + abs(t[1]) + abs(t[2]) < 1e-9):
                nd['translation'] = t
            if abs(abs(r[0]) - 1.0) > 1e-5:
                nd['rotation'] = r
            if abs(sc[0] - 1.0) + abs(sc[1] - 1.0) + abs(sc[2] - 1.0) > 1e-5:
                nd['scale'] = sc
        for k in ('translation', 'rotation', 'scale'):
            if k in src_nd:
                nd[k] = src_nd[k]
        kids = [new_index[c] for c in j['nodes'][old].get('children', []) if c in new_index]
        if kids:
            nd['children'] = kids
        out_nodes.append(nd)
    # Mirror the source layout exactly: the mesh lives on its own dedicated node (child of the
    # armature root), NOT merged into the armature node. Godot's glTF importer and the Khronos
    # validator both flag a skinned mesh on a non-root joint node.
    armature_idx = new_index[j['scenes'][0]['nodes'][0]]
    mesh_node_idx = len(out_nodes)
    out_nodes.append({'name': 'Mesh', 'mesh': 0, 'skin': 0})
    out_nodes[armature_idx].setdefault('children', []).append(mesh_node_idx)

    # ---- inverse bind matrices for the compacted joint list
    ibm_rows = []
    for old in used:
        M = ibm[old] if ibm is not None else np.eye(4)
        # glTF stores 4x4 matrices column-major -> transpose our row-major numpy form
        ibm_rows.append(M.T.reshape(-1).tolist())
    a_ibm = w.acc(np.array(ibm_rows, dtype=np.float64), 5126, 'MAT4', minmax=False)

    # ---- animations
    anims = write_animations(w, j, b, keep_names=keep_names, fps=fps)

    out = {
        'asset': {'version': '2.0', 'generator': 'cs15-godot import_models_repo.py'},
        'scene': 0,
        'scenes': [{'nodes': [armature_idx]}],
        'nodes': out_nodes,
        'meshes': [{'primitives': out_prims}],
        'skins': [{'joints': [new_index[x] for x in used_nodes], 'inverseBindMatrices': a_ibm}],
        'materials': new_mats,
        'textures': textures,
        'samplers': samplers,
        'images': new_images,
        'accessors': w.accessors,
        'bufferViews': w.views,
    }
    if anims:
        out['animations'] = anims
    save_glb(out, bytes(w.bin), dst)
    return dict(path=dst, size=os.path.getsize(dst), anims=[a['name'] for a in anims],
                verts=sum(len(p['V']) for p in prims), tris=sum(len(p['T']) for p in prims),
                bones=len(used))


def convert_static(src, dst, ratio=1.0, tex_size=256, world_rot=None):
    """w_*/p_* world & player models: baked static mesh, optionally re-oriented."""
    j, b = load_glb(src)
    w = Writer()
    W = world_mats(j)
    # images
    new_images = []
    imgs = []
    for ii, im in enumerate(j.get('images', [])):
        raw = acc_img(j, b, ii)
        img = None
        if raw:
            try:
                img = Image.open(io.BytesIO(raw)).convert('RGB')
            except Exception:
                img = None
        if img is not None and img.size[0] > tex_size:
            img = repack(img, uv_mask(j, b, ii, img.size), img.size[0] / float(tex_size))
        if img is None:
            new_images.append({'mimeType': 'image/png'})
            continue
        vi = w.blob(encode_jpeg(img, 82))
        new_images.append({'bufferView': vi, 'mimeType': 'image/jpeg', 'name': im.get('name', 'img')})
        imgs.append(img)

    prims_out = []
    for ni, n in enumerate(j['nodes']):
        if 'mesh' not in n:
            continue
        for pr in j['meshes'][n['mesh']]['primitives']:
            pos = acc_np(j, b, pr['attributes']['POSITION']).astype(np.float64)
            uv = acc_np(j, b, pr['attributes']['TEXCOORD_0']).astype(np.float64)
            nrm = acc_np(j, b, pr['attributes']['NORMAL']).astype(np.float64)
            tri = (acc_np(j, b, pr['indices']).astype(np.int64).reshape(-1)
                   if 'indices' in pr else np.arange(len(pos)))
            M = W[ni]
            pos = pos @ M[:3, :3].T + M[:3, 3]
            if world_rot is not None:
                pos = pos @ world_rot[:3, :3].T + world_rot[:3, 3]
            nrm = nrm @ np.linalg.inv(M[:3, :3]).T
            nl = np.linalg.norm(nrm, axis=1)
            nl[nl == 0] = 1
            nrm = nrm / nl[:, None]
            key = np.concatenate([np.round(pos, 5), np.round(uv, 5)], axis=1)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            first = np.zeros(len(uniq), dtype=np.int64)
            seen = np.zeros(len(uniq), dtype=bool)
            for i, u in enumerate(inv):
                if not seen[u]:
                    seen[u] = True
                    first[u] = i
            V = pos[first]; UVs = uv[first]; NRM = nrm[first]
            T = inv[tri].reshape(-1, 3)
            if ratio < 1.0:
                V2, T2 = decimate(V, T, ratio)
                srcidx = kd_lookup(V, V2)
                UV2, NRM2 = UVs[srcidx], NRM[srcidx]
            else:
                V2, T2, UV2, NRM2 = V, T, UVs, NRM
            T2 = fix_winding(V2, T2, NRM2)
            a_pos = w.acc(V2, 5126, 'VEC3', target=34962)
            a_nrm = w.acc(NRM2, 5126, 'VEC3', target=34962, minmax=False)
            a_uv = w.acc(UV2, 5126, 'VEC2', target=34962, minmax=False)
            a_i = w.acc(T2.reshape(-1).astype(np.uint16 if len(V2) < 65536 else np.uint32),
                        5123 if len(V2) < 65536 else 5125, 'SCALAR', target=34963, minmax=False)
            prims_out.append({'attributes': {'POSITION': a_pos, 'NORMAL': a_nrm, 'TEXCOORD_0': a_uv},
                              'indices': a_i, 'material': pr.get('material', 0), 'mode': 4})
    mats = json.loads(json.dumps(j.get('materials', [])))
    for m in mats:
        pbr = m.get('pbrMetallicRoughness', {})
        pbr.setdefault('metallicFactor', 0.0)
        pbr['roughnessFactor'] = 0.55
    out = {
        'asset': {'version': '2.0', 'generator': 'cs15-godot import_models_repo.py'},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [{'name': os.path.basename(dst)[:-4], 'mesh': 0}],
        'meshes': [{'primitives': prims_out}],
        'materials': mats,
        'textures': [{'sampler': 0, 'source': t.get('source', 0)} for t in j.get('textures', [])],
        'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}],
        'images': new_images,
        'accessors': w.accessors,
        'bufferViews': w.views,
    }
    save_glb(out, bytes(w.bin), dst)
    return dict(path=dst, size=os.path.getsize(dst),
                tris=int(sum(w.accessors[p['indices']]['count'] for p in prims_out)) // 3,
                verts=int(sum(w.accessors[p['attributes']['POSITION']]['count'] for p in prims_out)))


# --------------------------------------------------------------------------- sounds
def sound_src(bank, rel):
    """Resolve a catalogue entry to a WAV in the bank.

    Supported forms:  'folder/stem', 'archi:folder/stem' (Archimedes pack, e.g.
    'archi:cs2_galilar/clipin'), and 'folder/stem|fallback/stem2|...' chains.
    """
    for part in str(rel).split('|'):
        part = part.strip()
        if not part:
            continue
        if part.startswith('archi:'):
            cands = [os.path.join(bank, 'sounds', 'weapons', 'archi', part[6:] + '.wav')]
        else:
            rel = part[8:] if part.startswith('weapons/') else part
            cands = [os.path.join(bank, 'sounds', 'weapons', rel + '.wav')]
        for p in cands:
            if os.path.exists(p):
                return p
    return None


def build_sound_bank(bank, report):
    """Copy/downmix every shot + foley sample the catalogue asks for."""
    import weapon_catalog as CAT
    fire_root = os.path.join(SND, 'fire')
    foley_root = os.path.join(SND, 'foley')
    for d in (fire_root, foley_root):
        os.makedirs(d, exist_ok=True)
    written = 0
    total = 0

    for wid, (folder, stems) in CAT.FIRE.items():
        out_dir = os.path.join(fire_root, wid)
        for i, stem in enumerate(stems, start=1):
            src = sound_src(bank, folder + '/' + stem)
            if src is None:
                report['missing_sounds'].append(folder + '/' + stem)
                continue
            dst = os.path.join(out_dir, '%d.wav' % i)
            total += convert_sound(src, dst) or 0
            written += 1

    knife_like = {kid for kid, _lbl in CAT.KNIVES}
    for wid, events in CAT.FOLEY.items():
        out_dir = os.path.join(foley_root, wid)
        for ev, stem in events.items():
            src = sound_src(bank, stem)
            if src is None:
                report['missing_sounds'].append(stem)
                continue
            dst = os.path.join(out_dir, ev + '.wav')
            total += convert_sound(src, dst) or 0
            written += 1

    for name, stem in CAT.EXTRA.items():
        src = sound_src(bank, stem)
        if src is None:
            report['missing_sounds'].append(stem)
            continue
        dst = os.path.join(foley_root, name + '.wav')
        total += convert_sound(src, dst) or 0
        written += 1

    # impact / shell-dropping "clatter" samples reused by ImpactEffects
    for stem, name in (('weapons/ric_metal-1', 'ric_metal-1'), ('weapons/ric_metal-2', 'ric_metal-2'),
                       ('weapons/ric_conc-1', 'ric_conc-1'), ('weapons/ric_conc-2', 'ric_conc-2'),
                       ('weapons/ric1', 'ric1'), ('weapons/ric2', 'ric2'), ('weapons/ric3', 'ric3'),
                       ('weapons/ric4', 'ric4'), ('weapons/ric5', 'ric5'),
                       ('weapons/knife_hitwall1', 'knife_hitwall1'),
                       ('weapons/knife_slash1', 'knife_slash1'),
                       ('weapons/knife_slash2', 'knife_slash2'),
                       ('weapons/knife_stab', 'knife_stab'),
                       ('weapons/c4_beep1', 'c4_beep1'), ('weapons/c4_explode1', 'c4_explode1'),
                       ('weapons/c4_plant', 'c4_plant'),
                       ('weapons/zoom', 'zoom'), ('grenade_hit1', 'grenade_hit1'),
                       ('grenade_hit2', 'grenade_hit2'), ('grenade_hit3', 'grenade_hit3')):
        src = sound_src(bank, stem)
        if src is None:
            continue
        dst = os.path.join(foley_root, name + '.wav')
        total += convert_sound(src, dst) or 0
        written += 1

    # per-weapon extras (second inspect/"look-at" sample, throw, bounce, ...)
    for wid in sorted(knife_like):
        if wid == 'knife':
            continue
        src_dir = os.path.join(foley_root, 'knife')
        dst_dir = os.path.join(foley_root, wid)
        if not os.path.isdir(src_dir):
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if f.endswith('.wav'):
                shutil.copyfile(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    for wid, events in CAT.FOLEY.items():
        extra_src = {
            'lookat_1': 'cs2/knife/knife_slash1', 'lookat_2': 'cs2/knife/knife_slash2',
            'lookat_3': 'cs2/knife/knife_stab',
        }
    for stem, name in (('archi:cs2_knife/knife_hit1', 'knife_hit1_alt'),):
        src = sound_src(bank, stem)
        if src is None:
            continue
        total += convert_sound(src, os.path.join(foley_root, name + '.wav')) or 0
        written += 1

    report['sounds_written'] = written
    report['sounds_bytes'] = total


# --------------------------------------------------------------------------- database
def sound_list(rel_dir):
    d = os.path.join(SND, rel_dir)
    if not os.path.isdir(d):
        return []
    names = sorted(f for f in os.listdir(d) if f.endswith('.wav'))
    return ['weapons/%s/%s' % (rel_dir.replace(os.sep, '/'), n) for n in names]


def foley_path(wid, ev):
    p = 'weapons/foley/%s/%s.wav' % (wid, ev)
    return p if os.path.exists(os.path.join(SND, 'foley', wid, ev + '.wav')) else ''


def foley_events(wid):
    """event name -> sound path, straight from the files written for that weapon."""
    d = os.path.join(SND, 'foley', wid)
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith('.wav'):
            out[f[:-4]] = 'weapons/foley/%s/%s' % (wid, f)
    return out


def clip_durations_of(path):
    j, b = load_glb(path)
    out = {}
    for a in j.get('animations', []):
        d = 0.0
        for s in a['samplers']:
            acc = j['accessors'][s['input']]
            mx = acc.get('max')
            if mx:
                d = max(d, float(mx[0]))
            else:
                try:
                    times = acc_np(j, b, s['input']).reshape(-1)
                    if len(times):
                        d = max(d, float(times[-1]))
                except Exception:
                    pass
        out[a.get('name', '')] = round(d, 4)
    return out, [a.get('name', '') for a in j.get('animations', [])], j


def merge_weapons_json(exported, report):
    """Rewrite data/weapons.json: new sound banks, new models, new weapons."""
    import weapon_catalog as CAT
    path = os.path.join(ROOT, 'data', 'weapons.json')
    db = json.load(open(path, encoding='utf-8'))
    weapons = db['weapons']

    # --- retire the viewmodel that the bank does not provide
    weapons.pop('sg552_v', None)

    # --- existing weapons: swap models + sounds
    for wid, spec in weapons.items():
        if wid not in CAT.MODELS:
            continue
        v, w, p = CAT.MODELS[wid]
        spec['v_model'] = 'v_' + v
        spec['w_model'] = 'w_' + w
        spec['p_model'] = 'p_' + p
        slot = int(spec.get('slot', 1))
        if slot in (1, 2) or wid == 'knife':
            shots = sound_list('fire/%s' % wid)
            if shots:
                if wid == 'm4a1':
                    pass  # handled below (silenced / unsilenced split)
                elif wid == 'usp':
                    spec['silenced_sounds'] = shots
                else:
                    spec['fire_sounds'] = shots
        foley = foley_events(wid)
        if foley.get('draw'):
            spec['deploy_sound'] = foley['draw']
        if foley.get('clipin'):
            spec['reload_sound'] = foley['clipin']
        elif foley.get('shellinsert'):
            spec['reload_sound'] = foley['shellinsert']
        spec['foley'] = foley
        if wid in ('hegrenade', 'flashbang', 'smokegrenade'):
            spec['fire_sounds'] = [x for x in [foley.get('pinpull')] if x]
            if foley.get('throw'):
                spec['throw_sound'] = foley['throw']
        if wid == 'knife':
            spec['fire_sounds'] = ['weapons/foley/knife_slash1.wav', 'weapons/foley/knife_slash2.wav']
            spec['hit_sounds'] = ['weapons/foley/knife_hit1.wav', 'weapons/foley/knife_hit2.wav']
            spec['stab_sound'] = 'weapons/foley/knife_stab.wav'
            spec['wall_sound'] = 'weapons/foley/knife_hitwall1.wav'
            if foley.get('draw'):
                spec['deploy_sound'] = foley['draw']
        if wid == 'c4':
            spec['fire_sounds'] = [x for x in [foley.get('key1')] if x]
            spec['plant_sound'] = 'weapons/foley/c4_plant.wav' if os.path.exists(
                os.path.join(SND, 'foley', 'c4_plant.wav')) else spec.get('plant_sound', '')
            spec['explode_sounds'] = ['weapons/foley/c4_explode1.wav'] if os.path.exists(
                os.path.join(SND, 'foley', 'c4_explode1.wav')) else spec.get('explode_sounds', [])
            spec['beep_sounds'] = ['weapons/foley/c4_beep1.wav'] if os.path.exists(
                os.path.join(SND, 'foley', 'c4_beep1.wav')) else spec.get('beep_sounds', [])
        if wid in ('hegrenade', 'flashbang', 'smokegrenade'):
            bnc = sound_list('foley')  # not used, keeps flake quiet

    # --- silenced / unsilenced variants
    if 'm4a1' in weapons:
        unsil = sound_list('fire/m4a1_unsil')
        sil = sound_list('fire/m4a1')
        if unsil:
            weapons['m4a1']['fire_sounds'] = unsil
        if sil:
            weapons['m4a1']['silenced_sounds'] = sil
    if 'usp' in weapons:
        unsil = sound_list('fire/usp_unsil')
        if unsil:
            weapons['usp']['fire_sounds'] = unsil

    # --- new weapons that CS 1.5 never had
    for wid, st in CAT.STATS.items():
        if wid not in CAT.MODELS:
            continue
        base = {
            'id': wid, 'slot': 1, 'zoomed_speed': st.get('max_speed', 225.0),
            'spread_base': st.get('spread_base', 0.008), 'spread_move': st.get('spread_move', 0.09),
            'range': 8192.0, 'penetration': 1, 'reload_time': 2.5, 'automatic': True,
            'recoil_up': 2.0, 'recoil_lat': 0.6, 'vgui': '', 'alt_damage': 0,
            'description': CAT.DESCRIPTIONS.get(wid, ''),
        }
        base.update({k: v for k, v in st.items()})
        base['slot'] = 1 if wid not in ('p2000', 'cz75', 'tec9', 'revolver') else 2
        v, w, p_model = CAT.MODELS[wid]
        base['v_model'] = 'v_' + v
        base['w_model'] = 'w_' + w
        base['p_model'] = 'p_' + p_model
        shots = sound_list('fire/%s' % wid)
        base['fire_sounds'] = shots
        foley = foley_events(wid)
        base['foley'] = foley
        if foley.get('draw'):
            base['deploy_sound'] = foley['draw']
        if foley.get('clipin'):
            base['reload_sound'] = foley['clipin']
        elif foley.get('shellinsert'):
            base['reload_sound'] = foley['shellinsert']
        weapons[wid] = base

    # --- knife skins (slot 3 "inspect" variants; they share the CS 1.5 knife stats)
    knife_src = json.loads(json.dumps(weapons['knife']))
    for kid, label in CAT.KNIVES:
        if kid == 'knife':
            continue
        spec = json.loads(json.dumps(knife_src))
        spec['id'] = kid
        spec['name'] = label
        spec['v_model'] = 'v_' + kid
        spec['w_model'] = 'w_knife'
        spec['p_model'] = 'p_' + kid
        spec['buy_cat'] = 0
        spec['buy_num'] = 0
        spec['clips'] = {'slash': ['midslash1', 'slash1', 'slash2'], 'stab': ['stab', 'midslash1'],
                         'inspect': ['inspect', 'idle2', 'idle']}
        weapons[kid] = spec

    # --- per weapon viewmodel clip table (so the controller picks real clips)
    for wid, spec in weapons.items():
        if 'clips' not in spec:
            spec['clips'] = {k: list(v) for k, v in CAT.CLIPS.items()}

    db['weapons'] = weapons
    db['weapon_order'] = sorted(weapons.keys())
    json.dump(db, open(path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    report['weapons'] = len(weapons)


def merge_models_meta(report):
    path = os.path.join(ROOT, 'data', 'models_meta.json')
    meta = json.load(open(path, encoding='utf-8'))
    for f in os.listdir(MODELS):
        if not f.endswith('.glb') or not f.startswith('v_'):
            continue
        name = f[:-4]
        full = os.path.join(MODELS, f)
        durs, names, j = clip_durations_of(full)
        entry = meta.get('weapons/' + name, {})
        entry['sequences'] = names
        entry['seq_durations'] = durs
        bones = len(j['skins'][0]['joints']) if j.get('skins') else 0
        entry['bones'] = bones
        verts = sum(j['accessors'][pr['attributes']['POSITION']]['count']
                    for m in j['meshes'] for pr in m['primitives'])
        entry['vertices'] = verts
        entry['asset'] = 'ShadowNigth/models'
        meta['weapons/' + name] = entry
    json.dump(meta, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    report['meta_entries'] = sum(1 for k in meta if k.startswith('weapons/'))


# --------------------------------------------------------------------------- driver
#  Viewmodel clip names that the game actually asks for (everything else dropped so the
#  GLB stays small).  Inspect/look-at clips are always kept: `inspect` is a first-class
#  feature of this build (see scripts/weapons/WeaponController.gd).
KEEP_CLIPS = [
    'draw', 'draw_unsil', 'idle', 'idle_unsil',
    'shoot1', 'shoot', 'shoot2', 'shoot3', 'fire', 'shoot1_unsil', 'shoot_unsil', 'shoot2_unsil',
    'shoot3_unsil', 'reload', 'reload_unsil', 'reload_empty', 'start_reload', 'after_reload',
    'insert', 'deploy', 'prepare', 'attach', 'detach', 'add_silencer', 'detach_silencer',
    'inspect', 'inspect_2', 'inspect_unsil', 'lookat',
    'slash1', 'slash2', 'midslash1', 'midslash2', 'stab', 'stab_miss',
    'pullpin', 'throw', 'underhand', 'pressbutton', 'zoom_in', 'zoom_out', 'holster',
]


FOLEY_TEMPLATE = {
    # ---- rifles / smgs / pistols
    'draw':            [('draw', 0.22)],
    'draw_unsil':      [('draw', 0.22)],
    'deploy':          [('draw', 0.20)],
    'idle':            [], 'idle2': [], 'idle3': [], 'idle_unsil': [],
    'shoot1':          [], 'shoot2': [], 'shoot3': [],
    'shoot':           [], 'shoot_unsil': [], 'shoot1_unsil': [],
    'shoot2_unsil':    [], 'shoot3_unsil': [],
    'reload':          [('clipout', 0.20), ('clipin', 0.72)],
    'reload_unsil':    [('clipout', 0.20), ('clipin', 0.72)],
    'reload_empty':    [('clipout', 0.20), ('boltback', 0.66), ('boltforward', 0.86)],
    'insert':          [('shellinsert', 0.10)],
    'pump':            [('pump', 0.35)],
    'attach':          [('silencer_on', 0.55)],
    'detach':          [('silencer_off', 0.45)],
    'inspect':         [('lookat_1', 0.28), ('lookat_2', 0.60), ('lookat_3', 0.82)],
    'inspect_unsil':   [('lookat_1', 0.28), ('lookat_2', 0.60)],
    'inspect_2':       [('lookat_2', 0.35), ('lookat_3', 0.70)],
    # ---- knife
    'slash1':          [], 'slash2': [], 'stab': [], 'stab_miss': [],
    'midslash1':       [], 'midslash2': [],
    # ---- grenades / c4
    'pullpin':         [('pinpull_start', 0.08), ('pinpull', 0.32)],
    'throw':           [('throw', 0.35)],
    'underhand':       [('throw', 0.30)],
    'fire1':           [], 'fire2': [],
    'pressbutton':     [],
    'plant':           [],
    # ---- scopes
    'zoom_in':         [('zoom_in', 0.05)],
    'zoom_out':        [('zoom_out', 0.05)],
}

#  Events whose audio the first-person viewmodel must own (the world also plays a 3D
#  copy for bots / other players, so the player's own copy is muted in WeaponController).
VM_LOCAL_EVENTS = ('draw', 'deploy', 'clipout', 'clipin', 'boltback', 'boltforward',
                   'silencer_on', 'silencer_off', 'lookat_1', 'lookat_2', 'lookat_3',
                   'pump', 'pinpull')


def foley_events_for_model(v_model_path):
    """Per-clip event lists keyed by clip name, using only samples that exist."""
    j, b = load_glb(v_model_path)
    clips = {a.get('name', ''): a for a in j.get('animations', [])}
    durs = clip_durations_of(v_model_path)[0]
    wid = None
    out = {}
    for clip, items in FOLEY_TEMPLATE.items():
        if clip not in clips:
            continue
        dur = durs.get(clip, 1.0)
        evs = []
        for ev, frac in items:
            evs.append({'t': round(max(0.0, dur * frac), 3), 'event': ev})
        if evs:
            out[clip] = evs
    return out, list(clips.keys()), durs


def merge_foley_meta(report):
    """data/weapons.json -> spec.foley (sound paths) already done; here we publish the
    per-clip event tables in data/models_meta.json so PlayerController can schedule them."""
    path = os.path.join(ROOT, 'data', 'models_meta.json')
    meta = json.load(open(path, encoding='utf-8'))
    wpath = os.path.join(ROOT, 'data', 'weapons.json')
    wdb = json.load(open(wpath, encoding='utf-8'))
    # map v_model -> weapon id (first weapon that uses it)
    by_model = {}
    for wid, spec in wdb['weapons'].items():
        vm = spec.get('v_model', '')
        if vm and vm not in by_model:
            by_model[vm] = wid
    count = 0
    for vm, wid in sorted(by_model.items()):
        glb = os.path.join(MODELS, vm + '.glb')
        if not os.path.exists(glb):
            continue
        events, clips, durs = foley_events_for_model(glb)
        key = 'weapons/' + vm
        entry = meta.setdefault(key, {})
        spec = wdb['weapons'][wid]
        foley = spec.get('foley', {})
        # alias a few event names onto whatever the bank actually provides
        alias = {'pinpull_start': 'pinpull', 'draw': foley.get('draw', ''),
                 'shellinsert': foley.get('shellinsert', foley.get('insert', '')),
                 'lookat_1': foley.get('lookat_1', foley.get('sliderelease', foley.get('draw', ''))),
                 'lookat_2': foley.get('lookat_2', foley.get('sliderelease', '')),
                 'lookat_3': foley.get('lookat_3', foley.get('boltforward', '')),
                 }
        foley = dict(foley, **{k: v for k, v in alias.items() if v})
        seq_events = {}
        for clip, evs in events.items():
            lst = []
            for e in evs:
                snd = foley.get(e['event'], '')
                if not snd and e['event'] in ('lookat_1', 'lookat_2', 'lookat_3'):
                    snd = foley.get('draw', '')
                    e = dict(e, event='draw')
                if not snd:
                    continue
                if lst and lst[-1]['wav'] == snd:
                    continue
                lst.append({'t': e['t'], 'wav': snd, 'event': e['event']})
            if lst:
                seq_events[clip] = lst
        entry['seq_events'] = seq_events
        entry['clips'] = clips
        entry['vm_local_events'] = [e for e in VM_LOCAL_EVENTS]
        count += 1
    json.dump(meta, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    report['foley_models'] = count

def main(argv):
    import weapon_catalog as CAT
    bank = argv[1] if len(argv) > 1 else '/tmp/models_probe'
    if not os.path.isdir(os.path.join(bank, 'weapons')):
        print('usage: import_models_repo.py <checkout of ShadowNigth/models>')
        return 2
    report = {'missing_sounds': [], 'files': []}

    # ---- 1. viewmodels
    vsrc = sorted(f for f in os.listdir(os.path.join(bank, 'weapons')) if f.startswith('v_') and f.endswith('.glb'))
    for i, f in enumerate(vsrc, 1):
        name = f[:-4]
        src = os.path.join(bank, 'weapons', f)
        dst = os.path.join(MODELS, f)
        info = convert_viewmodel(src, dst, ratio=(0.20 if 'knife' in name else 0.26),
                                 arms_ratio=(0.40 if 'knife' in name else 0.44), keep_names=KEEP_CLIPS)
        report['files'].append((f, info['size'], info['verts'], info['tris']))
        print('[v %2d/%d] %-30s %6.2f MB  %6d tris  %2d clips' %
              (i, len(vsrc), f, info['size'] / 1e6, info['tris'], len(info['anims'])))

    # ---- 2. world + third person models (identical geometry -> one decimated copy each)
    #      Only the models the game actually instantiates are converted: per-weapon w_*/p_*,
    #      the shared knife body, shell casings and equipment backpacks.
    KEEP_WORLD = set()
    for _wid, _st in CAT.STATS.items():
        if _wid in CAT.MODELS:
            KEEP_WORLD.add('w_' + CAT.MODELS[_wid][1])
            KEEP_WORLD.add('p_' + CAT.MODELS[_wid][2])
    for _wid, _st in CAT.MODELS.items():
        KEEP_WORLD.add('w_' + _st[1])
        KEEP_WORLD.add('p_' + _st[2])
    KEEP_WORLD |= {'w_knife', 'p_knife', 'pshell', 'rshell', 'rshell_big',
                   'w_backpack', 'w_kevlar', 'w_thighpack'}
    wsrc = sorted(f for f in os.listdir(os.path.join(bank, 'weapons'))
                  if f.startswith('w_') and f.endswith('.glb') and f[:-4] in KEEP_WORLD)
    for i, f in enumerate(wsrc, 1):
        src = os.path.join(bank, 'weapons', f)
        dst = os.path.join(MODELS, f)
        convert_static(src, dst, ratio=0.22, tex_size=256)
        p_dst = os.path.join(MODELS, 'p_' + f[2:])
        convert_static(src, p_dst, ratio=0.12, tex_size=256)
        print('[w %2d/%d] %-30s ok' % (i, len(wsrc), f))

    # ---- 3. sounds
    build_sound_bank(bank, report)

    # ---- 4. databases
    merge_weapons_json(report, report)
    merge_models_meta(report)
    merge_foley_meta(report)

    total = sum(os.path.getsize(os.path.join(MODELS, f)) for f in os.listdir(MODELS) if f.endswith('.glb'))
    report['models_bytes'] = total
    print('\n== summary ==')
    print('models: %.1f MB   sounds: %d files / %.1f MB' %
          (total / 1e6, report['sounds_written'], report['sounds_bytes'] / 1e6))
    print('weapons in db: %d   viewmodel meta: %d   foley tables: %d' %
          (report['weapons'], report['meta_entries'], report.get('foley_models', 0)))
    if report['missing_sounds']:
        print('missing sound sources: %s' % sorted(set(report['missing_sounds'])))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

