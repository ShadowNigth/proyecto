#!/usr/bin/env python3
"""
Bakes weapon / equipment sprites for the weapons that the ShadowNigth/models bank adds.

The CS 1.5 sprite sheets shipped with the project only cover the 27 original weapons.
For every new weapon we render its world model (w_*.glb) with a tiny software rasteriser
and emit

  assets/ui/vgui/640_<wid>.png   buy-menu preview (side view, 256x96)
  assets/sprites/weapon_<wid>.png  HUD / killfeed icon (side view, 128x48, alpha)

and then patch data/hud_sprites.json so GameData.get_weapon_hud_atlas() resolves them.

Usage:  python3 tools/bake_weapon_icons.py [asset_bank_dir]
"""
import io
import json
import math
import os
import struct
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import import_models_repo as IMP  # noqa: E402  (glb helpers)

VGUI_SIZE = (256, 96)
HUD_SIZE = (128, 48)
SUPERSAMPLE = 3


# --------------------------------------------------------------------------- glb reading
def load(path):
    j, b = IMP.load_glb(path)
    return j, b


def world_matrices(j):
    W = [None] * len(j['nodes'])
    parent = {}

    def local(n):
        if 'matrix' in n:
            return np.array(n['matrix'], dtype=np.float64).reshape(4, 4).T
        M = np.eye(4)
        t = n.get('translation', [0, 0, 0])
        r = n.get('rotation', [0, 0, 0, 1])
        s = n.get('scale', [1, 1, 1])
        x, y, z, w = r
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        M[:3, :3] = R @ np.diag(s)
        M[:3, 3] = t
        return M

    for i, n in enumerate(j['nodes']):
        for c in n.get('children', []):
            parent[c] = i
    for i in range(len(j['nodes'])):
        chain = []
        k = i
        while k is not None and W[k] is None:
            chain.append(k)
            k = parent.get(k)
        for k2 in reversed(chain):
            m = local(j['nodes'][k2])
            W[k2] = (W[parent[k2]] @ m) if k2 in parent else m
    return W, parent


def collect(j, b):
    """Returns [(tris Nx3x3 world-space, uvs Nx3x2, texture image or None)]."""
    W, _parent = world_matrices(j)
    out = []
    for ni, n in enumerate(j['nodes']):
        if 'mesh' not in n:
            continue
        mesh = j['meshes'][n['mesh']]
        for pr in mesh['primitives']:
            if 'attributes' not in pr or 'POSITION' not in pr['attributes']:
                continue
            pos = IMP.acc_np(j, b, pr['attributes']['POSITION']).astype(np.float64)
            pos = pos @ W[ni][:3, :3].T + W[ni][:3, 3]
            uv = (IMP.acc_np(j, b, pr['attributes']['TEXCOORD_0']).astype(np.float64)
                  if 'TEXCOORD_0' in pr['attributes'] else np.zeros((len(pos), 2)))
            if 'indices' in pr:
                idx = IMP.acc_np(j, b, pr['indices']).astype(np.int64).reshape(-1)
            else:
                idx = np.arange(len(pos))
            tris = idx.reshape(-1, 3)
            img = None
            mat = j['materials'][pr['material']] if 'material' in pr else {}
            tex = mat.get('pbrMetallicRoughness', {}).get('baseColorTexture')
            if tex is not None:
                src = j['textures'][tex['index']].get('source')
                if src is not None:
                    raw = IMP.acc_img(j, b, src)
                    if raw:
                        try:
                            img = Image.open(io.BytesIO(raw)).convert('RGB')
                        except Exception:
                            img = None
            out.append((pos[tris], uv[tris], img))
    return out


# --------------------------------------------------------------------------- rasteriser
def render_ortho(parts, size, pad=0.06, yaw_deg=0.0, pitch_deg=0.0, shadow=False):
    """Orthographic side view, auto-framed, transparent background (RGBA uint8 array)."""
    ss = SUPERSAMPLE
    W, H = int(size[0] * ss), int(size[1] * ss)
    img = np.zeros((H, W, 3), dtype=np.float64)
    alpha = np.zeros((H, W), dtype=np.float64)
    zbuf = np.full((H, W), 1e9)

    allp = np.concatenate([p[0].reshape(-1, 3) for p in parts]) if parts else np.zeros((1, 3))
    lo, hi = allp.min(axis=0), allp.max(axis=0)

    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    Rz = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R = Rz @ Ry
    P = allp @ R.T
    lo, hi = P.min(axis=0), P.max(axis=0)
    ext = np.maximum(hi - lo, 1e-4)
    scale = min(W * (1 - 2 * pad) / ext[0], H * (1 - 2 * pad) / ext[1])
    center = (lo + hi) * 0.5
    L = np.array([0.35, 0.75, 0.55])
    L = L / np.linalg.norm(L)

    for (tris, uvs, timg) in parts:
        Q = tris.reshape(-1, 3) @ R.T
        Q = Q.reshape(-1, 3, 3)
        sx = (Q[:, :, 0] - center[0]) * scale + W * 0.5
        sy_ = H * 0.5 - (Q[:, :, 1] - center[1]) * scale
        z = -(Q[:, :, 2] - center[2])
        ta = None
        if timg is not None:
            ta = np.asarray(timg, dtype=np.float64) / 255.0
            th, tw = ta.shape[:2]
        nrm = np.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
        nl = np.linalg.norm(nrm, axis=1)
        nl[nl == 0] = 1.0
        nrm = nrm / nl[:, None]
        area = ((sx[:, 1] - sx[:, 0]) * (sy_[:, 2] - sy_[:, 0])
                - (sx[:, 2] - sx[:, 0]) * (sy_[:, 1] - sy_[:, 0]))
        keep = np.abs(area) > 1e-9
        for i in np.nonzero(keep)[0]:
            # two-sided shading so we never need to fix winding
            d = abs(float(nrm[i] @ L))
            shade = 0.72 + 0.62 * d
            shade *= 1.0 + 0.35 * abs(float(nrm[i][2]))  # lift the flat side-on faces
            minx = max(int(math.floor(sx[i].min())), 0)
            maxx = min(int(math.ceil(sx[i].max())), W - 1)
            miny = max(int(math.floor(sy_[i].min())), 0)
            maxy = min(int(math.ceil(sy_[i].max())), H - 1)
            if minx > maxx or miny > maxy:
                continue
            xs = np.arange(minx, maxx + 1) + 0.5
            ys = np.arange(miny, maxy + 1) + 0.5
            X, Y = np.meshgrid(xs, ys)
            x0, y0 = sx[i, 0], sy_[i, 0]
            x1, y1 = sx[i, 1], sy_[i, 1]
            x2, y2 = sx[i, 2], sy_[i, 2]
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            w0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
            w1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
            w2 = 1 - w0 - w1
            m = (w0 >= -0.0005) & (w1 >= -0.0005) & (w2 >= -0.0005)
            if not m.any():
                continue
            zz = w0 * z[i, 0] + w1 * z[i, 1] + w2 * z[i, 2]
            sub = zbuf[miny:maxy + 1, minx:maxx + 1]
            better = m & (zz < sub)
            if not better.any():
                continue
            if ta is not None:
                uu = w0 * uvs[i, 0, 0] + w1 * uvs[i, 1, 0] + w2 * uvs[i, 2, 0]
                vv = w0 * uvs[i, 0, 1] + w1 * uvs[i, 1, 1] + w2 * uvs[i, 2, 1]
                px = np.clip((uu * tw).astype(int), 0, tw - 1)
                py = np.clip((vv * th).astype(int), 0, th - 1)
                col = ta[py, px] * shade
            else:
                col = np.ones(X.shape + (3,)) * (np.array([0.62, 0.62, 0.66]) * shade)
            dst = img[miny:maxy + 1, minx:maxx + 1]
            dst[better] = col[better]
            a = alpha[miny:maxy + 1, minx:maxx + 1]
            a[better] = 1.0
            sub[better] = zz[better]

    img = np.clip(img, 0, 1) ** 0.72   # brighten the very dark CS2 albedo
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[:, :, :3] = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out[:, :, 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(out, 'RGBA')
    return im.resize(size, Image.LANCZOS)


# --------------------------------------------------------------------------- png / import
def write_png(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, optimize=True)


def write_import(path):
    """Godot 4 .import sidecar so the texture is importable without opening the editor."""
    res = 'res://' + os.path.relpath(path, ROOT).replace(os.sep, '/')
    uid = 'uid://' + __import__('hashlib').md5(res.encode()).hexdigest()[:13]
    txt = f'''[remap]

importer="texture"
type="CompressedTexture2D"
uid="{uid}"
path="res://.godot/imported/{os.path.basename(path)}-{__import__('hashlib').md5(res.encode()).hexdigest()}.ctex"

[deps]

source_file="{res}"

[params]

compress/mode=0
compress/high_quality=false
compress/lossy_quality=0.7
compress/hdr_compression=1
compress/normal_map=0
compress/channel_pack=0
mipmaps/generate=false
mipmaps/limit=-1
roughness/mode=0
roughness/src_normal=""
process/fix_alpha_border=true
process/premult_alpha=false
process/normal_map_invert_y=false
process/hdr_as_srgb=false
process/hdr_clamp_exposure=false
process/size_limit=0
detect_3d/compress_to=0
'''
    with open(path + '.import', 'w') as f:
        f.write(txt)


def main(argv):
    bank = argv[1] if len(argv) > 1 else '/tmp/models_probe'
    wdb = json.load(open(os.path.join(ROOT, 'data', 'weapons.json'), encoding='utf-8'))
    hdb_path = os.path.join(ROOT, 'data', 'hud_sprites.json')
    hdb = json.load(open(hdb_path, encoding='utf-8'))
    weapons_hud = hdb.setdefault('weapons', {})

    made = []
    for wid, spec in sorted(wdb['weapons'].items()):
        wmodel = spec.get('w_model', 'w_' + wid)
        src = os.path.join(ROOT, 'assets', 'models', 'weapons', wmodel + '.glb')
        if not os.path.exists(src):
            src = os.path.join(bank, 'weapons', wmodel + '.glb')
        if not os.path.exists(src):
            continue

        vgui_path = os.path.join(ROOT, 'assets', 'ui', 'vgui', '640_%s.png' % wid)
        hud_path = os.path.join(ROOT, 'assets', 'sprites', 'weapon_%s.png' % wid)
        # never re-bake the sprite sheet entry of the 27 authentic CS 1.5 weapons
        has_cs15_sheet = ('weapon_' + wid) in weapons_hud
        if wid in ('knife', 'c4'):
            has_cs15_sheet = False  # no CS 1.5 buy card exists for these two
        need_vgui = (not has_cs15_sheet) and (not os.path.exists(vgui_path) or not spec.get('vgui'))

        j, b = load(src)
        parts = collect(j, b)
        if not parts:
            continue

        if need_vgui:
            im = render_ortho(parts, VGUI_SIZE, pad=0.05)
            write_png(vgui_path, im)
            write_import(vgui_path)
            spec['vgui'] = '640_' + wid

        if has_cs15_sheet:
            made.append(wid + '(vgui only)' if need_vgui else wid)
            continue
        if spec.get('vgui', '') == '':
            spec['vgui'] = '640_' + wid

        # HUD/killfeed icon: transparent cut-out, same framing as the CS sprites
        im2 = render_ortho(parts, HUD_SIZE, pad=0.04)
        write_png(hud_path, im2)
        write_import(hud_path)
        weapons_hud['weapon_' + wid] = {
            'weapon_640': {'sprite': 'weapon_' + wid, 'rect': [0, 0, HUD_SIZE[0], HUD_SIZE[1]]},
            'weapon_s_640': {'sprite': 'weapon_' + wid, 'rect': [0, 0, HUD_SIZE[0], HUD_SIZE[1]]},
            'weapon_320': {'sprite': 'weapon_' + wid, 'rect': [0, 0, HUD_SIZE[0], HUD_SIZE[1]]},
            'weapon_s_320': {'sprite': 'weapon_' + wid, 'rect': [0, 0, HUD_SIZE[0], HUD_SIZE[1]]},
        }
        made.append(wid)

    # killfeed / death-notice entries (d_<wid>) that CSHud resolves through get_hud_atlas
    hud_entries = hdb.setdefault('hud', {})
    for wid in made:
        for res in (320, 640):
            key = 'd_%s_%d' % (wid, res)
            if key in hud_entries:
                continue
            hud_entries[key] = {'name': 'd_' + wid, 'res': res,
                                'sprite': 'weapon_' + wid,
                                'rect': [0, 0, HUD_SIZE[0], HUD_SIZE[1]]}

    json.dump(hdb, open(hdb_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    json.dump(wdb, open(os.path.join(ROOT, 'data', 'weapons.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('[icons] baked %d weapon icons: %s' % (len(made), ', '.join(made)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
