import math as m
import random
import time

import numpy as np 

SIZE = 8
FPS = 30
SS = 2
ASPECT = 1.40
MAXROWS = 30
TK_MS = 13.0

T_OPEN = 0.9
T_OPEN_D = 2.4
T_BLINK = 4.0
T_Z0 = 5.4
T_ZD = 28.0
ZEND = 14.0
T_SPACE = T_Z0 + T_ZD
T_HEART = T_Z0 + 3.5
T_HEART_D = 26.0

WO = 0.78
HU, HL = 0.30, 0.20
RI = 0.27
RP0 = 0.098
GS0 = 0.085
GS_MAX = 1.5

RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
NR = len(RAMP) - 1
CIDX = {c: i for i, c in enumerate(RAMP)}
QL = 24

rng = np.random.RandomState(5)
prnd = random.Random(9)


def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def zp(t):
    return ease((t - T_Z0) / T_ZD)


def mix(a, b, k):
    return a + (b - a) * k


def _up2(b, n):
    g = b.shape[0]
    idx = np.arange(n) * g / n
    i0 = idx.astype(np.int64) % g
    i1 = (i0 + 1) % g
    f = idx - np.floor(idx)
    f = (f * f * (3 - 2 * f)).astype(np.float32)
    t = b[:, i0] * (1 - f) + b[:, i1] * f
    return t[i0, :] * (1 - f)[:, None] + t[i1, :] * f[:, None]


def _norm(n, k=4.0):
    return np.clip(0.5 + (n - n.mean()) / (n.std() * k + 1e-9), 0, 1).astype(np.float32)


def tex2d(n, grids):
    out = np.zeros((n, n), np.float32)
    amp = 1.0
    for g in grids:
        out += amp * _up2(rng.rand(g, g).astype(np.float32), n)
        amp *= 0.55
    return _norm(out)


def noise1d(n, g):
    base = rng.rand(g)
    idx = np.arange(n) * g / n
    i0 = idx.astype(np.int64) % g
    i1 = (i0 + 1) % g
    f = idx - np.floor(idx)
    f = f * f * (3 - 2 * f)
    return (base[i0] * (1 - f) + base[i1] * f).astype(np.float32)


def bil(tex, u, v):
    n = tex.shape[0]
    u0 = np.floor(u)
    v0 = np.floor(v)
    fu = u - u0
    fv = v - v0
    i0 = u0.astype(np.int32) % n
    j0 = v0.astype(np.int32) % n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    return ((tex[j0, i0] * (1 - fu) + tex[j0, i1] * fu) * (1 - fv)
            + (tex[j1, i0] * (1 - fu) + tex[j1, i1] * fu) * fv)


print("Preparing textures...")
TS = tex2d(1024, (6, 20, 64, 200))
TG = tex2d(1024, (8, 24, 64, 160))
TW = rng.rand(512, 512).astype(np.float32)
_tv = tex2d(512, (5, 12, 30))
TV = 1 - np.abs(2 * _tv - 1)
_tv2 = tex2d(512, (3, 7, 16))
TV2 = 1 - np.abs(2 * _tv2 - 1)

NA = 4096
A1 = _norm(0.5 * noise1d(NA, 900) + 0.3 * noise1d(NA, 300) + 0.2 * noise1d(NA, 120), 3.0)
A2 = _norm(0.5 * noise1d(NA, 700) + 0.3 * noise1d(NA, 200) + 0.2 * noise1d(NA, 80), 3.0)
A3 = _norm(noise1d(NA, 13), 3.0)
A4 = _norm(noise1d(NA, 48), 3.0)

NS = 3200
SX = rng.uniform(-1, 1, NS).astype(np.float32)
SY = rng.uniform(-1, 1, NS).astype(np.float32)
SD = rng.uniform(0.06, 1.0, NS).astype(np.float32)
SB = (0.10 + 0.90 * rng.rand(NS) ** 3.5).astype(np.float32)
_tm = rng.rand(NS, 1).astype(np.float32)
SCOL = mix(np.array([0.70, 0.82, 1.0], np.float32), np.array([1.0, 0.80, 0.55], np.float32), _tm ** 1.5)
SPH = rng.uniform(0, 6.28, NS).astype(np.float32)
SF = rng.uniform(1.0, 3.5, NS).astype(np.float32)


GZ = [(0.0, 0.0, 0.0)]
_tc = 1.7
_gr = random.Random(4)
while _tc < 40:
    GZ.append((_tc, _gr.uniform(-0.045, 0.045), _gr.uniform(-0.022, 0.02)))
    _tc += _gr.uniform(0.9, 2.4)


_lr = random.Random(21)
UP_L = []
NUP = 44
for i in range(NUP):
    UP_L.append((-0.92 + 1.84 * i / (NUP - 1), _lr.uniform(0.92, 1.08), 0.0, 1.0))
LO_L = []
NLO = 20
for i in range(NLO):
    LO_L.append((-0.70 + 1.40 * i / (NLO - 1), _lr.uniform(0.90, 1.10), 0.0, 1.0))


D = {}


def init(cw, ch, rows):
    global CW, CH, ROWS, COLS, W, H, GW, GH, X, Y, RS, XC, YC, RSC, HSC, ARROWS_OK
    CW, CH = cw, ch
    ROWS = rows
    COLS = int(rows * ch * ASPECT / cw) // 2 * 2
    ROWS = ROWS // 2 * 2
    W, H = COLS * CW, ROWS * CH
    GW, GH = COLS * SS, ROWS * SS
    xs = (np.arange(GW, dtype=np.float32) + 0.5) * (CW / SS)
    ys = (np.arange(GH, dtype=np.float32) + 0.5) * (CH / SS)
    X = (((xs - W / 2) / H)[None, :] * np.ones((GH, 1), np.float32)).astype(np.float32)
    Y = (((ys - H / 2) / H)[:, None] * np.ones((1, GW), np.float32)).astype(np.float32)
    RS = np.sqrt(X * X + Y * Y)
    cx = (np.arange(COLS, dtype=np.float32) + 0.5) * CW
    cy = (np.arange(ROWS, dtype=np.float32) + 0.5) * CH
    XC = (((cx - W / 2) / H)[None, :] * np.ones((ROWS, 1), np.float32)).astype(np.float32)
    YC = (((cy - H / 2) / H)[:, None] * np.ones((1, COLS), np.float32)).astype(np.float32)
    RSC = np.sqrt(XC * XC + YC * YC)
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    HSC = (((rr * 73856093) ^ (cc * 19349663) ^ 0x5bd1e995) % 1009 / 1009.0).astype(np.float32)
    reset()


def blink_c(t, tb=T_BLINK):
    d = t - tb
    if d < 0 or d > 0.36:
        return 0.0
    if d < 0.09:
        return ease(d / 0.09)
    if d < 0.12:
        return 1.0
    return 1 - ease((d - 0.12) / 0.24)


def lid_open(t):
    k = ease((t - T_OPEN) / T_OPEN_D)
    k = k * (1 + 0.035 * m.sin(m.pi * k))
    return min(1.0, k) * (1 - 0.985 * blink_c(t))


def gaze(t, prog):
    prev = (0.0, 0.0)
    cur = (0.0, 0.0)
    te = 0.0
    for (t_, gx_, gy_) in GZ:
        if t_ <= t:
            prev = cur
            cur = (gx_, gy_)
            te = t_
        else:
            break
    k = ease((t - te) / 0.09)
    gx = prev[0] + (cur[0] - prev[0]) * k
    gy = prev[1] + (cur[1] - prev[1]) * k
    gx += 0.003 * m.sin(t * 1.3)
    gy += 0.002 * m.sin(t * 1.1 + 1)
    amp = 1 - ease((t - T_Z0) / 2.5)
    gy -= 0.03 * blink_c(t)
    return gx * amp, gy * amp


def lid_edges_scalar(u0, ou):
    s = u0 / WO
    wu = 0.88 if s < -0.12 else 1.12
    cu = max(0.0, 1 - ((s + 0.12) / wu) ** 2) ** 0.8
    cl = max(0.0, 1 - (s - 0.08) ** 2) ** 0.9
    vc = 0.03 * cl
    vl = vc + (HL * cl - vc) * (0.35 + 0.65 * ou)
    vu = vl + (-HU * cu - vl) * ou
    return vu, vl, cu, cl


def heart_nebula(xs, ys, sz, hm, t):
    hx = xs / sz
    hy = -ys / sz
    nC = bil(TG, hx * 16 + 400, hy * 16 + 100)
    nD = bil(TG, hx * 16 + 90, hy * 16 + 530)
    nA = bil(TG, hx * 42 + 300, hy * 42 + 700 + t * 0.8)
    nB = bil(TG, hx * 110 + 80, hy * 110 + 200)
    wob = 1 - 0.65 * hm
    hxw = hx + wob * 0.40 * (nC - 0.5) + 0.05 * (nA - 0.5)
    hyw = hy + wob * 0.40 * (nD - 0.5) + 0.05 * (nA - 0.5)
    qc = np.sqrt(hxw * hxw + hyw * hyw) / 0.80 - 1.0
    hyh = hyw + 0.12
    fh = (hxw * hxw + hyh * hyh - 1.0) ** 3 - hxw * hxw * hyh ** 3
    qh = 0.5 * np.cbrt(fh)
    q = qc * (1 - hm) + qh * hm
    ins = np.clip(-q / (0.80 - 0.35 * hm), 0, 1)
    k = 3.0 + 2.5 * hm
    body = np.where(q < 0, 0.50 + 0.50 * np.sqrt(ins), np.exp(-np.clip(q, 0, 10) * k) * 0.50)
    tex = 1 + (0.62 + 0.80 * nB - 1) * 0.8
    deep = np.array([0.92, 0.16, 0.40], np.float32)
    light = np.array([1.0, 0.58, 0.78], np.float32)
    hcol = mix(deep, light, (ins ** 0.8)[..., None])
    col = (body * tex)[..., None] * hcol

    halo = np.exp(-np.clip(q, 0, 10) * 1.6) * 0.20 * (0.5 + nA) * (q > 0)
    col = col + halo[..., None] * np.array([0.78, 0.10, 0.30], np.float32)

    rim = np.exp(-((q + 0.04) / 0.14) ** 2)
    rid = np.clip((bil(TV, hxw * 120 + 50, hyw * 120 + 300 + t * 0.6) - 0.86) * 7, 0, 1)
    red = np.array([0.95, 0.16, 0.34], np.float32)
    col = col + (rim * (0.20 + 0.9 * rid) * (0.5 + nA))[..., None] * red
    col = col + (rid * np.clip(-q, 0, 1) * 0.25)[..., None] * red
    return col.astype(np.float32)


def eye_scene(t, z, ou, gx, gy, Rp, sp):
    iz = 1.0 / z
    px = iz / GH
    u = X * iz
    v = Y * iz
    near = z < 3.4
    pm_u = u - gx
    pm_v = v - gy
    r = np.sqrt(pm_u * pm_u + pm_v * pm_v)
    d_in = None
    if near:
        s = u / WO
        sa = np.clip(s, -1.3, 1.3)
        wu = np.where(sa < -0.12, 0.88, 1.12)
        cu = np.clip(1 - ((sa + 0.12) / wu) ** 2, 0, 1) ** 0.8
        cl = np.clip(1 - (sa - 0.08) ** 2, 0, 1) ** 0.9
        vt = v + 0.055 * s
        vc = 0.03 * cl
        vl = vc + (HL * cl - vc) * (0.35 + 0.65 * ou)
        vu = vl + (-HU * cu - vl) * ou
        du_in = vt - vu
        dl_in = vl - vt
        d_in = np.minimum(du_in, dl_in)
        m_in = np.clip(d_in / (1.6 * px) + 0.5, 0, 1)[..., None]


    ia = (((np.arctan2(pm_v, pm_u) / (2 * m.pi)) % 1.0) * NA).astype(np.int32) % NA
    rn = np.clip((r - Rp) / (RI - Rp), 0, 1.3)
    rb = rn * 14
    i0 = np.floor(rb).astype(np.int32)
    fr = rb - i0
    fib = 0.55 * A1[ia] + 0.45 * (A2[(ia + i0 * 131) % NA] * (1 - fr) + A2[(ia + (i0 + 1) * 131) % NA] * fr)
    ang = (np.arctan2(pm_v, pm_u) / (2 * m.pi)) % 1.0
    zig = np.abs((ang * 34) % 1.0 - 0.5) * 2 - 0.5
    cr = 0.34 + 0.20 * (A3[ia] - 0.5) + 0.035 * zig
    mm = ss(-0.10, 0.22, rn - cr)[..., None]
    amber = np.array([0.80, 0.45, 0.10], np.float32)
    rust = np.array([0.42, 0.20, 0.06], np.float32)
    ir = mix(amber, rust, mm)
    ir = ir * (0.45 + 1.05 * fib)[..., None]
    ir += np.array([0.22, 0.11, 0.0], np.float32) * (np.exp(-((rn - cr) / 0.07) ** 2) * (0.2 + fib))[..., None]
    cry = (A4[ia] > 0.72) * ss(0.65, 0.18, rn)
    ir *= (1 - 0.55 * cry)[..., None]
    ir *= (1 - 0.82 * ss(0.62, 1.0, rn))[..., None]
    ir *= (0.55 + 0.45 * ss(0, 0.07, rn))[..., None]
    if near:
        ir *= (0.30 + 0.70 * ss(0, 0.16, du_in))[..., None]


    if near:
        cw = ss(0.40, 0.95, np.abs(s))
        sh = (0.38 + 0.62 * ss(0, 0.22, du_in)) * (0.72 + 0.28 * ss(0, 0.10, dl_in)) * (1 - 0.30 * cw)
        scl = np.array([0.80, 0.82, 0.86], np.float32) * sh[..., None]
        scl = mix(scl, np.array([0.86, 0.50, 0.46], np.float32) * sh[..., None], (cw * 0.35)[..., None])
        vein = np.clip((bil(TV2, u * 85 + 40, v * 85 + 70) - 0.925) * 14, 0, 1) * (0.25 + 0.75 * cw) * ss(0.12, 0.32, np.abs(pm_u))
        scl = mix(scl, np.array([0.62, 0.15, 0.14], np.float32), (vein * 0.65)[..., None])
        im = np.clip((RI - r) / (1.6 * px) + 0.5, 0, 1)[..., None]
        inside = mix(scl, ir, im)
    else:
        inside = ir


    Rps = Rp * z
    vig = 0.25 + 0.75 * ss(0, 0.30 * Rps, Rps - RS)
    pm = np.clip((Rp - r) / (1.6 * px) + 0.5, 0, 1)
    inside = mix(inside, sp * vig[..., None], pm[..., None])


    fade = max(0.0, min(1.0, 1 - (z - 1.6) / 2.4))
    if fade > 0:
        dd = np.sqrt((pm_u - 0.075) ** 2 + (pm_v + 0.040) ** 2)
        win = np.clip(1 - (dd - 0.030) / (1.4 * px + 0.004), 0, 1)
        dd2 = np.sqrt((pm_u + 0.045) ** 2 + (pm_v - 0.050) ** 2)
        win2 = np.clip(1 - (dd2 - 0.009) / (1.4 * px + 0.003), 0, 1) * 0.5
        inside = inside + ((win + win2) * fade)[..., None] * np.array([1.0, 0.98, 0.95], np.float32) * 1.4

    if not near:
        return inside, pm


    if z < 2.6:
        mid = 0.5 * (vu + vl)
        upper = vt < mid
        du = np.maximum(vu - vt, 0)
        dl = np.maximum(vt - vl, 0)
        dd = np.where(upper, du, dl)
        tx = bil(TS, u * 120 + 100, v * 120 + 100)
        base = np.array([0.62, 0.40, 0.29], np.float32)
        skin = base * (0.90 + 0.20 * tx)[..., None]
        sh = 0.50 + 0.50 * ss(0, 0.30, dd)
        sh *= 1 + 0.10 * np.exp(-((v + 0.20) / 0.09) ** 2)
        cr_pos = 0.075 + 0.03 * (1 - ou)
        sh *= 1 - 0.42 * np.exp(-((du - cr_pos) / 0.020) ** 2) * upper
        sh *= 1 + 0.08 * np.exp(-((dl - 0.09) / 0.04) ** 2) * (~upper)
        skin = skin * sh[..., None]
        lidzone = ss(cr_pos, cr_pos * 0.4, du) * upper
        skin = mix(skin, np.array([0.68, 0.40, 0.31], np.float32) * sh[..., None], (lidzone * 0.5)[..., None])

        sb = np.clip(1 - np.abs(s), 0, 1)
        eb = np.exp(-((vt + 0.42 + 0.10 * s * s) / (0.040 + 0.012 * sb)) ** 2)
        hair = 0.55 + 0.9 * bil(TW, u * 90, v * 30)
        skin = mix(skin, np.array([0.10, 0.06, 0.04], np.float32), np.clip(eb * hair, 0, 0.92)[..., None])

        line = np.exp(-(du / 0.010) ** 2) * upper
        skin = mix(skin, np.array([0.05, 0.03, 0.022], np.float32), np.clip(line * (1.0 - 0.3 * ou) * 0.9, 0, 1)[..., None])
        wl = ss(0.014, 0.0, dl) * (~upper)
        skin = mix(skin, np.array([0.74, 0.34, 0.30], np.float32), (wl * 0.7)[..., None])
        gl = np.exp(-((dl - 0.016) / 0.006) ** 2) * (~upper) * (0.35 + 0.65 * bil(TW, u * 60, v * 60))
        skin = skin + (gl * 0.55)[..., None] * np.array([1.0, 0.85, 0.75], np.float32) * ss(0.12, 0.5, ou)
        sp_ = (bil(TW, u * 150 + 7, v * 150 + 3) > 0.985) * 0.35 * (0.4 + tx) * ss(0.0, 0.3, dd)
        skin = skin + sp_[..., None] * np.array([1.0, 0.9, 0.8], np.float32)
        col = mix(skin, inside, m_in)

        cg = np.exp(-(((u + WO * 0.90) / 0.045) ** 2 + ((vt - 0.025) / 0.032) ** 2)) * ss(0.1, 0.5, ou)
        col = mix(col, np.array([0.82, 0.42, 0.38], np.float32), (cg * 0.85)[..., None])
        return col, pm
    return inside, pm


streaks = []
S = {"last": 0.0, "next_shoot": 1.0e9, "comet": False}


class Streak:
    def __init__(self, x, y, vx, vy, life, tail, col, soft=False):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life, self.tail, self.col, self.soft = life, tail, col, soft
        self.age = 0.0


def reset():
    S.update(last=0.0, next_shoot=1.0e9, comet=False)
    streaks.clear()
    SD[:] = rng.uniform(0.06, 1.0, NS).astype(np.float32)


def update_world(t, dt, prog):
    vf = 0.010 + 0.05 * prog + (0.015 if t > T_SPACE else 0.0)
    SD[:] -= vf * dt
    dead = SD < 0.06
    if dead.any():
        SD[dead] += 0.94
        n = int(dead.sum())
        SX[dead] = rng.uniform(-1, 1, n)
        SY[dead] = rng.uniform(-1, 1, n)
    if S["next_shoot"] > 1e8 and t > T_SPACE - 4.0:
        S["next_shoot"] = t + 1.0
    if t > S["next_shoot"]:
        S["next_shoot"] = t + prnd.uniform(2.2, 4.5)
        ang = m.radians(prnd.uniform(18, 62))
        sgn = prnd.choice((-1, 1))
        spd = prnd.uniform(0.7, 1.2)
        streaks.append(Streak(prnd.uniform(-0.7, 0.7), prnd.uniform(-0.45, -0.15),
                              sgn * m.cos(ang) * spd, m.sin(ang) * spd,
                              prnd.uniform(0.9, 1.4), prnd.uniform(0.14, 0.26),
                              np.array([0.88, 0.94, 1.0], np.float32)))
    if t > T_SPACE + 1.5 and not S["comet"]:
        S["comet"] = True
        streaks.append(Streak(-0.95, -0.28, 0.030, 0.008, 70.0, 0.22,
                              np.array([0.70, 0.86, 1.0], np.float32), soft=True))
    for s_ in streaks:
        s_.age += dt
    streaks[:] = [s_ for s_ in streaks if s_.age < s_.life]


def put(idx, fg, r, c, ch, color, a=1.0):
    if 0 <= r < ROWS and 0 <= c < COLS:
        idx[r, c] = CIDX[ch]
        fg[r, c] = color * a


def stamp_streak(s_, idx, fg, pmc, vigc):
    hx = s_.x + s_.vx * s_.age
    hy = s_.y + s_.vy * s_.age
    sp = m.hypot(s_.vx, s_.vy)
    dx, dy = s_.vx / sp, s_.vy / sp
    fade = m.sin(m.pi * min(1.0, s_.age / s_.life)) ** 0.7
    if fade < 0.02:
        return
    if abs(dy) < 0.38 * abs(dx):
        tc = "-"
    elif abs(dx) < 0.38 * abs(dy):
        tc = "|"
    elif dx * dy > 0:
        tc = "\\"
    else:
        tc = "/"
    n = max(6, int(s_.tail * H / (CW * 0.8)))
    for k in range(n):
        f = k / n
        c = int((W / 2 + (hx - dx * s_.tail * f) * H) / CW)
        r = int((H / 2 + (hy - dy * s_.tail * f) * H) / CH)
        b = (1 - f) ** 1.3 * fade
        if b < 0.08 or not (0 <= r < ROWS and 0 <= c < COLS) or pmc[r, c] < 0.5:
            continue
        if k == 0:
            ch = "@" if not s_.soft else "*"
        elif s_.soft:
            ch = ":" if b > 0.35 else "."
        else:
            ch = tc if b > 0.30 else "."
        put(idx, fg, r, c, ch, s_.col, min(1.0, 0.3 + b) * vigc[r, c])


def stamp_stars(idx, fg, L, t, gs, amt, pmc, vigc, offx, offy):
    if amt <= 0.003:
        return
    d = SD
    c = ((W * 0.5 + (SX * gs / d + offx) * H) / CW).astype(np.int32)
    r = ((H * 0.5 + (SY * gs / d + offy) * H) / CH).astype(np.int32)
    alpha = ss(1.0, 0.82, d) * (0.80 + 0.20 * np.sin(t * SF + SPH)) * amt
    ok = (c >= 0) & (c < COLS) & (r >= 0) & (r < ROWS) & (alpha > 0.04)
    if not ok.any():
        return
    ii = np.flatnonzero(ok)
    c, r = c[ii], r[ii]
    b = SB[ii] * alpha[ii] * (1 + 0.05 / d[ii])
    msk = pmc[r, c] > 0.5
    ii, c, r, b = ii[msk], c[msk], r[msk], b[msk]
    keep = b > L[r, c] * 0.7
    ii, c, r, b = ii[keep], c[keep], r[keep], b[keep]
    if len(b) == 0:
        return
    ch = np.where(b < 0.20, CIDX["."], np.where(b < 0.42, CIDX["+"], CIDX["*"]))
    idx[r, c] = ch
    fg[r, c] = SCOL[ii] * np.clip(0.35 + 0.9 * b, 0, 1)[:, None] * vigc[r, c][:, None]

    big = np.flatnonzero(b > 0.55)
    for j in big[:40]:
        cc_, rr_, bb = int(c[j]), int(r[j]), float(b[j])
        col = SCOL[ii[j]] * vigc[rr_, cc_]
        idx[rr_, cc_] = CIDX["@"] if bb > 0.85 else CIDX["*"]
        ln = int(2 + 6 * (bb - 0.5))
        for k in range(1, ln + 1):
            a = (1 - k / (ln + 1)) * bb
            for (dr, dc, chh) in ((0, k, "-"), (0, -k, "-"), (k // 2 + (1 if k > 2 else 0), 0, "|"), (-(k // 2 + (1 if k > 2 else 0)), 0, "|")):
                if dr == 0 and dc == 0:
                    continue
                r2, c2 = rr_ + dr, cc_ + dc
                if 0 <= r2 < ROWS and 0 <= c2 < COLS and pmc[r2, c2] > 0.5 and L[r2, c2] < a * 0.8:
                    idx[r2, c2] = CIDX[chh if a > 0.18 else "."]
                    fg[r2, c2] = col * min(1.0, 0.25 + a)


def stamp_lashes(idx, fg, t, z, ou, gx, gy):
    if z > 2.6 or ou < 0.0:
        return
    cl = 1 - ou ** 0.8

    def to_cell(ue, ve):
        return int((W / 2 + ue * z * H) / CW), int((H / 2 + ve * z * H) / CH)

    def pick(aa):
        ang = m.degrees(m.atan2(-m.sin(aa), m.cos(aa))) % 180
        if ang < 25 or ang >= 155:
            return "-"
        if ang < 70:
            return "/"
        if ang < 110:
            return "|"
        return "\\"

    def draw(base_u, base_v, a0, L, curl, c_root, c_tip, nmax, tip):
        n = int(max(2, min(nmax, L * z * H / (CW * 0.85))))
        step = L / n
        ue, ve = base_u, base_v
        for k in range(n):
            tt = k / max(1, n - 1)
            aa = a0 + curl * tt * tt
            ue += m.cos(aa) * step
            ve += m.sin(aa) * step
            c, r = to_cell(ue, ve)
            if not (0 <= r < ROWS and 0 <= c < COLS):
                continue
            ch = tip if (k == n - 1 and n >= 4) else pick(aa)
            idx[r, c] = CIDX[ch]
            fg[r, c] = c_root + (c_tip - c_root) * tt

    up_root = np.array([0.16, 0.10, 0.07], np.float32)
    up_tip = np.array([0.30, 0.19, 0.13], np.float32)
    lo_root = np.array([0.17, 0.11, 0.08], np.float32)
    lo_tip = np.array([0.30, 0.20, 0.14], np.float32)

    for (s0, lf_, ph, cv_) in UP_L:
        u0 = s0 * WO
        vu, vl, cu, cl_ = lid_edges_scalar(u0, ou)
        bv = vu - 0.055 * s0
        a_open = m.radians(-90 + 26 * s0)
        a_closed = m.radians(90 - 20 * s0) if s0 >= 0 else m.radians(-270 - 20 * s0)
        a0 = a_open + (a_closed - a_open) * cl
        edge = m.sqrt(max(0.0, 1 - s0 * s0))
        Lh = (0.070 + 0.050 * edge) * lf_
        curl = m.radians(-22 * s0) * (1 - cl)
        draw(u0, bv, a0, Lh, curl, up_root, up_tip, 14, "'")
    for (s0, lf_, ph, cv_) in LO_L:
        u0 = s0 * WO
        vu, vl, cu, cl_ = lid_edges_scalar(u0, ou)
        bv = vl - 0.055 * s0
        a0 = m.radians(90 - 20 * s0)
        Lh = 0.034 * (0.5 + 0.8 * cl_) * lf_
        draw(u0, bv, a0, Lh, m.radians(12 * (1 if s0 > 0 else -1)), lo_root, lo_tip, 6, ".")


def bloom(img):
    b = np.clip(img - 0.45, 0, None)
    out = img
    for f, w_ in ((2, 0.40), (4, 0.30)):
        s_ = b.reshape(GH // f, f, GW // f, f, 3).mean((1, 3))
        for _ in range(2):
            s_ = (s_ * 2 + np.roll(s_, 1, 0) + np.roll(s_, -1, 0) + np.roll(s_, 1, 1) + np.roll(s_, -1, 1)) / 6
        out = out + w_ * np.repeat(np.repeat(s_, f, 0), f, 1)
    return out


def render_cells(t, dt):
    S["last"] = t
    prog = zp(t)
    idle = ease(t / T_Z0)
    z = m.exp(m.log(ZEND) * prog) * (1 + 0.05 * idle + 0.004 * m.sin(t * 0.9))
    zsp = z if t < T_SPACE else ZEND * m.exp(0.03 * (t - T_SPACE))
    gs = min(6.0, GS0 * zsp ** 1.25)
    gh = min(1.15, GS0 * z ** 1.25)
    ou = lid_open(t)
    gx, gy = gaze(t, prog)
    Rp = RP0 * (0.76 + 0.24 * ease((t - 1.8) / 3.5)) * (1 + 0.025 * m.sin(1.9 * t)) * (1 + 0.30 * prog)
    Rps = Rp * z
    full = Rps * 0.72 > 1.03

    star_amt = ease((t - (T_Z0 + 1.5)) / 4.5)
    neb_amt = ease((t - (T_Z0 + 3.5)) / 8.0)
    update_world(t, dt, prog)

    offx, offy = gx * z, gy * z
    sp = np.empty((GH, GW, 3), np.float32)
    sp[:] = (0.004, 0.005, 0.012)
    if neb_amt > 0.002:
        drift = ease((t - T_SPACE) / 20.0)
        hm = ease((t - T_HEART) / T_HEART_D)
        beat = 1 + 0.03 * m.sin(t * 2.6) * hm * hm
        sp += heart_nebula(X - offx - 0.03 * drift * m.sin(t * 0.07), Y - offy,
                           0.17 * gh * beat, hm, t) * (1.25 * neb_amt ** 1.3)

    if full:
        img = sp
        pmc = np.ones((ROWS, COLS), np.float32)
        solid = np.zeros((ROWS, COLS), bool)
    else:
        img, pm = eye_scene(t, z, ou, gx, gy, Rp, sp)
        pmc = pm.reshape(ROWS, SS, COLS, SS).mean((1, 3))
        solid = pmc < 0.5

    img = bloom(img)
    img = 1 - np.exp(-2.0 * np.clip(img, 0, None))
    img = img * (1 - 0.30 * ss(0.4, 1.0, RS))[..., None] * ease(t / 1.2)
    cells = img.reshape(ROWS, SS, COLS, SS, 3).mean((1, 3))
    lum = 0.30 * cells[..., 0] + 0.60 * cells[..., 1] + 0.10 * cells[..., 2]
    li = np.clip(lum, 0, 1) ** 0.72
    ri = np.clip((li + (HSC - 0.5) * 0.07) * NR, 0, NR).astype(np.int32)
    ri = np.where(solid, np.maximum(ri, 3), ri)
    idx = ri.astype(np.int32)
    fg = np.clip(cells * 1.15, 0, 1).astype(np.float32)

    vigc = (0.25 + 0.75 * ss(0, 0.30 * Rps, Rps - RSC)).astype(np.float32)
    stamp_stars(idx, fg, li, t, gs, star_amt, pmc, vigc, offx, offy)
    for s_ in streaks:
        stamp_streak(s_, idx, fg, pmc, vigc)
    if not full:
        stamp_lashes(idx, fg, t, z, ou, gx, gy)

    q = np.clip((np.clip(fg, 0, 1) * (QL - 0.01)).astype(np.int16), 0, QL - 1)
    return idx, q


def plan_updates(idx, q, d_idx, d_q, budget):
    same = idx == d_idx
    delta = np.abs(q - d_q).max(axis=2)
    blank_both = (idx == 0) & (d_idx == 0)
    need = (~blank_both) & ((~same) | (delta >= 2))
    cells = np.flatnonzero(need)
    if len(cells) > budget:
        prio = ((~same).astype(np.int16) * 6 + delta).ravel()[cells]
        top = np.argpartition(-prio, budget)[:budget]
        cells = cells[top]
    return cells


def main():
    import tkinter as tk
    import tkinter.font as tkf

    root = tk.Tk()
    root.title("Eye -> Space")
    root.resizable(False, False)
    root.bind("<Escape>", lambda e: root.destroy())
    fams = set(tkf.families())
    mono = next((f for f in ("Cascadia Mono", "Consolas", "Courier New", "DejaVu Sans Mono", "Courier")
                 if f in fams), "Courier")
    scr_h = root.winfo_screenheight()
    fnt = tkf.Font(root=root, family=mono, size=SIZE, weight="bold")
    ch_ = fnt.metrics("linespace")
    cw_ = fnt.measure("M")
    rows = min(MAXROWS, (scr_h - 110) // ch_)
    init(cw_, ch_, rows)
    root.geometry(f"{W}x{H}+{(root.winfo_screenwidth() - W) // 2}+{max(0, (scr_h - H) // 2 - 20)}")
    cv = tk.Canvas(root, width=W, height=H, bg="#010203", highlightthickness=0)
    cv.pack()

    items = np.zeros((ROWS, COLS), dtype=object)
    items[:] = 0
    d_idx = np.zeros((ROWS, COLS), np.int32)
    d_q = np.zeros((ROWS, COLS, 3), np.int16)
    hexc = {}
    st = {"t0": time.perf_counter(), "budget": 2500, "cost": 0.012}

    def hx(rgb):
        k = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        s_ = hexc.get(k)
        if s_ is None:
            s_ = "#%02x%02x%02x" % tuple(v * 255 // (QL - 1) for v in k)
            hexc[k] = s_
        return s_

    def restart(_=None):
        st["t0"] = time.perf_counter()
        reset()

    root.bind("r", restart)
    root.bind("R", restart)

    def tick():
        t0 = time.perf_counter()
        t = t0 - st["t0"]
        dt = min(max(t - S["last"], 0.0), 0.1)
        idx, q = render_cells(t, dt)
        cells = plan_updates(idx, q, d_idx, d_q, st["budget"])
        t1 = time.perf_counter()
        for f in cells:
            r_, c_ = divmod(int(f), COLS)
            ch = RAMP[idx[r_, c_]]
            fill = hx(q[r_, c_])
            it = items[r_, c_]
            if it == 0:
                if ch == " ":
                    d_idx[r_, c_] = 0
                    d_q[r_, c_] = q[r_, c_]
                    continue
                items[r_, c_] = cv.create_text((c_ + 0.5) * CW, (r_ + 0.5) * CH, text=ch, fill=fill,
                                               font=fnt, anchor="center")
            else:
                cv.itemconfig(it, text=ch, fill=fill)
            d_idx[r_, c_] = idx[r_, c_]
            d_q[r_, c_] = q[r_, c_]
        if len(cells) > 50:
            per = (time.perf_counter() - t1) / len(cells)
            st["cost"] = 0.7 * st["cost"] + 0.3 * per
            st["budget"] = int(min(12000, max(400, (TK_MS / 1000.0) / max(st["cost"], 1e-6))))
        el = (time.perf_counter() - t0) * 1000
        root.after(max(1, int(1000 / FPS - el)), tick)

    root.after(100, tick)
    root.mainloop()


if __name__ == "__main__":
    main()