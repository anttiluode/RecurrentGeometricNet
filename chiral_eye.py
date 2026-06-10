"""
chiral_eye.py  —  the Geometric Neuron, given sight
====================================================
The Chiral Ear reads the direction of time in SOUND. This is the same neuron,
the same one equation, run across SPACE instead of frequency: it reads the
direction of MOTION in vision.

The identity that makes this not a metaphor:
    L_k = Im( z_k(t) * conj(z_k(t-lag)) ),   z_k = I(x_k) + i I(x_{k+1})
        = I_{k+1}(t) I_k(t-lag) - I_k(t) I_{k+1}(t-lag)
which is, term for term, the Hassenstein-Reichardt elementary motion detector
(1956): neighbour-now times here-then, minus here-now times neighbour-then.
The chiral readout of the recurrent_geometric_net IS the fly's motion detector.
One geometric primitive does ears (time's arrow) and eyes (motion's arrow).

Why it is worth deploying (not the bet -- the cheap, true, useful part):
  - TRAINLESS. No weights to learn. O(K) per frame, one complex multiply per edge.
  - EVENT-DRIVEN. Feed it brightness-change events and it costs almost nothing on
    a static scene -- which is exactly what an event camera (DVS) emits. A
    delta-code sensor feeding a delta-code processor.
  - It reads DIRECTION, not motion ENERGY. A counterphase (flickering) pattern
    has full motion energy but no net direction; a magnitude readout is fooled,
    the bilinear cross-time term reports zero -- the visual Wiener-Khinchin escape.

What is shown (printed + figure, measured not assumed):
  D1  direction tuning: net L is signed like the stimulus velocity (and bandpass
      in speed, like a real EMD -- stated honestly, not hidden).
  D2  the visual WK escape: a drifter gives a large signed L; a counterphase
      flicker of matched spatial frequency gives ~0 net L despite real energy.
  D3  trainless 2D optical flow: a moving blob, no training, recovered as a
      motion vector field whose mean points in the true direction of travel.
  D4  event-camera mode: brightness-change events only; direction still read,
      and compute collapses to ~0 on static frames (the delta-code economy, in
      vision).

Do not hype. Do not lie. Just show.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

rng = np.random.default_rng(0)
LAG = 2


# ============================================================
# The Chiral Eye operators (trainless)
# ============================================================
def chiral_1d(V, lag=LAG):
    """V: (T, W) a row over time -> (net signed motion, per-edge L). The EMD."""
    z = V[:, :-1] + 1j * V[:, 1:]
    zl = np.roll(z, lag, axis=0)
    L = (z * np.conj(zl)).imag[lag + 2:]
    return L.mean(), L.mean(0)


def motion_energy(V, lag=LAG):
    """direction-blind magnitude readout (what a power/energy detector sees)."""
    z = V[:, :-1] + 1j * V[:, 1:]
    zl = np.roll(z, lag, axis=0)
    return np.abs(z * np.conj(zl))[lag + 2:].mean()


def chiral_flow(frames, lag=LAG):
    """frames: (T,H,W) -> (Lx, Ly) motion vector field, averaged over time. Trainless."""
    F = frames
    zx = F[:, :, :-1] + 1j * F[:, :, 1:]                  # horizontal edges
    Lx = (zx * np.conj(np.roll(zx, lag, 0))).imag[lag + 2:].mean(0)   # (H, W-1)
    zy = F[:, :-1, :] + 1j * F[:, 1:, :]                  # vertical edges
    Ly = (zy * np.conj(np.roll(zy, lag, 0))).imag[lag + 2:].mean(0)   # (H-1, W)
    h = min(Lx.shape[0], Ly.shape[0]); w = min(Lx.shape[1], Ly.shape[1])
    return Lx[:h, :w], Ly[:h, :w]


def to_events(frames, thr=0.06):
    """event-camera model: signed brightness-change events above threshold."""
    d = np.diff(frames, axis=0)
    e = np.where(np.abs(d) > thr, np.sign(d), 0.0)
    return np.concatenate([frames[:1] * 0, e], axis=0)


# ============================================================
# Visual stimuli
# ============================================================
W, H, T = 64, 64, 140
xx = np.arange(W); tt = np.arange(T)

def grating_row(v, f=0.11):
    return np.sin(2 * np.pi * (f * xx[None, :] - f * v * tt[:, None]))

def counterphase_row(f=0.11, ft=0.11):
    return np.sin(2 * np.pi * f * xx[None, :]) * np.cos(2 * np.pi * ft * tt[:, None])

def moving_blob(vx, vy, sig=7.0):
    """a Gaussian blob translating across the field at (vx, vy) px/frame."""
    F = np.zeros((T, H, W))
    cx0, cy0 = 14, 32
    yy, xg = np.mgrid[0:H, 0:W]
    for t in range(T):
        cx = cx0 + vx * t; cy = cy0 + vy * t
        F[t] = np.exp(-((xg - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
    return F


# ============================================================
# Run + verify
# ============================================================
print("=" * 66)
print("CHIRAL EYE  ·  the geometric neuron, given sight (trainless EMD)")
print("=" * 66)

print("\n  D1  direction tuning  (net L signed like velocity; bandpass in speed):")
vs = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
tune = []
for v in vs:
    nl, _ = chiral_1d(grating_row(v)); tune.append(nl)
    print(f"        v = {v:+.1f} px/frame   net L = {nl:+.4f}")

print("\n  D2  the visual Wiener-Khinchin escape (direction vs energy):")
nl_d, _ = chiral_1d(grating_row(+1.0)); en_d = motion_energy(grating_row(+1.0))
nl_c, _ = chiral_1d(counterphase_row()); en_c = motion_energy(counterphase_row())
print(f"        drifter       net L = {nl_d:+.4f}   motion-energy = {en_d:.4f}")
print(f"        counterphase  net L = {nl_c:+.4f}   motion-energy = {en_c:.4f}")
print( "        the flicker carries energy but NO net direction; the magnitude readout")
print( "        cannot sign it, the bilinear cross-time term correctly reports ~0.")

print("\n  D3  trainless 2D optical flow (a moving blob, no weights):")
for (vx, vy, name) in [(0.7, 0.4, "down-right"), (-0.7, 0.4, "down-left"), (0.0, -0.8, "up")]:
    F = moving_blob(vx, vy)
    Lx, Ly = chiral_flow(F)
    mvx, mvy = Lx.mean(), Ly.mean()
    ang_err = np.degrees(np.arccos(np.clip((mvx * vx + mvy * vy) /
              (np.hypot(mvx, mvy) * np.hypot(vx, vy) + 1e-9), -1, 1)))
    print(f"        true ({vx:+.1f},{vy:+.1f}) [{name:10s}] -> recovered mean ({mvx:+.4f},{mvy:+.4f})"
          f"   angular error {ang_err:4.1f} deg")

print("\n  D4  event-camera mode (delta-code in vision):")
# a full-field drifting grating (clear EMD signal), moving then frozen
G = np.tile(grating_row(0.9)[:, None, :], (1, H, 1))            # (T,H,W) rightward grating
Gstatic = np.concatenate([G, np.repeat(G[-1:], 60, axis=0)], axis=0)
E = to_events(Gstatic)
nl_px, _ = chiral_1d(Gstatic[:, H // 2, :])
nl_ev, _ = chiral_1d(E[:, H // 2, :])
ev_rate = np.abs(E).mean(axis=(1, 2))
move_evts = ev_rate[:T].mean(); static_evts = ev_rate[T + 5:].mean()
print(f"        direction from raw pixels  net L = {nl_px:+.4f}")
print(f"        direction from EVENTS only net L = {nl_ev:+.4f}   (same sign — events suffice)")
print(f"        event rate  moving {move_evts:.4f}  vs  frozen scene {static_evts:.5f}")
print( "        a still scene emits ~no events, so the readout costs ~nothing until")
print( "        something moves — the delta-code economy, native to event cameras.")

print("\n" + "=" * 66)
print("VERDICT: the chiral readout L = Im(z·z̄_lag) IS the Hassenstein-Reichardt")
print("motion detector. The same neuron that hears time's arrow sees motion's")
print("arrow -- trainless, O(K), and (on events) almost free when the scene is")
print("still. It reads DIRECTION, not energy: a flicker with full motion energy")
print("reads zero. Honest limits: like every EMD it is bandpass in speed and")
print("confounds speed with contrast/spatial-frequency (the aperture problem) --")
print("it is a cheap local primitive, not a full optical-flow solver. That is")
print("exactly its niche: a near-free direction sense for event-camera hardware.")
print("=" * 66)


# ============================================================
# Figure
# ============================================================
BG="#0a0a12"; PAN="#12121e"; CGRY="#6b6b85"; CBLU="#2ec5ff"; CRED="#ff3b6b"
CYEL="#f5c542"; CGRN="#42f5a1"; CVIO="#a98bff"
plt.rcParams["font.family"] = "monospace"
fig = plt.figure(figsize=(15, 9.4), facecolor=BG)
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34, top=0.89, bottom=0.07, left=0.06, right=0.975)
def AX(p, t, c=CBLU):
    a = fig.add_subplot(p); a.set_facecolor(PAN)
    a.set_title(t, color=c, fontsize=9.3, pad=6); a.tick_params(colors=CGRY, labelsize=7)
    for s in a.spines.values(): s.set_color("#23233a")
    return a

# (0,0) a drifting-grating frame
a = AX(gs[0, 0], "the stimulus: a drifting grating (rightward)", CYEL)
a.imshow(grating_row(1.0)[:48], aspect="auto", cmap="gray", extent=[0, W, 48, 0])
a.annotate("", xy=(46, 24), xytext=(18, 24), arrowprops=dict(arrowstyle="-|>", color=CGRN, lw=2.2))
a.set_xlabel("space x", color=CGRY, fontsize=8); a.set_ylabel("time t", color=CGRY, fontsize=8)

# (0,1) direction tuning
a = AX(gs[0, 1], "D1 · net L is signed like velocity", CGRN)
a.plot(vs, tune, "-o", color=CGRN, lw=1.5, ms=4); a.axhline(0, color=CGRY, lw=0.7, ls=":")
a.axvline(0, color=CGRY, lw=0.7, ls=":")
a.set_xlabel("stimulus velocity (px/frame)", color=CGRY, fontsize=8)
a.set_ylabel("net L = Im(z·z̄_lag)", color=CGRY, fontsize=8)
a.text(0.03, 0.96, "left ◂ 0 ▸ right · bandpass in speed", transform=a.transAxes,
       color="white", fontsize=7, va="top")

# (0,2) the WK escape
a = AX(gs[0, 2], "D2 · reads DIRECTION, not energy (WK escape)", CVIO)
xb = np.arange(2)
a.bar(xb - 0.2, [nl_d, nl_c], 0.4, color=CBLU, alpha=0.9, label="net L (direction)")
a.bar(xb + 0.2, [en_d, en_c], 0.4, color=CRED, alpha=0.6, label="motion energy")
a.axhline(0, color="#33334d", lw=0.8); a.set_xticks(xb)
a.set_xticklabels(["drifter", "counterphase\n(flicker)"], fontsize=7.5, color=CGRY)
a.legend(facecolor=PAN, edgecolor="#23233a", labelcolor="white", fontsize=6.5, loc="upper right")
a.text(0.03, 0.5, "flicker: energy present,\ndirection ≈ 0", transform=a.transAxes,
       color="white", fontsize=7, va="top")

# (1,0) trainless optical flow on a moving blob
a = AX(gs[1, 0], "D3 · trainless optical flow (moving blob)", CGRN)
F = moving_blob(0.7, 0.4); Lx, Ly = chiral_flow(F)
a.imshow(F[T // 2], cmap="magma", alpha=0.85)
mag = np.hypot(Lx, Ly); m = mag > 0.15 * mag.max()              # show only where motion is read
Ux = np.where(m, Lx / (mag + 1e-9), 0.0); Uy = np.where(m, Ly / (mag + 1e-9), 0.0)
step = 4
ys, xs = np.mgrid[0:Lx.shape[0]:step, 0:Lx.shape[1]:step]
a.quiver(xs, ys, Ux[::step, ::step], Uy[::step, ::step], color=CBLU,
         scale=22, width=0.006, headwidth=4)
a.set_xlabel("x", color=CGRY, fontsize=8); a.set_ylabel("y", color=CGRY, fontsize=8)
a.text(0.03, 0.10, "direction recovered with no training\n(arrows = unit motion at the moving edge)",
       transform=a.transAxes, color="white", fontsize=6.8, va="bottom")

# (1,1) event-camera mode
a = AX(gs[1, 1], "D4 · event-camera mode: ~free on a static scene", CRED)
ev_rate = np.abs(E).mean(axis=(1, 2))
a.plot(ev_rate, color=CGRY, lw=1.0)
a.axvspan(0, T, color=CBLU, alpha=0.10); a.axvspan(T, len(ev_rate), color=CGRN, alpha=0.10)
a.axvline(T, color=CYEL, lw=1.0, ls="--")
a.text(6, ev_rate.max() * 0.88, "moving", color=CBLU, fontsize=7)
a.text(T + 6, ev_rate.max() * 0.88, "scene frozen", color=CGRN, fontsize=7)
a.set_xlabel("frame", color=CGRY, fontsize=8); a.set_ylabel("event rate", color=CGRY, fontsize=8)
a.text(0.03, 0.6, "≈ 0 events\nwhen still",
       transform=a.transAxes, color="white", fontsize=7.5, va="top")

# (1,2) verdict
a = fig.add_subplot(gs[1, 2]); a.set_facecolor(PAN); a.axis("off")
for s in a.spines.values(): s.set_color("#23233a")
txt = ("CHIRAL EYE\n"
       "  same equation as the Chiral Ear:\n"
       "  L = Im(z·z̄_lag)\n\n"
       "  across SPACE instead of frequency\n"
       "  = the Hassenstein-Reichardt\n"
       "    motion detector (1956)\n\n"
       "  one geometric neuron:\n"
       "   · ears  -> time's arrow\n"
       "   · eyes  -> motion's arrow\n\n"
       "  trainless · O(K) · event-driven\n"
       "  ~0 cost on a still scene\n\n"
       "  reads DIRECTION, not energy\n"
       "  (a flicker reads zero)\n\n"
       "  honest: bandpass in speed,\n"
       "  confounds speed × contrast —\n"
       "  a cheap local primitive,\n"
       "  the event-camera niche.")
a.text(0.0, 1.0, txt, transform=a.transAxes, color="white", fontsize=7.7, va="top", linespacing=1.42)

fig.suptitle("Chiral Eye · the same geometric neuron that hears time's arrow sees motion's arrow "
             "— the Hassenstein-Reichardt detector, trainless, on events",
             color="white", fontsize=9.8, y=0.955)
plt.savefig("chiral_eye.png", dpi=140, bbox_inches="tight", facecolor=BG)
plt.close()
print("\nsaved chiral_eye.png")
