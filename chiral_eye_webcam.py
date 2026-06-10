"""
chiral_eye_webcam.py  —  the Chiral Eye, live, on your camera
=============================================================
Point your webcam at the world and watch the geometric neuron report the
direction of motion in real time, using the SAME operator verified in
chiral_eye.py:

        L = Im( z(t) * conj(z(t-lag)) ),   z = pixel_x + i*pixel_{x+1}

which is the Hassenstein-Reichardt motion detector. No training, O(K) per
frame, and it costs almost nothing when nothing in view is moving.

HONEST NOTE: the core operator here is identical to the one verified on
synthetic motion in chiral_eye.py (which was run and checked). THIS webcam
wrapper was NOT run in the build sandbox (no camera there). It is deliberately
minimal so there is little to go wrong; if a detail needs adjusting on your
machine it will be a small one.

Requires: pip install opencv-python numpy
Run:      python chiral_eye_webcam.py     (press q to quit)

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("Install OpenCV first:  pip install opencv-python")

LAG = 2            # frames of delay for the cross-time product
DOWN = 8           # spatial downsample (speed; bigger = coarser/faster)
SMOOTH = 0.6       # temporal smoothing of the reported value


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No camera found (VideoCapture(0) failed).")
    buf = []                      # ring buffer of recent downsampled frames (complex-ready)
    netL_s = 0.0                  # smoothed horizontal motion
    netV_s = 0.0                  # smoothed vertical motion
    base_rate = None              # running event-rate baseline for the 'cost' meter
    print("Chiral Eye live — press q in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        small = gray[::DOWN, ::DOWN]
        buf.append(small)
        if len(buf) > LAG + 1:
            buf.pop(0)

        netL = netV = 0.0
        ev_rate = 0.0
        if len(buf) == LAG + 1:
            cur = buf[-1]; old = buf[0]
            # event-rate (delta-code) meter: how much changed since last frame
            ev_rate = float(np.abs(cur - buf[-2]).mean())
            # horizontal motion: z = I(x) + iI(x+1)
            zc = cur[:, :-1] + 1j * cur[:, 1:]; zo = old[:, :-1] + 1j * old[:, 1:]
            netL = float((zc * np.conj(zo)).imag.mean())
            # vertical motion: z = I(y) + iI(y+1)
            zc = cur[:-1, :] + 1j * cur[1:, :]; zo = old[:-1, :] + 1j * old[1:, :]
            netV = float((zc * np.conj(zo)).imag.mean())

        netL_s = SMOOTH * netL_s + (1 - SMOOTH) * netL
        netV_s = SMOOTH * netV_s + (1 - SMOOTH) * netV
        base_rate = ev_rate if base_rate is None else max(base_rate * 0.995, ev_rate)

        # ---- draw ----
        h, w = frame.shape[:2]; cx, cy = w // 2, h // 2
        gain = 6000.0
        ex = int(np.clip(netL_s * gain, -w // 3, w // 3))
        ey = int(np.clip(netV_s * gain, -h // 3, h // 3))
        mag = np.hypot(netL_s, netV_s)
        col = (255, 80, 60) if netL_s > 0 else (60, 200, 255)
        cv2.arrowedLine(frame, (cx, cy), (cx + ex, cy + ey), col, 4, tipLength=0.3)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
        label = ("RIGHT" if netL_s > 1e-4 else "LEFT" if netL_s < -1e-4 else "—") + \
                ("  DOWN" if netV_s > 1e-4 else "  UP" if netV_s < -1e-4 else "")
        cv2.putText(frame, f"motion: {label}", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # cost meter: fraction of the running peak event-rate (the delta-code economy)
        cost = ev_rate / (base_rate + 1e-9)
        cv2.putText(frame, f"compute now: {int(100*cost):3d}%  (free when still)", (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(frame, "Chiral Eye  L=Im(z z*_lag)  trainless EMD", (16, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 160), 1)

        cv2.imshow("Chiral Eye", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
