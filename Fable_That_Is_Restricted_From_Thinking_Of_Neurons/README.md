---
title: Chiral Net — the arrow of time
emoji: 🌀
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
license: mit
---

# Chiral Net — a network that knows the arrow of time

Two classes: a rising chirp, and the **exact time reversal** of a rising chirp.
Identical power spectra by construction. Four linear readouts train live in your
browser on the same data; only the feature algebra differs.

| arm | features | arrow task | pitch task |
|---|---|---|---|
| A | power (autocorrelation) | **blind — exactly** | sees it |
| B | chiral depth-1 ⟨Im(z·z̄τ)⟩ | **blind — exactly** | sees it |
| C | chiral depth-2 ⟨Im(ŵ·ŵ̄δ)⟩ | sees it (100% at 3 dB, 95% at −3 dB SNR) | **blind — exactly** |
| D | ChiralNet (d1 + d2) | sees it | sees it |

The blindness is not a training failure. With circular features it is an exact
symmetry: F(x′) = F(x) for power and depth-1, G(x′) = −G(x) for depth-2 —
verifiable to ~1e-16 with the "falsify it" button on the page, or by running
`node verify.js`.

Each chiral layer reads one more derivative of phase: |z| is power, layer 1 is
frequency, layer 2 is frequency's rate of change — the first quantity whose sign
flips with the arrow of time. The layer-1 operator is the same bilinear product
as the Chiral Eye motion detector; stacking it twice is what hears time.

Zero hidden units. Everything runs in this tab.

*do not hype · do not lie · just show* — PerceptionLab
