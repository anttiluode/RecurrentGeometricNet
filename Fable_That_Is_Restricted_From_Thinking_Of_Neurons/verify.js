'use strict';
/* ChiralNet verification.
   Claims to verify BEFORE building the demo:
   T1 (exact): circular autocorrelation A_tau is invariant under time reversal.
   T2 (exact): depth-1 chiral mean F_tau = <Im(z z̄_tau)> is invariant under reversal.
   T3 (exact): depth-2 chiral mean G_{tau,delta} = <Im(ŵ ŵ̄_delta)> is ANTIsymmetric.
   T4 (exact): depth-2 is blind to absolute frequency (G≈0 for pure tones).
   E1: arrow task (up vs reversed-up chirp): arms A,B at chance; C,D high.
   E2: pitch task (low vs high tone): arms A,B,D high; C at chance.
*/
const N = 256;

// ---- FFT (iterative radix-2) ----
function fft(re, im, inv) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inv ? 2 : -2) * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
  if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// analytic signal via FFT (circular Hilbert)
function analytic(x) {
  const re = Float64Array.from(x), im = new Float64Array(N);
  fft(re, im, false);
  for (let k = 0; k < N; k++) {
    let m = 0;
    if (k === 0 || k === N / 2) m = 1;
    else if (k < N / 2) m = 2;
    re[k] *= m; im[k] *= m;
  }
  fft(re, im, true);
  return { re, im };
}

const LAGS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32];
const PAIRS = [[2, 8], [4, 8], [4, 16], [8, 8], [8, 16], [8, 24], [12, 12], [16, 8], [16, 16]];

// arm A: circular autocorrelation (power-equivalent by Wiener-Khinchin)
function featA(x) {
  const f = new Float64Array(LAGS.length);
  for (let li = 0; li < LAGS.length; li++) {
    const tau = LAGS[li]; let s = 0;
    for (let t = 0; t < N; t++) s += x[t] * x[(t - tau + N) % N];
    f[li] = s / N;
  }
  return f;
}
// arm B: depth-1 chiral mean  <Im(z(t) z̄(t-tau))>
function featB(z) {
  const f = new Float64Array(LAGS.length);
  for (let li = 0; li < LAGS.length; li++) {
    const tau = LAGS[li]; let s = 0;
    for (let t = 0; t < N; t++) {
      const u = (t - tau + N) % N;
      s += z.im[t] * z.re[u] - z.re[t] * z.im[u]; // Im(z z̄_u)
    }
    f[li] = s / N;
  }
  return f;
}
// arm C: depth-2 chiral mean on phase-normalized w_tau
function featC(z) {
  const f = new Float64Array(PAIRS.length);
  const wr = new Float64Array(N), wi = new Float64Array(N);
  let lastTau = -1;
  for (let pi = 0; pi < PAIRS.length; pi++) {
    const [tau, del] = PAIRS[pi];
    if (tau !== lastTau) {
      for (let t = 0; t < N; t++) {
        const u = (t - tau + N) % N;
        let r = z.re[t] * z.re[u] + z.im[t] * z.im[u];   // Re(z z̄_u)
        let i = z.im[t] * z.re[u] - z.re[t] * z.im[u];   // Im(z z̄_u)
        const m = Math.hypot(r, i) + 1e-12;
        wr[t] = r / m; wi[t] = i / m;                     // ŵ = w/|w|
      }
      lastTau = tau;
    }
    let s = 0;
    for (let t = 0; t < N; t++) {
      const u = (t - del + N) % N;
      s += wi[t] * wr[u] - wr[t] * wi[u];                 // Im(ŵ ŵ̄_u)
    }
    f[pi] = s / N;
  }
  return f;
}

// ---- signals ----
function gauss() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function chirpUp(f0, f1, phase) {
  const x = new Float64Array(N);
  for (let t = 0; t < N; t++) {
    const ph = phase + 2 * Math.PI * (f0 * t + (f1 - f0) * t * t / (2 * N));
    x[t] = Math.cos(ph);
  }
  return x;
}
function tone(f, phase) {
  const x = new Float64Array(N);
  for (let t = 0; t < N; t++) x[t] = Math.cos(phase + 2 * Math.PI * f * t);
  return x;
}
function reverse(x) { const y = new Float64Array(N); for (let t = 0; t < N; t++) y[t] = x[N - 1 - t]; return y; }
function addNoise(x, sig) { const y = new Float64Array(N); for (let t = 0; t < N; t++) y[t] = x[t] + sig * gauss(); return y; }
function rmsNorm(x) { let s = 0; for (let t = 0; t < N; t++) s += x[t] * x[t]; s = Math.sqrt(s / N) + 1e-12; const y = new Float64Array(N); for (let t = 0; t < N; t++) y[t] = x[t] / s; return y; }

const F0 = 0.06, F1 = 0.22;       // chirp sweep, cycles/sample
const TLO = 0.09, THI = 0.19;     // tones

// =========== T1–T4: exact symmetry checks on CLEAN signals ===========
console.log('=== EXACT SYMMETRY CHECKS (clean chirp, machine precision expected) ===');
{
  const up = rmsNorm(chirpUp(F0, F1, 0.7));
  const dn = reverse(up);                       // exact time reversal, no new noise
  const zu = analytic(up), zd = analytic(dn);
  const Au = featA(up), Ad = featA(dn);
  const Bu = featB(zu), Bd = featB(zd);
  const Cu = featC(zu), Cd = featC(zd);
  const err = (a, b, sgn) => Math.max(...[...a].map((v, i) => Math.abs(v - sgn * b[i])));
  console.log('T1  max|A(rev) - A|      =', err(Ad, Au, +1).toExponential(2), ' (claim: ~0, invariant)');
  console.log('T2  max|B(rev) - B|      =', err(Bd, Bu, +1).toExponential(2), ' (claim: ~0, invariant)');
  console.log('T3  max|C(rev) + C|      =', err(Cd, Cu, -1).toExponential(2), ' (claim: ~0, ANTIsymmetric)');
  console.log('    C(up)  =', [...Cu].map(v => v.toFixed(4)).join(' '));
  console.log('    C(rev) =', [...Cd].map(v => v.toFixed(4)).join(' '));
  // T4: depth-2 blind to pitch
  const zl = analytic(rmsNorm(tone(TLO, 0.3))), zh = analytic(rmsNorm(tone(THI, 1.1)));
  const Cl = featC(zl), Ch = featC(zh);
  console.log('T4  max|C(tone)|         =', Math.max(err(Cl, Cl.map(() => 0), 0), err(Ch, Ch.map(() => 0), 0)).toExponential(2), ' (claim: ~0, pitch-blind)');
}

// =========== E1/E2: train 4 linear readouts ===========
function makeDataset(task, nTrain, nTest, sigma) {
  const make = (label) => {
    let clean;
    if (task === 'arrow') {
      const u = chirpUp(F0, F1, Math.random() * 2 * Math.PI);
      clean = label === 0 ? u : reverse(u);
    } else {
      clean = tone(label === 0 ? TLO : THI, Math.random() * 2 * Math.PI);
    }
    const x = rmsNorm(addNoise(clean, sigma));
    const z = analytic(x);
    const A = featA(x), B = featB(z), C = featC(z);
    const D = Float64Array.from([...B, ...C]);    // the ChiralNet: depth-1 + depth-2
    return { A, B, C, D, y: label };
  };
  const tr = [], te = [];
  for (let i = 0; i < nTrain; i++) tr.push(make(i % 2));
  for (let i = 0; i < nTest; i++) te.push(make(i % 2));
  return { tr, te };
}

function trainArm(tr, te, key, steps = 4000, lr = 0.1) {
  const d = tr[0][key].length;
  // standardize on train
  const mu = new Float64Array(d), sd = new Float64Array(d);
  for (const s of tr) for (let j = 0; j < d; j++) mu[j] += s[key][j] / tr.length;
  for (const s of tr) for (let j = 0; j < d; j++) sd[j] += (s[key][j] - mu[j]) ** 2 / tr.length;
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) + 1e-9;
  const X = arr => arr.map(s => { const v = new Float64Array(d); for (let j = 0; j < d; j++) v[j] = (s[key][j] - mu[j]) / sd[j]; return v; });
  const Xtr = X(tr), Xte = X(te);
  const w = new Float64Array(d); let b = 0;
  for (let s = 0; s < steps; s++) {
    const i = (Math.random() * tr.length) | 0;
    let z = b; for (let j = 0; j < d; j++) z += w[j] * Xtr[i][j];
    const p = 1 / (1 + Math.exp(-z)), g = p - tr[i].y;
    for (let j = 0; j < d; j++) w[j] -= lr * (g * Xtr[i][j] + 1e-4 * w[j]);
    b -= lr * g;
  }
  let ok = 0;
  for (let i = 0; i < te.length; i++) {
    let z = b; for (let j = 0; j < d; j++) z += w[j] * Xte[i][j];
    if ((z > 0 ? 1 : 0) === te[i].y) ok++;
  }
  return ok / te.length;
}

const SIGMA = 0.5;
console.log('\n=== TRAINING (600 train / 400 test, noise sigma=' + SIGMA + ', logistic SGD) ===');
for (const task of ['arrow', 'pitch']) {
  const { tr, te } = makeDataset(task, 600, 400, SIGMA);
  const out = {};
  for (const k of ['A', 'B', 'C', 'D']) out[k] = trainArm(tr, te, k);
  console.log(task.padEnd(6),
    'A(power):', (out.A * 100).toFixed(1) + '%',
    ' B(chiral d1):', (out.B * 100).toFixed(1) + '%',
    ' C(chiral d2):', (out.C * 100).toFixed(1) + '%',
    ' D(ChiralNet):', (out.D * 100).toFixed(1) + '%');
}
console.log('\nPredicted matrix: arrow -> A~50 B~50 C high D high | pitch -> A high B high C~50 D high');

console.log('\n=== NOISE SWEEP (arrow task, arm C and D) ===');
for (const sg of [0.5, 1.0, 1.5, 2.0, 3.0]) {
  const { tr, te } = makeDataset('arrow', 600, 400, sg);
  const a = trainArm(tr, te, 'A'), c = trainArm(tr, te, 'C'), d = trainArm(tr, te, 'D');
  // SNR in dB: signal RMS ~ 1/sqrt(2) for unit-amp cosine vs noise sigma
  const snr = 10 * Math.log10(0.5 / (sg * sg));
  console.log('sigma=' + sg.toFixed(1), '(SNR ' + snr.toFixed(0) + ' dB)',
    ' A:', (a * 100).toFixed(1) + '%', ' C:', (c * 100).toFixed(1) + '%', ' D:', (d * 100).toFixed(1) + '%');
}
