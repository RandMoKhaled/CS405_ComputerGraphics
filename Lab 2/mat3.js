// CS405 · Lab 2 — 3×3 matrices for 2D transforms (starter)
// Fill in the TODOs. main.js runs selfTest() on load: make all seven lines print ✓.
// Column-major: m[3*col + row], the same layout WGSL expects.

export const identity = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function multiply(a, b) {
  // TODO 1: out[3*c + r] = Σ_k  a[3*k + r] · b[3*c + k]
  //         (column-major, so the loops look "transposed" — write them carefully)
  const out = new Array(9);
  for (let c = 0; c < 3; c++) {
    for (let r = 0; r < 3; r++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += a[3 * k + r] * b[3 * c + k];
      }
      out[3 * c + r] = sum;
    }
  }

  return out;
}

export const mulAll = (...ms) => ms.reduce(multiply);

// The third column holds the translation.
export const translate = (x, y) => [1, 0, 0, 0, 1, 0, x, y, 1];

export const scale = (x, y = x) => [x, 0, 0, 0, y, 0, 0, 0, 1];

export function rotate(radians) {
  // The columns are where x̂ and ŷ land.
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return [cos, sin, 0, -sin, cos, 0, 0, 0, 1];
}

export function apply(m, x, y) {           // transform the point (x, y)
  return [m[0] * x + m[3] * y + m[6], m[1] * x + m[4] * y + m[7]];
}

// Invert a general 3×3 matrix with the adjugate and determinant.
export function invert(m) {
  const a = m[0], b = m[3], c = m[6];
  const d = m[1], e = m[4], f = m[7];
  const g = m[2], h = m[5], i = m[8];

  const c00 = e * i - f * h;
  const c01 = f * g - d * i;
  const c02 = d * h - e * g;
  const det = a * c00 + b * c01 + c * c02;
  if (Math.abs(det) < 1e-8) return null;

  const invDet = 1 / det;
  return [
    c00 * invDet,
    c01 * invDet,
    c02 * invDet,
    (c * h - b * i) * invDet,
    (a * i - c * g) * invDet,
    (b * g - a * h) * invDet,
    (b * f - c * e) * invDet,
    (c * d - a * f) * invDet,
    (a * e - b * d) * invDet,
  ];
}

// a mat3x3f in a uniform buffer is three vec4 columns: 48 bytes
export function pad3x3(m) {
  return new Float32Array([m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0]);
}

export function selfTest() {
  const near = (a, b, msg) => {
    const ok = a.every((v, i) => Math.abs(v - b[i]) < 1e-6);
    console.log(ok ? '✓' : '✗', msg, ok ? '' : `\n  got      ${a.map(v => v.toFixed(2))}\n  expected ${b.map(v => v.toFixed(2))}`);
    return ok;
  };
  
  let ok = true;
  const M = translate(3, -2);
  ok &= near(multiply(identity(), M), M, 'I · M = M');
  ok &= near(multiply(translate(2, 0), translate(3, 0)), translate(5, 0), 'translations add');
  ok &= near(apply(rotate(Math.PI / 2), 1, 0), [0, 1], 'rotate(90°) sends (1,0) to (0,1)');
  ok &= near(apply(multiply(translate(2, 0), scale(3)), 1, 1), [5, 3], 'T·S sends (1,1) to (5,3)');
  ok &= near(apply(multiply(scale(3), translate(2, 0)), 1, 1), [9, 3], 'S·T sends (1,1) to (9,3)');
  const W = [1.2, 0.5, 0, -0.3, 0.9, 0, 0.4, -0.2, 1];      // det = 1.23
  const Wi = invert(W) || identity();
  ok &= near(Wi, [0.9, -0.5, 0, 0.3, 1.2, 0, -0.3, 0.44, 1.23].map(v => v / 1.23), 'invert matches adj / det');
  ok &= near(apply(Wi, ...apply(W, 0.3, 0.8)), [0.3, 0.8], 'M⁻¹ undoes M on a point');
  console.log(ok ? 'mat3: all tests pass' : 'mat3: fix the failures above');
  return !!ok;
}
