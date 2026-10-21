// CS405 · Lab 3 — quaternions (STARTER)
//
// Fill in the five TODOs. main.js calls selfTest() on load: make all seven lines
// print ✓ in the console. On a fresh clone, six of the seven fail.
//
// Reference (Monday's lecture):
//   q = (n̂ sin(θ/2), cos(θ/2))          ← the HALF angle. This is the whole trick.
//   q1q2 = (w1v2 + w2v1 + v1×v2,  w1w2 − v1·v2)
//   slerp(a,b,t) = (sin((1−t)Ω)a + sin(tΩ)b) / sin Ω,   Ω = arccos(a·b)
// A quaternion is stored as [x, y, z, w]  with  q = (n̂ sin(θ/2), cos(θ/2)).

export const identity = () => [0, 0, 0, 1];

export function fromAxisAngle(axis, angle) {
  // TODO 1 — normalise the axis, then use the HALF angle.
  //          A very common wrong answer is sin(angle) instead of sin(angle/2):
  //          the self-test catches it immediately.
  const length = Math.hypot(axis[0], axis[1], axis[2]);
  if (length === 0) return identity();
  const half = angle / 2;
  const scale = Math.sin(half) / length;
  return [axis[0] * scale, axis[1] * scale, axis[2] * scale, Math.cos(half)];
}

// q1 · q2 — apply q2 first, then q1.  Note the cross product hiding in the
// vector part: that is exactly why quaternion multiplication does not commute.
export function multiply(a, b) {
  // TODO 2 — the Hamilton product. Write it out component by component;
  //          the cross product in the vector part is why it does not commute.
  const [ax, ay, az, aw] = a, [bx, by, bz, bw] = b;
  return [
    aw*bx + ax*bw + ay*bz - az*by,
    aw*by - ax*bz + ay*bw + az*bx,
    aw*bz + ax*by - ay*bx + az*bw,
    aw*bw - ax*bx - ay*by - az*bz,
  ];
}

export const mulAll = (...qs) => qs.reduce(multiply);

export function normalize(q) {
  const L = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0]/L, q[1]/L, q[2]/L, q[3]/L];
}

export const conjugate = q => [-q[0], -q[1], -q[2], q[3]];
export const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
export const negate = q => [-q[0], -q[1], -q[2], -q[3]];

// rotate a vector directly:  v' = q v q*
export function rotateVector(q, v) {
  const [x, y, z, w] = q;
  // t = 2 (qv × v);  v' = v + w t + qv × t     (the cheap expansion)
  const tx = 2 * (y*v[2] - z*v[1]);
  const ty = 2 * (z*v[0] - x*v[2]);
  const tz = 2 * (x*v[1] - y*v[0]);
  return [
    v[0] + w*tx + y*tz - z*ty,
    v[1] + w*ty + z*tx - x*tz,
    v[2] + w*tz + x*ty - y*tx,
  ];
}

// column-major 4×4, ready for WGSL
export function toMat4(q) {
  // TODO 3 — the rotation matrix from a unit quaternion, COLUMN-MAJOR:
  //            m[4*col + row].  If your cubes come out mirrored or transposed,
  //            you have written the rows where the columns go.
  //   row 0 :  1−2(y²+z²)     2(xy−wz)      2(xz+wy)
  //   row 1 :  2(xy+wz)       1−2(x²+z²)    2(yz−wx)
  //   row 2 :  2(xz−wy)       2(yz+wx)      1−2(x²+y²)
  const [x, y, z, w] = normalize(q);
  return [
    1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y), 0,
    2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x), 0,
    2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y), 0,
    0, 0, 0, 1,
  ];
}

export function axisAngleOf(q) {
  const n = normalize(q);
  const angle = 2 * Math.acos(Math.min(1, Math.max(-1, n[3])));
  const s = Math.sqrt(Math.max(0, 1 - n[3]*n[3]));
  if (s < 1e-6) return { axis: [0, 1, 0], angle: 0 };
  return { axis: [n[0]/s, n[1]/s, n[2]/s], angle };
}

// Spherical linear interpolation. Two things matter and both are on this slide's
// worth of code: the −q flip (take the short arc) and the small-angle fallback.
export function slerp(a, b, t) {
  a = normalize(a); b = normalize(b);
  let d = dot(a, b);
  // TODO 4a — if d < 0, negate b and d. (q and −q are the same rotation, but only
  //           one of them is on the short arc.)
  if (d < 0) {
    b = negate(b);
    d = -d;
  }
  // TODO 4b — if d > 0.9995 the angle is tiny and sin Ω → 0. Fall back to a plain
  //           lerp + normalize instead of dividing by nearly zero.
  if (d > 0.9995) {
    return normalize(a.map((value, i) => value + t * (b[i] - value)));
  }
  // TODO 4c — the formula above.
  d = Math.min(1, Math.max(-1, d));
  const omega = Math.acos(d);
  const sinOmega = Math.sin(omega);
  const weightA = Math.sin((1 - t) * omega) / sinOmega;
  const weightB = Math.sin(t * omega) / sinOmega;
  return normalize(a.map((value, i) => weightA * value + weightB * b[i]));
}

// plain lerp + normalize — same path as slerp, but not at constant speed
export function nlerp(a, b, t) {
  // TODO 5 — lerp the four components, then normalize. Remember the −q flip here
  //          too. Compare it with slerp in the browser: same path, different speed.
  a = normalize(a);
  b = normalize(b);
  if (dot(a, b) < 0) b = negate(b);
  return normalize(a.map((value, i) => value + t * (b[i] - value)));
}

// ---------------------------------------------------------------- self-test
export function selfTest() {
  let pass = 0, total = 0;
  const check = (ok, msg, got, want) => {
    total++; if (ok) pass++;
    console.log(ok ? '✓' : '✗', msg,
      ok ? '' : `\n  got      ${got}\n  expected ${want}`);
  };
  const close = (a, b, eps = 1e-5) => a.every((v, i) => Math.abs(v - b[i]) < eps);
  const fx = a => a.map(v => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(3)).join(', ');
  const Y = [0, 1, 0], Z = [0, 0, 1];
  const q90y = fromAxisAngle(Y, Math.PI / 2);

  check(close(q90y, [0, Math.SQRT1_2, 0, Math.SQRT1_2]),
    'fromAxisAngle(y, 90°) uses the HALF angle', fx(q90y), fx([0, .7071, 0, .7071]));

  const r = rotateVector(q90y, [1, 0, 0]);
  check(close(r, [0, 0, -1]),
    '90° about +y sends (1,0,0) to (0,0,−1)', fx(r), fx([0, 0, -1]));

  const twice = multiply(q90y, q90y), r2 = rotateVector(twice, [1, 0, 0]);
  check(close(r2, [-1, 0, 0]),
    'multiplying two 90° turns gives 180°', fx(r2), fx([-1, 0, 0]));

  const a = fromAxisAngle(Y, 1.1), b = fromAxisAngle(Z, 0.8);
  const nc = multiply(a, b), cn = multiply(b, a);
  check(!close(nc, cn, 1e-3), 'quaternion multiplication does NOT commute',
    'a·b == b·a', 'a·b ≠ b·a');

  const m = toMat4(q90y);
  check(close(m, [0,0,-1,0, 0,1,0,0, 1,0,0,0, 0,0,0,1]),
    'toMat4 agrees with rotateVector (column-major!)', fx(m), fx([0,0,-1,0, 0,1,0,0, 1,0,0,0, 0,0,0,1]));

  // literal quaternions here, so these two tests do not depend on TODO 1
  const Q90 = [0, 0.70710678, 0, 0.70710678];      // 90° about +y
  const Q45 = [0, 0.38268343, 0, 0.92387953];      // 45° about +y
  const half = slerp(identity(), Q90, 0.5);
  check(close(half, Q45),
    'slerp halfway to 90° is exactly 45°', fx(half), fx(Q45));

  const flipped = negate([0, 0.09983342, 0, 0.99500417]);   // −(0.2 rad about +y)
  const s = slerp(identity(), flipped, 1.0);
  check(Math.abs(axisAngleOf(s).angle % (2*Math.PI)) < 0.25 ||
        Math.abs(axisAngleOf(s).angle - 2*Math.PI) < 0.25,
    'slerp to −q takes the SHORT way (the dot<0 flip)',
    axisAngleOf(s).angle.toFixed(3), '≈ 0.2 rad, not 6.08');

  console.log(`%c quat.js — ${pass}/${total} `,
    `background:${pass === total ? '#2b6' : '#b33'};color:#fff;font-weight:700`);
  return pass === total;
}
