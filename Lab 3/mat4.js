// CS405 · Lab 3 — 4×4 matrices (PROVIDED IN FULL — you do not need to edit this file)
// Column-major, the layout WGSL's mat4x4f expects:  m[4*col + row]
//
//   | m0  m4  m8   m12 |
//   | m1  m5  m9   m13 |
//   | m2  m6  m10  m14 |
//   | m3  m7  m11  m15 |
//
// The last column (m12, m13, m14) is the translation. If a model matrix ever looks
// wrong, print it and read that column first — it should be the object's position.

export const identity = () => [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];

export function multiply(a, b) {            // returns a · b  (apply b first)
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[4*k + r] * b[4*c + k];
      o[4*c + r] = s;
    }
  return o;
}

export const mulAll = (...ms) => ms.reduce(multiply);

export const translate = (x, y, z) => [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1];
export const scale = (x, y = x, z = x)  => [x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1];

export const rotX = a => { const c = Math.cos(a), s = Math.sin(a);
  return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]; };
export const rotY = a => { const c = Math.cos(a), s = Math.sin(a);
  return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]; };
export const rotZ = a => { const c = Math.cos(a), s = Math.sin(a);
  return [c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1]; };

// transform a point (w = 1) and return [x, y, z, w] BEFORE the divide
export function apply(m, p) {
  const [x, y, z] = p;
  return [
    m[0]*x + m[4]*y + m[8]*z  + m[12],
    m[1]*x + m[5]*y + m[9]*z  + m[13],
    m[2]*x + m[6]*y + m[10]*z + m[14],
    m[3]*x + m[7]*y + m[11]*z + m[15],
  ];
}

const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const norm = a => { const L = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/L, a[1]/L, a[2]/L]; };

// The camera looks down −z.  w points BACKWARDS, away from the target.
export function lookAt(eye, target, up = [0, 1, 0]) {
  const w = norm(sub(eye, target));
  let uRaw = cross(up, w);
  // degenerate case: up is parallel to the view direction → cross product is zero
  if (Math.hypot(...uRaw) < 1e-6) uRaw = cross([0, 0, 1], w);
  const u = norm(uRaw);
  const v = cross(w, u);
  return [
    u[0], v[0], w[0], 0,
    u[1], v[1], w[1], 0,
    u[2], v[2], w[2], 0,
    -dot(u, eye), -dot(v, eye), -dot(w, eye), 1,
  ];
}

// WebGPU convention: NDC z runs from 0 (near) to 1 (far), NOT −1 to 1.
export function perspective(fovYRadians, aspect, near, far) {
  const f = 1 / Math.tan(fovYRadians / 2);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far / (near - far), -1,
    0, 0, (near * far) / (near - far), 0,
  ];
}

export const f32 = m => new Float32Array(m);   // ready for writeBuffer
