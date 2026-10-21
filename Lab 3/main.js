// CS405 · Lab 3 — a quaternion-driven hierarchy in WebGPU (STARTER)
//
// The WebGPU setup below is complete — it is Lab 1 and Lab 2 plus a depth buffer.
// Your work is in buildScene(), at the bottom. Do quat.js first.
//
// One pipeline, one draw call, 17 instances. Every object's world matrix lives in
// a storage buffer; the vertex shader indexes it with @builtin(instance_index).

import * as M from './mat4.js';
import * as Q from './quat.js';

Q.selfTest();

const canvas = document.querySelector('canvas');
const ui = {
  base:   document.querySelector('#base'),
  blend:  document.querySelector('#blend'),
  play:   document.querySelector('#play'),
  mode:   document.querySelector('#mode'),
  readout:document.querySelector('#readout'),
};

if (!navigator.gpu) throw new Error('WebGPU is not available. Use Chrome/Edge 113+ or Safari 18+.');
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('No WebGPU adapter — check webgpureport.org.');
const device = await adapter.requestDevice();
device.lost.then(info => console.error('device lost:', info.message));

const ctx = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format, alphaMode: 'opaque' });

// ------------------------------------------------------------------ geometry
// A unit cube centred on the origin: 36 vertices, each position + face normal.
function cube() {
  const f = [
    [[ 0, 0, 1], [[-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]]],
    [[ 0, 0,-1], [[.5,-.5,-.5],[-.5,-.5,-.5],[-.5,.5,-.5],[.5,.5,-.5]]],
    [[ 1, 0, 0], [[.5,-.5,.5],[.5,-.5,-.5],[.5,.5,-.5],[.5,.5,.5]]],
    [[-1, 0, 0], [[-.5,-.5,-.5],[-.5,-.5,.5],[-.5,.5,.5],[-.5,.5,-.5]]],
    [[ 0, 1, 0], [[-.5,.5,.5],[.5,.5,.5],[.5,.5,-.5],[-.5,.5,-.5]]],
    [[ 0,-1, 0], [[-.5,-.5,-.5],[.5,-.5,-.5],[.5,-.5,.5],[-.5,-.5,.5]]],
  ];
  const out = [];
  for (const [n, q] of f)
    for (const i of [0, 1, 2, 0, 2, 3]) out.push(...q[i], ...n);
  return new Float32Array(out);                    // 36 × 6 floats
}
const verts = cube();
const vbo = device.createBuffer({ size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(vbo, 0, verts);

// ------------------------------------------------------------------- shaders
const module = device.createShaderModule({ code: /* wgsl */`
struct Camera { viewProj : mat4x4f, eye : vec4f };
struct Node   { model : mat4x4f, colour : vec4f };

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<storage, read> nodes : array<Node>;

struct VSOut {
  @builtin(position) pos    : vec4f,
  @location(0)       normal : vec3f,
  @location(1)       colour : vec4f,
};

@vertex
fn vs(@location(0) p : vec3f,
      @location(1) n : vec3f,
      @builtin(instance_index) i : u32) -> VSOut {
  let node  = nodes[i];
  let world = node.model * vec4f(p, 1.0);          // w = 1: it is a point
  var o : VSOut;
  o.pos    = cam.viewProj * world;                 // the one line of the pipeline
  // Direction, so w = 0. (Non-uniform scale skews this — the proper fix is the
  // inverse-transpose normal matrix, which is Week 5.)
  o.normal = normalize((node.model * vec4f(n, 0.0)).xyz);
  o.colour = node.colour;
  return o;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  let L = normalize(vec3f(0.45, 0.85, 0.35));
  let d = max(dot(normalize(in.normal), L), 0.0);
  let shade = 0.30 + 0.70 * d;                     // ambient + diffuse, Week 6 properly
  return vec4f(in.colour.rgb * shade, 1.0);
}` });

const pipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: {
    module, entryPoint: 'vs',
    buffers: [{
      arrayStride: 6 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0,     format: 'float32x3' },
        { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' },
      ],
    }],
  },
  fragment: { module, entryPoint: 'fs', targets: [{ format }] },
  primitive: { topology: 'triangle-list', frontFace: 'ccw', cullMode: 'back' },
  depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
});

// ------------------------------------------------------------------- buffers
const N = 17;                                       // 5 scene nodes + 12 pillars
const NODE_FLOATS = 20;                             // mat4 (16) + colour (4)
const nodeData = new Float32Array(N * NODE_FLOATS);
const nodeBuf = device.createBuffer({
  size: nodeData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const camBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const bind = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: camBuf } },
    { binding: 1, resource: { buffer: nodeBuf } },
  ],
});

let depthTex = null;
function resize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width === w && canvas.height === h && depthTex) return;
  canvas.width = w; canvas.height = h;
  depthTex?.destroy();                              // the depth texture must match the canvas
  depthTex = device.createTexture({
    size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}
window.addEventListener('resize', resize);
resize();

// --------------------------------------------------------------- scene graph
// Every node keeps only its transform RELATIVE TO ITS PARENT.
// world = parentWorld · local, computed fresh every frame.

const COL = {
  floor:    [0.19, 0.20, 0.23, 1],
  pedestal: [0.55, 0.53, 0.49, 1],
  upper:    [0.12, 0.37, 0.55, 1],
  fore:     [0.23, 0.52, 0.42, 1],
  hand:     [0.71, 0.27, 0.11, 1],
  pillar:   [0.31, 0.29, 0.27, 1],
};

// two poses, each three joint rotations, blended with slerp
const POSE_A = {
  base:  Q.fromAxisAngle([0, 1, 0], -0.6),
  upper: Q.fromAxisAngle([0, 0, 1],  0.55),
  fore:  Q.fromAxisAngle([0, 0, 1], -1.15),
};
const POSE_B = {
  base:  Q.fromAxisAngle([0, 1, 0],  1.9),
  upper: Q.fromAxisAngle([0, 0, 1], -0.35),
  fore:  Q.fromAxisAngle([0, 0, 1],  0.95),
};

function poseAt(t, mode) {
  const f = mode === 'nlerp' ? Q.nlerp : Q.slerp;
  return {
    base:  f(POSE_A.base,  POSE_B.base,  t),
    upper: f(POSE_A.upper, POSE_B.upper, t),
    fore:  f(POSE_A.fore,  POSE_B.fore,  t),
  };
}

let slot = 0;
function push(world, colour) {
  nodeData.set(world, slot * NODE_FLOATS);
  nodeData.set(colour, slot * NODE_FLOATS + 16);
  slot++;
}

function buildScene(t, extraBaseSpin) {
  slot = 0;
  const p = poseAt(t, ui.mode.value);

  // floor — given, so that something appears before you write anything else
  push(M.mulAll(M.translate(0, -0.05, 0), M.scale(9, 0.1, 9)), COL.floor);

  // ---------------------------------------------------------------- TODO A
  // Build the LOCAL matrix of each joint:  translate(offset) · rotate(quaternion)
  // Use Q.toMat4(...) to turn a quaternion into a matrix, and M.mulAll(...) to
  // compose. The offsets are the link lengths:
  //     base    at (0, 0,    0)   rotation: p.base   (about y)
  //     upper   at (0, 0.55, 0)   rotation: p.upper  (about z)
  //     forearm at (0, 1.60, 0)   rotation: p.fore   (about z)
  //     hand    at (0, 1.30, 0)   no rotation of its own
  // Also fold `extraBaseSpin` (the slider) into the base rotation with
  // Q.multiply(Q.fromAxisAngle([0,1,0], extraBaseSpin), p.base).
  const baseRotation = Q.multiply(Q.fromAxisAngle([0, 1, 0], extraBaseSpin), p.base);
  const baseLocal  = M.mulAll(M.translate(0, 0, 0), Q.toMat4(baseRotation));
  const upperLocal = M.mulAll(M.translate(0, 0.55, 0), Q.toMat4(p.upper));
  const foreLocal  = M.mulAll(M.translate(0, 1.60, 0), Q.toMat4(p.fore));
  const handLocal  = M.translate(0, 1.30, 0);

  // ---------------------------------------------------------------- TODO B
  // Chain them:  world = parentWorld · local.  Four lines.
  const wBase  = baseLocal;
  const wUpper = M.multiply(wBase, upperLocal);
  const wFore  = M.multiply(wUpper, foreLocal);
  const wHand  = M.multiply(wFore, handLocal);

  // The SHAPE of each link is a separate scale applied AFTER the joint frame,
  // so scaling a link never scales its children. (This part is given — note how
  // the scale is on the RIGHT of the world matrix.)
  push(M.multiply(wBase,  M.mulAll(M.translate(0, 0.25, 0), M.scale(1.5, 0.5, 1.5))), COL.pedestal);
  push(M.multiply(wUpper, M.mulAll(M.translate(0, 0.80, 0), M.scale(0.45, 1.6, 0.45))), COL.upper);
  push(M.multiply(wFore,  M.mulAll(M.translate(0, 0.65, 0), M.scale(0.34, 1.3, 0.34))), COL.fore);
  push(M.multiply(wHand,  M.scale(0.5, 0.28, 0.5)), COL.hand);

  // ---------------------------------------------------------------- TODO C
  // Instancing. Twelve pillars in a ring of radius 4.2, each with its own height,
  // pushed into the same storage buffer — so they cost no extra draw calls.
  // Fill in a world matrix for each. Keep the loop count at 12: N is 17.
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * Math.PI * 2;
    const h = 0.6 + 0.45 * (1 + Math.sin(i * 1.7));
    const world = M.mulAll(
      M.translate(4.2 * Math.cos(a), h / 2, 4.2 * Math.sin(a)),
      M.scale(0.35, h, 0.35),
    );
    push(world, COL.pillar);
  }

  device.queue.writeBuffer(nodeBuf, 0, nodeData);
  return { wHand };
}

// ------------------------------------------------------------------ the loop
let playing = true, t0 = performance.now(), blendT = 0;
ui.play.addEventListener('click', () => {
  playing = !playing;
  ui.play.textContent = playing ? 'pause' : 'play';
  if (playing) { t0 = performance.now(); requestAnimationFrame(frame); }
});
ui.blend.addEventListener('input', () => { blendT = +ui.blend.value; if (!playing) frame(); });
ui.base.addEventListener('input', () => { if (!playing) frame(); });
ui.mode.addEventListener('change', () => { if (!playing) frame(); });

function frame() {
  resize();
  const time = (performance.now() - t0) / 1000;
  if (playing) { blendT = 0.5 - 0.5 * Math.cos(time * 0.9); ui.blend.value = blendT; }

  // camera
  const aspect = canvas.width / canvas.height;
  const eye = [Math.sin(time * 0.15) * 8.5, 4.2, Math.cos(time * 0.15) * 8.5];
  const view = M.lookAt(eye, [0, 1.4, 0], [0, 1, 0]);
  const proj = M.perspective(48 * Math.PI / 180, aspect, 0.5, 60);   // near out at 0.5!
  const vp = M.multiply(proj, view);
  const cam = new Float32Array(20);
  cam.set(vp, 0); cam.set([...eye, 1], 16);
  device.queue.writeBuffer(camBuf, 0, cam);

  const { wHand } = buildScene(blendT, +ui.base.value);

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      clearValue: { r: 0.055, g: 0.06, b: 0.075, a: 1 },
      loadOp: 'clear', storeOp: 'store',
    }],
    depthStencilAttachment: {
      view: depthTex.createView(),
      depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
    },
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.setVertexBuffer(0, vbo);
  pass.draw(36, N);                                  // ONE draw call, 17 instances
  pass.end();
  device.queue.submit([enc.finish()]);

  ui.readout.textContent =
    `blend t = ${blendT.toFixed(2)} · ${ui.mode.value} · ` +
    `hand at (${wHand[12].toFixed(2)}, ${wHand[13].toFixed(2)}, ${wHand[14].toFixed(2)}) · ` +
    `${N} instances, 1 draw call`;

  if (playing) requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
