// CS405 · Lab 1 — your first triangle in WebGPU (starter)  
// Work through the TODOs in order. After each one, check the matching  
// checkpoint on the lab slides. The reference solution is in ../lab1-solution/.  
  
const canvas = document.querySelector('canvas');  
if (!navigator.gpu) throw new Error('WebGPU not available');  
  
const adapter = await navigator.gpu.requestAdapter();  
const device  = await adapter.requestDevice();  
  
const ctx = canvas.getContext('webgpu');  
const format = navigator.gpu.getPreferredCanvasFormat(); 
ctx.configure({ device, format, alphaMode: 'opaque' });  
console.log('WebGPU ready:', format);  
  
const SHADER = `  
  struct U {  
    time:   f32,  
    aspect: f32,  
    mouse:  vec2f,  
  };  
  @group(0) @binding(0) var<uniform> u: U;  
  
  struct VSOut {  
    @builtin(position) pos: vec4f,  
    @location(0) colour: vec4f 
  };  
  
  @vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {  
    // Two triangles → a unit square, centred at origin.  
    // Vertex order:  0,1,2  and  0,2,3  (triangle-strip-friendly ordering not used here;  
    // we just list 6 vertices below so a single draw(6) works).  
    var p = array<vec2f, 6>(  
      vec2f(-0.5, -0.5),   // 0  
      vec2f( 0.5, -0.5),   // 1  
      vec2f(-0.5,  0.5),   // 2  → triangle 1: 0,1,2  
      vec2f(-0.5,  0.5),   // 3  
      vec2f( 0.5, -0.5),   // 4  
      vec2f( 0.5,  0.5)    // 5  → triangle 2: 3,4,5  
    );  
  
    var c = array<vec3f, 6>(  
      vec3f(1.0, 0.0, 0.0),  
      vec3f(0.0, 1.0, 0.0),  
      vec3f(0.0, 0.0, 1.0),  
      vec3f(0.0, 0.0, 1.0),  
      vec3f(0.0, 1.0, 0.0),  
      vec3f(1.0, 0.0, 0.0)  
    );  
  
    // TODO 5a: rotate around origin by u.time, then translate to mouse. 
    let a = u.time; 
 
    let R = mat2x2f( 
      cos(a),  sin(a), 
      -sin(a), cos(a) 
    ); 
 
    let S = mat2x2f( 
      1.5, 0.0, 
      0.0, 0.6 
    ); 
 
    let r = R * p[i]; 
 
    var q = r + u.mouse;  
  
    // TODO 5b: correct aspect ratio → divide x by aspect so the square  
    // stays a square no matter how wide the canvas is.  
    q.x = q.x / u.aspect;  
  
    var out: VSOut;  
    out.pos    = vec4f(q, 0.0, 1.0);  
    out.colour = vec4f(c[i], 1.0);  
    return out;  
  }  
  
  @fragment fn fs(in: VSOut) -> @location(0) vec4f {  
    return in.colour;  
  }  
`;  
  
const module = device.createShaderModule({ code: SHADER });  
  
const pipeline = device.createRenderPipeline({  
  layout: 'auto',  
  vertex:   { module, entryPoint: 'vs' },  
  fragment: { module, entryPoint: 'fs', targets: [{ format }] }  
});  
  
// Uniform layout: time(f32) | aspect(f32) | mouse(vec2f)  = 16 bytes  
const ubuf = device.createBuffer({  
  size: 16,  
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST  
});  
  
const bind = device.createBindGroup({  
  layout: pipeline.getBindGroupLayout(0),  
  entries: [{ binding: 0, resource: { buffer: ubuf } }]  
});  
  
// ---------------------------------------------------------------------------  
// TODO 5 — your turn: a square (two triangles), correct aspect ratio,  
//   and the shape following the mouse.  
// ---------------------------------------------------------------------------  
  
const t0 = performance.now();  
  
// Track mouse in normalized device coords (NDC):  x,y ∈ [-1, 1], y up.  
let mouseX = 0;  
let mouseY = 0;  
  
canvas.addEventListener('mousemove', (e) => {  
  const r = canvas.getBoundingClientRect();  
  // 0..1 across the canvas  
  const nx = (e.clientX - r.left) / r.width;  
  const ny = (e.clientY - r.top ) / r.height;  
  // map to NDC [-1,1]; flip Y because mouse y grows downward  
  mouseX =  nx * 2 - 1;  
  mouseY = -(ny * 2 - 1);  
});  
  
function resize() {  
  const dpr = Math.min(2, window.devicePixelRatio || 1);  
  const r = canvas.getBoundingClientRect();  
  canvas.width  = Math.round(r.width  * dpr);  
  canvas.height = Math.round(r.height * dpr);  
}  
window.addEventListener('resize', resize);  
resize();  
  
function frame() {  
  const t = (performance.now() - t0) * 0.001;  
  
  // Aspect ratio of the drawing surface (width / height).  
  const aspect = canvas.width / canvas.height;  
  
  // Uniform: [ time, aspect, mouseX, mouseY ]  
  device.queue.writeBuffer(  
    ubuf, 0,  
    new Float32Array([t, aspect, mouseX, mouseY])  
  );  
  
  const enc = device.createCommandEncoder();  
  const pass = enc.beginRenderPass({ colorAttachments: [{  
    view: ctx.getCurrentTexture().createView(),  
    clearValue: { r: 0.19, g: 0.2, b: 0.6, a: 1 },  
    loadOp: 'clear', storeOp: 'store' }] });  
  
  pass.setPipeline(pipeline);  
  pass.setBindGroup(0, bind);  
  pass.draw(6);          // 6 vertices = 2 triangles = 1 square  
  pass.end();  
  
  device.queue.submit([enc.finish()]);  
  requestAnimationFrame(frame);  
}  
frame();