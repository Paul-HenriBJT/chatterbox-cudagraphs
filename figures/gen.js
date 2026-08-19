const fs = require('fs');
const { createCanvas } = require('@napi-rs/canvas');
const rough = require('roughjs');

// Excalidraw-style: black outlines, muted pastel fills, colour for emphasis only.
const OUT = '#1e1e1e', GRAY = '#868e96';
const BLUE = '#a5d8ff', REDF = '#ffc9c9', GREENF = '#b2f2bb';
const RED = '#e03131', GREEN = '#2f9e44', ORANGE = '#e8590c';
const FONT = 'sans-serif';

function newCanvas(w, h) {
  const c = createCanvas(w, h);
  const ctx = c.getContext('2d');
  // transparent background — the page colour shows through (clean on any bg)
  return { c, ctx, rc: rough.canvas(c) };
}
const FS = 1.5; // canvases are downscaled to the text column; enlarge text to compensate
function text(ctx, s, x, y, size, color = OUT, weight = '', align = 'left') {
  const p = Math.round(size * FS);
  ctx.fillStyle = color; ctx.font = `${weight} ${p}px ${FONT}`.trim();
  ctx.textAlign = align; ctx.textBaseline = 'alphabetic';
  s.split('\n').forEach((l, i) => ctx.fillText(l, x, y + i * p * 1.25));
  ctx.textAlign = 'left';
}
function head(ctx, x1, y1, x2, y2, color) {
  const a = Math.atan2(y2 - y1, x2 - x1), s = 9;
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
  ctx.moveTo(x2, y2); ctx.lineTo(x2 - s * Math.cos(a - 0.4), y2 - s * Math.sin(a - 0.4));
  ctx.moveTo(x2, y2); ctx.lineTo(x2 - s * Math.cos(a + 0.4), y2 - s * Math.sin(a + 0.4));
  ctx.stroke();
}
function arrow(ctx, rc, x1, y1, x2, y2, color = OUT, dbl = false) {
  rc.line(x1, y1, x2, y2, { stroke: color, strokeWidth: 1.6, roughness: 0.9 });
  head(ctx, x1, y1, x2, y2, color);
  if (dbl) head(ctx, x2, y2, x1, y1, color);
}
const box = (rc, x, y, w, h, fill) =>
  rc.rectangle(x, y, w, h, { stroke: OUT, fill, fillStyle: 'solid', roughness: 1.0, strokeWidth: 1.5 });

// ---------- timeline ----------
function timeline(out) {
  const W = 980, H = 340; const { c, ctx, rc } = newCanvas(W, H);
  const ex = 230, kw = 44, iw = 50, n = 7, ey = 140, kh = 46;
  let x = ex;
  for (let i = 0; i < n; i++) {
    box(rc, x, ey, kw, kh, BLUE); x += kw;
    if (i < n - 1) { box(rc, x, ey, iw, kh, REDF); x += iw; }
  }
  const eagerEnd = x;
  text(ctx, 'eager', 175, ey + 30, 20, OUT, 'bold', 'right');

  const gy = 235; let gx = ex;
  for (let i = 0; i < n; i++) { box(rc, gx, gy, kw, kh, BLUE); gx += kw; }
  const graphEnd = gx;
  text(ctx, 'CUDA graph', 175, gy + 30, 20, OUT, 'bold', 'right');

  text(ctx, 'GPU idle', ex + kw + iw / 2, ey - 12, 14, RED, '', 'center');
  arrow(ctx, rc, ex + kw + iw / 2, ey - 6, ex + kw + iw / 2, ey + 4, RED);

  rc.line(graphEnd, gy - 6, graphEnd, ey - 34, { stroke: GRAY, strokeWidth: 1, roughness: 0.4, strokeLineDash: [5, 5] });
  rc.line(eagerEnd, ey - 6, eagerEnd, ey - 34, { stroke: GRAY, strokeWidth: 1, roughness: 0.4, strokeLineDash: [5, 5] });
  arrow(ctx, rc, graphEnd, ey - 34, eagerEnd, ey - 34, OUT, true);
  text(ctx, 'wall-clock saved', (graphEnd + eagerEnd) / 2, ey - 42, 14, OUT, '', 'center');

  box(rc, ex, H - 42, 18, 15, BLUE); text(ctx, 'GPU running a kernel', ex + 26, H - 30, 13, GRAY);
  box(rc, ex + 210, H - 42, 18, 15, REDF); text(ctx, 'GPU idle (launch overhead)', ex + 236, H - 30, 13, GRAY);
  fs.writeFileSync(out, c.toBuffer('image/png')); console.log('wrote', out);
}

// ---------- one step ----------
function oneStep(out) {
  const W = 980, H = 220; const { c, ctx, rc } = newCanvas(W, H);
  const x0 = 150, y0 = 70, h = 66, tot = 700, cw = Math.round(tot * 6.8 / 21.4);
  box(rc, x0, y0, cw, h, BLUE);
  box(rc, x0 + cw, y0, tot - cw, h, REDF);
  text(ctx, 'compute\n~7 ms', x0 + cw / 2, y0 + 28, 15, OUT, '', 'center');
  text(ctx, 'launch overhead / GPU idle   ~14 ms  (68%)', x0 + cw + (tot - cw) / 2, y0 + 40, 16, OUT, '', 'center');
  text(ctx, 'one decode step = 21 ms', x0, y0 - 16, 14, GRAY);
  text(ctx, '~1,404 tiny kernels — the GPU spends about two-thirds of the step waiting on launches',
    x0, y0 + h + 30, 14, GRAY);
  fs.writeFileSync(out, c.toBuffer('image/png')); console.log('wrote', out);
}

// ---------- results (near-crisp: data, not a sketch) ----------
function results(out) {
  const W = 1080, H = 440; const { c, ctx, rc } = newCanvas(W, H);
  const bs = [1, 8, 32, 64], eager = [21.37, 26.71, 26.97, 32.04],
        graph = [6.81, 8.30, 17.94, 30.60], speed = [3.14, 3.22, 1.50, 1.05];
  const R = 0.4;

  const lx = 90, rx = 500, ty = 95, by = 380, ymax = 35;
  const px = i => lx + i * (rx - lx) / 3, py = v => by - v / ymax * (by - ty);
  rc.line(lx, by, rx, by, { stroke: OUT, strokeWidth: 1.6, roughness: R });
  rc.line(lx, by, lx, ty, { stroke: OUT, strokeWidth: 1.6, roughness: R });
  for (let i = 0; i < 3; i++) {
    rc.line(px(i), py(eager[i]), px(i + 1), py(eager[i + 1]), { stroke: ORANGE, strokeWidth: 2.5, roughness: R });
    rc.line(px(i), py(graph[i]), px(i + 1), py(graph[i + 1]), { stroke: GREEN, strokeWidth: 2.5, roughness: R });
  }
  for (let i = 0; i < 4; i++) {
    rc.circle(px(i), py(eager[i]), 10, { stroke: ORANGE, fill: ORANGE, fillStyle: 'solid', roughness: R });
    rc.circle(px(i), py(graph[i]), 10, { stroke: GREEN, fill: GREEN, fillStyle: 'solid', roughness: R });
    text(ctx, String(bs[i]), px(i), by + 22, 13, GRAY, '', 'center');
  }
  text(ctx, 'per-step latency (ms)', lx - 4, ty - 20, 13, GRAY);
  text(ctx, 'eager', rx - 62, py(eager[3]) - 12, 15, ORANGE, 'bold');
  text(ctx, 'CUDA graph', px(1) - 10, py(graph[1]) + 26, 15, GREEN, 'bold');

  const sx = 640, ex2 = 1040, sty = 95, sby = 380, smax = 3.7, smin = 0.85;
  const spx = i => sx + i * (ex2 - sx) / 3, spy = v => sby - (v - smin) / (smax - smin) * (sby - sty);
  rc.line(sx, sby, ex2, sby, { stroke: OUT, strokeWidth: 1.6, roughness: R });
  rc.line(sx, sby, sx, sty, { stroke: OUT, strokeWidth: 1.6, roughness: R });
  rc.line(sx, spy(1), ex2, spy(1), { stroke: GRAY, strokeWidth: 1.2, roughness: 0.3, strokeLineDash: [6, 6] });
  text(ctx, '1× (no gain)', ex2 - 78, spy(1) - 8, 12, GRAY);
  for (let i = 0; i < 3; i++) rc.line(spx(i), spy(speed[i]), spx(i + 1), spy(speed[i + 1]), { stroke: OUT, strokeWidth: 2.5, roughness: R });
  for (let i = 0; i < 4; i++) {
    rc.circle(spx(i), spy(speed[i]), 10, { stroke: OUT, fill: OUT, fillStyle: 'solid', roughness: R });
    text(ctx, speed[i].toFixed(2) + '×', spx(i), spy(speed[i]) - 15, 13, OUT, 'bold', 'center');
    text(ctx, String(bs[i]), spx(i), sby + 22, 13, GRAY, '', 'center');
  }
  text(ctx, 'speedup (eager / graph)', sx - 4, sty - 20, 13, GRAY);

  text(ctx, 'batch size', W / 2, H - 12, 13, GRAY, '', 'center');
  fs.writeFileSync(out, c.toBuffer('image/png')); console.log('wrote', out);
}

const D = '/Users/paul-henri/Documents/personal-branding/chatterbox-cudagraphs/results';
timeline(D + '/timeline.png');
oneStep(D + '/one_step.png');
results(D + '/latency.png');
