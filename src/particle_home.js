const palette = ["#4976ff", "#7358e8", "#ec5061", "#f1ad39", "#629ae9"];
const turn = Math.PI * 2;
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

export default function mount({parentElement, setTriggerValue}) {
  const hero = parentElement.querySelector(".particle-hero");
  const canvas = hero.querySelector("canvas");
  const context = canvas.getContext("2d", {alpha:true});
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const touch = matchMedia("(pointer: coarse)");
  const events = new AbortController();
  const listen = (target, name, handler) =>
    target.addEventListener(name, handler, {signal:events.signal, passive:true});
  listen(hero.querySelector("button"), "click", () => setTriggerValue("check", true));

  // This flag follows the document's lifetime, including Streamlit reruns.
  const played = Symbol.for("jobshield.home.introPlayed");
  if (!window[played] && !reducedMotion.matches) {
    let index = 0;
    for (const line of hero.querySelectorAll(".particle-line > span")) {
      const letters = [...line.textContent].map(letter => {
        const glyph = document.createElement("span");
        glyph.textContent = letter;
        glyph.style.setProperty("--arrival", (90 + index++ * 22) + "ms");
        return glyph;
      });
      line.replaceChildren(...letters);
    }
    hero.dataset.intro = "";
  }
  window[played] = true;
  hero.dataset.ready = "";
  if (!context) return () => events.abort();

  let width = 0, height = 0, clock = 0, frame = 0, last = 0;
  let particles = [], pointer = null, visible = true;
  let centerX = 0, centerY = 0;
  function resize() {
    width = hero.clientWidth;
    height = hero.clientHeight;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    centerX = width / 2;
    centerY = height / 2;
    // Jittered cells keep the short strokes apart, even at the bright crest.
    const spacing = touch.matches ? 32 : 26;
    particles = [];
    for (let y = -spacing; y < height + spacing; y += spacing) {
      for (let x = -spacing; x < width + spacing; x += spacing) {
        particles.push({
          homeX:x + Math.random()*spacing, homeY:y + Math.random()*spacing,
          phase:Math.random()*turn, size:.65 + Math.random()*.85,
          dx:0, dy:0, x, y,
        });
      }
    }
    draw(0);
  }
  function draw(dt) {
    clock += dt;
    context.clearRect(0,0,width,height);
    context.lineCap = "round";
    const scale = Math.min(width, height);
    const follow = 1 - Math.exp(-dt * 2.1);
    const aimX = width*.5 + scale*.08*Math.sin(clock*.40) + (pointer ? (pointer.x-width*.5)*.35 : 0);
    const aimY = height*.49 + scale*.05*Math.sin(clock*.57) + (pointer ? (pointer.y-height*.49)*.35 : 0);
    centerX += (aimX-centerX)*follow;
    centerY += (aimY-centerY)*follow;
    const radius = scale*(.48 + .035*Math.sin(clock*.8));
    const decay = Math.exp(-dt*7);
    for (const p of particles) {
      const rx = p.homeX-centerX, ry = p.homeY-centerY;
      const angle = Math.atan2(ry,rx);
      const distance = Math.hypot(rx,ry);
      const ridge = radius + scale*.07*Math.sin(angle*3-clock*.56)
                           + scale*.035*Math.sin(angle*5+clock*.33);
      const offset = (distance-ridge)/(scale*.14);
      const crest = Math.exp(-offset*offset);
      const wobble = Math.sin(clock*.65+p.phase);
      const displacement = crest*(scale*.035 + 3*wobble);
      p.dx *= decay;
      p.dy *= decay;
      const direction = angle + .25*Math.sin(angle*4-clock*.55);
      const nx = Math.cos(direction), ny = Math.sin(direction);
      p.x = p.homeX + nx*displacement + p.dx + 2*wobble;
      p.y = p.homeY + ny*displacement + p.dy + 2*Math.cos(p.phase+clock*.45);
      const heading = Math.atan2(ny*4+p.dy, nx*4+p.dx);
      const half = p.size*(.20 + crest*2.30);
      const hx = half*Math.cos(heading), hy = half*Math.sin(heading);
      const hue = (angle + Math.PI + .33*Math.sin(clock*.5+distance/scale*3) + turn) % turn;
      context.strokeStyle = palette[Math.floor(hue/turn*palette.length)];
      context.globalAlpha = .10 + crest*.78 + .05*Math.sin(p.phase);
      context.lineWidth = p.size*(.60 + crest*.88);
      context.beginPath();
      context.moveTo(p.x-hx,p.y-hy);
      context.lineTo(p.x+hx,p.y+hy);
      context.stroke();
    }
  }
  function tick(now) {
    frame = 0;
    draw(last ? Math.min((now-last)/1000,.05) : 0);
    last = now;
    frame = requestAnimationFrame(tick);
  }
  function sync() {
    cancelAnimationFrame(frame); frame = 0; last = 0; pointer = null;
    if (reducedMotion.matches) {
      delete hero.dataset.intro;
      for (const p of particles) p.dx = p.dy = 0;
      draw(0);
    } else if (visible && !document.hidden) frame = requestAnimationFrame(tick);
  }
  function move(event) {
    if (reducedMotion.matches || event.pointerType !== "mouse") return;
    const bounds = hero.getBoundingClientRect();
    const point = {x:event.clientX-bounds.left, y:event.clientY-bounds.top};
    if (pointer) {
      const dx = point.x-pointer.x, dy = point.y-pointer.y;
      const squared = dx*dx + dy*dy;
      if (squared) {
        // Distance to the swept segment handles fast mouse moves in either direction.
        const gain = Math.min(.5, 15 / Math.sqrt(squared));
        for (const p of particles) {
          const t = clamp(((p.x-pointer.x)*dx+(p.y-pointer.y)*dy)/squared,0,1);
          const distance = Math.hypot(p.x-pointer.x-t*dx, p.y-pointer.y-t*dy);
          if (distance >= 120) continue;
          const weight = (1-distance/120)**2 * gain;
          p.dx = clamp(p.dx + dx*weight,-38,38);
          p.dy = clamp(p.dy + dy*weight,-38,38);
        }
      }
    }
    pointer = point;
  }
  listen(hero, "pointermove", move);
  listen(hero, "pointerleave", () => {pointer = null;});
  listen(reducedMotion, "change", sync);
  listen(document, "visibilitychange", sync);
  const resizer = new ResizeObserver(resize);
  resizer.observe(hero);
  const observer = new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting; sync();
  });
  observer.observe(hero);
  resize();
  sync();
  return () => {
    cancelAnimationFrame(frame);
    resizer.disconnect();
    observer.disconnect();
    events.abort();
  };
}
