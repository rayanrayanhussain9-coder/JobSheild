// This custom element is mounted and removed with the About page.
if (!customElements.get("jobshield-globe")) {
  class JobShieldGlobe extends HTMLElement {
    connectedCallback() {
      this.canvas = this.querySelector("canvas");
      if (!this.canvas) return;
      this.yaw = -0.34;
      this.pitch = 0.28;
      this.stage = this.closest(".js3d-stage");
      this.reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
      this.events = new AbortController();
      const listen = (target, type, handler) =>
        target.addEventListener(type, handler, {signal: this.events.signal});

      const gl = this.canvas.getContext("webgl", {alpha:true, antialias:true});
      if (gl) {
        this.gl = gl;
        this.prepare(gl);
        listen(this.canvas, "webglcontextlost", event => {
          event.preventDefault();
          this.enableRingView();
          this.draw();
        });
        listen(this.canvas, "webglcontextrestored", () => {
          this.classList.remove("ring-view");
          this.prepare(gl);
          this.draw();
        });
      } else {
        this.enableRingView();
      }
      this.classList.add("rendered");
      this.scrollHost = this.closest('[data-testid="stMain"]') || window;
      const scheduleDraw = () => {
        if (this.frame) return;
        this.frame = requestAnimationFrame(() => {
          this.frame = 0;
          this.updateFromScroll();
        });
      };
      this.scrollHost.addEventListener("scroll", scheduleDraw, {passive: true, signal: this.events.signal});
      listen(this.reduceMotion, "change", scheduleDraw);
      this.resize = new ResizeObserver(scheduleDraw);
      this.resize.observe(this);
      this.updateFromScroll();
    }

    updateFromScroll() {
      const y = this.scrollHost === window ? window.scrollY : this.scrollHost.scrollTop;
      const progress = this.reduceMotion.matches ? 0 : Math.min(1, Math.max(0, y / 680));
      this.yaw = -0.34 + progress * 2.4;
      this.pitch = 0.28 + progress * 0.25;
      this.stage.style.setProperty("--globe-rise", (-12 * progress) + "px");
      this.stage.style.setProperty("--dossier-rise", (-30 * progress) + "px");
      this.draw();
    }

    disconnectedCallback() {
      cancelAnimationFrame(this.frame);
      this.events?.abort();
      this.resize?.disconnect();
      if (this.gl) {
        for (const buffer of this.buffers || []) this.gl.deleteBuffer(buffer);
        if (this.program) this.gl.deleteProgram(this.program);
      }
    }

    enableRingView() {
      const rings = this.querySelector(".js3d-rings");
      if (!rings.childElementCount) {
        const addRing = transform => {
          const ring = document.createElement("i");
          ring.style.transform = transform;
          rings.appendChild(ring);
        };
        for (let i = 0; i < 5; i++) addRing('rotateY(' + i * 36 + 'deg)');
        for (const angle of [-55, -28, 0, 28, 55]) {
          const a = angle * Math.PI / 180;
          addRing('translateY(' + (-50 * Math.sin(a)) + '%) rotateX(90deg) scale(' + Math.cos(a) + ')');
        }
      }
      this.classList.add("ring-view");
    }

    prepare(gl) {
      const vertex = `
        attribute vec3 position, normal;
        uniform vec2 angle;
        varying vec3 n, p;
        vec3 turn(vec3 v) {
          vec2 c=cos(angle), s=sin(angle);
          vec3 a=vec3(c.x*v.x+s.x*v.z,v.y,-s.x*v.x+c.x*v.z);
          return vec3(a.x,c.y*a.y-s.y*a.z,s.y*a.y+c.y*a.z);
        }
        void main() {
          p=turn(position); n=turn(normal);
          float w=3.9-p.z;
          gl_Position=vec4(p.xy*3.45,-p.z,w);
        }`;
      const fragment = `
        precision mediump float;
        varying vec3 n, p;
        void main() {
          vec3 N=normalize(n), V=normalize(vec3(0.,0.,3.9)-p);
          vec3 light=normalize(vec3(-.45,.85,1.));
          float shade=.68+.36*dot(N, light);
          vec3 ink=mix(vec3(.32,.19,.80),vec3(.27,.59,.97),smoothstep(-.7,1.,p.x+.3*p.y));
          float shine=pow(max(dot(N,normalize(light+V)),0.),24.);
          float rim=pow(1.-abs(dot(N,V)),3.);
          gl_FragColor=vec4(ink*shade+.55*shine+.13*rim,1.);
        }`;
      const program = gl.createProgram();
      for (const [type, source] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        gl.attachShader(program, shader);
        gl.deleteShader(shader);
      }
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        gl.deleteProgram(program);
        this.enableRingView();
        return;
      }
      this.program = program;
      gl.useProgram(program);
      this.angles = gl.getUniformLocation(program, "angle");
      const mesh = createGlobeMesh();
      this.buffers = [];
      for (const [attribute, data] of [["position", mesh.positions], ["normal", mesh.normals]]) {
        const buffer = gl.createBuffer();
        this.buffers.push(buffer);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        const location = gl.getAttribLocation(program, attribute);
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 0, 0);
      }
      this.count = mesh.positions.length / 3;
      gl.enable(gl.DEPTH_TEST);
    }

    draw() {
      if (this.classList.contains("ring-view")) {
        this.querySelector(".js3d-rings").style.transform =
          'rotateX(' + this.pitch + 'rad) rotateY(' + this.yaw + 'rad)';
        return;
      }
      const gl = this.gl;
      if (!gl || !this.program) return;
      const size = Math.round(this.clientWidth * Math.min(devicePixelRatio, 2));
      if (!size) return;
      if (this.canvas.width !== size) this.canvas.width = this.canvas.height = size;
      gl.viewport(0,0,size,size);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.uniform2f(this.angles, this.yaw, this.pitch);
      gl.drawArrays(gl.TRIANGLES, 0, this.count);
    }
  }
  customElements.define("jobshield-globe", JobShieldGlobe);
}

export default function mountGlobe({parentElement}) {
  const placeholder = parentElement.querySelector("div.js3d-orb");
  if (!placeholder) return;
  const viewer = document.createElement("jobshield-globe");
  for (const attr of placeholder.attributes) viewer.setAttribute(attr.name, attr.value);
  viewer.append(...placeholder.childNodes);
  placeholder.replaceWith(viewer);
}
