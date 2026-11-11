// Rounded openings in a spherical shell. Neighboring patches share their
// boundary positions, so meridians and parallels meet without crossing tubes.
function createGlobeMesh() {
  const positions = [], normals = [];
  const longitudeCount = 7, latitudeCount = 5, edgeSteps = 8;
  const halfTurn = Math.PI, pole = 1.48;
  const innerRadius = 0.926, bevel = 0.014;

  function point(longitude, latitude) {
    const c = Math.cos(latitude);
    return [c * Math.cos(longitude), Math.sin(latitude), c * Math.sin(longitude)];
  }
  function vertex(direction, radius, normal) {
    return { p: direction.map(x => x * radius), n: normal };
  }
  function triangle(a, b, c) {
    const u = b.p.map((v, i) => v - a.p[i]);
    const v = c.p.map((v, i) => v - a.p[i]);
    const cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]];
    const winding = cross.reduce((sum, x, i) => sum + x * (a.n[i]+b.n[i]+c.n[i]), 0);
    for (const q of winding >= 0 ? [a,b,c] : [a,c,b]) {
      positions.push(...q.p);
      normals.push(...q.n);
    }
  }
  function stitch(a, b) {
    for (let i = 0; i < a.length; i++) {
      const j = (i + 1) % a.length;
      triangle(a[i], a[j], b[j]);
      triangle(a[i], b[j], b[i]);
    }
  }

  // Follow all four sides using identical spacing on shared boundaries.
  const perimeter = [];
  for (let side = 0; side < 4; side++) {
    for (let i = 0; i < edgeSteps; i++) {
      const t = 2*i/edgeSteps - 1;
      perimeter.push([ [t,-1], [1,t], [-t,1], [-1,-t] ][side]);
    }
  }
  const dx = halfTurn / longitudeCount, dy = pole / latitudeCount;
  for (let row = 0; row < latitudeCount; row++) {
    const lat = -pole + (2*row+1)*dy;
    for (let col = 0; col < longitudeCount; col++) {
      const lon = (2*col+1)*dx;
      const outer = [], inner = [];
      const rim = Array.from({length:8}, () => []);
      for (const [u,v] of perimeter) {
        const outside = point(lon + dx*u, lat + dy*v);
        outer.push(vertex(outside, 1, outside));
        inner.push(vertex(outside, innerRadius, outside.map(x => -x)));

        // A high-order superellipse gently rounds each window corner.
        const size = (Math.abs(u)**9 + Math.abs(v)**9)**(1/9);
        const x = u/size, y = v/size;
        const wx = .81*dx, wy = .78*dy;
        const phi = lon + wx*x, theta = lat + wy*y;
        const direction = point(phi, theta);
        const east = [-Math.sin(phi),0,Math.cos(phi)];
        const north = [-Math.sin(theta)*Math.cos(phi),Math.cos(theta),-Math.sin(theta)*Math.sin(phi)];
        const gx = x**7 * Math.abs(x)/(wx*Math.cos(theta)), gy = y**7*Math.abs(y)/wy;
        const length = Math.hypot(gx,gy);
        const outward = east.map((e,k) => (e*gx+north[k]*gy)/length);
        for (let ring = 0; ring < 8; ring++) {
          const top = ring < 4;
          const a = (top ? ring : 7-ring)*Math.PI/6;
          const offset = bevel*(1-Math.sin(a));
          const radial = direction.map((d,k) => d*Math.cos(offset) + outward[k]*Math.sin(offset));
          const sign = top ? 1 : -1;
          const radius = top ? 1-bevel*(1-Math.cos(a)) : innerRadius+bevel*(1-Math.cos(a));
          const normal = radial.map((d,k) => sign*d*Math.cos(a) - outward[k]*Math.sin(a));
          const norm = Math.hypot(...normal);
          rim[ring].push(vertex(radial, radius, normal.map(q => q/norm)));
        }
      }
      const rings = [outer, ...rim, inner];
      for (let i=1; i<rings.length; i++) stitch(rings[i-1],rings[i]);
    }
  }
  for (const sign of [-1,1]) {
    for (const radius of [innerRadius,1]) {
      const make = (lon,lat) => {
        const d = point(lon,lat);
        return vertex(d, radius, radius===1 ? d : d.map(x=>-x));
      };
      const cap = make(0, sign*Math.PI/2);
      const steps = edgeSteps*longitudeCount;
      for (let i=0; i<steps; i++)
        triangle(cap,make(2*halfTurn*i/steps,sign*pole),make(2*halfTurn*(i+1)/steps,sign*pole));
    }
  }
  return {positions:new Float32Array(positions), normals:new Float32Array(normals)};
}
