import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { AfterimagePass } from "three/addons/postprocessing/AfterimagePass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import gsap from "gsap";

export type OctaState = "idle" | "listening" | "thinking" | "speaking" | "noting";

/** Everything GSAP animates lives here; the render loop only reads it. */
interface Params {
  spinX: number;
  spinY: number;
  spinZ: number;
  wobble: number; // amplitude of a slow sinusoidal axis drift
  scale: number;
  glowStrength: number;
  edgeIntensity: number;
  damp: number; // afterimage feedback -> perceived motion blur
  floatAmp: number;
  gridPulse: number;
}

const DEFAULT_STATE_COLORS: Record<OctaState, string> = {
  idle: "#3a4044", // graphite
  listening: "#1d4433", // patina
  thinking: "#d94018", // signal
  speaking: "#4a9272", // patina light
  noting: "#8c8677", // stone
};

const BACKDROP_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const BACKDROP_FRAG = /* glsl */ `
  uniform vec3 uGlowColor;
  uniform vec3 uPaper;
  uniform vec3 uInk;
  uniform float uGlowStrength;
  uniform float uPulse;
  uniform float uTime;
  uniform float uAspect;
  varying vec2 vUv;

  float grid(vec2 p, float scale) {
    vec2 g = abs(fract(p * scale - 0.5) - 0.5) / fwidth(p * scale);
    return 1.0 - min(min(g.x, g.y), 1.0);
  }

  void main() {
    vec2 p = (vUv - 0.5) * vec2(uAspect, 1.0);
    float d = length(p);

    // whisper of the state color behind the octahedron
    float halo = exp(-d * d * 10.0) * uGlowStrength * (1.0 + uPulse * 0.5);
    vec3 col = mix(uPaper, uGlowColor, halo * 0.06);

    // hairline grid, strongest near the center — barely a texture
    float g = grid(p, 34.0);
    float gridMask = exp(-d * 2.6) * (0.018 + uPulse * 0.04);
    col = mix(col, uInk, g * gridMask);

    // barely-there edge shading
    col *= 1.0 - smoothstep(0.55, 1.2, d) * 0.06;

    gl_FragColor = vec4(col, 1.0);
  }
`;

const OCTA_VERT = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vView = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }
`;

const OCTA_FRAG = /* glsl */ `
  uniform vec3 uColor;
  uniform vec3 uBody;
  uniform float uFresnelPow;
  uniform float uIntensity;
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    float fresnel = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), uFresnelPow);
    vec3 rim = uColor * fresnel * uIntensity * 0.6;
    gl_FragColor = vec4(uBody + rim, 1.0);
  }
`;

export class OctaScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private composer: EffectComposer;
  private afterimage: AfterimagePass;

  private octa: THREE.Group;
  private octaMat: THREE.ShaderMaterial;
  private edgeMat: THREE.LineBasicMaterial;
  private backdropMat: THREE.ShaderMaterial;

  private glowColor = new THREE.Color(DEFAULT_STATE_COLORS.idle);
  private params: Params = {
    spinX: 0.05,
    spinY: 0.22,
    spinZ: 0.02,
    wobble: 0.15,
    scale: 1,
    glowStrength: 0.45,
    edgeIntensity: 1.2,
    damp: 0.05,
    floatAmp: 0.06,
    gridPulse: 0,
  };

  /** Live voice level, 0..1 — set every frame while listening/speaking. */
  level = 0;
  private smoothedLevel = 0;

  state: OctaState = "idle";
  private stateColors: Record<OctaState, string> = { ...DEFAULT_STATE_COLORS };
  private satPalette = ["#1d4433", "#d94018", "#3a4044", "#4a9272"];
  private themeDark = false;
  private frameColor = new THREE.Color("#0c1210");
  private clock = new THREE.Clock();
  // head-tracked parallax: targets set from face tracking, lerped per frame
  private parallax = { x: 0, y: 0, dolly: 0, tx: 0, ty: 0, tdolly: 0 };
  private headBaseSize = 0;
  private angularSpeed = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
    this.camera.position.z = 5;

    // backdrop plane — big enough to cover the frustum at its depth
    this.backdropMat = new THREE.ShaderMaterial({
      vertexShader: BACKDROP_VERT,
      fragmentShader: BACKDROP_FRAG,
      uniforms: {
        uGlowColor: { value: this.glowColor },
        uPaper: { value: new THREE.Color("#e5e8de") },
        uInk: { value: new THREE.Color("#0c1210") },
        uGlowStrength: { value: this.params.glowStrength },
        uPulse: { value: 0 },
        uTime: { value: 0 },
        uAspect: { value: 1 },
      },
      depthWrite: false,
    });
    const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.backdropMat);
    backdrop.position.z = -6;
    backdrop.frustumCulled = false;
    this.scene.add(backdrop);
    this.backdrop = backdrop;

    // octahedron: fresnel body + additive edges
    this.octa = new THREE.Group();
    const geo = new THREE.OctahedronGeometry(1);

    this.octaMat = new THREE.ShaderMaterial({
      vertexShader: OCTA_VERT,
      fragmentShader: OCTA_FRAG,
      uniforms: {
        uColor: { value: this.glowColor },
        uBody: { value: new THREE.Color("#0e1412") },
        uFresnelPow: { value: 2.2 },
        uIntensity: { value: 1.0 },
      },
    });
    this.octa.add(new THREE.Mesh(geo, this.octaMat));

    this.edgeMat = new THREE.LineBasicMaterial({
      color: this.glowColor,
      transparent: true,
      opacity: 0.9,
    });
    this.octa.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), this.edgeMat));
    this.scene.add(this.octa);

    // magic tail: onion-skin ghosts of the edge wireframe, lagging behind
    const ghostGeo = new THREE.EdgesGeometry(geo);
    for (let i = 0; i < 5; i++) {
      const m = new THREE.LineBasicMaterial({
        color: this.glowColor,
        transparent: true,
        opacity: 0.2 - i * 0.035,
        depthWrite: false,
      });
      const ghost = new THREE.LineSegments(ghostGeo, m);
      ghost.visible = false;
      this.scene.add(ghost);
      this.octaGhosts.push(ghost);
    }

    // shimmer dust: a slow shell of motes around the octahedron
    const dustGeo = new THREE.BufferGeometry();
    const dustPos = new Float32Array(48 * 3);
    for (let i = 0; i < 48; i++) {
      const r = 1.35 + Math.random() * 0.85;
      const a = Math.random() * Math.PI * 2;
      const b = Math.acos(2 * Math.random() - 1);
      dustPos[i * 3] = r * Math.sin(b) * Math.cos(a);
      dustPos[i * 3 + 1] = r * Math.cos(b);
      dustPos[i * 3 + 2] = r * Math.sin(b) * Math.sin(a);
    }
    dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
    this.dustMat = new THREE.PointsMaterial({
      color: new THREE.Color("#1d4433"),
      size: 0.035,
      transparent: true,
      opacity: 0.3,
      depthWrite: false,
    });
    this.dust = new THREE.Points(dustGeo, this.dustMat);
    this.scene.add(this.dust);

    // post: motion blur (afterimage feedback) + bloom
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.afterimage = new AfterimagePass(this.params.damp);
    this.composer.addPass(this.afterimage);
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.15, 0.6, 0.85);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.resize();
    addEventListener("resize", () => this.resize());
  }

  private backdrop!: THREE.Mesh;
  private bloom!: UnrealBloomPass;
  private octaGhosts: THREE.LineSegments[] = [];
  private octaHistory: { e: THREE.Euler; s: number; y: number }[] = [];
  private dust!: THREE.Points;
  private dustMat!: THREE.PointsMaterial;

  private resize() {
    const w = innerWidth;
    const h = innerHeight;
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.backdropMat.uniforms.uAspect.value = w / h;

    // size the backdrop plane to fill the view at its depth
    const dist = this.camera.position.z - this.backdrop.position.z;
    const height = 2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    this.backdrop.geometry.dispose();
    // generous margin so parallax camera moves never reveal the backdrop edge
    this.backdrop.geometry = new THREE.PlaneGeometry(height * (w / h) * 1.35, height * 1.35);
  }

  /** Bright panel textures shine under dark-mode bloom; scale them down there. */
  private panelToneScale(framed: boolean): number {
    return this.themeDark ? (framed ? 0.8 : 0.72) : 1.0;
  }

  /** Retheme the whole scene live — paper, ink, states, frames, trails. */
  setTheme(t: {
    dark: boolean;
    bg: string;
    fg: string;
    octaBody: string;
    frame: string;
    stateColors: Record<OctaState, string>;
    satelliteColors: string[];
  }) {
    this.themeDark = t.dark;
    this.stateColors = { ...t.stateColors };
    this.satPalette = [...t.satelliteColors];
    (this.backdropMat.uniforms.uPaper.value as THREE.Color).set(t.bg);
    (this.backdropMat.uniforms.uInk.value as THREE.Color).set(t.fg);
    (this.octaMat.uniforms.uBody.value as THREE.Color).set(t.octaBody);
    this.frameColor.set(t.frame);
    for (const p of this.panels.values()) {
      p.frameMat.color.set(t.frame);
      p.mat.color.setScalar(p.tone * this.panelToneScale(p.framed));
    }
    const c = new THREE.Color(this.stateColors[this.state]);
    gsap.to(this.glowColor, {
      r: c.r, g: c.g, b: c.b,
      duration: 0.8, ease: "power2.inOut",
      onUpdate: () => this.edgeMat.color.copy(this.glowColor),
    });
  }

  /** Head position (mirrored offset from frame center + face size) → parallax. */
  setHeadPosition(nx: number, ny: number, size: number) {
    if (size <= 0) {
      this.parallax.tx = this.parallax.ty = this.parallax.tdolly = 0;
      return;
    }
    this.parallax.tx = THREE.MathUtils.clamp(nx * 2.4, -0.85, 0.85);
    this.parallax.ty = THREE.MathUtils.clamp(-ny * 2.0, -0.55, 0.55);
    if (!this.headBaseSize) this.headBaseSize = size;
    this.headBaseSize += (size - this.headBaseSize) * 0.004; // slow neutral drift
    this.parallax.tdolly = THREE.MathUtils.clamp(
      (size / this.headBaseSize - 1) * -3.5,
      -0.9,
      0.7
    );
  }

  /** Spring-transition into a new behavioral state. */
  setState(state: OctaState) {
    if (state === this.state) return;
    this.state = state;

    gsap.killTweensOf(this.params);

    gsap.to(this.glowColor, {
      r: new THREE.Color(this.stateColors[state]).r,
      g: new THREE.Color(this.stateColors[state]).g,
      b: new THREE.Color(this.stateColors[state]).b,
      duration: 1.1,
      ease: "power2.inOut",
      onUpdate: () => this.edgeMat.color.copy(this.glowColor),
    });

    const spring = "elastic.out(1, 0.45)";

    switch (state) {
      case "idle":
        gsap.to(this.params, {
          spinX: 0.05, spinY: 0.22, spinZ: 0.02,
          wobble: 0.15, scale: 1, glowStrength: 0.45,
          edgeIntensity: 1.2, damp: 0.05, floatAmp: 0.06,
          duration: 1.6, ease: spring,
        });
        break;

      case "listening":
        // lean in: slow, attentive spin — the voice level does the moving
        gsap.to(this.params, {
          spinX: 0.02, spinY: 0.5, spinZ: 0,
          wobble: 0.05, scale: 1.06, glowStrength: 0.8,
          edgeIntensity: 1.8, damp: 0.05, floatAmp: 0.02,
          duration: 1.2, ease: spring,
        });
        break;

      case "thinking": {
        gsap.to(this.params, {
          wobble: 0.4, glowStrength: 1.0, edgeIntensity: 2.2,
          damp: 0.05, floatAmp: 0.04, scale: 0.92,
          duration: 0.8, ease: "power3.out",
        });
        // restless tumble: self-chaining randomized spin impulses on shifting axes
        const impulse = () => {
          if (this.state !== "thinking") return;
          const s = () => (Math.random() - 0.5) * (3 + Math.random() * 5);
          gsap.to(this.params, {
            spinX: s(), spinY: s(), spinZ: s(),
            duration: 0.55 + Math.random() * 0.7,
            ease: "power3.inOut",
            onComplete: impulse,
          });
        };
        impulse();
        break;
      }

      case "speaking":
        gsap.to(this.params, {
          spinX: 0.1, spinY: 0.8, spinZ: 0.05,
          wobble: 0.1, scale: 1.04, glowStrength: 0.85,
          edgeIntensity: 1.9, damp: 0.05, floatAmp: 0.03,
          duration: 1.0, ease: spring,
        });
        break;

      case "noting":
        // scribe persona: steady metronomic turn, warm and quiet
        gsap.to(this.params, {
          spinX: 0, spinY: 0.35, spinZ: 0,
          wobble: 0.02, scale: 0.9, glowStrength: 0.6,
          edgeIntensity: 1.5, damp: 0.05, floatAmp: 0.08,
          duration: 1.4, ease: spring,
        });
        break;
    }
  }

  /** Sharp springy pop — fired on word boundaries while speaking. */
  pulse(strength = 1) {
    gsap.fromTo(
      this.params,
      { gridPulse: 0.6 * strength },
      { gridPulse: 0, duration: 0.7, ease: "power2.out", overwrite: "auto" }
    );
    gsap.fromTo(
      this.pulseScale,
      { v: 1 + 0.12 * strength },
      { v: 1, duration: 0.9, ease: "elastic.out(1, 0.35)", overwrite: "auto" }
    );
  }

  private pulseScale = { v: 1 };

  // --- sub-agent satellites: small shapes orbiting the octahedron ---

  private satellites = new Map<string, {
    group: THREE.Group;
    ghosts: THREE.LineSegments[];
    history: { p: THREE.Vector3; e: THREE.Euler; s: number }[];
    angle: number;
    speed: number;
    radius: number;
    plane: THREE.Quaternion;
    spin: THREE.Vector3;
  }>();
  private satCount = 0;

  addSatellite(id: string) {
    if (this.satellites.has(id)) this.removeSatellite(id);
    const shapes = [
      () => new THREE.TetrahedronGeometry(0.17),
      () => new THREE.BoxGeometry(0.2, 0.2, 0.2),
      () => new THREE.IcosahedronGeometry(0.16),
      () => new THREE.OctahedronGeometry(0.16),
    ];
    const i = this.satCount++ % shapes.length;
    const geo = shapes[i]();
    const color = new THREE.Color(this.satPalette[i]);

    const mat = new THREE.ShaderMaterial({
      vertexShader: OCTA_VERT,
      fragmentShader: OCTA_FRAG,
      uniforms: {
        uColor: { value: color },
        uFresnelPow: { value: 2.2 },
        uIntensity: { value: 1.8 },
      },
    });
    const group = new THREE.Group();
    group.add(new THREE.Mesh(geo, mat));
    group.add(
      new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 })
      )
    );
    group.scale.setScalar(0.001);
    this.scene.add(group);

    // comet tail for the orbiters: a long run of tapering ghost frames
    const satGhostGeo = new THREE.EdgesGeometry(geo);
    const ghosts: THREE.LineSegments[] = [];
    for (let gi = 0; gi < 7; gi++) {
      const gm = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity: 0.26 * Math.pow(0.72, gi),
        depthWrite: false,
      });
      const ghost = new THREE.LineSegments(satGhostGeo, gm);
      ghost.visible = false;
      this.scene.add(ghost);
      ghosts.push(ghost);
    }

    // each satellite orbits in its own randomly tilted plane
    const plane = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(Math.random() * Math.PI, Math.random() * Math.PI, 0)
    );
    this.satellites.set(id, {
      group,
      ghosts,
      history: [],
      angle: Math.random() * Math.PI * 2,
      speed: 1.1 + Math.random() * 0.9,
      radius: 1.8 + Math.random() * 0.5,
      plane,
      spin: new THREE.Vector3(1 + Math.random() * 2, 1 + Math.random() * 2, Math.random()),
    });
    gsap.to(group.scale, { x: 1, y: 1, z: 1, duration: 1.1, ease: "elastic.out(1, 0.4)" });
  }

  clearSatellites() {
    for (const id of [...this.satellites.keys()]) this.removeSatellite(id);
  }

  removeSatellite(id: string) {
    const sat = this.satellites.get(id);
    if (!sat) return;
    this.satellites.delete(id);
    // the tail fades with its owner — never left frozen in the scene
    for (const ghost of sat.ghosts) {
      gsap.to(ghost.material as THREE.LineBasicMaterial, { opacity: 0, duration: 0.3 });
    }
    gsap.to(sat.group.scale, {
      x: 0.001, y: 0.001, z: 0.001,
      duration: 0.5,
      ease: "power3.in",
      onComplete: () => {
        this.scene.remove(sat.group);
        sat.group.traverse((o) => {
          if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments) {
            o.geometry.dispose();
            (o.material as THREE.Material).dispose();
          }
        });
        for (const ghost of sat.ghosts) {
          this.scene.remove(ghost);
          (ghost.material as THREE.Material).dispose();
        }
        sat.ghosts[0]?.geometry.dispose();
      },
    });
  }

  // --- floating artifact panels: textured planes living in the scene -------

  private panels = new Map<string, {
    group: THREE.Group;
    inner: THREE.Group;
    mat: THREE.MeshBasicMaterial;
    frameMat: THREE.LineBasicMaterial;
    desc: string;
    width: number;
    height: number;
    frameMax: number;
    focusOpacity: number;
    tone: number;
    framed: boolean;
    focused: boolean;
    phase: number;
    animate?: (elapsed: number) => boolean;
    animStart: number;
    animDone: boolean;
    shelfScale: number;
    // user view transform (pan/zoom) — current follows target with a fast lerp
    view: { x: number; y: number; zoom: number; tx: number; ty: number; tzoom: number };
  }>();
  private focusSlot = 0;

  /** Focus positions IN FRONT of the octahedron — the data overlays the scene,
   *  offset right/left but clamped inside the camera frustum. */
  private focusPos(slot: number, panelWidth: number): THREE.Vector3 {
    const z = 1.8;
    const dist = this.camera.position.z - z;
    const halfH = dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const halfW = halfH * this.camera.aspect;
    const side = slot % 2 === 0 ? 1 : -1;
    const x = side * Math.max(0, Math.min(1.15, halfW - panelWidth / 2 - 0.15));
    return new THREE.Vector3(x, 0.05 - Math.floor(slot / 2) * 1.1, z);
  }

  /** framed: photos get a faint frame; glowing line-work floats free. */
  addPanel(
    id: string,
    source: HTMLCanvasElement | HTMLImageElement,
    desc: string,
    opts: { framed?: boolean; animate?: (elapsed: number) => boolean; tone?: number } = {}
  ) {
    const tex =
      source instanceof HTMLCanvasElement
        ? new THREE.CanvasTexture(source)
        : new THREE.Texture(source);
    tex.needsUpdate = true;
    tex.colorSpace = THREE.SRGBColorSpace;

    const srcW = source instanceof HTMLImageElement ? source.naturalWidth : source.width;
    const srcH = source instanceof HTMLImageElement ? source.naturalHeight : source.height;
    const aspect = srcW / Math.max(1, srcH);
    let h = 1.35;
    let w = h * aspect;
    if (w > 2.6) { w = 2.6; h = w / aspect; }

    const geo = new THREE.PlaneGeometry(w, h);
    const framed = opts.framed ?? false;
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0 });
    const tone = opts.tone ?? (framed ? 0.96 : 1.0);
    // dark mode blooms harder — bright textures dim to stay crisp under it
    mat.color.setScalar(tone * this.panelToneScale(framed));
    const frameMat = new THREE.LineBasicMaterial({
      color: this.frameColor.clone(),
      transparent: true,
      opacity: 0,
    });
    const inner = new THREE.Group();
    inner.add(new THREE.Mesh(geo, mat));
    inner.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), frameMat));
    const group = new THREE.Group();
    group.add(inner);

    const pos = this.focusPos(this.focusSlot++, w);
    group.position.copy(pos).add(new THREE.Vector3(0, -0.4, 0));
    group.scale.setScalar(0.55);
    this.scene.add(group);

    const frameMax = framed ? 0.35 : 0;
    const focusOpacity = framed ? 0.9 : 0.96; // photos slightly translucent in focus
    this.panels.set(id, {
      group, inner, mat, frameMat, desc, width: w, height: h, frameMax, focusOpacity,
      tone, framed,
      focused: true,
      phase: Math.random() * Math.PI * 2,
      animate: opts.animate,
      animStart: this.clock.elapsedTime,
      animDone: !opts.animate,
      shelfScale: 0.5,
      view: { x: 0, y: 0, zoom: 1, tx: 0, ty: 0, tzoom: 1 },
    });

    gsap.to(group.position, { x: pos.x, y: pos.y, z: pos.z, duration: 1.1, ease: "back.out(1.5)" });
    gsap.to(group.scale, { x: 1, y: 1, z: 1, duration: 1.2, ease: "elastic.out(1, 0.5)" });
    gsap.to(mat, { opacity: focusOpacity, duration: 0.8 });
    gsap.to(frameMat, { opacity: frameMax, duration: 0.8 });
  }

  /** Push every focused panel into the dimmed background arc. */
  shelvePanels() {
    for (const p of this.panels.values()) {
      p.focused = false;
      p.view.tx = p.view.ty = 0;
      p.view.tzoom = 1;
    }
    this.focusSlot = 0;
    this.layoutShelf();
  }

  // --- gestures: raycast hit-testing + direct pan/zoom on the view ----------

  private raycaster = new THREE.Raycaster();
  private ndc = new THREE.Vector2();

  hitPanel(clientX: number, clientY: number): { id: string; focused: boolean; uv?: THREE.Vector2 } | null {
    this.ndc.set((clientX / innerWidth) * 2 - 1, -(clientY / innerHeight) * 2 + 1);
    this.raycaster.setFromCamera(this.ndc, this.camera);
    const byMesh = new Map<THREE.Object3D, string>();
    const meshes: THREE.Object3D[] = [];
    for (const [id, p] of this.panels) {
      const mesh = p.inner.children[0];
      meshes.push(mesh);
      byMesh.set(mesh, id);
    }
    const hit = this.raycaster.intersectObjects(meshes, false)[0];
    if (!hit) return null;
    const id = byMesh.get(hit.object)!;
    return { id, focused: this.panels.get(id)!.focused, uv: hit.uv };
  }

  /** Pan by a pixel delta — converted to world units at the panel's depth. */
  panPanel(id: string, dxPx: number, dyPx: number) {
    const p = this.panels.get(id);
    if (!p) return;
    const dist = this.camera.position.z - p.group.position.z;
    const worldPerPx =
      (2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) / innerHeight;
    p.view.tx = THREE.MathUtils.clamp(p.view.tx + dxPx * worldPerPx, -2.5, 2.5);
    p.view.ty = THREE.MathUtils.clamp(p.view.ty + dyPx * worldPerPx, -2, 2);
  }

  /** Zoom by a factor, anchored so the point under the cursor stays fixed. */
  zoomPanel(id: string, factor: number, uv?: THREE.Vector2) {
    const p = this.panels.get(id);
    if (!p) return;
    const z1 = p.view.tzoom;
    const z2 = THREE.MathUtils.clamp(z1 * factor, 0.6, 4);
    if (uv) {
      const px = (uv.x - 0.5) * p.width;
      const py = (uv.y - 0.5) * p.height;
      p.view.tx += px * (z1 - z2);
      p.view.ty += py * (z1 - z2);
    }
    p.view.tzoom = z2;
  }

  resetPanelView(id: string) {
    const p = this.panels.get(id);
    if (!p) return;
    p.view.tx = p.view.ty = 0;
    p.view.tzoom = 1;
  }

  /** Shelve one panel only, leaving the rest of the focus intact. */
  shelfPanel(id: string) {
    const p = this.panels.get(id);
    if (!p || !p.focused) return;
    p.focused = false;
    p.view.tx = p.view.ty = 0;
    p.view.tzoom = 1;
    this.focusSlot = [...this.panels.values()].filter((x) => x.focused).length;
    this.layoutShelf();
  }

  /** After a drag/pinch release: a panel flicked upward goes back to the shelf. */
  maybeShelfAfterDrag(id: string): boolean {
    const p = this.panels.get(id);
    if (!p || !p.focused || p.view.ty < 0.85) return false;
    this.shelfPanel(id);
    return true;
  }

  /** Bring one panel to focus. Additive: up to two share the stage; beyond
   *  that, the previous holders go to the shelf. */
  recallPanel(id: string): boolean {
    const p = this.panels.get(id);
    if (!p) return false;
    const others = [...this.panels.values()].filter((x) => x.focused && x !== p);
    if (p.focused || others.length >= 2) {
      for (const other of others) {
        other.focused = false;
        other.view.tx = other.view.ty = 0;
        other.view.tzoom = 1;
      }
      this.focusSlot = 0;
    }
    if (this.highlightedId === id) this.highlightedId = null;
    p.focused = true;
    p.view.tx = p.view.ty = 0;
    p.view.tzoom = 1;
    this.focusSlot = [...this.panels.values()].filter((x) => x.focused && x !== p).length;
    const pos = this.focusPos(this.focusSlot++, p.width);
    gsap.to(p.group.position, { x: pos.x, y: pos.y, z: pos.z, duration: 1.1, ease: "back.out(1.4)" });
    gsap.to(p.group.scale, { x: 1, y: 1, z: 1, duration: 1.2, ease: "elastic.out(1, 0.5)" });
    gsap.to(p.group.rotation, { x: 0, y: 0, z: 0, duration: 1.0, ease: "power3.inOut" });
    gsap.to(p.mat, { opacity: p.focusOpacity, duration: 0.8 });
    gsap.to(p.frameMat, { opacity: p.frameMax, duration: 0.8 });
    this.layoutShelf();
    return true;
  }

  /** Bento-box shelf: uniform-height rows behind the octahedron, natural
   *  widths, greedily packed and centered — structure instead of a loose arc. */
  private layoutShelf() {
    const shelved = [...this.panels.values()].filter((p) => !p.focused);
    if (!shelved.length) return;
    const tileH = 1.05;
    const gap = 0.3;
    const budget = 7.4; // row width in world units at shelf depth

    const rows: { items: { p: (typeof shelved)[0]; w: number; s: number }[]; width: number }[] = [
      { items: [], width: 0 },
    ];
    for (const p of shelved) {
      const s = tileH / p.height;
      const w = p.width * s;
      let row = rows[rows.length - 1];
      if (row.items.length && row.width + gap + w > budget) {
        row = { items: [], width: 0 };
        rows.push(row);
      }
      row.width += (row.items.length ? gap : 0) + w;
      row.items.push({ p, w, s });
    }

    const topY = 2.0;
    rows.forEach((row, r) => {
      let x = -row.width / 2;
      const y = topY - r * (tileH + gap);
      for (const { p, w, s } of row.items) {
        const cx = x + w / 2;
        x += w + gap;
        p.shelfScale = s;
        gsap.to(p.group.position, { x: cx, y, z: -3.4, duration: 1.2, ease: "power3.inOut" });
        gsap.to(p.group.scale, { x: s, y: s, z: s, duration: 1.2, ease: "power3.inOut" });
        gsap.to(p.group.rotation, { y: cx * -0.03, duration: 1.2, ease: "power3.inOut" });
        gsap.to(p.mat, { opacity: 0.3, duration: 1.0 });
        gsap.to(p.frameMat, { opacity: Math.max(p.frameMax * 0.4, 0.1), duration: 1.0 });
      }
    });
  }

  removePanel(id: string) {
    const p = this.panels.get(id);
    if (!p) return;
    if (this.highlightedId === id) this.highlightedId = null;
    this.panels.delete(id);
    gsap.to(p.group.scale, { x: 0.01, y: 0.01, z: 0.01, duration: 0.5, ease: "power3.in" });
    gsap.to(p.mat, {
      opacity: 0, duration: 0.5,
      onComplete: () => {
        this.scene.remove(p.group);
        p.group.traverse((o) => {
          if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments) {
            o.geometry.dispose();
            (o.material as THREE.Material).dispose();
          }
        });
        p.mat.map?.dispose();
        this.layoutShelf();
      },
    });
  }

  clearPanels() {
    for (const id of [...this.panels.keys()]) this.removePanel(id);
  }

  private highlightedId: string | null = null;

  /** Hover glow on a shelved panel — the current selection candidate. */
  highlightPanel(id: string | null) {
    if (id === this.highlightedId) return;
    const prev = this.highlightedId ? this.panels.get(this.highlightedId) : null;
    this.highlightedId = id;
    if (prev && !prev.focused) {
      const s = prev.shelfScale;
      gsap.to(prev.mat, { opacity: 0.3, duration: 0.3 });
      gsap.to(prev.frameMat, { opacity: Math.max(prev.frameMax * 0.4, 0.1), duration: 0.3 });
      gsap.to(prev.group.scale, { x: s, y: s, z: s, duration: 0.35 });
    }
    const p = id ? this.panels.get(id) : null;
    if (p && !p.focused) {
      const s = p.shelfScale * 1.15;
      gsap.to(p.mat, { opacity: 0.62, duration: 0.25 });
      gsap.to(p.frameMat, { opacity: 0.5, duration: 0.25 });
      gsap.to(p.group.scale, { x: s, y: s, z: s, duration: 0.3, ease: "back.out(2)" });
    }
  }

  highlightedPanel(): string | null {
    return this.highlightedId;
  }

  /** Magnetic targeting: nearest shelved tile within maxPx of a screen point. */
  nearestShelfPanel(clientX: number, clientY: number, maxPx = 140): string | null {
    let best: string | null = null;
    let bestD = maxPx;
    const v = new THREE.Vector3();
    for (const [id, p] of this.panels) {
      if (p.focused) continue;
      v.copy(p.group.position).project(this.camera);
      const sx = (v.x * 0.5 + 0.5) * innerWidth;
      const sy = (-v.y * 0.5 + 0.5) * innerHeight;
      const d = Math.hypot(sx - clientX, sy - clientY);
      if (d < bestD) {
        bestD = d;
        best = id;
      }
    }
    return best;
  }

  /** The most recently focused panel — the target for gestures. */
  focusedPanelId(): string | null {
    let last: string | null = null;
    for (const [id, p] of this.panels) if (p.focused) last = id;
    return last;
  }

  listPanels(): { id: string; desc: string }[] {
    return [...this.panels.entries()].map(([id, p]) => ({ id, desc: p.desc }));
  }

  /** Angular speed magnitude — feeds the soundscape. */
  getMotion() {
    return { angularSpeed: this.angularSpeed, level: this.smoothedLevel, state: this.state };
  }

  render() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;
    const p = this.params;

    // floaty parallax follow — the scene gains depth as the head moves
    const pa = 1 - Math.exp(-dt * 4);
    this.parallax.x += (this.parallax.tx - this.parallax.x) * pa;
    this.parallax.y += (this.parallax.ty - this.parallax.y) * pa;
    this.parallax.dolly += (this.parallax.tdolly - this.parallax.dolly) * pa;
    this.camera.position.set(this.parallax.x, this.parallax.y, 5 + this.parallax.dolly);
    this.camera.lookAt(0, 0, 0);

    this.smoothedLevel += (this.level - this.smoothedLevel) * Math.min(1, dt * 12);
    const lv = this.smoothedLevel;

    // integrate rotation; wobble drifts the axes so motion never looks canned
    const wob = p.wobble;
    this.octa.rotation.x += (p.spinX + Math.sin(t * 0.7) * wob * 0.3) * dt;
    this.octa.rotation.y += (p.spinY + Math.sin(t * 0.4 + 2) * wob * 0.2) * dt;
    this.octa.rotation.z += (p.spinZ + Math.cos(t * 0.55) * wob * 0.25) * dt;

    // voice pushes spin + scale while listening
    const voiceSpin = this.state === "listening" ? lv * 3.5 : 0;
    this.octa.rotation.y += voiceSpin * dt;

    const voiceScale =
      this.state === "listening" || this.state === "speaking" || this.state === "noting"
        ? lv * 0.22
        : 0;
    const breathe = this.state === "thinking" ? Math.sin(t * 3.2) * 0.03 : 0;
    const s = (p.scale + voiceScale + breathe) * this.pulseScale.v;
    this.octa.scale.setScalar(s);

    this.octa.position.y = Math.sin(t * 1.1) * p.floatAmp;

    // panels: bob + fast-reactive pan/zoom lerp; streaming ones redraw
    const viewAlpha = 1 - Math.exp(-dt * 22);
    for (const panel of this.panels.values()) {
      const v = panel.view;
      v.x += (v.tx - v.x) * viewAlpha;
      v.y += (v.ty - v.y) * viewAlpha;
      v.zoom += (v.tzoom - v.zoom) * viewAlpha;
      panel.inner.position.x = v.x;
      panel.inner.position.y = v.y + Math.sin(t * 1.05 + panel.phase) * 0.035;
      panel.inner.scale.setScalar(v.zoom);
      if (panel.animate && !panel.animDone) {
        panel.animDone = panel.animate(t - panel.animStart);
        if (panel.mat.map) panel.mat.map.needsUpdate = true;
      }
    }

    // satellites orbit the octahedron, each in its own tilted plane
    for (const sat of this.satellites.values()) {
      sat.angle += sat.speed * dt;
      sat.group.position
        .set(Math.cos(sat.angle) * sat.radius, 0, Math.sin(sat.angle) * sat.radius)
        .applyQuaternion(sat.plane)
        .add(this.octa.position);
      sat.group.rotation.x += sat.spin.x * dt;
      sat.group.rotation.y += sat.spin.y * dt;
      sat.group.rotation.z += sat.spin.z * dt;

      // orbit tail
      sat.history.push({
        p: sat.group.position.clone(),
        e: sat.group.rotation.clone(),
        s: sat.group.scale.x,
      });
      if (sat.history.length > 44) sat.history.shift();
      sat.ghosts.forEach((ghost, gi) => {
        const snap = sat.history[sat.history.length - 1 - (gi + 1) * 5];
        if (snap) {
          ghost.visible = true;
          ghost.position.copy(snap.p);
          ghost.rotation.copy(snap.e);
          // comet taper: the tail thins toward its tip and breathes slightly
          ghost.scale.setScalar(snap.s * (1 - gi * 0.09) * (1 + 0.04 * Math.sin(t * 7 + gi)));
        } else {
          ghost.visible = false;
        }
      });
    }

    // octahedron tail: replay lagged snapshots through the ghost wireframes
    this.octaHistory.push({ e: this.octa.rotation.clone(), s, y: this.octa.position.y });
    if (this.octaHistory.length > 32) this.octaHistory.shift();
    this.octaGhosts.forEach((ghost, gi) => {
      const snap = this.octaHistory[this.octaHistory.length - 1 - (gi + 1) * 5];
      const mat = ghost.material as THREE.LineBasicMaterial;
      if (snap) {
        ghost.visible = true;
        ghost.rotation.copy(snap.e);
        ghost.scale.setScalar(snap.s * (1 + gi * 0.012));
        ghost.position.y = snap.y;
        mat.color.copy(this.edgeMat.color);
        mat.opacity = (0.2 - gi * 0.035) * (0.6 + this.angularSpeed * 0.25);
      } else {
        ghost.visible = false;
      }
    });

    // shimmer: living flicker on edges and rim, dust breathing around it
    const flicker = Math.sin(t * 11.3) * 0.04 + Math.sin(t * 17.7 + 1.3) * 0.03;
    this.edgeMat.opacity = Math.min(1, 0.55 + p.edgeIntensity * 0.2 + lv * 0.5 + flicker);
    this.octaMat.uniforms.uIntensity.value =
      p.edgeIntensity * (0.8 + lv * 0.8) * (1 + 0.08 * Math.sin(t * 9.1));
    this.dust.rotation.y += dt * 0.06;
    this.dust.rotation.z = Math.sin(t * 0.13) * 0.15;
    this.dust.position.y = this.octa.position.y * 0.5;
    this.dustMat.opacity = 0.18 + 0.1 * Math.sin(t * 1.7) + lv * 0.25 + p.glowStrength * 0.08;
    this.dustMat.color.lerp(this.edgeMat.color, 0.02);

    this.angularSpeed =
      Math.abs(p.spinX) + Math.abs(p.spinY) + Math.abs(p.spinZ) + voiceSpin + wob +
      this.satellites.size * 0.6;

    // uniforms
    this.backdropMat.uniforms.uGlowStrength.value = p.glowStrength * (1 + lv * 0.9);
    this.backdropMat.uniforms.uPulse.value = p.gridPulse + lv * 0.4;
    this.backdropMat.uniforms.uTime.value = t;
    (this.afterimage.uniforms as { damp: { value: number } }).damp.value = this.themeDark
      ? 0.8
      : p.damp;
    this.bloom.strength = this.themeDark
      ? 0.25 + p.glowStrength * 0.2 + lv * 0.25
      : 0.08 + p.glowStrength * 0.08 + lv * 0.1;

    this.composer.render();
  }
}
