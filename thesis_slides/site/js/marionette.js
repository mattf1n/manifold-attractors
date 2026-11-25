// "The marionette model of steering" slide: a flightless bird (a cassowary, in a dinosaur's posture: body level, tail out
// behind, neck up in front) hanging from a two-bar marionette control, seen from three-quarters front and a little above.
// The control has two separate bars, each with its own strings. The main bar runs front to back: its front end holds the head
// string, its back end the body string. The leg bar runs side to side, held just below the main bar over the feet: a string
// hangs from each end to a foot. Both hang in place; they only turn.
//   The pointer drives one bar at a time, like a joystick around the middle of the stage; the other bar eases back to level.
//   Button up (just moving): the leg bar.
//     Left/right: yaw (turns the leg bar about the vertical). Its ends swing forward and back, and the feet with them.
//     Up/down: roll (rocks the leg bar). Pointer up lifts the near leg, pointer down the far leg.
//     Circles: each foot follows the pointer's path (near foot: right = forward, up = up; the far foot the same, half a turn
//       later), so circling clockwise steps the feet forward and counter-clockwise backward.
//   Button held down (dragging): the main bar.
//     Left/right: yaw (turns the main bar about the vertical): the head and body strings swing sideways, turning the bird.
//     Up/down: pitch, about the bar's middle. Pointer down lowers the head and raises the body, pointer up the reverse.
//   Left alone, the leg bar circles clockwise on its own, so the bird steps (CONFIG.autoplay); any pointer input takes over,
//   and it resumes idleDelay after the last.
// Physics, in 3-D: the bird's weight (gravity), the strings' tension (they only pull) and a little air drag so it comes to rest.
// Everything else is structure: the body, head, neck bones and each leg's thigh, shin and toes are rigid bodies (they move
// and turn freely in 3-D), joined at joints, some with limited bending (CONFIG.joints). There is one spring, at the head-neck
// joint (CONFIG.joints.headSpring), so the neck always leaves the head through its back; no floor or walls.
// Each step, every constraint (a joint holding two bodies together, a joint limit, a taut string) is projected onto the bodies
// it involves in proportion to their inverse masses and inverse moments of inertia, so it neither makes nor loses momentum
// (position-based rigid-body dynamics: Muller et al., "Detailed rigid body simulation with extended position based dynamics",
// 2020). Units are bird units (1 = a bit less than the body's half length); x is forward, y down, z toward the viewer's side
// of the bird (its near side). The loop sleeps when everything is still (and the autoplay is paused). Edit CONFIG to tweak.
(() => {
  const CONFIG = {
    // Stage.
    size: 0.11,             // one bird unit, as a fraction of the stage height
    minAspect: 0.75,        // stages narrower than this (width / height) size the bird by their width instead, as if they were this wide
    // Camera (orthographic): it looks at the bird's near side, turned `turn` degrees toward the bird's front (0 = pure
    // profile, 90 = head-on), from `elevation` degrees above.
    camera: { turn: 35, elevation: 15 },

    // Physics: weight, string tension and a little air drag (no friction, springs, floor or walls).
    stepRate: 960,          // simulation steps per second (fixed, so the feel is the same on every display). Small steps keep the
                            // joints tight more cheaply than extra passes, since light legs hang from a heavy body
    iterations: 3,          // constraint passes per step: more = stiffer joints and strings, more CPU
    gravity: 150,           // bird units per second squared: lower = floatier puppet. Below about 100 the legs swing slower than
                            // a quick circle of the pointer and lag behind it, so a clockwise circle can step them backward
    drag: 1.5,              // air drag: the share of velocity (and spin) lost per second is 1 - exp(-drag); 0 = it never stops
    masses: { body: 14, head: 1.5, neck: 0.6, thigh: 0.1, shin: 0.1, toes: 0.06 },   // the neck mass is shared by its bones. Heavier legs swing
                            // less far for the same turn of the leg bar (their own weight holds the feet forward, under the
                            // backward-bent ankle); lighter legs follow the strings more

    // Control: two separate bars (lengths in bird units).
    control: {
      pitchAt: 0.5,         // where the main bar pivots when pitched: 0.5 = in its middle (the head and body move opposite ways),
                            // 0 = at the body string (only the head rises and falls), 1 = at the head string. The main bar
                            // runs from straight above the body string to straight above the casque
      legHalf: 1.4,         // half the leg bar: the foot strings hang from its ends (longer = more swing and lift per turn)
      legDrop: 0.9,         // how far below the main bar the leg bar is held (enough that the two read as separate bars)
      thickness: 0.16
    },
    // Joint limits, in degrees. The neck has none: its two bones hang free between the body and the head. Ball joints (hips)
    // may swing their bone up to `swing` away from how it hangs at rest, any way, and twist it up to `twist` either way about
    // itself. Hinges (ankles, toes) turn about the side-to-side axis only, within [min, max].
    joints: {
      // thigh against the body. footYaw: how far (degrees) the leg's side-to-side axis (its ankle hinge) may turn away from
      // square to the body's forward, so the feet keep pointing forward; twist is centred on the feet pointing forward
      hip: { swing: 110, twist: 15, footYaw: 5 },
      // The only spring: at the head-neck joint, turning the last neck bone back toward leaving the back of the head straight
      // down (when the head is level). Higher = firmer; 0 = no spring (the joint swings freely).
      headSpring: 200,
      // Damping of the head on its neck, per second: how quickly the head stops turning relative to the body (its relative
      // spin is cut by this fraction of itself each second, up to all of it). 0 = undamped (the head wobbles longer).
      headDamping: 20,
      ankle: [-180, -25],   // hinge, measured from a straight leg (not from rest): 0 = straight, negative = bent with the heel
                            // pointing back, as a bird's. It may fold all the way (-180); it never straightens past 25 degrees,
                            // and never bends the other way
      toe: [-80, 45]        // hinge, toes against the shin, from their resting angle (positive = drooping). Wide, so the toe
                            // weight can keep them level
    },
    strings: {
      head: 1.8,            // the main bar's resting height above the top of the head (sets every string's length)
      bodySlack: 0,         // extra length of the body string
      foot: 4.84,           // length of each foot string. With the leg bar where it is, shorter strings lift the feet higher and
                            // fold the ankles further: at 4.84 the ankles hang at about 78 degrees, well clear of their stop
                            // (CONFIG.joints.ankle), so they bend both ways as the leg bar moves
      neckSlack: 0.75       // the head string is cut so that at rest the head hangs with the neck's base-to-head distance at this
                            // share of the neck's full length. The rest is slack: how far the head can be raised (pitching the
                            // main bar back) before the neck goes taut and starts to lift the front of the body
    },
    // Pointer: with the button up it drives the leg bar, with it held down the main bar. The bar turns as the pointer moves
    // away from the middle of the stage.
    reach: [5, 4],          // pointer distance from the middle of the stage (bird units, [left/right, up/down]) for the largest turn
    maxYaw: 1.2,            // leg bar: largest yaw either way, in radians (pointer right swings the near leg forward)
    maxRoll: 1.0,           // leg bar: largest roll either way, in radians (pointer up lifts the near leg)
    maxMainYaw: 0.6,        // main bar: largest yaw either way, in radians (turns the bird)
    maxPitch: 0.9,          // main bar: largest pitch either way, in radians (pointer down lowers the head)
    turnFollow: 12,         // how fast a bar turns to where the pointer puts it, or back to level when the other bar takes over, per second
    // The leg bar moves on its own while nobody steers: it is
    // already moving when the slide appears, tracing a clockwise circle (as a pointer circling clockwise would: yaw and roll a
    // quarter turn apart) so the feet step. Any pointer input on the stage takes over at once; idleDelay ms after the last
    // input it eases back in from wherever the bars are (over easeIn seconds). Shown again after being hidden: it plays at once.
    // period: seconds per circle; yaw, roll: its size, in radians. null turns it off.
    autoplay: { idleDelay: 10000, period: 4, yaw: 0.7, roll: 0.6, easeIn: 1.5 },

    // Bird, facing +x, in bird units from the body's centre (x forward, y down) in its middle plane (z = 0). Each leg is the
    // same, at +-legSpacing to the side.
    bird: {
      body: [1.25, 0.8], tilt: -0.1,     // half length and half height of the body; its tilt (radians, negative = chest up)
      bodyDepth: 0.95,                   // half the body's width (wide enough that the hips, legSpacing to each side, sit inside it)
      back: [0.05, -0.78],               // where the body string ties on: on top of the body, straight above its balance point,
                                         // so the body hangs level from it and pitching the main bar mostly raises or lowers the head
      tail: [-2.05, -0.45], tailRoot: [[-0.9, -0.5], [-1.0, 0.25]],   // tail tip, and where its upper and lower edges leave the body
      wing: { at: [-0.1, -0.15], size: [0.6, 0.28], angle: -0.2 },   // each folded wing, on the body's side: centre, half length and height, angle
      neckBase: [0.95, -0.35], neckMid: [1.5, -0.95], neckWidth: [0.3, 0.15, 0.09],   // neck curve: base, a point it bends toward; half widths at base, middle and head
      neckEnd: [-0.55, 0.05],            // where the neck joins the head, from the head's centre in head radii (back, down): at the
                                         // back of the head, near its middle height. The neck leaves it straight down when the head is level
      neckBones: 5,                      // bones in the neck (more = a more supple neck)
      head: [1.95, -1.45], headRadius: 0.2,
      casque: [1.92, -1.87], casqueWidth: 0.17, casqueThickness: 0.07,   // top of the casque (the head string ties here); its half width (front to back) and half thickness (side to side)
      beak: [2.5, -1.35], beakDepth: 0.12, beakWidth: 0.1,   // beak tip; the beak's depth (top to bottom) and width (side to side) where it leaves the head
      eye: { at: [0.05, -0.05], radius: 0.07, pupil: 0.038 },   // from the head's centre (front/back, up/down); each eye sits on the head's surface, one per side
      wattle: { at: [-0.04, 0.17], length: 0.3, width: 0.07 },  // hangs from under the head
      legSpacing: 0.75,                  // each hip's distance to the side of the body's middle plane
      // Each leg: hip - thigh - ankle (heel back) - shin - foot joint - toes. Where the foot strings tie on: the foot joint.
      hip: [0.1, 0.4], ankle: [-0.167, 1.6], foot: [0.1, 2.8], toe: [0.55, 2.8],
      thighWidth: 0.34, legWidth: 0.1, toeWidth: 0.07, toeSplay: 0.4,   // toeSplay: angle (radians) of the second toe, turned outward
      // Toe weight: an unseen weight fixed under each foot's toes, placed so the toes and weight together balance straight
      // below the foot joint when the toes are level. Gravity then turns the toes level (their plane's normal vertical),
      // the way a keel rights a boat. Deeper or heavier = the toes return to level more firmly.
      toeWeight: { depth: 0.6, mass: 0.5 }
    },
    colors: {
      body: '#202a36', wing: '#334155', neck: [42, 111, 219], wattle: [214, 40, 57], casque: [236, 160, 32], beak: [32, 42, 54],
      legs: [236, 160, 32], farLegs: [196, 128, 18],   // the far leg is a shade darker
      eye: '#ffffff', pupil: '#202a36',
      string: 'rgba(32,42,54,0.7)',
      control: [40, 160, 110]   // both bars
    },
    stringWidth: 1,         // CSS pixels

    // Sleep: the loop stops when every body is slower than this (bird units per second, spin included) for settleFrames frames.
    sleepSpeed: 0.05,
    settleFrames: 20,
    maxFrameTime: 1 / 20    // longest stretch simulated after a slow frame, in seconds (avoids a jump after a stall)
  };

  const slide = document.querySelector('#marionette-slide'), canvas = document.querySelector('#marionette-stage'), ctx = canvas.getContext('2d');
  const B = CONFIG.bird, X = CONFIG.control, C = CONFIG.colors, M = CONFIG.masses, deg = Math.PI / 180;
  const rgb = c => typeof c === 'string' ? c : `rgb(${c[0]},${c[1]},${c[2]})`;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  // ---- 3-D math: vectors [x, y, z], quaternions [w, x, y, z] ----
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = a => Math.hypot(a[0], a[1], a[2]);
  const unit = a => mul(a, 1 / (len(a) || 1));
  const perp = (v, n) => unit(sub(v, mul(n, dot(v, n))));   // v made perpendicular to the unit vector n
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
  function rot(q, v) {   // v turned by the unit quaternion q
    const [w, x, y, z] = q, tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
    return [v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx];
  }
  const rotInv = (q, v) => rot([q[0], -q[1], -q[2], -q[3]], v);
  const qmul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
  function turnBy(q, r) {   // turn q (in place) by the small rotation vector r: q <- q + 1/2 [0, r] q, renormalised
    const [w, x, y, z] = q;
    const nw = w - 0.5 * (r[0] * x + r[1] * y + r[2] * z), nx = x + 0.5 * (r[0] * w + r[1] * z - r[2] * y);
    const ny = y + 0.5 * (r[1] * w + r[2] * x - r[0] * z), nz = z + 0.5 * (r[2] * w + r[0] * y - r[1] * x), n = Math.hypot(nw, nx, ny, nz);
    q[0] = nw / n; q[1] = nx / n; q[2] = ny / n; q[3] = nz / n;
  }
  const rotAxis = (v, n, a) => add(add(mul(v, Math.cos(a)), mul(cross(n, v), Math.sin(a))), mul(n, dot(n, v) * (1 - Math.cos(a))));   // v turned by a about the unit axis n
  function signedAngle(n, n1, n2) {   // angle (about the unit axis n) from n1 to n2, both perpendicular to n
    let a = Math.asin(clamp(dot(cross(n1, n2), n), -1, 1));
    if (dot(n1, n2) < 0) a = Math.PI - a;
    return wrap(a);
  }
  function inverse3(m) {   // inverse of a 3x3 matrix stored row by row
    const [a, b, c, d, e, f, g, h, i] = m, A = e * i - f * h, Bc = -(d * i - f * g), Cc = d * h - e * g, det = a * A + b * Bc + c * Cc;
    return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det, Bc / det, (a * i - c * g) / det, -(a * f - c * d) / det,
      Cc / det, -(a * h - b * g) / det, (a * e - b * d) / det];
  }
  const v3 = ([x, y], z = 0) => [x, y, z];

  // ---- Bodies ----
  // A rigid body: mass m, centre of mass x, orientation q (identity in the resting pose, so its frame has the world's axes
  // at rest), velocity v, spin w, and the inverse of its inertia tensor in its own frame. Its mass and inertia come from a
  // few point masses placed in the resting pose.
  const bodies = [], joints = [], ropes = [];
  function makeBody(points) {   // points: [[position], mass] in the resting pose
    const m = points.reduce((s, [, pm]) => s + pm, 0), x = mul(points.reduce((s, [p, pm]) => add(s, mul(p, pm)), [0, 0, 0]), 1 / m);
    const I = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    let size = 0;
    for (const [p, pm] of points) {
      const r = sub(p, x), rr = dot(r, r); size = Math.max(size, Math.sqrt(rr));
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) I[3 * i + j] += pm * ((i === j ? rr : 0) - r[i] * r[j]);
    }
    for (let i = 0; i < 3; i++) I[4 * i] += 0.02 * m;   // a little inertia about every axis, so thin bones don't spin infinitely fast about themselves
    const b = { m, invI: inverse3(I), x, q: [1, 0, 0, 0], v: [0, 0, 0], w: [0, 0, 0], px: x, pq: [1, 0, 0, 0], size };
    bodies.push(b); return b;
  }
  const local = (b, p) => sub(p, b.x);   // a resting-pose point in b's frame (only while b is in its resting pose)
  const world = (b, r) => add(b.x, rot(b.q, r));
  const invIw = (b, v) => { const u = rotInv(b.q, v), m = b.invI; return rot(b.q, [m[0] * u[0] + m[1] * u[1] + m[2] * u[2], m[3] * u[0] + m[4] * u[1] + m[5] * u[2], m[6] * u[0] + m[7] * u[1] + m[8] * u[2]]); };

  // Body: an ellipsoid (point masses at the ends of its three half-axes), turned by its tilt.
  const [bw, bh] = B.body, bd = B.bodyDepth, tiltV = (x, y) => [x * Math.cos(B.tilt) - y * Math.sin(B.tilt), x * Math.sin(B.tilt) + y * Math.cos(B.tilt), 0];
  const torso = makeBody([[tiltV(bw, 0)], [tiltV(-bw, 0)], [tiltV(0, bh)], [tiltV(0, -bh)], [[0, 0, bd]], [[0, 0, -bd]]].map(([p]) => [p, M.body / 6]));
  const bone = (a, b, m) => makeBody([[a, m / 2], [b, m / 2]]);   // a bone: half its mass at each end
  const r = B.headRadius, headBackP = v3([B.head[0] + B.neckEnd[0] * r, B.head[1] + B.neckEnd[1] * r]);   // the neck's end, at the back of the head
  // Neck: a chain of bones along a curve (cubic Bezier) from its base on the body, bending toward neckMid, and rising straight
  // up into the back of the head (its last control point is directly below the end).
  const neckCurve = t => { const p0 = v3(B.neckBase), p3 = headBackP, p1 = v3(B.neckMid), p2 = add(p3, [0, 0.5 * Math.abs(p3[1] - p0[1]), 0]), u = 1 - t;
    return add(add(mul(p0, u ** 3), mul(p1, 3 * u * u * t)), add(mul(p2, 3 * u * t * t), mul(p3, t ** 3))); };
  const neckAt = Array.from({ length: B.neckBones + 1 }, (_, i) => neckCurve(i / B.neckBones));
  const neckBones = neckAt.slice(1).map((p, i) => bone(neckAt[i], p, M.neck / B.neckBones));
  const head = makeBody([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]].map(d => [add(v3(B.head), mul(d, 1.5 * r)), M.head / 6]));
  const legs = [-1, 1].map(s => {   // s = -1 far, +1 near
    const z = s * B.legSpacing, hip = v3(B.hip, z), ankle = v3(B.ankle, z), foot = v3(B.foot, z), toe = v3(B.toe, z);
    const K = B.toeWeight, keel = [B.foot[0] - (B.toe[0] - B.foot[0]) * M.toes / K.mass, B.foot[1] + K.depth, z];   // balances the toes about the foot joint
    const thigh = bone(hip, ankle, M.thigh), shin = bone(ankle, foot, M.shin), toes = makeBody([[toe, M.toes], [keel, K.mass]]);
    const splay = rotAxis(sub(toe, foot), [0, 1, 0], -s * B.toeSplay);   // the second toe, turned outward about the vertical
    return { s, thigh, shin, toes, hip: local(torso, hip), ankle: local(thigh, ankle), foot: local(shin, foot),
      toeTip: local(toes, toe), toe2: local(toes, add(foot, mul(splay, 0.8))) };
  });
  // The neck's drawn points (its base on the body, then the far end of each bone), and half widths along it.
  const neckPts = [[torso, local(torso, neckAt[0])], ...neckBones.map((b, i) => [b, local(b, neckAt[i + 1])])];
  const neckWidths = neckAt.map((_, i) => { const t = 2 * i / B.neckBones, [w0, w1, w2] = B.neckWidth; return t <= 1 ? w0 + (w1 - w0) * t : w1 + (w2 - w1) * (t - 1); });

  // ---- Joints ----
  // A ball joint lets the child body swing its bone `axis` up to `swing` from the parent's copy of it and twist up to
  // `twist` about it; a hinge keeps the two bodies' side-to-side axes lined up and limits the angle between their bones.
  const J = CONFIG.joints;
  function ball(b1, b2, at, axis, lim) {   // lim null: a free ball joint (it only holds the two bodies together)
    axis = unit(axis);
    const ref = perp([0, 0, 1], axis);
    joints.push({ kind: lim ? 'ball' : 'free', b1, b2, r1: local(b1, at), r2: local(b2, at), a1: axis, a2: axis, c1: ref, c2: ref,
      swing: lim && lim.swing * deg, twist: lim && lim.twist * deg, footYaw: lim && lim.footYaw !== undefined ? lim.footYaw * deg : undefined });
  }
  function hinge(b1, b2, at, parentDir, childDir, range, fromStraight) {
    const axis = [0, 0, 1], c1 = unit(parentDir), c2 = unit(childDir), rest = fromStraight ? 0 : signedAngle(axis, c1, c2);
    joints.push({ kind: 'hinge', b1, b2, r1: local(b1, at), r2: local(b2, at), a1: axis, a2: axis, c1, c2, min: rest + range[0] * deg, max: rest + range[1] * deg });
  }
  neckBones.forEach((b, i) => ball(i ? neckBones[i - 1] : torso, b, neckAt[i], sub(neckAt[i + 1], neckAt[i]), null));   // the neck: free joints
  ball(neckBones[neckBones.length - 1], head, headBackP, sub(v3(B.beak), headBackP), null);
  const headJoint = joints[joints.length - 1];   // its spring is applied once per step (see substep)
  // The spring's rest: the last neck bone turned (by the smallest rotation) to point straight up into the head, in the head's
  // frame, so with the head level the neck leaves its back vertically.
  const lastBone = sub(neckAt[neckAt.length - 1], neckAt[neckAt.length - 2]), d = unit(lastBone), up = [0, -1, 0];
  const neckRest = (q => q.map(v => v / Math.hypot(...q)))([1 + dot(d, up), ...cross(d, up)]);
  legs.forEach(l => {
    const z = l.s * B.legSpacing, hip = v3(B.hip, z), ankle = v3(B.ankle, z), foot = v3(B.foot, z), toe = v3(B.toe, z);
    ball(torso, l.thigh, hip, sub(ankle, hip), J.hip);
    hinge(l.thigh, l.shin, ankle, sub(ankle, hip), sub(foot, ankle), J.ankle, true);
    hinge(l.shin, l.toes, foot, sub(foot, ankle), sub(toe, foot), J.toe, false);
    l.ankleJoint = joints[joints.length - 2];
  });
  const ankleBend = l => { const j = l.ankleJoint, n = rot(j.b1.q, j.a1); return signedAngle(n, perp(rot(j.b1.q, j.c1), n), perp(rot(j.b2.q, j.c2), n)); };   // radians, from straight

  // ---- Constraint projection ----
  function moveBody(b, P, rr, s) { b.x = add(b.x, mul(P, s / b.m)); turnBy(b.q, mul(invIw(b, cross(rr, P)), s)); }
  // Bring point p1 (on b1) to `rest` distance from point p2 (on b2, or fixed when b2 is null); a string only pulls.
  function positional(b1, p1, b2, p2, rest, string) {
    const d = sub(p2, p1), l = len(d), c = l - rest;
    if (l < 1e-9 || (string && c <= 0)) return;
    const n = mul(d, 1 / l), r1 = sub(p1, b1.x), k1 = cross(r1, n);
    let w = 1 / b1.m + dot(k1, invIw(b1, k1)), r2;
    if (b2) { r2 = sub(p2, b2.x); const k2 = cross(r2, n); w += 1 / b2.m + dot(k2, invIw(b2, k2)); }
    const P = mul(n, c / w);
    moveBody(b1, P, r1, 1); if (b2) moveBody(b2, P, r2, -1);
  }
  // Turn b1 by the rotation vector `corr` relative to b2, shared by their inverse moments of inertia about its axis. With a
  // compliance (the inverse of a spring's stiffness, per step h), only part of the turn is made: a spring (XPBD).
  function angular(b1, b2, corr, compliance = 0, h = 1) {
    const th = len(corr); if (th < 1e-9) return;
    const n = mul(corr, 1 / th), P = mul(n, th / (dot(n, invIw(b1, n)) + dot(n, invIw(b2, n)) + compliance / (h * h)));
    turnBy(b1.q, invIw(b1, P)); turnBy(b2.q, mul(invIw(b2, P), -1));
  }
  // Keep the angle (about n) from n1 (on b1) to n2 (on b2) within [min, max]. Past a limit, it is turned back to the nearer
  // limit (the nearer way round the circle) by exactly the overshoot.
  function limitAngle(b1, b2, n, n1, n2, min, max) {
    const a = signedAngle(n, n1, n2);
    if (a >= min && a <= max) return;
    const pastMin = wrap(a - min), pastMax = wrap(a - max);
    angular(b1, b2, mul(n, Math.abs(pastMin) < Math.abs(pastMax) ? pastMin : pastMax));
  }
  function solveJoint(j) {
    positional(j.b1, world(j.b1, j.r1), j.b2, world(j.b2, j.r2), 0, false);   // the two bodies stay joined
    if (j.kind === 'hinge') {
      // The hinge axes stay lined up, turned together by the exact angle between them (even when far apart).
      const A1 = rot(j.b1.q, j.a1), A2 = rot(j.b2.q, j.a2), c = cross(A1, A2), s = len(c);
      if (s > 1e-9 || dot(A1, A2) < 0) angular(j.b1, j.b2, mul(s > 1e-6 ? mul(c, 1 / s) : perp([1, 0, 0], A1), Math.atan2(s, dot(A1, A2))));
      const n = rot(j.b1.q, j.a1);
      limitAngle(j.b1, j.b2, n, perp(rot(j.b1.q, j.c1), n), perp(rot(j.b2.q, j.c2), n), j.min, j.max);
    } else if (j.kind === 'ball') {
      const A1 = rot(j.b1.q, j.a1), A2 = rot(j.b2.q, j.a2), swing = Math.acos(clamp(dot(A1, A2), -1, 1));
      if (swing > j.swing) angular(j.b1, j.b2, mul(unit(cross(A1, A2)), swing - j.swing));
      const n = unit(add(rot(j.b1.q, j.a1), rot(j.b2.q, j.a2)));
      limitAngle(j.b1, j.b2, n, perp(rot(j.b1.q, j.c1), n), perp(rot(j.b2.q, j.c2), n), -j.twist, j.twist);
      if (j.footYaw !== undefined) {   // the child's side-to-side axis stays within footYaw of square to the parent's forward
        const F = rot(j.b1.q, [1, 0, 0]), A = rot(j.b2.q, [0, 0, 1]), d = dot(A, F), e = Math.abs(Math.asin(clamp(d, -1, 1)));
        if (e > j.footYaw) angular(j.b1, j.b2, mul(unit(cross(A, F)), Math.sign(d) * (e - j.footYaw)));
      }
    }
  }
  const solveString = s => positional(s.b, world(s.b, s.r), null, s.anchor, s.len, true);

  // ---- The control and its strings ----
  const bar = { x: 0, y: 0, yaw: 0, roll: 0, mainYaw: 0, pitch: 0 }, target = { yaw: 0, roll: 0, mainYaw: 0, pitch: 0 };   // yaw, roll: leg bar
  // Where a string ties onto the control: 'front' or 'back' (main bar), a leg bar end (-1 far, +1 near), or 0 for the leg
  // bar's centre. Returns [x, y, z]. The main bar pitches (forward/back) about its pivot (CONFIG.control.pitchAt), then yaws;
  // the leg bar, held legDrop below the main bar, rolls (about the front-to-back axis), then yaws.
  function anchor(at, b = bar) {
    let x, y = 0, z;
    if (at === 'front' || at === 'back') {
      const arm = arms[at], q = b.mainYaw;
      x = arm * Math.cos(b.pitch); y = arm * Math.sin(b.pitch);
      [x, z] = [x * Math.cos(q), -x * Math.sin(q)];
      return [b.x + x, b.y + y, z];
    }
    z = at * X.legHalf;
    [y, z] = [-z * Math.sin(b.roll), z * Math.cos(b.roll)];
    [x, z] = [z * Math.sin(b.yaw), z * Math.cos(b.yaw)];
    return [b.x + crossAt + x, b.y + X.legDrop + y, z];
  }
  // At rest both bars are level: the main bar's ends straight above the body string and the casque, its pivot between them.
  const pivotX = B.back[0] + X.pitchAt * (B.casque[0] - B.back[0]), arms = { front: B.casque[0] - pivotX, back: B.back[0] - pivotX };
  const restBar = { x: pivotX, y: B.casque[1] - CONFIG.strings.head, yaw: 0, roll: 0, mainYaw: 0, pitch: 0 };
  let crossAt = 0;   // where the leg bar sits along the main bar: straight above the feet as they hang at rest (set when rigging)
  // A string from point r (in body b's frame) to the control at `at`, as long as it is in the current pose plus slack.
  const string = (b, r, at, slack) => { const s = { b, r, at }; s.len = len(sub(world(b, r), anchor(at, restBar))) + slack; ropes.push(s); return s; };
  string(head, local(head, v3(B.casque)), 'front', 0);
  string(torso, local(torso, v3(B.back)), 'back', CONFIG.strings.bodySlack);

  // ---- One step of h seconds: move every body by its velocity and spin (after weight and drag), then project the
  // constraints, then take the new velocities from how far the bodies actually moved. Returns the fastest body's speed. ----
  function substep(h, drag, cross_) {
    const keep = Math.exp(-drag * h);
    for (const b of bodies) {
      b.v = mul(b.v, keep); b.v[1] += CONFIG.gravity * h; b.w = mul(b.w, keep);
      b.px = b.x; b.pq = b.q.slice();
      b.x = add(b.x, mul(b.v, h)); turnBy(b.q, mul(b.w, h));
    }
    for (const s of ropes) s.anchor = anchor(s.at, cross_);
    if (CONFIG.joints.headSpring > 0) {   // the head-neck spring: turns the last neck bone toward its rest in the head's frame (neckRest)
      const { b1, b2 } = headJoint, dq = qmul(qmul(b2.q, neckRest), [b1.q[0], -b1.q[1], -b1.q[2], -b1.q[3]]), s = dq[0] < 0 ? -1 : 1, v = [dq[1] * s, dq[2] * s, dq[3] * s], sv = len(v);
      if (sv > 1e-9) angular(b1, b2, mul(v, 2 * Math.atan2(sv, Math.abs(dq[0])) / sv), 1 / CONFIG.joints.headSpring, h);
    }
    for (let it = 0; it < CONFIG.iterations; it++) { ropes.forEach(solveString); joints.forEach(solveJoint); }
    let fastest = 0;
    for (const b of bodies) {
      b.v = mul(sub(b.x, b.px), 1 / h);
      const dq = qmul(b.q, [b.pq[0], -b.pq[1], -b.pq[2], -b.pq[3]]), k = (dq[0] < 0 ? -2 : 2) / h;
      b.w = [dq[1] * k, dq[2] * k, dq[3] * k];
    }
    // Neck damping: slow the head's spin relative to the body (equal and opposite, shared by their inertia). Damping only
    // the head-neck joint does little: the light neck bones turn with the head, so the wobble is head against body.
    if (CONFIG.joints.headDamping > 0) {
      const b1 = torso, b2 = head, dw = sub(b2.w, b1.w), m = len(dw);
      if (m > 1e-9) { const n = mul(dw, 1 / m), P = mul(n, m * Math.min(CONFIG.joints.headDamping * h, 1) / (dot(n, invIw(b1, n)) + dot(n, invIw(b2, n))));
        b1.w = add(b1.w, invIw(b1, P)); b2.w = sub(b2.w, invIw(b2, P)); }
    }
    for (const b of bodies) {
      fastest = Math.max(fastest, len(b.v) + len(b.w) * b.size);
    }
    return fastest;
  }

  // ---- Rigging (once, in idle time after the page loads, or when the slide is first shown). First the bird hangs from its
  // head and body strings alone (damped) until it is still. Then the foot strings are tied on, the leg bar straight above
  // the hips (so the strings draw the feet in under the hips), and brought to CONFIG.strings.foot: at rest they carry part of each leg's weight, so the
  // ankle hangs between its limits, and a foot swung off any way is lifted by its string and brought back by its weight.
  // It simulates about 9 seconds, which takes a few hundred milliseconds. ----
  let rigged = false;
  function rig() {
    if (rigged) return;
    const h = 1 / 240, damping = 6;   // its own step rate: rest needs no fine steps
    const relax = (seconds, after) => { for (let i = 0; i < seconds * 240; i++) { substep(h, damping, restBar); after(); } };
    // The head string: lengthened (head lower) while the neck is straighter than neckSlack allows, shortened while more bent.
    const neckFull = neckAt.slice(1).reduce((s, p, i) => s + len(sub(p, neckAt[i])), 0);   // all neck bones, end to end
    relax(3, () => {
      const chord = len(sub(world(...neckPts[0]), world(...neckPts[neckPts.length - 1])));
      ropes[0].len += clamp(0.2 * (chord - CONFIG.strings.neckSlack * neckFull), -0.01, 0.01);
    });
    const overHips = () => world(torso, legs[0].hip)[0] - restBar.x;
    crossAt = overHips();
    legs.forEach(l => { l.string = string(l.shin, l.foot, l.s, 0); });
    // Bring each foot string to its length a little per step (so the feet rise smoothly), keeping the leg bar over the hips,
    // until the legs hang still.
    relax(6, () => {
      crossAt = overHips();
      legs.forEach(l => { l.string.len += clamp(CONFIG.strings.foot - l.string.len, -0.01, 0.01); });
    });
    bodies.forEach(b => { b.v = [0, 0, 0]; b.w = [0, 0, 0]; b.px = b.x; b.pq = b.q.slice(); });
    // The hips hang where the legs' weight puts them; make that their neutral swing, so each swings the same amount either
    // way from how it hangs at rest. Their twist stays centred on the drawn template, where the feet point forward.
    joints.forEach(j => { if (j.kind === 'ball') j.a2 = rotInv(j.b2.q, rot(j.b1.q, j.a1)); });
    rigged = true;
  }
  if (window.requestIdleCallback) requestIdleCallback(rig); else setTimeout(rig, 0);

  // ---- World state (bird units; x right, y down; origin at the top-left corner of the stage) ----
  let scale = 0, dpr = 1, worldW = 0, worldH = 0, feetY = 0, placed = false;   // feetY: where the feet hang at rest (for console inspection)
  const shift = d => { bodies.forEach(b => { b.x = add(b.x, d); b.px = add(b.px, d); }); bar.x += d[0]; bar.y += d[1]; };
  // Autoplay (CONFIG.autoplay): `auto` is set by the loop; autoTime runs only while it plays; autoBlend eases from `held`
  // (the bars' targets when it took over) into the circle.
  let auto = false, autoTime = 0, autoBlend = 0, held = { ...target };
  function step(h) {
    if (auto) {
      const A = CONFIG.autoplay, a = 2 * Math.PI * autoTime / A.period;
      autoTime += h; autoBlend = Math.min(1, autoBlend + h / A.easeIn);
      const e = autoBlend * autoBlend * (3 - 2 * autoBlend);   // smoothstep: starts and arrives gently
      const circle = { yaw: A.yaw * Math.sin(a), roll: A.roll * Math.cos(a), mainYaw: 0, pitch: 0 };
      for (const k in target) target[k] = held[k] + (circle[k] - held[k]) * e;
    }
    // The bars turn toward where the pointer (or the autoplay) puts them.
    const rate = 1 - Math.exp(-CONFIG.turnFollow * h);
    for (const k in target) bar[k] += (target[k] - bar[k]) * rate;
    return substep(h, CONFIG.drag, bar);
  }

  // ---- Drawing: an orthographic camera. Screen x and y and depth (toward the camera) are the rows of `cam`. ----
  const T = CONFIG.camera.turn * deg, E = CONFIG.camera.elevation * deg;
  const camX = [Math.cos(T), 0, -Math.sin(T)], camY = [Math.sin(E) * Math.sin(T), Math.cos(E), Math.sin(E) * Math.cos(T)];
  const camZ = [Math.cos(E) * Math.sin(T), -Math.sin(E), Math.cos(E) * Math.cos(T)];
  let pivot = [0, 0, 0];   // the camera turns about the middle of the stage
  const project = p => { const q = sub(p, pivot); return { x: pivot[0] + dot(camX, q), y: pivot[1] + dot(camY, q), d: dot(camZ, q) }; };
  const flat = v => [dot(camX, v), dot(camY, v)];   // a direction on screen
  const facing = v => dot(camZ, v) > 0;              // whether a direction points toward the camera
  // Draw in a body's own plane: template coordinates (x, y) at depth z map to the screen (set as a canvas transform).
  function inPlane(b, origin, z) {
    const u = flat(rot(b.q, [1, 0, 0])), v = flat(rot(b.q, [0, 1, 0])), o = project(world(b, [-origin[0], -origin[1], z]));
    ctx.transform(u[0], u[1], v[0], v[1], o.x, o.y);
  }
  // A tapered limb through screen points ps with half-widths ws: round joints joined by straight-sided segments.
  function limb(ps, ws) {
    ps.forEach((p, i) => { ctx.beginPath(); ctx.arc(p.x, p.y, ws[i], 0, 2 * Math.PI); ctx.fill(); });
    for (let i = 1; i < ps.length; i++) {
      const a = ps[i - 1], b = ps[i], d = Math.hypot(b.x - a.x, b.y - a.y) || 1e-9, nx = -(b.y - a.y) / d, ny = (b.x - a.x) / d;
      ctx.beginPath();
      ctx.moveTo(a.x + nx * ws[i - 1], a.y + ny * ws[i - 1]); ctx.lineTo(b.x + nx * ws[i], b.y + ny * ws[i]);
      ctx.lineTo(b.x - nx * ws[i], b.y - ny * ws[i]); ctx.lineTo(a.x - nx * ws[i - 1], a.y - ny * ws[i - 1]); ctx.closePath();
      ctx.fill();
    }
  }
  function drawLeg(l) {
    const hip = project(world(torso, l.hip)), ankle = project(world(l.thigh, l.ankle)), foot = project(world(l.shin, l.foot));
    const toe = project(world(l.toes, l.toeTip)), toe2 = project(world(l.toes, l.toe2));
    const thigh = { x: hip.x + 0.45 * (ankle.x - hip.x), y: hip.y + 0.45 * (ankle.y - hip.y) };
    ctx.fillStyle = rgb(l.s < 0 ? C.farLegs : C.legs);
    limb([hip, ankle, foot], [B.legWidth / 2, B.legWidth / 2, B.legWidth / 2]);
    limb([foot, toe], [B.toeWidth / 2, B.toeWidth / 3]); limb([foot, toe2], [B.toeWidth / 2, B.toeWidth / 3]);
    ctx.fillStyle = C.body; limb([hip, thigh], [B.thighWidth / 2, B.thighWidth * 0.3]);   // the feathered drumstick over the top of the leg
  }
  function drawString(s) {   // straight when taut, sagging (downward) when slack: a parabola as long as the string
    const a3 = s.anchor || anchor(s.at), p3 = world(s.b, s.r), d = len(sub(p3, a3));
    const sag = s.len > d ? Math.min(Math.sqrt(3 * d * (s.len - d) / 8), s.len / 2) : 0;
    const a = project(a3), p = project(p3), m = project(add(mul(add(a3, p3), 0.5), [0, 2 * sag, 0]));
    ctx.strokeStyle = C.string; ctx.lineWidth = CONFIG.stringWidth / scale;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo(m.x, m.y, p.x, p.y); ctx.stroke();
  }
  function drawNeck() {
    ctx.fillStyle = rgb(C.neck); limb(neckPts.map(([b, p]) => project(world(b, p))), neckWidths);
  }
  function drawTorso() {
    ctx.fillStyle = C.body;
    // Tail, in the body's middle plane.
    const [[ux, uy], [lx, ly]] = B.tailRoot, [tx, ty] = B.tail;
    ctx.save(); inPlane(torso, [0, 0], 0);
    ctx.beginPath(); ctx.moveTo(ux, uy); ctx.quadraticCurveTo((ux + tx) / 2, uy - 0.1, tx, ty); ctx.quadraticCurveTo((lx + tx) / 2, ly, lx, ly); ctx.closePath(); ctx.fill();
    ctx.restore();
    // Body: an ellipsoid.
    ellipsoid(torso.x, [tiltV(bw, 0), tiltV(0, bh), [0, 0, bd]].map(v => rot(torso.q, v)));
    // Wings, folded on the body's sides: only the one facing the camera shows.
    const W = B.wing;
    for (const side of [-1, 1]) {
      if (!facing(rot(torso.q, [0, 0, side]))) continue;
      ctx.save(); inPlane(torso, [0, 0], side * bd * 0.92);
      ctx.fillStyle = C.wing; ctx.beginPath(); ctx.ellipse(W.at[0], W.at[1], W.size[0], W.size[1], W.angle, 0, 2 * Math.PI); ctx.fill();
      ctx.restore();
    }
  }
  // An ellipsoid (centre c, three half-axis vectors, in world space), filled: its outline is the ellipse with shape matrix
  // A A^T (A: the half-axes on screen), drawn as the unit circle under a matrix L with L L^T = A A^T (Cholesky).
  function ellipsoid(c, axes) {
    const a = axes.map(flat), c11 = a.reduce((s, v) => s + v[0] * v[0], 0), c12 = a.reduce((s, v) => s + v[0] * v[1], 0), c22 = a.reduce((s, v) => s + v[1] * v[1], 0);
    const l11 = Math.sqrt(c11) || 1e-9, l21 = c12 / l11, l22 = Math.sqrt(Math.max(1e-9, c22 - l21 * l21)), o = project(c);
    ctx.save(); ctx.transform(l11, l21, 0, l22, o.x, o.y); ctx.beginPath(); ctx.arc(0, 0, 1, 0, 2 * Math.PI); ctx.restore(); ctx.fill();
  }
  const shade = (c, k) => rgb(c.map(v => Math.round(v + (255 - v) * k)));   // a colour mixed with k of white
  // The head and its features, in 3-D: a sphere, with the casque and wattle as ellipsoids, the beak as a four-sided pyramid,
  // and an eyeball (with its pupil) on each side. Parts behind the head are drawn before it, parts in front after it.
  function drawHead() {
    const [hx, hy] = B.head, [cx, cy] = B.casque, Wt = B.wattle, E2 = B.eye, o = project(head.x);
    const at = (x, y, z = 0) => world(head, [x - hx, y - hy, z]);   // a point given in template coordinates (z: to the side)
    const dirH = (x, y, z) => rot(head.q, [x, y, z]);                 // a direction in the head's frame, in world space
    const top = hy - r, casqueHalf = (top - cy) / 2 + 0.04;
    ctx.fillStyle = rgb(C.casque);
    ellipsoid(at(cx, cy + casqueHalf), [dirH(B.casqueWidth * 0.75, 0, 0), dirH(0, casqueHalf, 0), dirH(0, 0, B.casqueThickness)]);
    ctx.fillStyle = rgb(C.wattle);
    ellipsoid(at(hx + Wt.at[0], hy + Wt.at[1] + Wt.length / 2), [dirH(Wt.width, 0, 0), dirH(0, Wt.length / 2, 0), dirH(0, 0, Wt.width)]);
    // Beak: tip, and four corners of its base just inside the head's front.
    const [bx, by] = B.beak, base = [hx + 0.12, hy + 0.02], d = B.beakDepth / 2, w = B.beakWidth / 2;
    const tip = at(bx, by), corners = [at(base[0], base[1] - d), at(base[0], base[1], w), at(base[0], base[1] + d), at(base[0], base[1], -w)];
    const mid = at(base[0], base[1]);
    const drawBeak = () => corners.forEach((c1, i) => {
      const c2 = corners[(i + 1) % 4];
      let n = cross(sub(c1, tip), sub(c2, tip));
      if (dot(n, sub(add(c1, c2), mul(mid, 2))) < 0) n = mul(n, -1);   // outward
      if (!facing(n)) return;
      ctx.fillStyle = shade(C.beak, 0.25 * Math.max(0, -unit(n)[1]));   // faces turned up catch a little light
      const [p0, p1, p2] = [tip, c1, c2].map(project);
      ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.closePath(); ctx.fill();
    });
    const beakInFront = project(tip).d > o.d;
    if (!beakInFront) drawBeak();
    ctx.fillStyle = rgb(C.neck); ctx.beginPath(); ctx.arc(o.x, o.y, r, 0, 2 * Math.PI); ctx.fill();
    if (beakInFront) drawBeak();
    // Eyes: an eyeball on the head's surface on each side, its pupil looking out and a little forward; shown when facing the camera.
    const ez = Math.sqrt(Math.max(0, r * r - E2.at[0] ** 2 - E2.at[1] ** 2));
    for (const side of [-1, 1]) {
      const out = unit(dirH(E2.at[0], E2.at[1], side * ez));
      if (!facing(out)) continue;
      const eye = at(hx + E2.at[0], hy + E2.at[1], side * ez * 0.85), look = unit(add(out, dirH(0.6, 0, 0)));
      const pe = project(eye), pp = project(add(eye, mul(look, E2.radius * 0.75)));
      ctx.fillStyle = C.eye; ctx.beginPath(); ctx.arc(pe.x, pe.y, E2.radius, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = C.pupil; ctx.beginPath(); ctx.arc(pp.x, pp.y, E2.pupil, 0, 2 * Math.PI); ctx.fill();
    }
  }
  function draw() {
    const k = scale * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(k, 0, 0, k, 0, 0);
    // Back to front by depth. The legs and neck start inside the body, so they always go behind it.
    const bodyD = project(torso.x).d, items = [];
    legs.forEach(l => items.push({ d: bodyD - 1 + project(world(l.shin, [0, 0, 0])).d * 1e-3, draw: () => drawLeg(l) }));
    // The neck goes in front of the body (its base overlaps the body) and behind the head.
    const headD = Math.max(bodyD + 0.02, project(head.x).d);
    items.push({ d: bodyD, draw: drawTorso }, { d: bodyD + 0.01, draw: drawNeck }, { d: headD, draw: drawHead });
    ropes.forEach(s => items.push({ d: project(world(s.b, s.r)).d + (s.b === torso || s.b === head ? 1 : 0.01), draw: () => drawString(s) }));
    items.sort((a, b) => a.d - b.d).forEach(i => i.draw());
    // The control: far half of the leg bar, the main bar (back end to front end), near half of the leg bar.
    ctx.strokeStyle = rgb(C.control); ctx.lineWidth = X.thickness; ctx.lineCap = 'round';
    const mid = project(anchor(0)), seg = (a, b) => { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); };
    seg(mid, project(anchor(-1))); seg(project(anchor('back')), project(anchor('front'))); seg(mid, project(anchor(1)));
  }

  // ---- Sizing: the canvas fills the space under the heading; the world keeps the bird's size tied to the stage ----
  function resize() {
    const rect = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);   // capped: 3x canvases are costly
    canvas.width = Math.max(1, Math.round(rect.width * dpr)); canvas.height = Math.max(1, Math.round(rect.height * dpr));
    if (rect.width < 2 || rect.height < 2) return;
    const oldW = worldW, oldH = worldH;
    scale = Math.min(rect.height, rect.width / CONFIG.minAspect) * CONFIG.size;   // CSS pixels per bird unit
    worldW = rect.width / scale; worldH = rect.height / scale;
    pivot = [worldW / 2, worldH / 2, 0];
    if (!placed) {   // first time: hanging at rest, centred on the stage
      rig();
      shift([worldW / 2, worldH / 2, 0]);
      bar.x = worldW / 2 + restBar.x; bar.y = worldH / 2 + restBar.y;
      // Centre what is seen at rest (tail tip to beak tip, the feet and both bars) on the stage. The camera shortens forward
      // distances on screen by cos(turn) and heights by cos(elevation), so a world shift of d moves the picture by that much less.
      const seen = [world(torso, v3(B.tail)), world(head, sub(v3(B.beak), v3(B.head))), ...legs.flatMap(l => [world(l.shin, l.foot), world(l.toes, l.toeTip)]),
        ...['front', 'back', -1, 1].map(a => anchor(a))].map(project);
      const mid = k => (Math.min(...seen.map(p => p[k])) + Math.max(...seen.map(p => p[k]))) / 2;
      shift([(worldW / 2 - mid('x')) / Math.cos(T), (worldH / 2 - mid('y')) / Math.cos(E), 0]);
      feetY = Math.max(...legs.map(l => world(l.shin, l.foot)[1])) + 0.07;
      placed = true;
    } else { shift([(worldW - oldW) / 2, (worldH - oldH) / 2, 0]); feetY += (worldH - oldH) / 2; }   // stay centred
    if (!slide.hidden) { draw(); wake(); }   // resizing clears the canvas, so repaint even if asleep
  }
  new ResizeObserver(resize).observe(canvas);

  // ---- Loop: runs while the slide is shown and anything moves (or the autoplay plays) ----
  let running = false, calm = 0, last = 0, carry = 0;
  let lastInput = -Infinity, resumeTimer = 0;   // when the pointer last steered; the timer that wakes the loop to resume autoplay
  const autoplayDue = now => !!CONFIG.autoplay && now - lastInput >= CONFIG.autoplay.idleDelay;
  function wake() {
    calm = 0;
    if (running || !placed || slide.hidden || document.hidden) return;
    running = true; last = performance.now(); carry = 0; requestAnimationFrame(frame);
  }
  function frame(now) {
    if (slide.hidden || document.hidden || canvas.width < 2) { running = false; return; }
    const h = 1 / CONFIG.stepRate;
    carry = Math.min(carry + (now - last) / 1000, CONFIG.maxFrameTime); last = now;
    const playing = autoplayDue(now);
    if (playing && !auto) { held = { ...target }; autoBlend = 0; }   // (re)starting: ease in from where the bars are headed
    auto = playing;
    const before = bodies.map(b => [b.x, b.q.slice()]);
    let steps = 0;
    while (carry >= h) { step(h); carry -= h; steps++; }
    draw();
    // How fast anything actually moved this frame (net movement, plus turning times size): the solver's own velocities never
    // quite reach zero while the strings hold the bird up against its weight.
    let fastest = 0;
    if (steps) bodies.forEach((b, i) => { const [x, q] = before[i], dq = qmul(b.q, [q[0], -q[1], -q[2], -q[3]]);
      fastest = Math.max(fastest, (len(sub(b.x, x)) + 2 * Math.acos(Math.min(1, Math.abs(dq[0]))) * b.size) / (steps * h)); });
    const still = Object.keys(target).every(k => Math.abs(target[k] - bar[k]) < 0.001);
    calm = fastest < CONFIG.sleepSpeed && still ? calm + 1 : 0;
    if (calm >= CONFIG.settleFrames && !auto) { running = false; return; }
    requestAnimationFrame(frame);
  }

  // ---- Input ----
  // Pointer: the button picks the bar (up: leg bar, held down: main bar); the offset from the middle of the stage turns it.
  function aim(event) {
    if (!placed) return;
    const rect = canvas.getBoundingClientRect();
    const dx = clamp(((event.clientX - rect.left) / scale - worldW / 2) / CONFIG.reach[0], -1, 1);
    const dy = clamp(((event.clientY - rect.top) / scale - worldH / 2) / CONFIG.reach[1], -1, 1);
    if (event.buttons & 1) Object.assign(target, { yaw: 0, roll: 0, mainYaw: CONFIG.maxMainYaw * dx, pitch: CONFIG.maxPitch * dy });
    else Object.assign(target, { yaw: CONFIG.maxYaw * dx, roll: -CONFIG.maxRoll * dy, mainYaw: 0, pitch: 0 });
    lastInput = performance.now(); auto = false;   // the pointer takes over; the autoplay resumes idleDelay after the last input
    clearTimeout(resumeTimer);
    if (CONFIG.autoplay) resumeTimer = setTimeout(wake, CONFIG.autoplay.idleDelay + 50);
    wake();
  }
  canvas.addEventListener('pointermove', aim);
  canvas.addEventListener('pointerdown', event => { event.preventDefault(); canvas.setPointerCapture(event.pointerId); aim(event); });   // keeps the drag while the pointer leaves the stage
  canvas.addEventListener('pointerup', aim);
  canvas.addEventListener('pointercancel', aim);

  new MutationObserver(() => {   // shown: the autoplay plays at once; hidden: nothing pending
    if (slide.hidden) { clearTimeout(resumeTimer); return; }
    lastInput = -Infinity; wake();
  }).observe(slide, { attributes: true, attributeFilter: ['hidden'] });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && !slide.hidden) wake(); });   // paused while the tab is hidden
  window.marionetteDemoState = () => ({   // for console inspection
    running, autoplaying: auto, bar: { ...bar }, target: { ...target }, feetY,
    body: { x: torso.x, q: torso.q }, head: { x: head.x, q: head.q },
    feet: legs.map(l => { const p = world(l.shin, l.foot); return { x: p[0], z: p[2], lift: feetY - 0.07 - p[1], ankle: ankleBend(l) / deg }; })
  });
  if (!slide.hidden) wake();
})();
