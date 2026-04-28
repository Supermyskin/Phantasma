import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { HandTracker } from "./core/HandTracker.js";
// import { Particle } from "./effects/Particle.js";  // commented out for now

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const tracker = new HandTracker();
// const particles = [];  // commented out for now

async function initializeMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
  const handLandmarker = await HandLandmarker.createFromOptions(
    vision,
    {
      baseOptions: {
        modelAssetPath: "/public/hand_landmarker.task"
      },
      numHands: 2,
      runningMode: "video"
    }
  );
  return handLandmarker;
}

const handLandmarker = await initializeMediaPipe();

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });
    video.srcObject = stream;
    video.onloadeddata = () => renderLoop();
    console.log("Camera started successfully");
  } catch (error) {
    console.error("Error accessing webcam: ", error);
  }
}

// Finger joint chains: [tip, ..., base]
const FINGER_CHAINS = [
  [4, 3, 2, 1],    // thumb
  [8, 7, 6, 5],    // index
  [12, 11, 10, 9], // middle
  [16, 15, 14, 13],// ring
  [20, 19, 18, 17] // pinky
];

const fingerColors = {
  thumb: '#FF5733',
  index: '#33FF57',
  middle: '#3357FF',
  ring: '#F333FF',
  pinky: '#FFFF33'
};
const FINGER_COLOR_LIST = [
  fingerColors.thumb,
  fingerColors.index,
  fingerColors.middle,
  fingerColors.ring,
  fingerColors.pinky
];

function toCanvas(point, w, h) {
  return { x: point.x * w, y: point.y * h };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function drawLine(ctx, a, b, color, lineWidth = 1.5) {
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function drawPoint(ctx, p, color, radius = 3) {
  ctx.beginPath();
  ctx.fillStyle = color;
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

let lastVideoTime = -1;

function renderLoop() {
  if (video.currentTime !== lastVideoTime) {
    const now = performance.now();
    const detections = handLandmarker.detectForVideo(video, now);
    lastVideoTime = video.currentTime;

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const W = canvas.width, H = canvas.height;

    ctx.drawImage(video, 0, 0);

    // --- Velocity labels ---
    if (detections.worldLandmarks) {
      detections.worldLandmarks.forEach((hand, handIndex) => {
        const velocity = tracker.calculateVelocity(handIndex, hand, now);
        if (velocity) {
          ctx.fillStyle = 'white';
          ctx.font = '12px Arial';
          [4, 8, 12, 16, 20].forEach((idx, i) => {
            const pos = detections.landmarks[handIndex][idx];
            ctx.fillText(velocity.fingertips[i].toFixed(2), pos.x * W + 10, pos.y * H);
          });
        }
      });
    }

    if (detections.landmarks) {
      // Collect all fingertips across all hands for cross-hand line
      const allTips = detections.landmarks.map(hand => ({
        thumb: toCanvas(hand[4], W, H),
        pinky: toCanvas(hand[20], W, H),
        index: toCanvas(hand[8], W, H),
      }));

      detections.landmarks.forEach((hand, handIndex) => {
        // --- Draw joint connections (finger chains) ---
        FINGER_CHAINS.forEach((chain, fi) => {
          const color = FINGER_COLOR_LIST[fi];
          for (let j = 0; j < chain.length - 1; j++) {
            const a = toCanvas(hand[chain[j]], W, H);
            const b = toCanvas(hand[chain[j + 1]], W, H);
            drawLine(ctx, a, b, color, 2);
          }
          // Connect base of each finger to wrist (landmark 0)
          const base = toCanvas(hand[chain[chain.length - 1]], W, H);
          const wrist = toCanvas(hand[0], W, H);
          drawLine(ctx, base, wrist, color, 1);
        });

        // --- Connect knuckles (5→9→13→17) ---
        ctx.beginPath();
        ctx.strokeStyle = 'cyan';
        ctx.lineWidth = 2;
        const knuckleIds = [5, 9, 13, 17];
        const knuckles = knuckleIds.map(id => toCanvas(hand[id], W, H));
        ctx.moveTo(knuckles[0].x, knuckles[0].y);
        for (let i = 1; i < knuckles.length; i++) {
          ctx.lineTo(knuckles[i].x, knuckles[i].y);
        }
        ctx.stroke();

        // --- Draw all landmark dots ---
        hand.forEach((point) => {
          drawPoint(ctx, toCanvas(point, W, H), 'white', 3);
        });

        // --- Fingertip distances ---
        const tipIds = [4, 8, 12, 16, 20];
        const tipNames = ['Th', 'Idx', 'Mid', 'Rng', 'Pnk'];
        const tips = tipIds.map(id => toCanvas(hand[id], W, H));

        // Draw distance between each adjacent pair of fingertips
        for (let i = 0; i < tips.length - 1; i++) {
          const a = tips[i], b = tips[i + 1];
          const d = dist(a, b);
          drawLine(ctx, a, b, 'rgba(255,255,255,0.4)', 1);
          const mid = midpoint(a, b);
          ctx.fillStyle = 'white';
          ctx.font = '11px Arial';
          ctx.fillText(`${d.toFixed(0)}px`, mid.x + 4, mid.y - 4);
        }

        // --- Fingertip dots (colored, larger) ---
        tips.forEach((tip, i) => {
          drawPoint(ctx, tip, FINGER_COLOR_LIST[i], 6);
        });

        // --- Particles commented out ---
        // [4, 8, 12, 16, 20].forEach((idx, i) => {
        //   const color = FINGER_COLOR_LIST[i];
        //   const point = hand[idx];
        //   particles.push(new Particle(point.x * W, point.y * H, color));
        // });
      });

      // --- Cross-hand rectangle: connect closest fingertips between two hands ---
      if (allTips.length === 2) {
        const [left, right] = allTips;

        // Find the two closest fingertips between the hands
        const candidates = [
          { a: left.thumb, b: right.thumb, la: 'Th', ra: 'Th' },
          { a: left.thumb, b: right.index, la: 'Th', ra: 'Idx' },
          { a: left.index, b: right.thumb, la: 'Idx', ra: 'Th' },
          { a: left.index, b: right.index, la: 'Idx', ra: 'Idx' },
          { a: left.pinky, b: right.pinky, la: 'Pnk', ra: 'Pnk' },
          { a: left.pinky, b: right.index, la: 'Pnk', ra: 'Idx' },
          { a: left.index, b: right.pinky, la: 'Idx', ra: 'Pnk' },
        ];

        // Sort by distance, pick the closest pair
        candidates.sort((x, y) => dist(x.a, x.b) - dist(y.a, y.b));
        const closest = candidates[0];

        // Draw the connecting line
        drawLine(ctx, closest.a, closest.b, 'rgba(255, 200, 0, 0.9)', 2);

        // Draw rectangle using those two points as opposite corners
        // Find the other two fingertips to form a proper quad
        // Use thumb+pinky from each hand as the two sides
        const L1 = left.thumb, L2 = left.pinky;
        const R1 = right.thumb, R2 = right.pinky;

        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 200, 0, 0.85)';
        ctx.lineWidth = 2;
        ctx.moveTo(L1.x, L1.y);
        ctx.lineTo(R1.x, R1.y);
        ctx.lineTo(R2.x, R2.y);
        ctx.lineTo(L2.x, L2.y);
        ctx.closePath();
        ctx.stroke();

        // Label the rectangle sides with distances
        const sides = [
          { a: L1, b: R1 },
          { a: R1, b: R2 },
          { a: R2, b: L2 },
          { a: L2, b: L1 },
        ];
        ctx.fillStyle = 'rgba(255, 200, 0, 1)';
        ctx.font = '12px Arial';
        sides.forEach(({ a, b }) => {
          const d = dist(a, b);
          const mid = midpoint(a, b);
          ctx.fillText(`${d.toFixed(0)}px`, mid.x + 4, mid.y - 4);
        });
      }
    }

    // --- Particles update/draw commented out ---
    // for (let i = particles.length - 1; i >= 0; i--) {
    //   const p = particles[i];
    //   p.update();
    //   p.draw(ctx);
    //   if (p.lifespan <= 0) particles.splice(i, 1);
    // }
  }

  requestAnimationFrame(renderLoop);
}

startCamera();