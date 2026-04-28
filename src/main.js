import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { HandTracker } from "./core/HandTracker.js";

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const tracker = new HandTracker();
const showLines = document.getElementById('showLines');
const showDots = document.getElementById('showDots');

async function initializeMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
  const handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/public/hand_landmarker.task" },
    numHands: 2,
    runningMode: "video"
  });
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
  } catch (error) {
    console.error("Error accessing webcam: ", error);
  }
}

const FINGER_CHAINS = [
  [4, 3, 2, 1],
  [8, 7, 6, 5],
  [12, 11, 10, 9],
  [16, 15, 14, 13],
  [20, 19, 18, 17]
];

const FINGER_COLOR_LIST = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FFFF33'];
const TIP_IDS = [4, 8, 12, 16, 20];

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

function drawMirroredText(ctx, text, x, y) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(-1, 1);
  ctx.fillText(text, 0, 0);
  ctx.restore();
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

    if (detections.worldLandmarks) {
      detections.worldLandmarks.forEach((hand, handIndex) => {
        const velocity = tracker.calculateVelocity(handIndex, hand, now);
        if (velocity) {
          ctx.fillStyle = 'white';
          ctx.font = '12px Arial';
          TIP_IDS.forEach((idx, i) => {
            const pos = detections.landmarks[handIndex][idx];
            drawMirroredText(ctx, velocity.fingertips[i].toFixed(2), pos.x * W + 10, pos.y * H);
          });
        }
      });
    }

    if (detections.landmarks) {
      detections.landmarks.forEach((hand, handIndex) => {
        if (showLines.checked) {
          FINGER_CHAINS.forEach((chain, fi) => {
            const color = FINGER_COLOR_LIST[fi];
            for (let j = 0; j < chain.length - 1; j++) {
              drawLine(ctx, toCanvas(hand[chain[j]], W, H), toCanvas(hand[chain[j + 1]], W, H), color, 2);
            }
            drawLine(ctx, toCanvas(hand[chain[chain.length - 1]], W, H), toCanvas(hand[0], W, H), color, 1);
          });

          ctx.beginPath();
          ctx.strokeStyle = 'cyan';
          ctx.lineWidth = 2;
          const knuckles = [5, 9, 13, 17].map(id => toCanvas(hand[id], W, H));
          ctx.moveTo(knuckles[0].x, knuckles[0].y);
          knuckles.slice(1).forEach(k => ctx.lineTo(k.x, k.y));
          ctx.stroke();

          const tips = TIP_IDS.map(id => toCanvas(hand[id], W, H));
          for (let i = 0; i < tips.length - 1; i++) {
            const a = tips[i], b = tips[i + 1];
            drawLine(ctx, a, b, 'rgba(255,255,255,0.4)', 1);
            const mid = midpoint(a, b);
            ctx.fillStyle = 'white';
            ctx.font = '11px Arial';
            drawMirroredText(ctx, `${dist(a, b).toFixed(0)}px`, mid.x + 4, mid.y - 4);
          }
        }

        if (showDots.checked) {
          hand.forEach(point => drawPoint(ctx, toCanvas(point, W, H), 'white', 3));
          const tips = TIP_IDS.map(id => toCanvas(hand[id], W, H));
          tips.forEach((tip, i) => drawPoint(ctx, tip, FINGER_COLOR_LIST[i], 6));
        }
      });

      if (detections.landmarks.length === 2) {
        const handA = detections.landmarks[0];
        const handB = detections.landmarks[1];

        const L1 = toCanvas(handA[4], W, H);
        const L2 = toCanvas(handA[12], W, H);
        const R1 = toCanvas(handB[4], W, H);
        const R2 = toCanvas(handB[12], W, H);

        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 200, 0, 0.85)';
        ctx.lineWidth = 2;
        ctx.moveTo(L1.x, L1.y);
        ctx.lineTo(R1.x, R1.y);
        ctx.lineTo(R2.x, R2.y);
        ctx.lineTo(L2.x, L2.y);
        ctx.closePath();
        ctx.stroke();

        const sides = [
          { a: L1, b: R1 },
          { a: R1, b: R2 },
          { a: R2, b: L2 },
          { a: L2, b: L1 },
        ];
        ctx.fillStyle = 'rgba(255, 200, 0, 1)';
        ctx.font = '12px Arial';
        sides.forEach(({ a, b }) => {
          const mid = midpoint(a, b);
          drawMirroredText(ctx, `${dist(a, b).toFixed(0)}px`, mid.x + 4, mid.y - 4);
        });
      }
    }
  }

  requestAnimationFrame(renderLoop);
}

startCamera();