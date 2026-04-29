import { HandLandmarker, FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";
import { HandTracker } from "./core/HandTracker.js";

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const tracker = new HandTracker();
const showLines = document.getElementById('showLines');
const showDots = document.getElementById('showDots');
const faceRecognitionToggle = document.getElementById('faceRecognition');
const warpModeButton = document.getElementById('warpMode');
const cubeModeButton = document.getElementById('cubeMode');
const facePixelateModeButton = document.getElementById('facePixelateMode');
const offscreenCanvas = document.createElement('canvas');
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
const pixelationCanvas = document.createElement('canvas');
const pixelationCtx = pixelationCanvas.getContext('2d');

const modeState = {
  selected: null
};

async function initializeMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
  const handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "public/hand_landmarker.task" },
    numHands: 2,
    runningMode: "video"
  });

  let faceDetector = null;
  try {
    faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: "public/face_recognition.task" },
      runningMode: "video",
      minDetectionConfidence: 0.55
    });
  } catch (error) {
    console.warn("Face detector failed to initialize:", error);
  }

  return { handLandmarker, faceDetector };
}

const { handLandmarker, faceDetector } = await initializeMediaPipe();

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

function lerp(start, end, amount) {
  return start + (end - start) * amount;
}

function handCenter(hand, width, height) {
  const ids = [0, 5, 9, 13, 17];
  const total = ids.reduce((sum, id) => {
    sum.x += hand[id].x * width;
    sum.y += hand[id].y * height;
    return sum;
  }, { x: 0, y: 0 });
  return { x: total.x / ids.length, y: total.y / ids.length };
}

function averagePoint(points) {
  const total = points.reduce((sum, point) => addPoint(sum, point), { x: 0, y: 0 });
  return {
    x: total.x / points.length,
    y: total.y / points.length
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sortHands(leftHand, rightHand, width, height) {
  const centerA = handCenter(leftHand, width, height);
  const centerB = handCenter(rightHand, width, height);
  return centerA.x <= centerB.x
    ? { left: leftHand, right: rightHand, leftCenter: centerA, rightCenter: centerB }
    : { left: rightHand, right: leftHand, leftCenter: centerB, rightCenter: centerA };
}

function createBridgeRegion(handA, handB, width, height) {
  const { left, right, leftCenter, rightCenter } = sortHands(handA, handB, width, height);
  const leftPalm = leftCenter;
  const rightPalm = rightCenter;
  const leftMiddle = toCanvas(left[12], width, height);
  const rightMiddle = toCanvas(right[12], width, height);
  const polygon = [leftPalm, rightPalm, rightMiddle, leftMiddle];
  return { polygon, leftCenter, rightCenter };
}

function scalePoint(origin, target, factor) {
  return {
    x: origin.x + (target.x - origin.x) * factor,
    y: origin.y + (target.y - origin.y) * factor
  };
}

function addPoint(a, b) {
  return { x: a.x + b.x, y: a.y + b.y };
}

function subtractPoint(a, b) {
  return { x: a.x - b.x, y: a.y - b.y };
}

function measureHandClosedness(hand, width, height) {
  const palm = averagePoint([0, 5, 9, 13, 17].map(id => toCanvas(hand[id], width, height)));
  const palmSpan = Math.max(
    dist(toCanvas(hand[5], width, height), toCanvas(hand[17], width, height)),
    24
  );
  const tipDistances = TIP_IDS.map(id => dist(toCanvas(hand[id], width, height), palm));
  const normalizedAverage = tipDistances.reduce((sum, value) => sum + value, 0) / (tipDistances.length * palmSpan);
  const openness = clamp((normalizedAverage - 0.55) / 0.75, 0, 1);
  return 1 - openness;
}

function orderPointsClockwise(points) {
  const center = points.reduce((sum, point) => addPoint(sum, point), { x: 0, y: 0 });
  center.x /= points.length;
  center.y /= points.length;
  return [...points].sort((a, b) => {
    const angleA = Math.atan2(a.y - center.y, a.x - center.x);
    const angleB = Math.atan2(b.y - center.y, b.x - center.x);
    return angleA - angleB;
  });
}

function createHandFace(hand, width, height) {
  const palm = handCenter(hand, width, height);
  const thumb = toCanvas(hand[4], width, height);
  const middle = toCanvas(hand[12], width, height);
  const pinky = toCanvas(hand[20], width, height);
  const thumbOuter = scalePoint(palm, thumb, 1.08);
  const middleOuter = scalePoint(palm, middle, 1.06);
  const pinkyOuter = scalePoint(palm, pinky, 1.08);
  const palmOuter = scalePoint(middleOuter, palm, 1.22);
  return [palmOuter, thumbOuter, middleOuter, pinkyOuter];
}

function createMassiveCubeGeometry(handA, handB, width, height) {
  const { left, right } = sortHands(handA, handB, width, height);
  const leftFace = createHandFace(left, width, height);
  const rightFace = createHandFace(right, width, height);
  return { leftFace, rightFace };
}

function polygonBounds(points, width, height, padding = 24) {
  const xs = points.map(point => point.x);
  const ys = points.map(point => point.y);
  const minX = clamp(Math.floor(Math.min(...xs) - padding), 0, width - 1);
  const minY = clamp(Math.floor(Math.min(...ys) - padding), 0, height - 1);
  const maxX = clamp(Math.ceil(Math.max(...xs) + padding), 0, width);
  const maxY = clamp(Math.ceil(Math.max(...ys) + padding), 0, height);
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / ((yj - yi) || 0.000001) + xi);
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}

function sampleBilinear(data, width, height, x, y) {
  const px = clamp(x, 0, width - 1);
  const py = clamp(y, 0, height - 1);
  const x0 = Math.floor(px);
  const y0 = Math.floor(py);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const tx = px - x0;
  const ty = py - y0;
  const i00 = (y0 * width + x0) * 4;
  const i10 = (y0 * width + x1) * 4;
  const i01 = (y1 * width + x0) * 4;
  const i11 = (y1 * width + x1) * 4;
  const color = [0, 0, 0, 0];

  for (let channel = 0; channel < 4; channel++) {
    const top = data[i00 + channel] * (1 - tx) + data[i10 + channel] * tx;
    const bottom = data[i01 + channel] * (1 - tx) + data[i11 + channel] * tx;
    color[channel] = top * (1 - ty) + bottom * ty;
  }

  return color;
}

function applyBridgeWarp(ctx, width, height, polygon, leftCenter, rightCenter, strength, timestamp) {
  const bounds = polygonBounds(polygon, width, height, 32);
  if (bounds.width < 2 || bounds.height < 2) {
    return;
  }

  const source = ctx.getImageData(bounds.minX, bounds.minY, bounds.width, bounds.height);
  const output = new ImageData(new Uint8ClampedArray(source.data), bounds.width, bounds.height);
  const mid = midpoint(leftCenter, rightCenter);
  const axis = {
    x: rightCenter.x - leftCenter.x,
    y: rightCenter.y - leftCenter.y
  };
  const axisLength = Math.max(Math.hypot(axis.x, axis.y), 1);
  const tangent = { x: axis.x / axisLength, y: axis.y / axisLength };
  const normal = { x: -tangent.y, y: tangent.x };
  const maxOffset = 18 + axisLength * 0.06;
  const oscillation = timestamp * 0.012;

  for (let y = 0; y < bounds.height; y++) {
    for (let x = 0; x < bounds.width; x++) {
      const worldX = bounds.minX + x;
      const worldY = bounds.minY + y;
      if (!pointInPolygon(worldX, worldY, polygon)) {
        continue;
      }

      const relX = worldX - mid.x;
      const relY = worldY - mid.y;
      const along = (relX * tangent.x + relY * tangent.y) / (axisLength * 0.5);
      const across = (relX * normal.x + relY * normal.y) / (axisLength * 0.28 + 24);
      const envelope = Math.max(0, 1 - Math.abs(along)) * Math.max(0, 1 - Math.abs(across));
      const wave = Math.sin(along * 7.5 - oscillation) * Math.cos(across * 4.5 + oscillation * 0.7);
      const pinch = (1 - Math.abs(along)) * strength;
      const offsetAlong = -along * maxOffset * 0.12 * strength;
      const offsetAcross = wave * maxOffset * envelope * strength + across * maxOffset * 0.18 * pinch;
      const sampleX = x + offsetAlong * tangent.x + offsetAcross * normal.x;
      const sampleY = y + offsetAlong * tangent.y + offsetAcross * normal.y;
      const color = sampleBilinear(source.data, bounds.width, bounds.height, sampleX, sampleY);
      const index = (y * bounds.width + x) * 4;
      output.data[index] = color[0];
      output.data[index + 1] = color[1];
      output.data[index + 2] = color[2];
      output.data[index + 3] = color[3];
    }
  }

  ctx.putImageData(output, bounds.minX, bounds.minY);
}

function drawPolygon(ctx, points, fillStyle, strokeStyle, lineWidth = 1) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.closePath();
  if (fillStyle) {
    ctx.fillStyle = fillStyle;
    ctx.fill();
  }
  if (strokeStyle) {
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }
}

function drawGlassCube(ctx, handA, handB, width, height) {
  const { leftFace, rightFace } = createMassiveCubeGeometry(handA, handB, width, height);
  const sideFaces = [
    [leftFace[0], leftFace[1], rightFace[1], rightFace[0]],
    [leftFace[1], leftFace[2], rightFace[2], rightFace[1]],
    [leftFace[2], leftFace[3], rightFace[3], rightFace[2]],
    [leftFace[3], leftFace[0], rightFace[0], rightFace[3]]
  ];

  ctx.save();
  drawPolygon(ctx, rightFace, 'rgba(170, 238, 255, 0.08)', 'rgba(190, 245, 255, 0.24)', 1.2);
  sideFaces.forEach((face, index) => {
    const alpha = 0.06 + index * 0.015;
    drawPolygon(ctx, face, `rgba(190, 245, 255, ${alpha})`, 'rgba(220, 250, 255, 0.16)', 1);
  });
  drawPolygon(ctx, leftFace, 'rgba(210, 248, 255, 0.14)', 'rgba(245, 253, 255, 0.52)', 1.4);

  ctx.beginPath();
  for (let i = 0; i < leftFace.length; i++) {
    ctx.moveTo(leftFace[i].x, leftFace[i].y);
    ctx.lineTo(rightFace[i].x, rightFace[i].y);
  }
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
  ctx.lineWidth = 1;
  ctx.stroke();

  const glow = ctx.createLinearGradient(leftFace[0].x, leftFace[0].y, rightFace[2].x, rightFace[2].y);
  glow.addColorStop(0, 'rgba(255, 255, 255, 0.24)');
  glow.addColorStop(0.45, 'rgba(255, 255, 255, 0.04)');
  glow.addColorStop(1, 'rgba(160, 236, 255, 0.18)');
  drawPolygon(ctx, [leftFace[0], leftFace[1], rightFace[2], rightFace[3]], glow, null);
  ctx.restore();
}

function drawMirroredText(ctx, text, x, y) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(-1, 1);
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function getExpandedFaceBounds(boundingBox, width, height) {
  if (!boundingBox) {
    return null;
  }

  const expandX = boundingBox.width * 0.18;
  const expandY = boundingBox.height * 0.28;
  const minX = clamp(Math.floor(boundingBox.originX - expandX), 0, width - 1);
  const minY = clamp(Math.floor(boundingBox.originY - expandY), 0, height - 1);
  const maxX = clamp(Math.ceil(boundingBox.originX + boundingBox.width + expandX), 1, width);
  const maxY = clamp(Math.ceil(boundingBox.originY + boundingBox.height + expandY), 1, height);

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY
  };
}

function applyPixelationToFace(ctx, sourceCanvas, faceBounds, amount) {
  if (!faceBounds || amount <= 0.01 || faceBounds.width < 2 || faceBounds.height < 2) {
    return;
  }

  const cellSize = Math.round(lerp(1, 26, amount));
  const scaledWidth = Math.max(1, Math.round(faceBounds.width / cellSize));
  const scaledHeight = Math.max(1, Math.round(faceBounds.height / cellSize));

  pixelationCanvas.width = scaledWidth;
  pixelationCanvas.height = scaledHeight;
  pixelationCtx.clearRect(0, 0, scaledWidth, scaledHeight);
  pixelationCtx.drawImage(
    sourceCanvas,
    faceBounds.x,
    faceBounds.y,
    faceBounds.width,
    faceBounds.height,
    0,
    0,
    scaledWidth,
    scaledHeight
  );

  ctx.save();
  ctx.globalAlpha = amount;
  ctx.imageSmoothingEnabled = false;
  ctx.beginPath();
  ctx.ellipse(
    faceBounds.x + faceBounds.width / 2,
    faceBounds.y + faceBounds.height / 2,
    faceBounds.width / 2,
    faceBounds.height / 2,
    0,
    0,
    Math.PI * 2
  );
  ctx.clip();
  ctx.drawImage(pixelationCanvas, 0, 0, scaledWidth, scaledHeight, faceBounds.x, faceBounds.y, faceBounds.width, faceBounds.height);
  ctx.restore();
}

let lastVideoTime = -1;

function setSelectedMode(mode) {
  modeState.selected = modeState.selected === mode ? null : mode;
  warpModeButton.classList.toggle('active', modeState.selected === 'warp');
  cubeModeButton.classList.toggle('active', modeState.selected === 'cube');
  facePixelateModeButton.classList.toggle('active', modeState.selected === 'facePixelate');
  warpModeButton.setAttribute('aria-pressed', String(modeState.selected === 'warp'));
  cubeModeButton.setAttribute('aria-pressed', String(modeState.selected === 'cube'));
  facePixelateModeButton.setAttribute('aria-pressed', String(modeState.selected === 'facePixelate'));
}

warpModeButton.addEventListener('click', () => {
  setSelectedMode('warp');
});

cubeModeButton.addEventListener('click', () => {
  setSelectedMode('cube');
});

facePixelateModeButton.addEventListener('click', () => {
  setSelectedMode('facePixelate');
});

function renderLoop() {
  if (video.currentTime !== lastVideoTime) {
    const now = performance.now();
    const detections = handLandmarker.detectForVideo(video, now);
    lastVideoTime = video.currentTime;

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const W = canvas.width, H = canvas.height;
    offscreenCanvas.width = W;
    offscreenCanvas.height = H;

    offscreenCtx.drawImage(video, 0, 0, W, H);
    ctx.drawImage(offscreenCanvas, 0, 0, W, H);

    const faceRecognitionEnabled = faceRecognitionToggle.checked && faceDetector;
    const faceDetections = faceRecognitionEnabled
      ? faceDetector.detectForVideo(video, now).detections ?? []
      : [];
    const handClosedness = detections.landmarks?.length
      ? Math.max(...detections.landmarks.map(hand => measureHandClosedness(hand, W, H)))
      : 0;

    if (modeState.selected === 'facePixelate' && faceDetections.length > 0) {
      faceDetections.forEach((face) => {
        const faceBounds = getExpandedFaceBounds(face.boundingBox, W, H);
        applyPixelationToFace(ctx, offscreenCanvas, faceBounds, handClosedness);
      });
    }

    if (modeState.selected === 'warp' && detections.landmarks?.length === 2) {
      const { polygon, leftCenter, rightCenter } = createBridgeRegion(detections.landmarks[0], detections.landmarks[1], W, H);
      const bridgeSize = dist(leftCenter, rightCenter);
      const breathing = 0.72 + Math.sin(now * 0.008) * 0.14;
      const strength = clamp(0.55 + 140 / Math.max(bridgeSize, 120), 0.45, 1.2) * breathing;
      applyBridgeWarp(ctx, W, H, polygon, leftCenter, rightCenter, strength, now);
    }

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
      if (modeState.selected === 'cube' && detections.landmarks.length === 2) {
        drawGlassCube(ctx, detections.landmarks[0], detections.landmarks[1], W, H);
      }

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

      if (modeState.selected === 'warp' && detections.landmarks.length === 2) {
        const { polygon } = createBridgeRegion(detections.landmarks[0], detections.landmarks[1], W, H);
        const [L1, R1, R2, L2] = polygon;

        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 150, 40, 0.95)';
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

      if (modeState.selected === 'facePixelate') {
        ctx.fillStyle = 'rgba(255, 211, 90, 0.95)';
        ctx.font = '13px Arial';
        drawMirroredText(ctx, `Hand closed: ${(handClosedness * 100).toFixed(0)}%`, 18, 28);
      }
    }

    if (faceRecognitionToggle.checked && !faceDetector) {
      ctx.fillStyle = 'rgba(255, 120, 120, 0.95)';
      ctx.font = '13px Arial';
      drawMirroredText(ctx, 'Face model unavailable', 18, H - 16);
    }
  }

  requestAnimationFrame(renderLoop);
}

startCamera();
