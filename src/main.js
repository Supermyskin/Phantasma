import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { HandTracker } from "./core/HandTracker.js";

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const tracker = new HandTracker();

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

let lastVideoTime = -1;
function renderLoop() {
  if (video.currentTime !== lastVideoTime) {
    const now = performance.now();
    const detections = handLandmarker.detectForVideo(video, now);
    lastVideoTime = video.currentTime;
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    if (detections.worldLandmarks) {
      detections.worldLandmarks.forEach((hand, index) => {
        const velocity = tracker.calculateVelocity(index, hand, now);
        if (velocity) {
          const ctx = canvas.getContext('2d');
          ctx.fillStyle = 'white';
          ctx.font = '12px Arial';

          [4, 8, 12, 16, 20].forEach((idx, i) => {
            const pos = detections.landmarks[index][idx];
            ctx.fillText(velocity.fingertips[i].toFixed(2), pos.x * canvas.width + 10, pos.y * canvas.height);
          });

          const wristPos = detections.landmarks[index][0];
          ctx.fillText(`Palm: ${velocity.palm.toFixed(2)}`, wristPos.x * canvas.width + 10, wristPos.y * canvas.height - 10);
        }
      });
    }

    if (detections.landmarks) {
      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20],
        [0, 17]
      ];

      for (const hand of detections.landmarks) {
        ctx.beginPath();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        for (const [start, end] of connections) {
          ctx.moveTo(hand[start].x * canvas.width, hand[start].y * canvas.height);
          ctx.lineTo(hand[end].x * canvas.width, hand[end].y * canvas.height);
        }
        ctx.stroke();

        hand.forEach((point, index) => {
          ctx.beginPath();
          ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
          ctx.fillStyle = [4, 8, 12, 16, 20].includes(index) ? 'red' : 'lime';
          ctx.fill();
        });
      }
    }
  }

  requestAnimationFrame(renderLoop);
}
startCamera();
