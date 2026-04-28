import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { HandTracker } from "./core/HandTracker.js";
import { Particle } from "./effects/Particle.js";

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const tracker = new HandTracker();
const particles = [];

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

    const fingerColors = {
      thumb: '#FF5733',
      index: '#33FF57',
      middle: '#3357FF',
      ring: '#F333FF',
      pinky: '#FFFF33'
    };

    if (detections.worldLandmarks) {
      detections.worldLandmarks.forEach((hand, handIndex) => {
        const velocity = tracker.calculateVelocity(handIndex, hand, now);
        if (velocity) {
          ctx.fillStyle = 'white';
          ctx.font = '12px Arial';
          [4, 8, 12, 16, 20].forEach((idx, i) => {
            const pos = detections.landmarks[handIndex][idx];
            ctx.fillText(velocity.fingertips[i].toFixed(2), pos.x * canvas.width + 10, pos.y * canvas.height);
          });
        }
      });
    }

    if (detections.landmarks) {
      detections.landmarks.forEach((hand) => {
        // Spawn particles at fingertips
        [4, 8, 12, 16, 20].forEach((idx, i) => {
          const color = [fingerColors.thumb, fingerColors.index, fingerColors.middle, fingerColors.ring, fingerColors.pinky][i];
          const point = hand[idx];
          particles.push(new Particle(point.x * canvas.width, point.y * canvas.height, color));
        });

        const fingerGroups = [
          { joints: [0, 1, 2, 3, 4], color: fingerColors.thumb },
          { joints: [0, 5, 6, 7, 8], color: fingerColors.index },
          { joints: [0, 9, 10, 11, 12], color: fingerColors.middle },
          { joints: [0, 13, 14, 15, 16], color: fingerColors.ring },
          { joints: [0, 17, 18, 19, 20], color: fingerColors.pinky }
        ];

        fingerGroups.forEach(group => {
          ctx.beginPath();
          ctx.strokeStyle = group.color;
          ctx.lineWidth = 3;
          for (let i = 0; i < group.joints.length - 1; i++) {
            const start = hand[group.joints[i]];
            const end = hand[group.joints[i + 1]];
            ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
            ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
          }
          ctx.stroke();
        });

        hand.forEach((point) => {
          ctx.beginPath();
          ctx.fillStyle = 'white';
          ctx.arc(point.x * canvas.width, point.y * canvas.height, 3, 0, Math.PI * 2);
          ctx.fill();
        });
      });
    }

    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.update();
      p.draw(ctx);
      if (p.lifespan <= 0) particles.splice(i, 1);
    }
  }

  requestAnimationFrame(renderLoop);
}

startCamera();
