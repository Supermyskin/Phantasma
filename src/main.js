import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');

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
    const detections = handLandmarker.detectForVideo(video, performance.now());
    lastVideoTime = video.currentTime;
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    if (detections.landmarks) {
      for (const hand of detections.landmarks) {
        for (const point of hand) {
          ctx.beginPath();
          ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
          ctx.fillStyle = 'lime';
          ctx.fill();
        }
      }
    }
  }

  requestAnimationFrame(renderLoop);
}
startCamera();
