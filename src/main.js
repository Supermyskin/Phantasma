const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });
    video.srcObject = stream;
    console.log("Camera started successfully");
  } catch (error) {
    console.error("Error accessing webcam: ", error);
  }
}

startCamera();
