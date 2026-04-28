export class HandTracker {
  constructor() {
    this.prevLandmarks = new Map(); // hand index -> landmarks
    this.prevTimestamp = new Map(); // hand index -> timestamp
  }

  calculateVelocity(handIndex, currentLandmarks, timestamp) {
    const prev = this.prevLandmarks.get(handIndex);
    const prevTime = this.prevTimestamp.get(handIndex);
    
    if (!prev || !prevTime) {
      this.prevLandmarks.set(handIndex, [...currentLandmarks]);
      this.prevTimestamp.set(handIndex, timestamp);
      return null;
    }

    const dt = (timestamp - prevTime) / 1000; // seconds
    const velocities = [4, 8, 12, 16, 20].map(idx => {
      const p1 = currentLandmarks[idx];
      const p2 = prev[idx];
      const dist = Math.sqrt(
        Math.pow(p1.x - p2.x, 2) + 
        Math.pow(p1.y - p2.y, 2) + 
        Math.pow(p1.z - p2.z, 2)
      );
      return dist / dt;
    });

    // Palm center (simplified as wrist)
    const wristVel = Math.sqrt(
      Math.pow(currentLandmarks[0].x - prev[0].x, 2) +
      Math.pow(currentLandmarks[0].y - prev[0].y, 2) +
      Math.pow(currentLandmarks[0].z - prev[0].z, 2)
    ) / dt;

    this.prevLandmarks.set(handIndex, [...currentLandmarks]);
    this.prevTimestamp.set(handIndex, timestamp);

    return { 
      fingertips: velocities, 
      palm: wristVel 
    };
  }
}
