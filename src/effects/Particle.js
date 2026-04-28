export class Particle {
  constructor(x, y, color = null) {
    this.x = x;
    this.y = y;
    this.vx = (Math.random() - 0.5) * 4;
    this.vy = (Math.random() - 0.5) * 4;
    this.lifespan = 1.0;
    this.decay = 0.02 + Math.random() * 0.03;
    this.color = color || this.getRandomColor();
  }

  getRandomColor() {
    const colors = ['#ff0000', '#1eff00', '#00c8ff', '#ff00b3', '#fff700', '#3216ff'];
    return colors[Math.floor(Math.random() * colors.length)];
  }

  update() {
    this.x += this.vx;
    this.y += this.vy;
    this.lifespan -= this.decay;
  }

  draw(ctx) {
    ctx.globalAlpha = this.lifespan;
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1.0;
  }
}
