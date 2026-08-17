/* Rising-bubble canvas background — evokes a chemical reaction beaker. */
(function () {
  const canvas = document.getElementById('bubble-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, bubbles;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }

  function makeBubble(randomY) {
    const r = 4 + Math.random() * 22;
    return {
      x: Math.random() * w,
      y: randomY ? Math.random() * h : h + r + Math.random() * 200,
      r,
      speed: 0.25 + Math.random() * 0.9,
      drift: (Math.random() - 0.5) * 0.4,
      alpha: 0.05 + Math.random() * 0.22,
      hue: 195 + Math.random() * 25,
    };
  }

  function init() {
    resize();
    const count = Math.max(24, Math.floor((w * h) / 45000));
    bubbles = Array.from({ length: count }, () => makeBubble(true));
  }

  function tick() {
    ctx.clearRect(0, 0, w, h);
    for (const b of bubbles) {
      b.y -= b.speed;
      b.x += Math.sin(b.y * 0.01) * b.drift;

      const gradient = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
      gradient.addColorStop(0, `hsla(${b.hue}, 90%, 75%, ${b.alpha})`);
      gradient.addColorStop(1, `hsla(${b.hue}, 90%, 65%, 0)`);
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
      ctx.fill();

      if (b.y < -b.r - 20) Object.assign(b, makeBubble(false));
    }
    requestAnimationFrame(tick);
  }

  window.addEventListener('resize', () => {
    resize();
  });

  init();
  requestAnimationFrame(tick);
})();
