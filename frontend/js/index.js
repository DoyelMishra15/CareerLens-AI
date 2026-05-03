// CareerLens — Landing Page JS
// Subtle animations and interactions

document.addEventListener('DOMContentLoaded', () => {
  // Animate score ring on landing
  const arc = document.querySelector('.ring-svg circle:last-child');
  if (arc) {
    arc.style.strokeDashoffset = '314';
    setTimeout(() => {
      arc.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(.4,0,.2,1)';
      arc.style.strokeDashoffset = '94'; // 70%
    }, 400);
  }

  // Animate score number
  const num = document.querySelector('.ring-num');
  if (num) {
    animateNumber(num, 0, 70, 1500);
  }

  // Stagger pill animations
  document.querySelectorAll('.pill').forEach((pill, i) => {
    pill.style.opacity = '0';
    pill.style.transform = 'translateY(10px)';
    setTimeout(() => {
      pill.style.transition = 'all .3s ease-out';
      pill.style.opacity = '1';
      pill.style.transform = 'translateY(0)';
    }, 800 + i * 100);
  });
});

function animateNumber(el, from, to, duration) {
  const start = performance.now();
  function update(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}