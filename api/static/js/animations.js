/* Scroll-reveal, animated counters, and active-nav tracking. */
(function () {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- Reveal-on-scroll ----
  const revealTargets = document.querySelectorAll('.reveal, .reveal-stagger');
  if (revealTargets.length) {
    if (reduceMotion) {
      revealTargets.forEach((el) => el.classList.add('in-view'));
    } else {
      const io = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add('in-view');
              io.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
      );
      revealTargets.forEach((el) => io.observe(el));
    }
  }

  // ---- Animated counters (data-count-target="93.7" data-count-suffix="%") ----
  const counters = document.querySelectorAll('[data-count-target]');
  const animateCounter = (el) => {
    const target = parseFloat(el.getAttribute('data-count-target'));
    const suffix = el.getAttribute('data-count-suffix') || '';
    const decimals = el.getAttribute('data-count-decimals') !== null
      ? parseInt(el.getAttribute('data-count-decimals'), 10)
      : (target % 1 !== 0 ? 1 : 0);
    const duration = 1200;
    const start = performance.now();

    function frame(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const value = target * eased;
      el.textContent = value.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      }) + suffix;
      if (t < 1) requestAnimationFrame(frame);
    }
    if (reduceMotion) {
      el.textContent = target.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix;
    } else {
      requestAnimationFrame(frame);
    }
  };

  if (counters.length) {
    const io2 = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateCounter(entry.target);
            io2.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    counters.forEach((el) => io2.observe(el));
  }

  // ---- Feature-importance bar fill-in ----
  const impBars = document.querySelectorAll('.imp-fill');
  if (impBars.length) {
    const io3 = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const el = entry.target;
            el.style.width = el.getAttribute('data-width') + '%';
            io3.unobserve(el);
          }
        });
      },
      { threshold: 0.3 }
    );
    impBars.forEach((el) => io3.observe(el));
  }

  // ---- Active nav link tracking ----
  const navLinks = Array.from(document.querySelectorAll('.nav-links a[href^="#"]'));
  const sections = navLinks
    .map((a) => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  if (navLinks.length && sections.length) {
    const setActive = (id) => {
      navLinks.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === `#${id}`));
    };
    const io4 = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) setActive(entry.target.id);
        });
      },
      { rootMargin: '-45% 0px -50% 0px', threshold: 0 }
    );
    sections.forEach((s) => io4.observe(s));
  }
})();
