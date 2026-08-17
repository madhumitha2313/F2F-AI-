/* Polls the existing /health and /stats endpoints to keep the live panel current. */
(function () {
  const panel = document.getElementById('live-panel');
  if (!panel) return;

  const dot = document.getElementById('live-dot');
  const statusText = document.getElementById('live-status-text');
  const elTotal = document.getElementById('live-total');
  const elFailures = document.getElementById('live-failures');
  const elRate = document.getElementById('live-rate');
  const elConf = document.getElementById('live-conf');

  function flash(el, text) {
    if (el.textContent === text) return;
    el.textContent = text;
    el.classList.add('flash');
    setTimeout(() => el.classList.remove('flash'), 500);
  }

  async function pollHealth() {
    try {
      const res = await fetch('/health');
      const data = await res.json();
      const up = res.ok && data.status === 'healthy';
      dot.classList.toggle('up', up);
      dot.classList.toggle('down', !up);
      statusText.textContent = up ? `${data.model} model live` : 'Model unavailable';
    } catch {
      dot.classList.remove('up');
      dot.classList.add('down');
      statusText.textContent = 'Connection lost';
    }
  }

  async function pollStats() {
    try {
      const res = await fetch('/stats');
      if (!res.ok) return;
      const data = await res.json();
      flash(elTotal, String(data.total_predictions ?? 0));
      flash(elFailures, String(data.flagged_failures ?? 0));
      flash(elRate, data.failure_rate != null ? `${(data.failure_rate * 100).toFixed(1)}%` : '—');
      flash(elConf, data.average_confidence != null ? `${(data.average_confidence * 100).toFixed(1)}%` : '—');
    } catch {
      /* keep last known values on transient failure */
    }
  }

  pollHealth();
  pollStats();
  setInterval(pollHealth, 15000);
  setInterval(pollStats, 5000);

  // Refresh stats immediately after this tab scores a reaction.
  document.addEventListener('f2f:prediction-logged', pollStats);
})();
