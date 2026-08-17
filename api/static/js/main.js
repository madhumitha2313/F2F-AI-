/* Try-it-live form: calls the real /predict endpoint. */
(function () {
  const form = document.getElementById('predict-form');
  if (!form) return;

  const btn = document.getElementById('predict-btn');
  const resultBox = document.getElementById('predict-result');
  const outcomeEl = document.getElementById('result-outcome');
  const probEl = document.getElementById('result-prob');
  const confEl = document.getElementById('result-conf');
  const warningEl = document.getElementById('result-warning');
  const errorEl = document.getElementById('result-error');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorEl.textContent = '';
    warningEl.textContent = '';
    btn.disabled = true;
    btn.textContent = 'Scoring…';

    const payload = {
      ligand: form.ligand.value,
      base: form.base.value,
      additive: form.additive.value,
      aryl_halide: form.aryl_halide.value,
    };

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || data.error || 'Prediction failed');
      }

      resultBox.classList.remove('is-empty');
      outcomeEl.textContent = data.prediction;
      outcomeEl.parentElement.classList.remove('outcome-success', 'outcome-failure');
      outcomeEl.parentElement.classList.add(
        data.prediction === 'Success' ? 'outcome-success' : 'outcome-failure'
      );
      probEl.textContent = `${(data.failure_probability * 100).toFixed(1)}%`;
      confEl.textContent = `${(data.confidence * 100).toFixed(1)}%`;

      if (data.warnings && data.warnings.length) {
        warningEl.textContent = data.warnings.join(' ');
      }
    } catch (err) {
      errorEl.textContent = err.message || 'Something went wrong while scoring this reaction.';
    } finally {
      btn.disabled = false;
      btn.innerHTML = 'Predict Failure Risk &nearr;';
    }
  });
})();
