document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("loan-form");
    const submitBtn = document.getElementById("submit-btn");
    const modelSelect = document.getElementById("model-select");
    const modelBadge = document.getElementById("model-badge");
    const metricsBar = document.getElementById("model-metrics-bar");
    const chipAcc = document.getElementById("chip-acc");
    const chipF1 = document.getElementById("chip-f1");
    const chipAuc = document.getElementById("chip-auc");

    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultContent = document.getElementById("result-content");
    const resultError = document.getElementById("result-error");
    const resetBtn = document.getElementById("reset-btn");

    let modelsData = [];
    let defaultModel = "";

    async function loadModels() {
        try {
            const res = await fetch("/api/models");
            const json = await res.json();
            defaultModel = json.default;
            modelsData = json.models;

            modelSelect.innerHTML = "";
            modelsData.forEach((m) => {
                const opt = document.createElement("option");
                opt.value = m.key;
                opt.textContent = m.display_name + (m.is_default ? "  ★" : "");
                if (m.is_default) opt.selected = true;
                modelSelect.appendChild(opt);
            });

            updateMetricsBar();
        } catch (err) {
            modelSelect.innerHTML = '<option value="">Models unavailable</option>';
        }
    }

    function updateMetricsBar() {
        const key = modelSelect.value;
        const m = modelsData.find((x) => x.key === key);
        if (!m) return;

        metricsBar.classList.remove("hidden");

        chipAcc.textContent = "Acc " + (m.accuracy * 100).toFixed(1) + "%";
        chipF1.textContent = "F1 " + (m.f1 * 100).toFixed(1) + "%";
        chipAuc.textContent = "AUC " + (m.roc_auc * 100).toFixed(1) + "%";

        if (m.is_default) {
            modelBadge.classList.remove("hidden");
        } else {
            modelBadge.classList.add("hidden");
        }
    }

    modelSelect.addEventListener("change", updateMetricsBar);
    loadModels();

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const payload = {};
        for (const [key, value] of formData.entries()) {
            payload[key] = value;
        }
        
        payload.model = modelSelect.value;

        submitBtn.classList.add("loading");
        submitBtn.disabled = true;

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok) {
                showError(data.error || "An unknown error occurred.");
                return;
            }

            showResult(data, payload);
        } catch (err) {
            showError("Could not connect to the server. Is the Flask app running?");
        } finally {
            submitBtn.classList.remove("loading");
            submitBtn.disabled = false;
        }
    });

    resetBtn.addEventListener("click", () => {
        resultContent.classList.add("hidden");
        resultError.classList.add("hidden");
        resultPlaceholder.classList.remove("hidden");

        const gaugeFill = document.getElementById("gauge-fill");
        gaugeFill.style.strokeDasharray = "0 251.33";

        form.reset();

        if (defaultModel) {
            modelSelect.value = defaultModel;
            updateMetricsBar();
        }

        window.scrollTo({ top: 0, behavior: "smooth" });
    });

    function showResult(data, payload) {
        resultPlaceholder.classList.add("hidden");
        resultError.classList.add("hidden");
        resultContent.classList.remove("hidden");

        const isApproved = data.prediction === "Approved";
        const prob = data.probability;

        const statusIcon = document.getElementById("status-icon");
        statusIcon.className = "status-icon " + (isApproved ? "approved" : "rejected");
        statusIcon.innerHTML = isApproved ? "✓" : "✗";

        const statusText = document.getElementById("status-text");
        statusText.textContent = data.prediction;
        statusText.className = "status-text " + (isApproved ? "approved" : "rejected");

        const modelLabel = document.getElementById("model-used-label");
        modelLabel.textContent = `Model: ${data.model_used}`;

        const gaugeFill = document.getElementById("gauge-fill");
        const gaugeValue = document.getElementById("gauge-value");

        const arcLength = 251.33;
        const fillLength = (prob / 100) * arcLength;

        let gaugeColor;
        if (prob >= 60) gaugeColor = "#10b981";
        else if (prob >= 35) gaugeColor = "#f59e0b";
        else gaugeColor = "#ef4444";

        gaugeFill.style.strokeDasharray = "0 251.33";
        setTimeout(() => {
            gaugeFill.style.stroke = gaugeColor;
            gaugeFill.style.strokeDasharray = `${fillLength} ${arcLength}`;
        }, 100);

        animateCounter(gaugeValue, 0, prob, 1000);

        const details = document.getElementById("result-details");
        const loanPercent =
            payload.person_income > 0
                ? ((payload.loan_amnt / payload.person_income) * 100).toFixed(1)
                : "N/A";

        details.innerHTML = `
            <div class="detail-row">
                <span class="label">Loan Amount</span>
                <span class="value">$${Number(payload.loan_amnt).toLocaleString()}</span>
            </div>
            <div class="detail-row">
                <span class="label">Annual Income</span>
                <span class="value">$${Number(payload.person_income).toLocaleString()}</span>
            </div>
            <div class="detail-row">
                <span class="label">Loan-to-Income Ratio</span>
                <span class="value">${loanPercent}%</span>
            </div>
            <div class="detail-row">
                <span class="label">Credit Score</span>
                <span class="value">${payload.credit_score}</span>
            </div>
            <div class="detail-row">
                <span class="label">Interest Rate</span>
                <span class="value">${payload.loan_int_rate}%</span>
            </div>
            <div class="detail-row">
                <span class="label">Model ROC-AUC</span>
                <span class="value">${data.model_roc_auc ? (data.model_roc_auc * 100).toFixed(1) + '%' : '-'}</span>
            </div>
        `;

        if (window.innerWidth <= 900) {
            document.getElementById("result-section").scrollIntoView({ behavior: "smooth" });
        }
    }

    function showError(msg) {
        resultPlaceholder.classList.add("hidden");
        resultContent.classList.add("hidden");
        resultError.classList.remove("hidden");
        document.getElementById("error-message").textContent = msg;
    }

    function animateCounter(element, start, end, duration) {
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const value = start + (end - start) * eased;
            element.textContent = value.toFixed(1) + "%";
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }
});
