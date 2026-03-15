/* ============================================================
   Value Similarity Engine  -  Frontend Logic
   ============================================================ */

document.addEventListener("DOMContentLoaded", loadExamples);

/* ---- Examples ---- */
let _examples = [];

function loadExamples() {
    fetch("/api/examples")
        .then(r => r.json())
        .then(d => {
            _examples = d.examples || [];
            const wrap = document.getElementById("example-chips");
            wrap.innerHTML = "";
            _examples.forEach((ex, i) => {
                const ch = document.createElement("span");
                ch.className = "ex-chip";
                ch.textContent = ex.label;
                ch.title = ex.desc
                    ? `${ex.name_a} vs ${ex.name_b} — ${ex.desc}`
                    : `${ex.name_a} vs ${ex.name_b}`;
                ch.onclick = () => fillAndRun(i);
                wrap.appendChild(ch);
            });
        });
}

function fillAndRun(idx) {
    const ex = _examples[idx];
    if (!ex) return;
    document.getElementById("name-a").value = ex.name_a;
    document.getElementById("name-b").value = ex.name_b;
    document.getElementById("values-a").value = formatValues(ex.values_a);
    document.getElementById("values-b").value = formatValues(ex.values_b);

    // Highlight the active chip
    document.querySelectorAll(".ex-chip").forEach((c, i) => {
        c.classList.toggle("active", i === idx);
    });

    // Auto-run the comparison
    runCompare();
}

function formatValues(arr) {
    return arr.map(v => (v === null || v === undefined ? "" : String(v))).join("\n");
}

/* ---- Parse values from textarea ---- */
function parseValues(text) {
    // Split by newlines, then commas if single-line
    let lines = text.trim().split(/\n/);
    if (lines.length === 1 && lines[0].indexOf(",") !== -1) {
        lines = lines[0].split(",");
    }
    return lines.map(l => {
        const s = l.trim();
        if (s === "" || s.toLowerCase() === "null" || s.toLowerCase() === "none") return null;
        // Try numeric
        const n = Number(s.replace(/,/g, ""));
        if (!isNaN(n) && s !== "") return n;
        return s;
    }).filter(v => v !== undefined);
}

/* ---- Compare ---- */
function runCompare() {
    const nameA = document.getElementById("name-a").value.trim() || "Column A";
    const nameB = document.getElementById("name-b").value.trim() || "Column B";
    const rawA = document.getElementById("values-a").value;
    const rawB = document.getElementById("values-b").value;

    const valsA = parseValues(rawA);
    const valsB = parseValues(rawB);

    if (valsA.length === 0 || valsB.length === 0) {
        alert("Please enter values for both columns.");
        return;
    }

    // UI loading
    const btn = document.getElementById("btn-compare");
    btn.querySelector(".btn-text").style.display = "none";
    btn.querySelector(".btn-spinner").style.display = "inline-block";
    btn.disabled = true;

    fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            values_a: valsA,
            values_b: valsB,
            name_a: nameA,
            name_b: nameB,
        }),
    })
        .then(r => r.json())
        .then(data => {
            btn.querySelector(".btn-text").style.display = "";
            btn.querySelector(".btn-spinner").style.display = "none";
            btn.disabled = false;
            if (data.error) { alert(data.error); return; }
            renderResults(data);
        })
        .catch(err => {
            btn.querySelector(".btn-text").style.display = "";
            btn.querySelector(".btn-spinner").style.display = "none";
            btn.disabled = false;
            alert("Error: " + err);
        });
}

/* ---- Render Results ---- */
function renderResults(data) {
    const container = document.getElementById("results-container");
    container.style.display = "block";
    container.scrollIntoView({ behavior: "smooth", block: "start" });

    // Highlight architecture nodes sequentially
    highlightArch();

    // Score
    setTimeout(() => renderScore(data), 200);
    // Profiles
    setTimeout(() => renderProfiles(data), 500);
    // Signals
    setTimeout(() => renderSignals(data), 800);
}

function highlightArch() {
    const ids = ["an-input", "an-norm", "an-type", "an-prof", "an-sig", "an-score"];
    ids.forEach((id, i) => {
        setTimeout(() => {
            const el = document.getElementById(id);
            el.classList.add("active");
            setTimeout(() => el.classList.remove("active"), 1200);
        }, i * 250);
    });
}

/* ---- Score Section ---- */
function renderScore(data) {
    const sec = document.getElementById("score-section");
    sec.style.animation = "none";
    void sec.offsetWidth;
    sec.style.animation = "";

    const score = data.final_score;
    const pct = Math.round(score * 100);

    // Color class
    const wrap = document.querySelector(".score-circle-wrap");
    const parent = document.querySelector(".score-hero");
    parent.classList.remove("score-high", "score-mid", "score-low");
    if (score >= 0.7) parent.classList.add("score-high");
    else if (score >= 0.4) parent.classList.add("score-mid");
    else parent.classList.add("score-low");

    // Ring animation
    const ring = document.getElementById("ring-fill");
    const circum = 2 * Math.PI * 62; // ~389.56
    ring.style.strokeDashoffset = circum;
    setTimeout(() => {
        ring.style.strokeDashoffset = circum * (1 - score);
    }, 50);

    // Number animation
    const numEl = document.getElementById("score-number");
    animateNumber(numEl, 0, pct, 1200, v => v + "%");

    // Meta
    document.getElementById("meta-type").textContent = data.detected_type;
    const matchEl = document.getElementById("meta-match");
    if (data.reason) {
        matchEl.textContent = data.reason;
        matchEl.className = "meta-value c-low";
    } else if (data.type_match) {
        matchEl.textContent = "Yes";
        matchEl.className = "meta-value c-high";
    } else {
        matchEl.textContent = "No (type mismatch)";
        matchEl.className = "meta-value c-low";
    }
    document.getElementById("meta-runtime").textContent = data.runtime_ms + " ms";
}

/* ---- Profiles ---- */
function renderProfiles(data) {
    renderProfileCard("profile-card-a", data.profile_a);
    renderProfileCard("profile-card-b", data.profile_b);
    const sec = document.getElementById("profiles-section");
    sec.style.animation = "none";
    void sec.offsetWidth;
    sec.style.animation = "";
}

function renderProfileCard(containerId, p) {
    const el = document.getElementById(containerId);
    const typeClass = p.detected_type === "numeric" ? "numeric" : p.detected_type === "categorical" ? "categorical" : "unknown";

    let html = `<h4>${esc(p.name)} <span class="type-badge ${typeClass}">${p.detected_type}</span></h4>`;
    html += `<div class="stat-grid">`;
    html += statRow("Total Rows", p.total_rows);
    html += statRow("Valid Count", p.valid_count);
    html += statRow("Null Fraction", (p.null_frac * 100).toFixed(1) + "%");
    html += statRow("Distinct", p.n_distinct);
    html += statRow("Distinct Ratio", (p.distinct_ratio * 100).toFixed(1) + "%");

    if (p.detected_type === "numeric") {
        html += statRow("Min", fmt(p.min));
        html += statRow("Max", fmt(p.max));
        html += statRow("Q05", fmt(p.q05));
        html += statRow("Median (Q50)", fmt(p.q50));
        html += statRow("Q95", fmt(p.q95));
        html += statRow("Integer Ratio", (p.integer_ratio * 100).toFixed(1) + "%");
        html += statRow("Monotonicity", (p.monotonic * 100).toFixed(1) + "%");
        html += statRow("Outlier Ratio", (p.outlier_ratio * 100).toFixed(1) + "%");
    } else if (p.detected_type === "categorical") {
        html += statRow("Entropy", p.entropy);
        html += statRow("Avg Length", p.avg_len);
        html += statRow("Std Length", p.std_len);
    }
    html += `</div>`;

    // Sample preview
    if (p.sample_preview && p.sample_preview.length > 0) {
        html += `<div class="sample-preview"><h5>Sample Values</h5><div class="sample-tags">`;
        p.sample_preview.forEach(s => {
            html += `<span class="sample-tag" title="${esc(s)}">${esc(s || "null")}</span>`;
        });
        html += `</div></div>`;
    }

    // Top values (categorical)
    if (p.top_values && p.top_values.length > 0) {
        const maxFreq = p.top_values[0].freq;
        html += `<div class="top-values"><h5>Top Values (frequency)</h5>`;
        p.top_values.forEach(tv => {
            const w = maxFreq > 0 ? (tv.freq / maxFreq * 100) : 0;
            html += `<div class="tv-row">
                <span class="tv-label" title="${esc(tv.value)}">${esc(tv.value)}</span>
                <div class="tv-bar-bg"><div class="tv-bar-fill" style="width:0" data-w="${w}%"></div></div>
                <span class="tv-pct">${(tv.freq * 100).toFixed(1)}%</span>
            </div>`;
        });
        html += `</div>`;
    }

    // Histogram (numeric)
    if (p.histogram) {
        const maxC = Math.max(...p.histogram.counts, 1);
        html += `<div class="histogram-container"><h5>Value Distribution</h5><div class="histogram-bars">`;
        p.histogram.counts.forEach(c => {
            const h = Math.max(2, (c / maxC) * 70);
            html += `<div class="hist-bar" style="height:0" data-h="${h}px" title="count: ${c}"></div>`;
        });
        html += `</div></div>`;
    }

    el.innerHTML = html;

    // Animate bars
    setTimeout(() => {
        el.querySelectorAll(".tv-bar-fill").forEach(b => { b.style.width = b.dataset.w; });
        el.querySelectorAll(".hist-bar").forEach(b => { b.style.height = b.dataset.h; });
    }, 100);
}

function statRow(key, val) {
    return `<div class="stat-item"><span class="stat-key">${esc(key)}</span><span class="stat-val">${esc(String(val))}</span></div>`;
}

/* ---- Signals ---- */
function renderSignals(data) {
    const sec = document.getElementById("signals-section");
    sec.style.animation = "none";
    void sec.offsetWidth;
    sec.style.animation = "";

    const table = document.getElementById("signals-table");
    if (!data.signals || data.signals.length === 0) {
        table.innerHTML = `<div style="text-align:center;color:var(--text3);padding:1rem;">No signals computed (${data.reason || 'type mismatch'})</div>`;
        return;
    }

    // Sort by contribution descending
    const sorted = [...data.signals].sort((a, b) => b.contribution - a.contribution);
    const maxContrib = Math.max(...sorted.map(s => s.contribution), 0.001);

    let html = `<div class="signal-row header">
        <span class="sig-label">Signal</span>
        <span class="sig-value">Score</span>
        <span class="sig-weight">Weight</span>
        <span class="sig-label">Contribution</span>
        <span class="sig-contrib">Weighted</span>
    </div>`;

    sorted.forEach((s, i) => {
        const colorClass = s.value >= 0.7 ? "c-high" : s.value >= 0.4 ? "c-mid" : "c-low";
        const barColor = s.value >= 0.7 ? "var(--success)" : s.value >= 0.4 ? "var(--warn)" : "var(--accent)";
        const barW = (s.contribution / maxContrib * 100).toFixed(1);
        html += `<div class="signal-row" style="animation-delay:${i * 60}ms">
            <span class="sig-label">${esc(s.label)}</span>
            <span class="sig-value ${colorClass}">${s.value.toFixed(3)}</span>
            <span class="sig-weight">${(s.weight * 100).toFixed(0)}%</span>
            <div class="sig-bar-bg"><div class="sig-bar-fill" style="width:0;background:${barColor}" data-w="${barW}%"></div></div>
            <span class="sig-contrib ${colorClass}">${s.contribution.toFixed(4)}</span>
        </div>`;
    });

    table.innerHTML = html;

    // Animate bars
    setTimeout(() => {
        table.querySelectorAll(".sig-bar-fill").forEach(b => { b.style.width = b.dataset.w; });
    }, 150);
}

/* ---- Helpers ---- */
function esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
}

function fmt(v) {
    if (v === null || v === undefined) return "-";
    if (typeof v === "number") {
        if (Number.isInteger(v)) return v.toLocaleString();
        return v.toLocaleString(undefined, { maximumFractionDigits: 4 });
    }
    return String(v);
}

function animateNumber(el, from, to, duration, formatter) {
    const start = performance.now();
    function step(ts) {
        const elapsed = ts - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = Math.round(from + (to - from) * eased);
        el.textContent = formatter ? formatter(current) : current;
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}
