/**
 * AI Text Detector — Web UI (served at /app/, calls FastAPI /api/v1/analyze)
 */
(function () {
  const STORAGE_KEY = "aidet_api_key";

  const el = {
    apiKey: document.getElementById("api-key"),
    rememberKey: document.getElementById("remember-key"),
    text: document.getElementById("text"),
    includeSentences: document.getElementById("include-sentences"),
    includeMetadata: document.getElementById("include-metadata"),
    btn: document.getElementById("analyze-btn"),
    status: document.getElementById("status"),
    err: document.getElementById("error"),
    result: document.getElementById("result"),
    verdict: document.getElementById("verdict"),
    prob: document.getElementById("prob"),
    confidence: document.getElementById("confidence"),
    components: document.getElementById("components"),
    sentences: document.getElementById("sentence-list"),
  };

  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) el.apiKey.value = saved;

  el.btn.addEventListener("click", analyze);

  async function analyze() {
    el.err.textContent = "";
    el.result.classList.add("hidden");
    const key = el.apiKey.value.trim();
    if (!key) {
      el.err.textContent = "Enter your API key (same as server / .api_keys.json or API_KEY_1 in .env).";
      return;
    }
    if (el.rememberKey.checked) {
      localStorage.setItem(STORAGE_KEY, key);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }

    const text = el.text.value;
    if (!text.trim()) {
      el.err.textContent = "Paste some text to analyze.";
      return;
    }

    el.btn.disabled = true;
    el.status.textContent = "Analyzing…";

    try {
      const res = await fetch("/api/v1/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": key,
        },
        body: JSON.stringify({
          text,
          include_sentences: el.includeSentences.checked,
          include_metadata: el.includeMetadata.checked,
        }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || data.error || res.statusText || "Request failed");
      }

      renderResult(data);
      el.status.textContent = "";
    } catch (e) {
      el.status.textContent = "";
      el.err.textContent = String(e.message || e);
    } finally {
      el.btn.disabled = false;
    }
  }

  function renderResult(data) {
    const v = data.verdict || "Unknown";
    const prob = data.ai_probability ?? 0;

    el.verdict.className = "verdict";
    if (v === "AI") el.verdict.classList.add("ai");
    else if (v === "Human") el.verdict.classList.add("human");
    else if (v === "Mixed") el.verdict.classList.add("mixed");
    else el.verdict.classList.add("unknown");

    el.verdict.textContent = "Verdict: " + v;
    el.prob.textContent = (prob * 100).toFixed(1) + "% AI probability";
    el.confidence.textContent =
      (data.confidence || "—") + " — " + (data.confidence_category || "");

    const comp = data.components || {};
    if (comp.perplexity || comp.burstiness || comp.classifier) {
      el.components.innerHTML = "";
      const dl = document.createElement("dl");
      dl.className = "components";
      if (comp.perplexity) {
        dl.appendChild(dt("Perplexity P(AI)"));
        dl.appendChild(dd(fmtNum(comp.perplexity.ai_probability)));
      }
      if (comp.burstiness) {
        dl.appendChild(dt("Burstiness P(AI)"));
        dl.appendChild(dd(fmtNum(comp.burstiness.ai_probability)));
      }
      if (comp.classifier) {
        dl.appendChild(dt("Classifier P(AI)"));
        dl.appendChild(dd(fmtNum(comp.classifier.ai_probability)));
      }
      el.components.appendChild(dl);
    }

    el.sentences.innerHTML = "";
    const sents = data.sentences || [];
    if (sents.length) {
      const ul = document.createElement("ul");
      for (const s of sents) {
        const p = s.ai_probability ?? 0.5;
        const li = document.createElement("li");
        if (p > 0.6) li.classList.add("ai");
        else if (p < 0.4) li.classList.add("human");
        else li.classList.add("uncertain");
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = Math.round(p * 100) + "% AI";
        li.appendChild(tag);
        li.appendChild(document.createTextNode(s.text || ""));
        ul.appendChild(li);
      }
      el.sentences.appendChild(ul);
    } else {
      el.sentences.innerHTML = "<p class=\"meta\">No sentence breakdown (disabled or empty).</p>";
    }

    el.result.classList.remove("hidden");
  }

  function dt(t) {
    const e = document.createElement("dt");
    e.textContent = t;
    return e;
  }
  function dd(t) {
    const e = document.createElement("dd");
    e.textContent = t;
    return e;
  }
  function fmtNum(x) {
    if (typeof x !== "number") return "—";
    return (x * 100).toFixed(1) + "%";
  }
})();
