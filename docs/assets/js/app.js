/* =====================================================================
   Bosque de decisiones · dashboard interactivo
   Datos: docs/data/dashboard.json + modelos exportados (JSON)
   ===================================================================== */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "—");
  const pct = (v, d = 1) => (Number.isFinite(v) ? (v * 100).toFixed(d) + "%" : "—");
  const col = { teal: "#0f766e", amber: "#b45309", violet: "#6d28d9", rose: "#be123c", mute: "#94a3b8" };

  let DATA = null;
  let ET = null, ADA = null, CAT = null;

  /* ------------------------------------------------------------------ */
  /*  DOM helpers                                                        */
  /* ------------------------------------------------------------------ */
  function h(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) node.setAttribute(k, v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined) continue;
      node.append(c.nodeType ? c : document.createTextNode(String(c)));
    }
    return node;
  }

  /* ------------------------------------------------------------------ */
  /*  Navigation, progress, reveal, lightbox                             */
  /* ------------------------------------------------------------------ */
  function initChrome() {
    const nav = $(".site-nav");
    const progress = $("#scrollProgress");
    const sections = $$("main section[id]");
    const links = $$(".nav-links a");

    const onScroll = () => {
      const y = window.scrollY;
      nav.classList.toggle("scrolled", y > 24);
      const h = document.documentElement.scrollHeight - window.innerHeight;
      progress.style.width = `${h > 0 ? (y / h) * 100 : 0}%`;
      let current = "";
      for (const s of sections) {
        if (s.getBoundingClientRect().top <= 140) current = s.id;
      }
      links.forEach(a => a.classList.toggle("active", a.getAttribute("href") === `#${current}`));
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    $("#navToggle")?.addEventListener("click", () => $(".nav-links")?.classList.toggle("open"));
    $$(".nav-links a").forEach(a => a.addEventListener("click", () => $(".nav-links")?.classList.remove("open")));

    // reveal
    const targets = $$(".section-head, .dataset-card, .pipeline, .model-head, .model-body, .results-grid, .simulator, .application, .compare-grid, .decision-card, .lessons, .conclusion-main, .conclusion-aside, .report-card, .repro");
    targets.forEach(t => t.classList.add("reveal"));
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("visible"); io.unobserve(e.target); } });
    }, { threshold: 0.08, rootMargin: "0px 0px -40px 0px" });
    targets.forEach(t => io.observe(t));

    // lightbox
    const box = $("#lightbox"), img = $("#lightboxImg"), cap = $("#lightboxCaption");
    $$(".figure").forEach(fig => {
      fig.addEventListener("click", () => {
        const im = $("img", fig);
        if (!im) return;
        img.src = im.src;
        cap.textContent = $("figcaption", fig)?.textContent || "";
        box.classList.add("open");
        box.setAttribute("aria-hidden", "false");
      });
    });
    const close = () => { box.classList.remove("open"); box.setAttribute("aria-hidden", "true"); };
    $("#lightboxClose")?.addEventListener("click", close);
    box?.addEventListener("click", e => { if (e.target === box) close(); });
    window.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
  }

  /* ------------------------------------------------------------------ */
  /*  Controls factory                                                   */
  /* ------------------------------------------------------------------ */
  function fieldWrap(label, control) {
    return h("div", { class: "control" }, h("label", {}, h("span", { text: label }), control.readout || ""), control.el);
  }

  function rangeControl({ id, label, min, max, step, value, onChange, format = v => v }) {
    const readout = h("b", { text: format(value) });
    const input = h("input", {
      type: "range", id, min, max, step, value,
      oninput: () => {
        const v = Number(input.value);
        readout.textContent = format(v);
        onChange(v);
      },
    });
    return {
      el: input, readout,
      get: () => Number(input.value),
      set: v => {
        input.value = v;
        readout.textContent = format(Number(v));
        input.style.setProperty("--fill", `${((v - min) / (max - min)) * 100}%`);
      },
    };
  }

  function selectControl({ id, label, options, value, onChange, labels = {} }) {
    const sel = h("select", { id, onchange: () => onChange(sel.value) },
      ...options.map(o => h("option", { value: o, selected: String(o) === String(value) ? "selected" : null },
        labels[o] !== undefined ? labels[o] : o)));
    return { el: sel, readout: "", get: () => sel.value, set: v => { sel.value = v; } };
  }

  function toggleControl({ id, label, checked, onChange }) {
    const input = h("input", { type: "checkbox", id, checked: checked ? "checked" : null, onchange: () => onChange(input.checked) });
    return {
      el: h("label", { class: "toggle", for: id }, input, h("span", { text: label })),
      readout: "", get: () => input.checked, set: v => { input.checked = !!v; },
    };
  }

  function group(title) {
    const fs = h("fieldset", { class: "control-group" }, h("legend", { text: title }));
    return fs;
  }

  function addControl(parent, spec) {
    const c = spec.type === "range" ? rangeControl(spec)
      : spec.type === "select" ? selectControl(spec)
      : toggleControl(spec);
    if (spec.type === "toggle") parent.append(c.el);
    else parent.append(fieldWrap(spec.label, c));
    if (spec.key) {
      c.el.dataset.key = spec.key;
      const inner = c.el.matches("input,select") ? c.el : c.el.querySelector("input,select");
      if (inner) inner.dataset.key = spec.key;
    }
    return c;
  }

  function refreshControls(host, state) {
    $$("[data-key]", host).forEach(el => {
      const key = el.dataset.key;
      if (!(key in state)) return;
      const v = state[key];
      if (el.type === "checkbox") el.checked = !!v;
      else if (el.tagName === "SELECT") el.value = String(v);
      else {
        el.value = v;
        el.dispatchEvent(new Event("input"));
      }
    });
  }

  /* ------------------------------------------------------------------ */
  /*  Chart defaults                                                     */
  /* ------------------------------------------------------------------ */
  function initChartDefaults() {
    if (!window.Chart) return;
    Chart.defaults.font.family = 'Inter, "Segoe UI", system-ui, sans-serif';
    Chart.defaults.font.size = 10;
    Chart.defaults.color = "#64748b";
    Chart.defaults.plugins.legend.labels.boxWidth = 10;
    Chart.defaults.plugins.legend.labels.boxHeight = 10;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(10,25,40,.94)";
    Chart.defaults.plugins.tooltip.padding = 9;
    Chart.defaults.plugins.tooltip.cornerRadius = 7;
    Chart.defaults.elements.line.tension = 0.3;
  }

  /* ------------------------------------------------------------------ */
  /*  Training replay                                                    */
  /* ------------------------------------------------------------------ */
  const replays = {};

  function makeReplay(key, labels, series, opts = {}) {
    const canvas = $(`#replay-${key}`);
    if (!canvas || !window.Chart) return;
    const x = labels;
    const base = series.map(s => ({
      label: s.label, data: s.data, borderColor: s.color,
      backgroundColor: s.color, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4,
      yAxisID: s.axis || "y", tension: 0.25, fill: false,
    }));
    const cursor = series.map(s => ({
      label: s.label + " cursor", data: new Array(x.length).fill(null),
      borderColor: s.color, backgroundColor: s.color, borderWidth: 0,
      pointRadius: 5, pointHoverRadius: 6, showLine: false, yAxisID: s.axis || "y",
    }));
    const chart = new Chart(canvas, {
      type: "line",
      data: { labels: x, datasets: [...base, ...cursor] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "bottom", labels: { filter: it => !it.text.includes("cursor") } },
          tooltip: { callbacks: { title: it => `${opts.xLabel || "iteración"} ${it[0].label}` } },
        },
        scales: {
          x: { title: { display: true, text: opts.xLabel || "iteración" }, grid: { color: "rgba(148,163,184,.16)" }, ticks: { maxTicksLimit: 8 } },
          y: {
            title: { display: true, text: opts.yLabel || "métrica" },
            grid: { color: "rgba(148,163,184,.16)" },
            min: opts.yMin, max: opts.yMax,
          },
          ...(opts.y2 ? { y1: { position: "right", title: { display: true, text: opts.y2 }, grid: { display: false } } } : {}),
        },
      },
    });
    const slider = $(`#slider-${key}`);
    const play = $(`#play-${key}`);
    const readout = $(`#readout-${key}`);
    if (slider) slider.max = x.length - 1;

    let idx = 0, timer = null;
    const draw = () => {
      const n = Math.min(x.length, idx + 1);
      chart.data.datasets.forEach((ds, di) => {
        if (di >= series.length) {
          const si = di - series.length;
          const arr = new Array(x.length).fill(null);
          arr[idx] = series[si].data[idx];
          ds.data = arr;
        }
      });
      chart.update("none");
      if (slider) {
        slider.value = idx;
        slider.style.setProperty("--fill", `${(idx / Math.max(x.length - 1, 1)) * 100}%`);
      }
      if (readout) {
        const vals = series.map(s => `${s.label.split(" ")[0]} ${fmt(s.data[idx], opts.digits ?? 3)}`).join(" · ");
        readout.textContent = `${opts.xShort || "it"} ${x[idx]} · ${vals}`;
      }
    };
    const tick = () => { if (idx >= x.length - 1) { stop(); return; } idx++; draw(); };
    const start = () => { stop(); if (idx >= x.length - 1) idx = 0; timer = setInterval(tick, 70); play.textContent = "❚❚"; };
    const stop = () => { if (timer) clearInterval(timer); timer = null; if (play) play.textContent = "▶"; };

    slider?.addEventListener("input", () => { stop(); idx = Number(slider.value); draw(); });
    play?.addEventListener("click", () => timer ? stop() : start());
    draw();
    replays[key] = { chart, series, draw };
  }

  function initReplays() {
    const e = DATA.models.extra_trees.learning;
    makeReplay("extra", e.trees, [
      { label: "RMSE validación", data: e.val_rmse, color: col.teal },
      { label: "MAE validación", data: e.val_mae, color: "#14b8a6" },
    ], { xLabel: "árboles acumulados", yLabel: "error (unidades/día)", xShort: "árboles", digits: 3 });

    const a = DATA.models.adaboost.learning;
    makeReplay("ada", a.estimators, [
      { label: "ROC-AUC", data: a.roc_auc, color: col.amber },
      { label: "F1", data: a.f1, color: "#f59e0b" },
      { label: "Accuracy", data: a.accuracy, color: "#0f766e" },
    ], { xLabel: "estimadores débiles", yLabel: "métrica", yMin: 0.5, yMax: 0.9, xShort: "estimadores", digits: 3 });

    const c = DATA.models.catboost.learning;
    makeReplay("cat", c.trees, [
      { label: "F1 macro", data: c.f1_macro, color: col.violet },
      { label: "Accuracy", data: c.accuracy, color: "#0f766e" },
      { label: "Log-loss", data: c.log_loss, color: "#a78bfa" },
    ], { xLabel: "iteraciones", yLabel: "métrica", yMin: 0.5, yMax: 1.0, xShort: "iter", digits: 3 });
  }

  /* ------------------------------------------------------------------ */
  /*  Radar                                                              */
  /* ------------------------------------------------------------------ */
  function initRadar() {
    const canvas = $("#radar");
    if (!canvas || !window.Chart) return;
    new Chart(canvas, {
      type: "radar",
      data: {
        labels: ["Desempeño", "Interpretabilidad", "Categóricas", "Velocidad", "Robustez a outliers", "Facilidad de tuning"],
        datasets: [
          { label: "Extra Trees", data: [4, 3, 2, 4.5, 5, 4], borderColor: col.teal, backgroundColor: "rgba(15,118,110,.12)", pointBackgroundColor: col.teal, borderWidth: 2, pointRadius: 3 },
          { label: "AdaBoost", data: [3, 4, 1.5, 5, 2, 3.5], borderColor: col.amber, backgroundColor: "rgba(180,83,9,.10)", pointBackgroundColor: col.amber, borderWidth: 2, pointRadius: 3 },
          { label: "CatBoost", data: [4.5, 3, 5, 3, 4, 3], borderColor: col.violet, backgroundColor: "rgba(109,40,217,.10)", pointBackgroundColor: col.violet, borderWidth: 2, pointRadius: 3 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          r: {
            min: 0, max: 5, ticks: { stepSize: 1, backdropColor: "transparent", color: "#94a3b8", font: { size: 8 } },
            grid: { color: "rgba(148,163,184,.22)" }, angleLines: { color: "rgba(148,163,184,.22)" },
            pointLabels: { color: "#475569", font: { size: 9, weight: "600" } },
          },
        },
      },
    });
  }

  /* ------------------------------------------------------------------ */
  /*  Extra Trees simulator                                              */
  /* ------------------------------------------------------------------ */
  function initExtraSim() {
    const host = $("#sim-extra-controls");
    if (!host) return;
    const feats = ET.feature_names;
    const cat = ET.categorical;
    const levels = ET.cat_levels;
    const med = ET.medians;

    const state = {
      store_id: levels.store_id[0], dept_id: levels.dept_id[2] || levels.dept_id[0],
      cat_id: levels.cat_id[0], lag_7: 0, lag_1: 0, roll_mean_7: 0.4,
      roll_mean_28: 0.5, roll_mean_90: 0.5, sell_price: med.sell_price ?? 3.6,
      wday: 3, month: 5, snap: false, event_flag: false,
    };
    state.lag_1 = state.lag_7;

    const f1 = group("Contexto de la serie");
    const storeSel = addControl(f1, {
      type: "select", label: "Tienda", options: levels.store_id, value: state.store_id, key: "store_id",
      onChange: v => { state.store_id = v; update(); },
    });
    addControl(f1, {
      type: "select", label: "Departamento", options: levels.dept_id, value: state.dept_id, key: "dept_id",
      onChange: v => { state.dept_id = v; update(); },
    });
    addControl(f1, {
      type: "select", key: "cat_id", label: "Categoría", options: levels.cat_id, value: state.cat_id,
      onChange: v => { state.cat_id = v; update(); },
    });

    const f2 = group("Demanda reciente");
    const lag7 = addControl(f2, {
      type: "range", key: "lag_7", label: "Ventas de hace 7 días (lag_7)", min: 0, max: 20, step: 1,
      value: state.lag_7, format: v => `${v} u.`,
      onChange: v => { state.lag_7 = v; state.lag_1 = v; update(); },
    });
    addControl(f2, {
      type: "range", label: "Media móvil 7 días", min: 0, max: 20, step: 0.1,
      value: state.roll_mean_7, key: "roll_mean_7", format: v => fmt(v, 1),
      onChange: v => { state.roll_mean_7 = v; update(); },
    });
    addControl(f2, {
      type: "range", label: "Media móvil 28 días", min: 0, max: 20, step: 0.1,
      value: state.roll_mean_28, key: "roll_mean_28", format: v => fmt(v, 1),
      onChange: v => { state.roll_mean_28 = v; update(); },
    });
    addControl(f2, {
      type: "range", label: "Media móvil 90 días", min: 0, max: 15, step: 0.1,
      value: state.roll_mean_90, key: "roll_mean_90", format: v => fmt(v, 1),
      onChange: v => { state.roll_mean_90 = v; update(); },
    });

    const f3 = group("Precio y calendario");
    addControl(f3, {
      type: "range", label: "Precio de venta ($)", min: 0.5, max: 15, step: 0.1,
      value: state.sell_price, key: "sell_price", format: v => `$${fmt(v, 2)}`,
      onChange: v => { state.sell_price = v; update(); },
    });
    addControl(f3, {
      type: "range", label: "Día de la semana (1=sáb)", min: 1, max: 7, step: 1,
      value: state.wday, key: "wday", format: v => ["", "sáb", "dom", "lun", "mar", "mié", "jue", "vie"][v],
      onChange: v => { state.wday = v; update(); },
    });
    addControl(f3, {
      type: "range", label: "Mes", min: 1, max: 12, step: 1,
      value: state.month, key: "month", format: v => `mes ${v}`,
      onChange: v => { state.month = v; update(); },
    });
    addControl(f3, {
      type: "toggle", label: "Día SNAP (ayuda alimentaria)", checked: state.snap, key: "snap",
      onChange: v => { state.snap = v; update(); },
    });
    addControl(f3, {
      type: "toggle", label: "Evento especial", checked: state.event_flag, key: "event_flag",
      onChange: v => { state.event_flag = v; update(); },
    });

    host.append(f1, f2, f3);

    const presets = h("div", { class: "preset-row" });
    (DATA.models.extra_trees.test_sample || []).slice(0, 3).forEach((s, i) => {
      const b = h("button", { class: "preset", type: "button", text: `Escenario ${i + 1} · real ${fmt(s.actual, 0)}` });
      b.addEventListener("click", () => {
        state.lag_7 = clamp(Math.round(s.lag_7), 0, 20);
        state.lag_1 = state.lag_7;
        state.roll_mean_7 = clamp(s.lag_7, 0, 20);
        state.roll_mean_28 = clamp(s.roll_mean_28, 0, 20);
        state.roll_mean_90 = clamp(s.roll_mean_28 * 1.05, 0, 15);
        state.sell_price = clamp(s.price, 0.5, 15);
        if (s.store && levels.store_id.includes(s.store)) state.store_id = s.store;
        const parts = String(s.item || "").split("_");
        const dept = parts.slice(0, 2).join("_");
        const cat = parts[0];
        if (levels.dept_id.includes(dept)) state.dept_id = dept;
        if (levels.cat_id.includes(cat)) state.cat_id = cat;
        refreshControls(host, state);
        update();
      });
      presets.append(b);
    });
    host.append(h("div", { class: "control-group" }, h("legend", { text: "Escenarios reales de test" }), presets));

    function vector(overrides = {}) {
      const x = new Array(feats.length).fill(0);
      feats.forEach((f, i) => { if (med[f] !== undefined && med[f] !== null) x[i] = med[f]; });
      const s = { ...state, ...overrides };
      const set = (f, v) => { const i = feats.indexOf(f); if (i >= 0) x[i] = v; };
      set("dept_id", levels.dept_id.indexOf(s.dept_id));
      set("cat_id", levels.cat_id.indexOf(s.cat_id));
      set("store_id", levels.store_id.indexOf(s.store_id));
      const stateId = String(s.store_id).split("_")[0];
      set("state_id", levels.state_id.indexOf(stateId));
      set("wday", s.wday);
      set("is_weekend", (s.wday === 1 || s.wday === 2) ? 1 : 0);
      set("month", s.month);
      set("dayofyear", Math.round((s.month - 0.5) * 30.4));
      set("snap", s.snap ? 1 : 0);
      set("event_flag", s.event_flag ? 1 : 0);
      set("event_type_code", s.event_flag ? 1 : 0);
      set("sell_price", s.sell_price);
      set("price_change_28", 0);
      set("price_ratio_item", s.sell_price / Math.max(med.item_price_mean || 1, 0.01));
      set("lag_1", s.lag_1);
      set("lag_7", s.lag_7);
      set("lag_14", s.lag_7);
      set("lag_28", s.roll_mean_28);
      set("lag_364", s.roll_mean_90 * 30);
      set("roll_mean_7", s.roll_mean_7);
      set("roll_mean_28", s.roll_mean_28);
      set("roll_mean_90", s.roll_mean_90);
      set("roll_max_28", Math.max(s.roll_mean_28 * 3, s.lag_7));
      set("roll_std_7", Math.max(s.lag_7 * 0.7, 0));
      set("roll_std_28", Math.max(s.roll_mean_28 * 1.2, 0));
      set("zero_ratio_28", s.roll_mean_28 <= 0.01 ? 0.95 : clamp(1 - s.roll_mean_28 / 3, 0.02, 0.95));
      return x;
    }

    function predict(x) {
      const preds = [];
      for (const t of ET.trees) {
        let i = 0;
        while (t.l[i] >= 0) i = x[t.f[i]] <= t.t[i] ? t.l[i] : t.r[i];
        preds.push(t.v[i]);
      }
      const mean = preds.reduce((a, b) => a + b, 0) / preds.length;
      const sd = Math.sqrt(preds.reduce((a, b) => a + (b - mean) ** 2, 0) / preds.length);
      return { mean: Math.max(0, mean), sd };
    }

    let curveChart = null;
    function drawCurve() {
      const xs = Array.from({ length: 21 }, (_, i) => i);
      const ys = xs.map(v => predict(vector({ lag_7: v, lag_1: v })).mean);
      if (!curveChart) {
        curveChart = new Chart($("#extra-curve"), {
          type: "line",
          data: {
            labels: xs,
            datasets: [{
              label: "predicción", data: ys, borderColor: "#5eead4",
              backgroundColor: "rgba(94,234,212,.12)", fill: true, borderWidth: 2,
              pointRadius: 0, tension: 0.35,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { title: { display: true, text: "lag_7 (unidades)", color: "#8fa6b8" }, ticks: { color: "#8fa6b8", maxTicksLimit: 6 }, grid: { color: "rgba(255,255,255,.08)" } },
              y: { title: { display: true, text: "unidades/día", color: "#8fa6b8" }, ticks: { color: "#8fa6b8" }, grid: { color: "rgba(255,255,255,.08)" } },
            },
          },
        });
      } else {
        curveChart.data.datasets[0].data = ys;
        curveChart.update("none");
      }
    }

    function update() {
      const { mean, sd } = predict(vector());
      $("#extra-pred").textContent = fmt(mean, 2);
      $("#extra-band").textContent = `±1σ entre árboles: ${fmt(Math.max(0, mean - sd), 2)} – ${fmt(mean + sd, 2)}`;
      $("#extra-gauge").style.width = `${clamp(mean / 12 * 100, 2, 100)}%`;
      drawCurve();
    }

    // sync all range fills
    $$("input[type=range]", host).forEach(inp => {
      const upd = () => inp.style.setProperty("--fill", `${((inp.value - inp.min) / (inp.max - inp.min)) * 100}%`);
      inp.addEventListener("input", upd); upd();
    });
    update();
  }

  /* ------------------------------------------------------------------ */
  /*  AdaBoost simulator                                                 */
  /* ------------------------------------------------------------------ */
  const ADA_LABELS = {
    HighBP: "Presión alta", HighChol: "Colesterol alto", CholCheck: "Chequeo de colesterol",
    BMI: "IMC", Smoker: "Fumador", Stroke: "Derrame", HeartDiseaseorAttack: "Enfermedad cardíaca",
    PhysActivity: "Actividad física", Fruits: "Consume fruta", Veggies: "Consume verdura",
    HvyAlcoholConsump: "Alcohol elevado", AnyHealthcare: "Cobertura de salud", NoDocbcCost: "Sin médico por costo",
    GenHlth: "Salud general (1–5)", MentHlth: "Días de salud mental", PhysHlth: "Días de salud física",
    DiffWalk: "Dificultad para caminar", Sex: "Sexo (1=hombre)", Age: "Grupo de edad (1–13)",
    Education: "Educación (1–6)", Income: "Ingreso (1–8)", Comorbidity: "Comorbilidad",
    HealthyHabits: "Hábitos saludables", BMI_cat: "Categoría de IMC", MentPhysDays: "Días mentales+físicos",
  };

  function initAdaSim() {
    const host = $("#sim-ada-controls");
    if (!host) return;
    const feats = ADA.feature_names;
    const def = ADA.defaults;
    const state = {};
    for (const f of feats) state[f] = Number(def[f] ?? 0);
    for (const f of ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
      "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"]) {
      state[f] = Math.round(state[f]) ? 1 : 0;
    }

    const toggles = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
      "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"];
    const ranges = [
      ["BMI", 12, 60, 1, v => fmt(v, 0)],
      ["GenHlth", 1, 5, 1, v => `${v}/5`],
      ["Age", 1, 13, 1, v => `${v}/13`],
      ["MentHlth", 0, 30, 1, v => `${v} días`],
      ["PhysHlth", 0, 30, 1, v => `${v} días`],
      ["Education", 1, 6, 1, v => `${v}/6`],
      ["Income", 1, 8, 1, v => `${v}/8`],
    ];

    const g1 = group("Indicadores clínicos y hábitos");
    toggles.forEach(f => addControl(g1, {
      type: "toggle", label: ADA_LABELS[f] || f, checked: state[f], key: f,
      onChange: v => { state[f] = v ? 1 : 0; update(); },
    }));
    const g2 = group("Variables continuas y ordinales");
    ranges.forEach(([f, min, max, step, format]) => addControl(g2, {
      type: "range", label: ADA_LABELS[f] || f, min, max, step, value: state[f], format, key: f,
      onChange: v => { state[f] = v; update(); },
    }));
    host.append(g1, g2);

    // presets
    const presetRow = h("div", { class: "preset-row" });
    const presets = {
      "Perfil saludable": { BMI: 23, GenHlth: 1, Age: 4, HighBP: 0, HighChol: 0, PhysActivity: 1, Fruits: 1, Veggies: 1, Income: 8, Education: 6, DiffWalk: 0, Smoker: 0, HeartDiseaseorAttack: 0, Stroke: 0, MentHlth: 0, PhysHlth: 0 },
      "Riesgo medio": { BMI: 29, GenHlth: 3, Age: 8, HighBP: 1, HighChol: 0, PhysActivity: 0, Fruits: 0, Veggies: 1, Income: 5, Education: 4, DiffWalk: 0, Smoker: 1, HeartDiseaseorAttack: 0, Stroke: 0, MentHlth: 3, PhysHlth: 2 },
      "Alto riesgo": { BMI: 38, GenHlth: 5, Age: 11, HighBP: 1, HighChol: 1, PhysActivity: 0, Fruits: 0, Veggies: 0, Income: 3, Education: 3, DiffWalk: 1, Smoker: 1, HeartDiseaseorAttack: 1, Stroke: 0, MentHlth: 8, PhysHlth: 10 },
    };
    Object.entries(presets).forEach(([name, vals]) => {
      const b = h("button", { class: "preset", type: "button", text: name });
      b.addEventListener("click", () => {
        Object.assign(state, vals);
        refreshControls(host, state);
        update();
      });
      presetRow.append(b);
    });
    host.append(h("div", { class: "control-group" }, h("legend", { text: "Perfiles de ejemplo" }), presetRow));

    function vector(overrides = {}) {
      const s = { ...state, ...overrides };
      s.Comorbidity = s.HighBP + s.HighChol + s.Stroke + s.HeartDiseaseorAttack;
      s.HealthyHabits = s.PhysActivity + s.Fruits + s.Veggies - s.Smoker - s.HvyAlcoholConsump;
      s.BMI_cat = s.BMI <= 18.5 ? 0 : s.BMI <= 25 ? 1 : s.BMI <= 30 ? 2 : s.BMI <= 35 ? 3 : 4;
      s.MentPhysDays = s.MentHlth + s.PhysHlth;
      return feats.map(f => Number(s[f] ?? 0));
    }

    function predict(x) {
      const mean = ADA.scaler.mean, scale = ADA.scaler.scale;
      const z = x.map((v, i) => (v - mean[i]) / scale[i]);
      let total = 0, sw = 0;
      const drivers = {};
      for (let ti = 0; ti < ADA.trees.length; ti++) {
        const t = ADA.trees[ti], w = ADA.weights[ti];
        sw += w;
        let i = 0;
        const path = [];
        while (t.l[i] >= 0) {
          const f = t.f[i];
          path.push(f);
          i = z[f] <= t.t[i] ? t.l[i] : t.r[i];
        }
        const sign = t.v[i] >= 0.5 ? 1 : -1;
        total += w * sign;
        path.forEach(f => {
          drivers[f] = (drivers[f] || 0) + (w * sign) / Math.max(path.length, 1);
        });
      }
      const f = 2 * total / sw;
      return { p: 1 / (1 + Math.exp(-f)), drivers };
    }

    const names = ADA.feature_names;
    let curveChart = null;
    function drawCurve() {
      const xs = Array.from({ length: 41 }, (_, i) => 15 + i);
      const ys = xs.map(v => predict(vector({ BMI: v })).p);
      if (!curveChart) {
        curveChart = new Chart($("#ada-curve"), {
          type: "line",
          data: {
            labels: xs,
            datasets: [{
              label: "P(diabetes)", data: ys, borderColor: "#fbbf24",
              backgroundColor: "rgba(251,191,36,.12)", fill: true, borderWidth: 2,
              pointRadius: 0, tension: 0.35,
            }, {
              label: "umbral 0.38", data: xs.map(() => ADA.threshold),
              borderColor: "rgba(255,255,255,.55)", borderDash: [5, 4], borderWidth: 1,
              pointRadius: 0, fill: false,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { title: { display: true, text: "IMC", color: "#8fa6b8" }, ticks: { color: "#8fa6b8", maxTicksLimit: 7 }, grid: { color: "rgba(255,255,255,.08)" } },
              y: { title: { display: true, text: "probabilidad", color: "#8fa6b8" }, min: 0, max: 1, ticks: { color: "#8fa6b8" }, grid: { color: "rgba(255,255,255,.08)" } },
            },
          },
        });
      } else {
        curveChart.data.datasets[0].data = ys;
        curveChart.update("none");
      }
    }

    function update() {
      const { p, drivers } = predict(vector());
      $("#ada-pred").textContent = pct(p, 1);
      const label = p < 0.35 ? "riesgo bajo" : p < 0.6 ? "riesgo medio" : "riesgo alto";
      $("#ada-risk-label").textContent = `${label} · umbral operativo 0.38 → ${p >= ADA.threshold ? "derivar a confirmación" : "sin alerta"}`;
      $("#ada-meter").style.width = `${clamp(p * 100, 1, 100)}%`;

      const top = Object.entries(drivers)
        .map(([f, v]) => ({ f: names[Number(f)], v }))
        .sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 6);
      const max = Math.max(...top.map(d => Math.abs(d.v)), 1e-9);
      const list = $("#ada-drivers");
      list.innerHTML = "<h4>Votos que más empujan la decisión</h4>";
      top.forEach(d => {
        const positive = d.v > 0;
        list.append(h("div", { class: "driver" },
          h("span", { text: ADA_LABELS[d.f] || d.f }),
          h("b", { text: (positive ? "+" : "") + fmt(d.v, 3), style: `color:${positive ? "#fbbf24" : "#5eead4"};font-family:var(--mono);font-size:10px;text-align:right` }),
          h("div", { class: "bar" }, h("i", { class: positive ? "pos" : "neg", style: `width:${Math.abs(d.v) / max * 100}%` })),
        ));
      });
      drawCurve();
    }

    $$("input[type=range]", host).forEach(inp => {
      const upd = () => inp.style.setProperty("--fill", `${((inp.value - inp.min) / (inp.max - inp.min)) * 100}%`);
      inp.addEventListener("input", upd); upd();
    });
    update();
  }

  /* ------------------------------------------------------------------ */
  /*  CatBoost simulator                                                 */
  /* ------------------------------------------------------------------ */
  const CAT_LABELS = {
    Marital_status: "Estado civil", Application_mode: "Modo de solicitud", Application_order: "Orden de solicitud",
    Course: "Curso", Daytime_evening_attendance: "Turno", Previous_qualification: "Cualificación previa",
    Nacionality: "Nacionalidad", Mothers_qualification: "Cualificación madre", Fathers_qualification: "Cualificación padre",
    Mothers_occupation: "Ocupación madre", Fathers_occupation: "Ocupación padre", Displaced: "Desplazado",
    Educational_special_needs: "Necesidades especiales", Debtor: "Deudor", Tuition_fees_up_to_date: "Matrícula al día",
    Gender: "Género", Scholarship_holder: "Becario", International: "Internacional",
    Age_at_enrollment: "Edad al matricularse", Admission_grade: "Nota de admisión",
    Previous_qualification_grade: "Nota de cualificación previa",
    Curricular_units_1st_sem_approved: "Aprobadas 1er semestre", Curricular_units_2nd_sem_approved: "Aprobadas 2do semestre",
    Curricular_units_1st_sem_enrolled: "Matriculadas 1er semestre", Curricular_units_2nd_sem_enrolled: "Matriculadas 2do semestre",
    Curricular_units_1st_sem_grade: "Nota media 1er semestre", Curricular_units_2nd_sem_grade: "Nota media 2do semestre",
    Approved_ratio_1st: "Ratio aprobación 1er semestre", Approved_ratio_2nd: "Ratio aprobación 2do semestre",
    Grade_delta: "Cambio de nota entre semestres",
  };

  function initCatSim() {
    const host = $("#sim-cat-controls");
    if (!host) return;
    const feats = CAT.feature_names;
    const num = new Set(CAT.numeric);
    const state = {};
    for (const f of feats) state[f] = CAT.defaults[f];
    for (const f of CAT.categorical) state[f] = String(CAT.defaults[f]);

    const g1 = group("Trayectoria académica");
    [
      ["Curricular_units_1st_sem_approved", 0, 20, 1],
      ["Curricular_units_1st_sem_enrolled", 0, 20, 1],
      ["Curricular_units_1st_sem_grade", 0, 20, 0.1],
      ["Curricular_units_2nd_sem_approved", 0, 20, 1],
      ["Curricular_units_2nd_sem_enrolled", 0, 20, 1],
      ["Curricular_units_2nd_sem_grade", 0, 20, 0.1],
      ["Admission_grade", 90, 200, 0.5],
      ["Previous_qualification_grade", 90, 200, 0.5],
    ].forEach(([f, min, max, step]) => addControl(g1, {
      type: "range", label: CAT_LABELS[f] || f, min, max, step, value: state[f], key: f,
      format: v => fmt(v, step < 1 ? 1 : 0),
      onChange: v => { state[f] = v; update(); },
    }));

    const g2 = group("Contexto y apoyo");
    const yesno = { "0": "No", "1": "Sí" };
    addControl(g2, {
      type: "select", label: "Matrícula al día", options: CAT.cat_levels.Tuition_fees_up_to_date,
      value: state.Tuition_fees_up_to_date, labels: yesno, key: "Tuition_fees_up_to_date",
      onChange: v => { state.Tuition_fees_up_to_date = v; update(); },
    });
    addControl(g2, {
      type: "select", label: "Becario", options: CAT.cat_levels.Scholarship_holder,
      value: state.Scholarship_holder, labels: yesno, key: "Scholarship_holder",
      onChange: v => { state.Scholarship_holder = v; update(); },
    });
    addControl(g2, {
      type: "select", label: "Deudor", options: CAT.cat_levels.Debtor,
      value: state.Debtor, labels: yesno, key: "Debtor",
      onChange: v => { state.Debtor = v; update(); },
    });
    addControl(g2, {
      type: "select", label: "Curso", options: CAT.cat_levels.Course,
      value: state.Course, labels: Object.fromEntries(CAT.cat_levels.Course.map(c => [c, `Curso ${c}`])), key: "Course",
      onChange: v => { state.Course = v; update(); },
    });
    addControl(g2, {
      type: "select", label: "Modo de solicitud", options: CAT.cat_levels.Application_mode,
      value: state.Application_mode, key: "Application_mode",
      onChange: v => { state.Application_mode = v; update(); },
    });
    addControl(g2, {
      type: "select", label: "Género", options: CAT.cat_levels.Gender,
      value: state.Gender, labels: { "0": "Femenino", "1": "Masculino" }, key: "Gender",
      onChange: v => { state.Gender = v; update(); },
    });
    addControl(g2, {
      type: "range", label: "Edad al matricularse", min: 17, max: 60, step: 1,
      value: state.Age_at_enrollment, format: v => `${v} años`, key: "Age_at_enrollment",
      onChange: v => { state.Age_at_enrollment = v; update(); },
    });

    const g3 = group("Contexto familiar (avanzado)");
    ["Marital_status", "Mothers_qualification", "Fathers_qualification", "Mothers_occupation", "Fathers_occupation"].forEach(f => {
      addControl(g3, {
        type: "select", label: CAT_LABELS[f] || f, options: CAT.cat_levels[f],
        value: state[f], key: f,
        onChange: v => { state[f] = v; update(); },
      });
    });
    host.append(g1, g2, g3);

    const presetRow = h("div", { class: "preset-row" });
    (DATA.models.catboost.test_sample || []).slice(0, 3).forEach((s, i) => {
      const b = h("button", { class: "preset", type: "button", text: `Estudiante ${i + 1} · ${s.actual}` });
      b.addEventListener("click", () => {
        state.Curricular_units_1st_sem_approved = s.approved1;
        state.Curricular_units_2nd_sem_approved = s.approved2;
        state.Curricular_units_1st_sem_enrolled = Math.max(6, s.approved1);
        state.Curricular_units_2nd_sem_enrolled = Math.max(6, s.approved2);
        state.Curricular_units_1st_sem_grade = s.grade1;
        state.Curricular_units_2nd_sem_grade = s.grade2;
        state.Tuition_fees_up_to_date = String(s.tuition);
        state.Scholarship_holder = String(s.scholarship);
        state.Age_at_enrollment = s.age;
        state.Course = String(s.course);
        refreshControls(host, state);
        update();
      });
      presetRow.append(b);
    });
    host.append(h("div", { class: "control-group" }, h("legend", { text: "Estudiantes reales de test" }), presetRow));

    function vector(overrides = {}) {
      const s = { ...state, ...overrides };
      const v = new Array(CAT.columns.length).fill(0);
      const idx = new Map(CAT.columns.map((c, i) => [c, i]));
      for (const f of feats) {
        if (CAT.categorical.includes(f)) {
          const key = `${f}_${s[f]}`;
          if (idx.has(key)) v[idx.get(key)] = 1;
        } else if (idx.has(f)) {
          v[idx.get(f)] = Number(s[f] ?? 0);
        }
      }
      // derived ratios
      const set = (k, val) => { if (idx.has(k)) v[idx.get(k)] = val; };
      set("Approved_ratio_1st", s.Curricular_units_1st_sem_enrolled > 0 ? s.Curricular_units_1st_sem_approved / s.Curricular_units_1st_sem_enrolled : 0);
      set("Approved_ratio_2nd", s.Curricular_units_2nd_sem_enrolled > 0 ? s.Curricular_units_2nd_sem_approved / s.Curricular_units_2nd_sem_enrolled : 0);
      set("Grade_delta", s.Curricular_units_2nd_sem_grade - s.Curricular_units_1st_sem_grade);
      return v;
    }

    function predict(x) {
      const nc = CAT.classes.length;
      const raw = new Array(nc).fill(0);
      for (const t of CAT.trees) {
        let li = 0;
        for (let l = 0; l < t.splits.length; l++) {
          const [fi, border] = t.splits[l];
          if (x[fi] > border) li |= (1 << l);
        }
        for (let c = 0; c < nc; c++) raw[c] += t.leaves[li * nc + c];
      }
      for (let c = 0; c < nc; c++) raw[c] = raw[c] * CAT.scale + CAT.bias[c];
      const mx = Math.max(...raw);
      const e = raw.map(v => Math.exp(v - mx));
      const z = e.reduce((a, b) => a + b, 0);
      return e.map(v => v / z);
    }

    const imp = (DATA.models.catboost.importance || []).slice(0, 6);
    const impMax = Math.max(...imp.map(x => x.importance), 1e-9);
    const impBox = $("#cat-importance");
    impBox.innerHTML = "<h4>Variables más influyentes (modelo completo)</h4>";
    imp.forEach(x => {
      impBox.append(h("div", { class: "imp-mini-row" },
        h("span", { text: CAT_LABELS[x.feature] || x.feature }),
        h("div", { class: "imp-mini-bar" }, h("i", { style: `width:${x.importance / impMax * 100}%` })),
      ));
    });

    function update() {
      const p = predict(vector());
      const order = CAT.classes.map((c, i) => ({ c, p: p[i] })).sort((a, b) => b.p - a.p);
      $("#cat-pred").textContent = order[0].c;
      const pd = p[CAT.classes.indexOf("Dropout")] ?? 0;
      const risk = pd < 0.33 ? "riesgo bajo" : pd < 0.5 ? "riesgo medio" : "riesgo alto";
      $("#cat-risk").textContent = `${risk} · P(Dropout) = ${pct(pd, 1)}`;
      const bars = $("#cat-bars");
      bars.innerHTML = "";
      CAT.classes.forEach((c, i) => {
        bars.append(h("div", { class: "prob-row" },
          h("div", { class: "prob-head" }, h("span", { text: c }), h("b", { text: pct(p[i], 1) })),
          h("div", { class: "prob-track" }, h("div", { class: "prob-fill", style: `width:${p[i] * 100}%` })),
        ));
      });
    }

    $$("input[type=range]", host).forEach(inp => {
      const upd = () => inp.style.setProperty("--fill", `${((inp.value - inp.min) / (inp.max - inp.min)) * 100}%`);
      inp.addEventListener("input", upd); upd();
    });
    update();
  }

  /* ------------------------------------------------------------------ */
  /*  Boot                                                               */
  /* ------------------------------------------------------------------ */
  async function boot() {
    initChartDefaults();
    initChrome();

    try {
      const [dashboard, et, ada, cat] = await Promise.all([
        fetch("data/dashboard.json").then(r => r.json()),
        fetch("models/extra_trees.json").then(r => r.json()),
        fetch("models/adaboost.json").then(r => r.json()),
        fetch("models/catboost.json").then(r => r.json()),
      ]);
      DATA = dashboard; ET = et; ADA = ada; CAT = cat;
      initRadar();
      initReplays();
      initExtraSim();
      initAdaSim();
      initCatSim();
      console.log("Modelos cargados. Verificación de inferencia:", dashboard.inference_checks);
    } catch (err) {
      console.error("No se pudieron cargar los datos del dashboard:", err);
      const host = $("#sim-extra-controls");
      if (host) host.innerHTML = `<p style="color:#fca5a5;font-size:12px">No se pudieron cargar los JSON. Abre la página con un servidor local (python -m http.server) o revisa la ruta docs/data.</p>`;
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
