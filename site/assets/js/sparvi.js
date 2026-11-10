/* Sparvi shared components. Plain JS on top of d3 + Observable Plot (both self-hosted). */
(function () {
  const S = {};
  const cache = new Map();

  S.load = (path) => {
    if (!cache.has(path)) cache.set(path, fetch(path).then((r) => {
      if (!r.ok) throw new Error(`${path}: ${r.status}`);
      return r.json();
    }));
    return cache.get(path);
  };

  // ---- formatting -------------------------------------------------------
  const dash = "—";
  S.fmt = {
    pct: (v, d = 1) => (v == null ? dash : (100 * v).toFixed(d) + "%"),
    num: (v, d = 1) => (v == null ? dash : (+v).toFixed(d)),
    int: (v) => (v == null ? dash : Math.round(v).toLocaleString()),
    woba: (v) => (v == null ? dash : (+v).toFixed(3).replace(/^0/, "")),
    signed: (v, d = 1) => (v == null ? dash : (v > 0 ? "+" : "") + (+v).toFixed(d)),
    pts: (v) => (v == null ? dash : (v > 0 ? "+" : "") + (100 * v).toFixed(1)),
  };

  // Diverging scale: blue (worse) -> gray (average) -> orange (better). Domain 0-100 percentile.
  const div = d3.scaleLinear().domain([0, 50, 100]).range(["#2a78d6", "#d9d8d3", "#eb6834"]).interpolate(d3.interpolateLab).clamp(true);
  S.pctColor = (p) => (p == null ? "transparent" : div(p));
  // Stuff+ style scale (100 = average, 10 = 1 SD) mapped onto the same diverging ramp.
  S.plusColor = (v) => (v == null ? "transparent" : div(50 + (v - 100) * 2.5));
  S.textOn = (bg) => { const c = d3.lab(bg); return c.l < 62 ? "#fff" : "#16181d"; };

  S.corr = (rows, x, y) => {
    const d = rows.filter((r) => r[x] != null && r[y] != null);
    const mx = d3.mean(d, (r) => r[x]), my = d3.mean(d, (r) => r[y]);
    let sxy = 0, sxx = 0, syy = 0;
    d.forEach((r) => { const a = r[x] - mx, b = r[y] - my; sxy += a * b; sxx += a * a; syy += b * b; });
    return sxy / Math.sqrt(sxx * syy);
  };
  S.params = () => Object.fromEntries(new URLSearchParams(location.search));
  S.el = (tag, attrs = {}, ...kids) => {
    kids = kids.flat().filter((k) => k != null && k !== false);
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else if (v != null) n.setAttribute(k, v);
    }
    kids.flat().forEach((k) => k != null && n.append(k.nodeType ? k : document.createTextNode(k)));
    return n;
  };

  // ---- controls ---------------------------------------------------------
  S.select = (label, options, value, onchange) => {
    const sel = S.el("select", { onchange: (e) => onchange(e.target.value) },
      options.map(([v, t]) => S.el("option", { value: v, ...(String(v) === String(value) ? { selected: "" } : {}) }, t)));
    return S.el("label", {}, label, sel);
  };
  S.search = (label, placeholder, onchange) =>
    S.el("label", {}, label, S.el("input", { type: "search", placeholder, oninput: (e) => onchange(e.target.value.trim().toLowerCase()) }));
  S.number = (label, value, step, onchange) =>
    S.el("label", {}, label, S.el("input", { type: "number", value, step, min: 0, style: "width:6.5rem", onchange: (e) => onchange(+e.target.value) }));
  S.seg = (options, value, onchange) => {
    const wrap = S.el("div", { class: "seg" });
    options.forEach(([v, t]) => {
      const b = S.el("button", { type: "button", class: v === value ? "on" : "" , onclick: () => {
        wrap.querySelectorAll("button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); onchange(v);
      } }, t);
      wrap.append(b);
    });
    return wrap;
  };

  // ---- sortable, paged table ---------------------------------------------
  // columns: [{key, label, fmt, align:'l', heat:'pct'|'plus', html:(row)=>string, sortKey}]
  S.table = (host, columns, { sortKey, desc = true, pageSize = 25 } = {}) => {
    let rows = [], page = 0;
    const state = { sortKey, desc };
    const wrap = S.el("div", { class: "s-table-wrap" });
    const table = S.el("table", { class: "s-table" });
    const pager = S.el("div", { class: "pager" });
    wrap.append(table); host.replaceChildren(wrap, pager);

    function render() {
      const col = columns.find((c) => c.key === state.sortKey);
      const get = (r) => (col && col.sortVal ? col.sortVal(r) : r[state.sortKey]);
      const sorted = rows.slice().sort((a, b) => {
        const x = get(a), y = get(b);
        if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
        return (x < y ? -1 : x > y ? 1 : 0) * (state.desc ? -1 : 1);
      });
      const pages = Math.max(1, Math.ceil(sorted.length / pageSize));
      page = Math.min(page, pages - 1);
      const slice = sorted.slice(page * pageSize, (page + 1) * pageSize);
      const thead = S.el("thead", {}, S.el("tr", {}, columns.map((c) => S.el("th", {
        class: [c.align === "l" ? "l" : "", c.key === state.sortKey ? "sorted" + (state.desc ? "" : " asc") : ""].join(" "),
        title: c.title || "",
        onclick: () => { if (state.sortKey === c.key) state.desc = !state.desc; else { state.sortKey = c.key; state.desc = c.align !== "l"; } render(); },
      }, c.label))));
      const tbody = S.el("tbody", {}, slice.map((r) => S.el("tr", {}, columns.map((c) => {
        const v = r[c.key];
        const td = S.el("td", { class: (c.align === "l" ? "l " : "") + (c.heat ? "heat" : "") });
        if (c.html) td.innerHTML = c.html(r); else td.textContent = c.fmt ? c.fmt(v, r) : (v ?? "—");
        if (c.heat) {
          const hv = c.heatVal ? c.heatVal(r) : v;
          const bg = c.heat === "plus" ? S.plusColor(hv) : S.pctColor(hv);
          if (hv != null) { td.style.background = bg; td.style.color = S.textOn(bg); }
        }
        return td;
      }))));
      table.replaceChildren(thead, tbody);
      pager.replaceChildren(
        S.el("span", {}, `${sorted.length.toLocaleString()} rows`),
        S.el("span", {},
          S.el("button", { type: "button", ...(page === 0 ? { disabled: "" } : {}), onclick: () => { page--; render(); } }, "‹ Prev"),
          ` ${page + 1} / ${pages} `,
          S.el("button", { type: "button", ...(page >= pages - 1 ? { disabled: "" } : {}), onclick: () => { page++; render(); } }, "Next ›")));
    }
    return { update(r) { rows = r; page = 0; render(); } };
  };

  // ---- responsive Plot ----------------------------------------------------
  // One ResizeObserver per host; calling S.plot again on the same host swaps in the new chart
  // function instead of stacking another observer that would redraw a stale chart on resize.
  S.plot = (host, make) => {
    host._sparviMake = make;
    const draw = () => {
      const w = Math.max(280, host.clientWidth || 640);
      host.replaceChildren(host._sparviMake(w));
    };
    draw();
    if (!host._sparviObserver) {
      let t, lastW = host.clientWidth;
      host._sparviObserver = new ResizeObserver(() => {
        if (host.clientWidth === lastW) return;  // ignore height-only changes from our own redraw
        lastW = host.clientWidth;
        clearTimeout(t); t = setTimeout(draw, 120);
      });
      host._sparviObserver.observe(host);
    }
    return { redraw: draw };
  };
  S.plotDefaults = (w) => ({
    width: w, marginLeft: 50, style: { fontFamily: "Inter, sans-serif", fontSize: "12px", background: "transparent", color: "#4d4f55" },
  });

  // ---- percentile bars ----------------------------------------------------
  // items: [{label, pctl, value}]
  S.pbars = (host, items) => {
    const grid = S.el("div", { class: "pbars" });
    items.forEach((it) => {
      const c = S.pctColor(it.pctl);
      const track = S.el("div", { class: "track" });
      if (it.pctl != null) {
        track.append(S.el("div", { class: "fill", style: `width:${it.pctl}%;background:${c}` }),
          S.el("div", { class: "dot", style: `left:${it.pctl}%;background:${c};color:${S.textOn(c)}` }, String(Math.round(it.pctl))));
      }
      grid.append(S.el("div", { class: "lab" }, it.label), track, S.el("div", { class: "val" }, it.value));
    });
    host.replaceChildren(grid);
  };

  S.playerLink = (id, name) => `<a href="/baseball/player.html?id=${id}">${name ?? id}</a>`;
  window.Sparvi = S;
})();
