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
    // AP-style date from YYYY-MM-DD, matching the prose: "Nov. 17", "June 5", "Sept. 30".
    apDate: (iso) => {
      if (!iso) return dash;
      const [, m, d] = iso.split("-").map(Number);
      return ["Jan.", "Feb.", "March", "April", "May", "June", "July", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."][m - 1] + " " + d;
    },
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
  // Pass the current value when controls are rebuilt, or the box shows empty while the filter still applies.
  S.search = (label, placeholder, onchange, value = "") =>
    S.el("label", {}, label, S.el("input", { type: "search", placeholder, value: value || null, oninput: (e) => onchange(e.target.value.trim().toLowerCase()) }));
  // value null = empty box showing `placeholder`; clearing the box calls onchange(null) ("use the default").
  S.number = (label, value, step, onchange, placeholder) =>
    S.el("label", {}, label, S.el("input", {
      type: "number", value: value ?? null, step, min: 0, placeholder, style: "width:6.5rem",
      onchange: (e) => onchange(e.target.value === "" ? null : +e.target.value),
    }));
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

  // ---- pitcher roles -------------------------------------------------------
  // Segmented All · Starters · Relievers. Swing pitchers always show under All, and get their own
  // button only when the current filters leave some (opts.swing > 0) or Swing is already selected.
  S.ROLES = { all: "All", SP: "Starters", RP: "Relievers", Swing: "Swing" };
  S.roleFilter = (value, onchange, opts = {}) => {
    const keys = ["all", "SP", "RP"].concat((opts.swing > 0 || value === "Swing") ? ["Swing"] : []);
    return S.el("label", {}, "Role", S.seg(keys.map((k) => [k, S.ROLES[k]]), value, onchange));
  };
  // Role-aware pitch minimum: an explicit override applies to everyone; otherwise each pitcher is held
  // to his own role's minimum from meta.qualifiers[level][role], prorated by `factor` (the share of
  // the schedule played by the selected date; see S.prorate) and never below 25.
  const QFLOOR = 25;
  const qmin = (q, factor) => Math.max(QFLOOR, Math.round(q * factor));
  S.qualifies = (quals, level, override, pitches, role, factor = 1) =>
    pitches != null && pitches >= (override ?? qmin(quals[level][role] ?? quals[level].SP, factor));
  S.qualifierText = (quals, level, override, role, factor = 1) => {
    if (override != null) return `${override}+ pitches`;
    const q = quals[level], pro = factor < 1 ? ", prorated to games played" : "";
    if (role !== "all") return `${qmin(q[role], factor)}+ pitches (${S.ROLES[role].toLowerCase()} minimum${pro})`;
    return `${qmin(q.SP, factor)}+ pitches for starters, ${qmin(q.RP, factor)}+ for relievers and swingmen${pro}`;
  };
  // Share of the full schedule played in a window as of snapshot i (1 at season's end).
  S.prorate = (meta, level, i, win = "std") => {
    const c = meta.snapshots[level], g = (win === "std" ? c.team_games : c.team_games_last)[i];
    return Math.min(1, g / c.full_season_games);
  };

  // ---- snapshots + date slider --------------------------------------------------
  // Snapshot files are columnar: {dates, fields, players: [{id, name, series: {std|last30: {field: [..]}}}]}.
  S.WINDOWS = { std: "Season to date", last30: "Last 30 days" };
  S.snapIndex = (dates, iso) => { let i = -1; dates.forEach((d, k) => { if (d <= iso) i = k; }); return i; };
  // Rows at snapshot i: one object per player with every field's value on that date. A player is
  // included only if `present` (a field name) has a value then, i.e. he had appeared by that date.
  S.snapRows = (file, win, i, present, extra = {}) => {
    const out = [];
    for (const p of file.players || file.pitches) {
      const s = win ? p.series[win] : p.series;
      if (!s || s[present] == null || s[present][i] == null) continue;
      const r = { id: p.id, name: p.name, ...(p.static || {}), ...(p.pt ? { pt: p.pt, group: p.group } : {}) };
      for (const f of file.fields) r[f] = s[f] ? s[f][i] : null;
      out.push(Object.assign(r, extra[p.id] || {}));
    }
    return out;
  };
  // League snapshot files are split by window; the trailing-window file loads only when asked for.
  S.loadSnap = (stem, win = "std") => S.load(`/data/baseball/${stem}${win === "last30" ? "_last30" : ""}.json`);
  S.dateLabel = (iso) => new Date(iso + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" });
  S.asOfMark = (iso) => Plot.ruleX([new Date(iso + "T12:00:00")], { stroke: "#0f5257", strokeWidth: 1.5, strokeDasharray: "4,3" });
  // Range input over snapshot dates (or NFL weeks) with step buttons and play. Keyboard: the slider
  // takes arrow keys/Home/End natively; buttons are real buttons. Returns {el, set(i)}; onchange(i) on
  // every move. opts.label(i) sets the readout (default "Through <date>"), opts.aria the slider's name.
  S.dateSlider = (dates, value, onchange, opts = {}) => {
    let i = Math.max(0, Math.min(dates.length - 1, value)), timer = null;
    const label = S.el("span", { class: "ds-label", "aria-live": "polite" });
    const range = S.el("input", { type: "range", min: 0, max: dates.length - 1, step: 1, value: i,
      "aria-label": opts.aria || "Snapshot date", oninput: (e) => go(+e.target.value) });
    const btn = (txt, aria, fn) => S.el("button", { type: "button", class: "ds-btn", "aria-label": aria, onclick: fn }, txt);
    const prev = btn("◀", "Previous week", () => { stop(); go(i - 1); });
    const next = btn("▶", "Next week", () => { stop(); go(i + 1); });
    const play = btn("▶ Play", "Play through the season", () => (timer ? stop() : start()));
    function render() {
      range.value = i; label.textContent = opts.label ? opts.label(i) : `Through ${S.dateLabel(dates[i])}`;
      range.setAttribute("aria-valuetext", label.textContent);
      prev.disabled = i === 0; next.disabled = i === dates.length - 1;
    }
    function go(k) { k = Math.max(0, Math.min(dates.length - 1, k)); if (k === i) { render(); return; } i = k; render(); onchange(i); }
    function stop() { if (timer) { clearInterval(timer); timer = null; } play.textContent = "▶ Play"; play.setAttribute("aria-pressed", "false"); }
    function start() {
      if (i === dates.length - 1) go(0);
      play.textContent = "❚❚ Pause"; play.setAttribute("aria-pressed", "true");
      timer = setInterval(() => { if (i >= dates.length - 1) stop(); else go(i + 1); }, 600);
    }
    render();
    const el = S.el("div", { class: "date-slider", role: "group", "aria-label": "Date" }, label, S.el("div", { class: "ds-row" }, prev, range, next, play));
    return { el, set: (k) => { stop(); go(k); }, get: () => i, stop };
  };
  S.roleTag = (role, g, gs) => (role ? `${role} · ${g} G, ${gs} GS` : "");

  // ---- URL state --------------------------------------------------------------
  // Shareable views: read ?key=value, and write back with replaceState (no history spam).
  // Values equal to the view's default are dropped so URLs stay short.
  S.urlState = {
    get: (key, fallback) => new URLSearchParams(location.search).get(key) ?? fallback,
    set: (obj, defaults = {}) => {
      const u = new URL(location.href);
      for (const [k, v] of Object.entries(obj)) {
        if (v == null || v === "" || v === defaults[k]) u.searchParams.delete(k); else u.searchParams.set(k, v);
      }
      history.replaceState(null, "", u.pathname + (u.searchParams.toString() ? "?" + u.searchParams : "") + u.hash);
    },
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
    // Screen readers: the chart is a figure described by the first caption after it in its explorer
    // (captions are rewritten as controls change, so the description stays current).
    if (!host.hasAttribute("role")) {
      host.setAttribute("role", "figure");
      const box = host.closest(".explorer") || host.parentElement;
      const cap = [...box.querySelectorAll(".caption[id]")]
        .find((c) => host.compareDocumentPosition(c) & Node.DOCUMENT_POSITION_FOLLOWING);
      if (cap) host.setAttribute("aria-describedby", cap.id);
      else host.setAttribute("aria-label", "Chart");
    }
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
  // ---- axes that stay put while a date/week slider moves --------------------------------------------
  // A chart behind a slider must keep one scale for the whole season, or points look like they move
  // when only the axis moved. S.stableDomain pools a variable over *every* date of the current view and
  // takes the 2nd-98th percentile (early weeks' tiny samples would otherwise stretch it), padded.
  // S.onScale pins anything beyond the domain to the edge (_x/_y) and marks it (_off) so the chart can
  // draw it as an open circle; tooltips keep the true values.
  S.stableDomain = (vals, { q = [0.02, 0.98], pad = 0.08, include = [] } = {}) => {
    const v = vals.filter(x => x != null && Number.isFinite(x)).sort(d3.ascending);
    if (!v.length) return undefined;
    let lo = d3.quantileSorted(v, q[0]), hi = d3.quantileSorted(v, q[1]);
    for (const x of include) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
    const p = (hi - lo) * pad || Math.abs(hi) * 0.1 || 1;
    return [lo - p, hi + p];
  };
  S.onScale = (rows, xk, yk, xd, yd) => rows.map(d => {
    const cx = xd ? Math.max(xd[0], Math.min(xd[1], d[xk])) : d[xk], cy = yd ? Math.max(yd[0], Math.min(yd[1], d[yk])) : d[yk];
    return { ...d, _x: cx, _y: cy, _off: cx !== d[xk] || cy !== d[yk] };
  });
  // The marks for pinned points: open circles at the edge, so nothing silently drops off the chart.
  S.offScaleDots = (rows, opts = {}) => Plot.dot(rows.filter(d => d._off), { x: "_x", y: "_y", r: 4, stroke: "#8a8b8f", fill: "none", clip: false, ...opts });

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
  // NFL team pages are static, one per team: /nfl/teams/kc/ (query string carries the view state).
  S.nflTeamHref = (id, qs = "") => `/nfl/teams/${String(id).toLowerCase()}/${qs ? "?" + qs : ""}`;
  S.nflTeamLink = (id, label, qs = "") => `<a href="${S.nflTeamHref(id, qs)}">${label ?? id}</a>`;
  // Rank percentile (0-100) of each row's value within rows, higher = better unless lowerBetter.
  S.rankPct = (rows, key, lowerBetter = false) => {
    const v = rows.map((r) => r[key]).filter((x) => x != null).sort(d3.ascending), n = v.length;
    return (x) => (x == null || n < 2 ? null : (() => {
      const below = d3.bisectLeft(v, x), above = n - d3.bisectRight(v, x);
      const p = 100 * (below + (n - below - above - 1) / 2) / (n - 1);
      return lowerBetter ? 100 - p : p;
    })());
  };
  window.Sparvi = S;
})();
