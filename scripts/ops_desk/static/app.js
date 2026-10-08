(() => {
  const REFRESH_MS = 20000;
  let activeTab = "ops";
  let pollTimer = null;
  let signalPollTimer = null;

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function pct(v, digits = 2) {
    if (v == null || Number.isNaN(Number(v))) return "—";
    return `${(Number(v) * 100).toFixed(digits)}%`;
  }

  function num(v, digits = 2) {
    if (v == null || Number.isNaN(Number(v))) return "—";
    return Number(v).toFixed(digits);
  }

  function codeChips(codes, cls) {
    if (!codes || !codes.length) return `<span class="empty">无</span>`;
    return `<div class="codes">${codes
      .map((c) => `<span class="chip ${cls || ""}">${esc(c)}</span>`)
      .join("")}</div>`;
  }

  function ordersTable(orders) {
    const rows = orders?.rows || [];
    if (!rows.length) return `<p class="empty">暂无调仓清单</p>`;
    const body = rows
      .map((r) => {
        const action = r.action || "";
        return `<tr>
          <td class="action-${esc(action)}">${esc(action)}</td>
          <td class="code">${esc(r.code)}</td>
          <td class="num">${r.ref_price != null && r.ref_price !== "" ? esc(r.ref_price) : "—"}</td>
          <td class="num">${r.shares != null && r.shares !== "" ? esc(r.shares) : "—"}</td>
          <td class="num">${r.amount != null && r.amount !== "" ? esc(r.amount) : "—"}</td>
        </tr>`;
      })
      .join("");
    return `<p class="hint">${esc(orders.file || "")} · 买 ${orders.buy_count || 0} / 卖 ${orders.sell_count || 0} · 买入金额 ${esc(orders.buy_amount || 0)}</p>
      <table>
        <thead><tr><th>动作</th><th>代码</th><th>参考价</th><th>股数</th><th>金额</th></tr></thead>
        <tbody>${body}</tbody>
      </table>`;
  }

  function industryBlock(ind) {
    if (!ind || !ind.n) return `<p class="empty">无持仓行业</p>`;
    const rows = (ind.breakdown || [])
      .map(
        (b) =>
          `<tr><td>${esc(b.industry)}</td><td class="num">${b.count}</td><td class="num">${pct(b.share)}</td></tr>`
      )
      .join("");
    return `<p class="hint">行业表 ${esc(ind.industry_asof || "—")} · Top ${esc(ind.top)} ${pct(ind.top_share)} · HHI ${num(ind.hhi, 3)}</p>
      <table><thead><tr><th>行业</th><th>只数</th><th>占比</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderTrack(track) {
    const h = track.holdings || {};
    const p = track.paper || {};
    return `
      <h2>${esc(track.label)}</h2>
      <p class="hint">${esc(track.live_dir)} · 信号日 ${esc(h.date || p.signal_date || "—")} · 更新 ${esc(track.holdings_mtime || "—")}</p>
      <h3>目标持仓</h3>
      ${codeChips(h.codes)}
      <h3>本期买卖（paper）</h3>
      <div>买 ${codeChips(p.buy, "buy")}</div>
      <div>卖 ${codeChips(p.sell, "sell")}</div>
      <h3>行业集中度</h3>
      ${industryBlock(track.industry)}
      <h3>最新订单表</h3>
      ${ordersTable(track.orders)}
    `;
  }

  function renderOps(snap, signalStatus) {
    const data = snap.data || {};
    const st = signalStatus || {};
    const pills = [
      { k: "日线 data_daily", v: data.daily_last || "无", ok: data.daily_ok },
      { k: "状态 data_ud_new", v: data.ud_last || "无", ok: data.ud_ok },
      { k: "日历 date.pkl", v: data.calendar_last || "无", ok: true },
      { k: "主合成", v: data.composite_last || "无", ok: data.composite_ok },
      { k: "短线合成", v: data.composite_st_last || "无", ok: data.composite_st_ok },
      { k: "行业", v: data.industry_last || "无", ok: data.industry_ok },
      { k: "财务 PIT", v: data.fundamental_last || "无", ok: data.fundamental_ok },
      { k: "估值", v: data.valuation_last || "无", ok: data.valuation_ok },
    ]
      .map(
        (it) =>
          `<div class="pill ${it.ok ? "ok" : "bad"}"><div class="k">${esc(it.k)}</div><div class="v">${esc(it.v)}</div></div>`
      )
      .join("");
    const busy = !!st.running;
    const sigBtns = [
      ["btnSigMain", "刷新主轨信号"],
      ["btnSigSt", "刷新短线信号"],
      ["btnSigBoth", "两个都刷"],
      ["btnPipeMain", "含管线刷新主轨"],
      ["btnPipeSt", "含管线刷新短线"],
    ]
      .map(
        ([id, label]) =>
          `<button type="button" class="${id.startsWith("btnPipe") ? "" : "primary"}" id="${id}" ${busy ? "disabled" : ""}>${label}</button>`
      )
      .join("");
    return `
      <div class="toolbar">
        ${sigBtns}
        <span class="hint" id="sigStatus">${esc(
          st.message || "仅信号=用现有合成分；含管线=先日更合成分再出清单（主轨约 20+ 分钟）"
        )}</span>
      </div>
      <div class="data-bar">${pills}</div>
      <div class="tracks">
        <section class="track">${renderTrack(snap.main)}</section>
        <section class="track">${renderTrack(snap.st)}</section>
      </div>`;
  }

  function metricCards(m, title) {
    const items = [
      ["总收益", pct(m.total_return)],
      ["年化", pct(m.annual_return)],
      ["夏普", num(m.sharpe_ratio)],
      ["最大回撤", pct(m.max_drawdown)],
      ["平均换手", pct(m.avg_turnover)],
      ["总成本", m.total_cost != null ? num(m.total_cost, 1) : "—"],
      ["IC均值", num(m.ic_mean, 4)],
      ["IR", num(m.ir, 3)],
    ];
    return `<div class="card"><h2>${esc(title)}</h2>
      <div class="grid-3">${items
        .map(
          ([k, v]) =>
            `<div class="pill metric"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`
        )
        .join("")}</div></div>`;
  }

  function drawEquity(svg, seriesA, seriesB) {
    const w = svg.clientWidth || 640;
    const h = svg.clientHeight || 260;
    const pad = { t: 16, r: 12, b: 28, l: 44 };
    const all = [...(seriesA || []), ...(seriesB || [])];
    if (!all.length) {
      svg.innerHTML = `<text x="20" y="40" fill="#8b97a8">暂无净值曲线，请先刷新回测</text>`;
      return;
    }
    const dates = (seriesA.length ? seriesA : seriesB).map((p) => p.date);
    const vals = all.map((p) => p.nav);
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const span = maxV - minV || 1;
    const x = (i, n) => pad.l + ((w - pad.l - pad.r) * i) / Math.max(n - 1, 1);
    const y = (v) => pad.t + (1 - (v - minV) / span) * (h - pad.t - pad.b);
    function path(series, color) {
      if (!series.length) return "";
      const d = series
        .map((p, i) => `${i ? "L" : "M"}${x(i, series.length).toFixed(1)},${y(p.nav).toFixed(1)}`)
        .join(" ");
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.8"/>`;
    }
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.innerHTML = `
      <rect width="${w}" height="${h}" fill="#121820"/>
      ${path(seriesA, "#c4a35a")}
      ${path(seriesB, "#6b9bd1")}
      <text x="${pad.l}" y="14" fill="#8b97a8" font-size="11">金=扣费后 蓝=扣费前</text>
      <text x="${pad.l}" y="${h - 8}" fill="#8b97a8" font-size="10">${esc(dates[0] || "")}</text>
      <text x="${w - pad.r}" y="${h - 8}" fill="#8b97a8" font-size="10" text-anchor="end">${esc(dates[dates.length - 1] || "")}</text>
      <text x="8" y="${y(maxV) + 4}" fill="#8b97a8" font-size="10">${num(maxV, 2)}</text>
      <text x="8" y="${y(minV) + 4}" fill="#8b97a8" font-size="10">${num(minV, 2)}</text>
    `;
  }

  function renderBacktestTrack(block, title) {
    if (!block || !block.ok) {
      return `<div class="card"><h2>${esc(title)}</h2><p class="empty">尚无缓存，请点对应刷新按钮</p></div>`;
    }
    const wc = block.with_cost || {};
    const nc = block.no_cost || {};
    const svgId = title.includes("短") ? "equitySvgSt" : "equitySvgMain";
    return `<div class="card">
      <h2>${esc(title)}</h2>
      <p class="hint">${esc(wc.strategy || "")} · Top${esc(wc.top_n)} / 每${esc(wc.rebalance)}日 · ${esc(wc.start)} → ${esc(wc.end)} · ${wc.n_days || 0} 日 · ${esc(block.generated_at || "")}</p>
      <div class="grid-2">
        ${metricCards(wc.metrics || {}, "扣费后")}
        ${metricCards(nc.metrics || {}, "扣费前")}
      </div>
      <svg class="chart-box tall" id="${svgId}" role="img" aria-label="净值"></svg>
    </div>`;
  }

  function renderBacktest(bt, status) {
    const st = status || {};
    const toolbar = `<div class="toolbar">
      <button type="button" class="primary" id="btnRefreshBtMain">刷新主轨回测</button>
      <button type="button" class="primary" id="btnRefreshBtSt">刷新短线回测</button>
      <span class="hint" id="btStatus">${esc(st.message || (bt.ok ? `缓存 ${bt.generated_at}` : bt.message || ""))}</span>
    </div>`;
    if (!bt.ok) {
      return `${toolbar}<p class="empty">${esc(bt.message || "无回测数据")}</p>`;
    }
    const ind = bt.industry_live || {};
    return `${toolbar}
      <div class="grid-2">
        ${renderBacktestTrack(bt.main, "主轨 composite")}
        ${renderBacktestTrack(bt.st, "短线 composite_st")}
      </div>
      <div class="grid-2" style="margin-top:1rem">
        <div class="card"><h2>主轨行业集中度（live）</h2>${industryBlock(ind.main)}</div>
        <div class="card"><h2>短线行业集中度（live）</h2>${industryBlock(ind.st)}</div>
      </div>`;
  }

  function barList(items, valueKey, labelKey = "name") {
    const vals = items.map((x) => Math.abs(Number(x[valueKey]) || 0));
    const max = Math.max(...vals, 1e-9);
    return items
      .map((it) => {
        const v = Number(it[valueKey]) || 0;
        const w = (Math.abs(v) / max) * 100;
        return `<div class="bar-row">
          <div class="name" title="${esc(it[labelKey])}">${esc(it[labelKey])}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${v >= 0 ? "var(--hold)" : "var(--sell)"}"></div></div>
          <div class="num">${valueKey.includes("return") || valueKey.includes("mean") ? (valueKey === "ic_mean" ? num(v, 3) : valueKey.includes("return") ? pct(v) : num(v, 3)) : num(v, 3)}</div>
        </div>`;
      })
      .join("");
  }

  function renderFactors(fac) {
    const rows = (fac.factors || [])
      .map(
        (r) => `<tr>
        <td class="code">${esc(r.name)}</td>
        <td>${esc(r.status || "")}</td>
        <td>${esc(r.health || "")}</td>
        <td class="num">${num(r.ic_mean, 4)}</td>
        <td class="num">${num(r.ic_ir, 3)}</td>
        <td class="num">${pct(r.turnover_mean, 2)}</td>
        <td>${r.in_st ? "Y" : ""}</td>
      </tr>`
      )
      .join("");
    const best = fac.loo_st_best || {};
    const pairs = (fac.corr?.top_pairs || [])
      .slice(0, 10)
      .map(
        (p) =>
          `<tr><td class="code">${esc(p.a)}</td><td class="code">${esc(p.b)}</td><td class="num">${num(p.corr, 3)}</td></tr>`
      )
      .join("");
    const loo = (fac.loo_main?.rows || [])
      .slice(0, 10)
      .map(
        (r) =>
          `<tr><td class="code">${esc(r.dropped || r.tag)}</td><td class="num">${pct(r.total_return)}</td><td class="num">${num(r.sharpe, 2)}</td></tr>`
      )
      .join("");
    return `
      <p class="hint">冻结池 ${fac.universe_n} 只 · 相关窗口 ${(fac.corr && fac.corr.dates) || 0} 日 · asof ${esc(fac.corr?.asof || "—")}</p>
      <div class="grid-2">
        <div class="card">
          <h2>IC_IR 排名</h2>
          ${barList(fac.factors || [], "ic_ir")}
        </div>
        <div class="card">
          <h2>短线 LOO 优胜</h2>
          <p class="hint">去掉 ${esc(best.dropped || "—")} · 收益 ${pct(best.total_return)} · 夏普 ${num(best.sharpe)}</p>
          <h3>主池 LOO（掉谁更好）</h3>
          <table><thead><tr><th>去掉</th><th>收益</th><th>夏普</th></tr></thead><tbody>${loo}</tbody></table>
        </div>
      </div>
      <div class="grid-2" style="margin-top:1rem">
        <div class="card">
          <h2>高相关因子对</h2>
          <table><thead><tr><th>A</th><th>B</th><th>相关</th></tr></thead><tbody>${pairs}</tbody></table>
        </div>
        <div class="card">
          <h2>冻结池明细</h2>
          <table>
            <thead><tr><th>因子</th><th>状态</th><th>健康</th><th>IC</th><th>IR</th><th>换手</th><th>ST</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>`;
  }

  function renderCharts(ch) {
    const scripts = (ch.viz_scripts || [])
      .map((s) => `<option value="${esc(s.id)}">${esc(s.title)}</option>`)
      .join("");
    const imgs = (ch.images || [])
      .map(
        (im) => `<figure>
        <img src="${esc(im.url)}" alt="${esc(im.name)}" loading="lazy" />
        <figcaption>${esc(im.name)} · ${esc(im.mtime)}</figcaption>
      </figure>`
      )
      .join("");
    return `
      <div class="card">
        <h2>内嵌图：扣费前后净值</h2>
        <svg class="chart-box tall" id="chartEquity" role="img" aria-label="净值"></svg>
      </div>
      <div class="grid-2" style="margin-top:1rem">
        <div class="card">
          <h2>冻结池 IC_IR</h2>
          ${barList(ch.ic_bars || [], "ic_ir")}
        </div>
        <div class="card">
          <h2>LOO 去掉后收益</h2>
          ${barList(ch.loo_bars || [], "total_return")}
        </div>
      </div>
      <div class="card" style="margin-top:1rem">
        <h2>course_viz 脚本</h2>
        <div class="toolbar">
          <select id="vizSelect">${scripts}</select>
          <button type="button" class="primary" id="btnRunViz">运行出图</button>
          <span class="hint" id="vizMsg">输出到 outputs/ops_desk/plots</span>
        </div>
        <div class="gallery">${imgs || '<p class="empty">暂无图片，可先跑脚本或刷新回测</p>'}</div>
      </div>`;
  }

  async function fetchJson(url, opts) {
    const res = await fetch(url, { cache: "no-store", ...opts });
    return res.json();
  }

  function pollSignal() {
    if (signalPollTimer) clearInterval(signalPollTimer);
    signalPollTimer = setInterval(async () => {
      const st = await fetchJson("/api/signal/status");
      const el = document.getElementById("sigStatus");
      if (el) el.textContent = st.message || "";
      if (!st.running) {
        clearInterval(signalPollTimer);
        signalPollTimer = null;
        if (activeTab === "ops") loadTab("ops");
      }
    }, 2000);
  }

  async function startSignalRefresh(track, runPipeline) {
    const el = document.getElementById("sigStatus");
    if (el) el.textContent = runPipeline
      ? `含管线刷新（${track}）…`
      : `正在刷新信号（${track}）…`;
    ["btnSigMain", "btnSigSt", "btnSigBoth", "btnPipeMain", "btnPipeSt"].forEach((id) => {
      const b = document.getElementById(id);
      if (b) b.disabled = true;
    });
    await fetchJson("/api/signal/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track, run_pipeline: !!runPipeline }),
    });
    pollSignal();
  }

  async function loadTab(tab) {
    const gen = document.getElementById("generated");
    if (tab === "ops") {
      const [snap, st] = await Promise.all([
        fetchJson("/api/ops"),
        fetchJson("/api/signal/status"),
      ]);
      gen.textContent = `更新 ${snap.generated_at}`;
      document.getElementById("panel-ops").innerHTML = renderOps(snap, st);
      document.getElementById("btnSigMain")?.addEventListener("click", () =>
        startSignalRefresh("main", false)
      );
      document.getElementById("btnSigSt")?.addEventListener("click", () =>
        startSignalRefresh("st", false)
      );
      document.getElementById("btnSigBoth")?.addEventListener("click", () =>
        startSignalRefresh("both", false)
      );
      document.getElementById("btnPipeMain")?.addEventListener("click", () =>
        startSignalRefresh("main", true)
      );
      document.getElementById("btnPipeSt")?.addEventListener("click", () =>
        startSignalRefresh("st", true)
      );
      if (st.running) pollSignal();
      return;
    }
    if (tab === "backtest") {
      const [bt, st] = await Promise.all([
        fetchJson("/api/backtest"),
        fetchJson("/api/backtest/status"),
      ]);
      gen.textContent = bt.generated_at ? `回测缓存 ${bt.generated_at}` : "回测未缓存";
      const panel = document.getElementById("panel-backtest");
      panel.innerHTML = renderBacktest(bt, st);
      const svgMain = document.getElementById("equitySvgMain");
      if (svgMain && bt.main) {
        drawEquity(svgMain, bt.main.with_cost?.equity || [], bt.main.no_cost?.equity || []);
      }
      const svgSt = document.getElementById("equitySvgSt");
      if (svgSt && bt.st) {
        drawEquity(svgSt, bt.st.with_cost?.equity || [], bt.st.no_cost?.equity || []);
      }
      const startBt = async (track) => {
        await fetchJson("/api/backtest/refresh", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ track }),
        });
        const el = document.getElementById("btStatus");
        if (el) el.textContent = `回测运行中（${track}）…`;
        pollBacktest();
      };
      document.getElementById("btnRefreshBtMain")?.addEventListener("click", () => startBt("main"));
      document.getElementById("btnRefreshBtSt")?.addEventListener("click", () => startBt("st"));
      return;
    }
    if (tab === "factors") {
      const fac = await fetchJson("/api/factors");
      gen.textContent = `更新 ${fac.generated_at}`;
      document.getElementById("panel-factors").innerHTML = renderFactors(fac);
      return;
    }
    if (tab === "charts") {
      const ch = await fetchJson("/api/charts");
      gen.textContent = `更新 ${ch.generated_at}`;
      document.getElementById("panel-charts").innerHTML = renderCharts(ch);
      const svg = document.getElementById("chartEquity");
      if (svg) {
        drawEquity(
          svg,
          ch.equity?.with_cost || [],
          ch.equity?.no_cost || []
        );
      }
      document.getElementById("btnRunViz")?.addEventListener("click", async () => {
        const id = document.getElementById("vizSelect").value;
        document.getElementById("vizMsg").textContent = "运行中…";
        const r = await fetchJson("/api/viz/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id }),
        });
        document.getElementById("vizMsg").textContent = r.message || (r.ok ? "完成" : "失败");
        if (r.ok) loadTab("charts");
      });
    }
  }

  function pollBacktest() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      const st = await fetchJson("/api/backtest/status");
      const el = document.getElementById("btStatus");
      if (el) el.textContent = st.message || "";
      if (!st.running) {
        clearInterval(pollTimer);
        pollTimer = null;
        if (activeTab === "backtest") loadTab("backtest");
      }
    }, 3000);
  }

  function switchTab(tab) {
    activeTab = tab;
    document.querySelectorAll(".tab").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === tab);
    });
    document.querySelectorAll(".panel").forEach((p) => {
      p.classList.toggle("active", p.id === `panel-${tab}`);
    });
    loadTab(tab).catch((e) => {
      document.getElementById("generated").textContent = String(e);
    });
  }

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  document.getElementById("refreshBtn").addEventListener("click", () => {
    loadTab(activeTab).catch((e) => {
      document.getElementById("generated").textContent = String(e);
    });
  });

  switchTab("ops");
  setInterval(() => {
    if (activeTab === "ops") loadTab("ops").catch(() => {});
  }, REFRESH_MS);
})();
