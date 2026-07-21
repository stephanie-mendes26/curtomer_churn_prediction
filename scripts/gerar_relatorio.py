# -*- coding: utf-8 -*-
"""
Gera reports/lista_alertas_churn.html do zero a cada execução — nunca editado
manualmente (foi editar esse arquivo à mão que corrompeu o relatório antes,
ver CLAUDE.md decisão 51). Roda sozinho ou como último passo de
scripts/scoring_mensal.py.

Lê scores_2026.parquet + features_scoring_atual.parquet (gerados por
scoring_mensal.py) — nunca df_model_teste.parquet, que é um snapshot congelado
e reintroduziria o mesmo problema de dado desatualizado que o Passo 2 corrigiu.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists())
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR  # noqa: E402
from src.model_prep import preparar_X  # noqa: E402

OUT_PATH = PROJECT_ROOT / "reports" / "lista_alertas_churn.html"

MAPA_CATEGORIA_LABEL = {0: "Baixo", 1: "Médio", 2: "Alto", 3: "Premium"}


def fmt_brl(v: float) -> str:
    return f"R$ {v:,.0f}".replace(",", ".")


def risk_tier(p: float):
    if p >= 0.7:
        return "critico", "Risco crítico"
    if p >= 0.4:
        return "alto", "Risco alto"
    return "moderado", "Risco moderado"


def phrase_for(feat: str, row: pd.Series):
    if feat == "meses_sem_pedido_pre":
        m = int(row["meses_sem_pedido_pre"])
        return f"não faz pedidos há {m} {'mês' if m == 1 else 'meses'}"
    if feat == "sem_historico_itens":
        return "histórico de itens comprados curto demais para avaliar tendência" if row["sem_historico_itens"] == 1 else None
    if feat == "n_meses_ativos":
        n = int(row["n_meses_ativos"])
        return f"só teve {n} {'mês' if n == 1 else 'meses'} de atividade no período analisado"
    if feat == "intervalo_medio":
        return f"intervalo médio entre compras de {row['intervalo_medio']:.1f} meses"
    if feat == "n_itens_portfolio":
        n = int(row["n_itens_portfolio"])
        return f"portfólio reduzido — só {n} {'item diferente comprado' if n == 1 else 'itens diferentes comprados'} no período"
    if feat == "slope_portfolio_medio":
        return "tendência de queda no volume de itens comprados" if row["slope_portfolio_medio"] < 0 else "tendência de crescimento no volume de itens comprados"
    if feat == "ticket_medio":
        return f"ticket médio de {fmt_brl(row['ticket_medio'])} por pedido"
    if feat == "pct_itens_queda":
        return f"{row['pct_itens_queda'] * 100:.0f}% dos itens comprados em queda de consumo"
    if feat == "total_valor":
        return f"faturamento de {fmt_brl(row['total_valor'])} no período analisado"
    if feat == "media_pedidos_mes":
        return f"média de {row['media_pedidos_mes']:.1f} pedidos por mês"
    if feat == "cv_pedidos":
        return "frequência de pedidos irregular mês a mês"
    if feat == "total_pedidos":
        n = int(row["total_pedidos"])
        return f"{n} {'pedido' if n == 1 else 'pedidos'} no total durante o período"
    if feat == "n_intervalos":
        n = int(row["n_intervalos"])
        return "histórico curto demais para calcular regularidade de compra" if n == 0 else f"apenas {n} intervalos de compra registrados"
    if feat == "DIASINADIMPLENTE":
        d = row["DIASINADIMPLENTE"]
        if not d or d <= 0:
            return None
        if d >= 999:
            return "está com atraso crônico no pagamento (limite de registro do sistema)"
        return f"{int(d)} dias em atraso no pagamento"
    if feat == "max_intervalo":
        return f"já ficou até {row['max_intervalo']:.1f} meses sem comprar em algum momento"
    if feat == "itens_por_pedido":
        return f"média de {row['itens_por_pedido']:.1f} itens por pedido"
    if feat == "sem_historico_cadencia":
        return "histórico de frequência de compra curto demais para avaliar regularidade" if row["sem_historico_cadencia"] == 1 else None
    return None


def montar_motivo(top_features_shap: str, feat_row: pd.Series) -> str:
    feats = [f.strip() for f in top_features_shap.split(",") if f.strip()]
    frases = []
    for f in feats:
        p = phrase_for(f, feat_row)
        if p and p not in frases:
            frases.append(p)
        if len(frases) == 2:
            break
    if not frases:
        return "Combinação de sinais de comportamento de compra."
    texto = "Este cliente " + " e ".join(frases) + "."
    return texto[0].upper() + texto[1:]


def calcular_auc_teste(modelo, feature_cols) -> float:
    """AUC-ROC do modelo no conjunto de teste — recalculada na hora, nunca hardcoded."""
    df_teste = pd.read_parquet(PROCESSED_DIR / "df_model_teste.parquet")
    X_teste = preparar_X(df_teste, feature_cols)
    p_teste = modelo.predict_proba(X_teste)[:, 1]
    return roc_auc_score(df_teste["churn"], p_teste)


def montar_dados():
    with open(PROCESSED_DIR / "modelo_churn.pkl", "rb") as f:
        artefatos = pickle.load(f)
    modelo_nome = artefatos["modelo_nome"]
    auc_roc = calcular_auc_teste(artefatos["modelo"], artefatos["features"])

    scores = pd.read_parquet(PROCESSED_DIR / "scores_2026.parquet")
    feats = pd.read_parquet(PROCESSED_DIR / "features_scoring_atual.parquet")

    n_total = len(scores)
    sel = (
        scores[scores["entra_na_lista"]]
        .sort_values("valor_em_risco", ascending=False)
        .merge(feats, on="CLIENTE", how="left", suffixes=("", "_feat"))
    )

    rows = []
    for _, r in sel.iterrows():
        categoria = r["categoria_cliente"]
        if categoria in MAPA_CATEGORIA_LABEL:
            categoria = MAPA_CATEGORIA_LABEL[categoria]
        elif isinstance(categoria, str):
            categoria = categoria.capitalize()

        tier_key, tier_label = risk_tier(r["p_churn_calibrada"])
        motivo = montar_motivo(r["top_features_shap"], r)

        rows.append({
            "cliente": int(r["CLIENTE"]),
            "nome": r["NOME"].strip(),
            "categoria": categoria,
            "p": round(float(r["p_churn_calibrada"]) * 100, 1),
            "tier_key": tier_key,
            "tier_label": tier_label,
            "receita_anual": round(float(r["receita_anual"]), 2),
            "valor_em_risco": round(float(r["valor_em_risco"]), 2),
            "historico_insuficiente": bool(r["historico_insuficiente"]),
            "motivo": motivo,
        })

    n_clientes = len(rows)
    summary = {
        "n_clientes": n_clientes,
        "n_total": n_total,
        "receita_total": round(sum(r["receita_anual"] for r in rows), 2),
        "valor_risco_total": round(sum(r["valor_em_risco"] for r in rows), 2),
        "risco_medio": round(sum(r["p"] for r in rows) / n_clientes, 1) if n_clientes else 0,
        "auc_roc": round(auc_roc * 100, 1),
        "piso_risco": 11,
        "n_capacidade": 25,
        "modelo_nome": modelo_nome,
    }
    return {"rows": rows, "summary": summary}


TEMPLATE = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Lista de Alertas de Churn</title>
<style>
  :root {
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page-plane:     #f9f9f7;
    --card-surface:   #ffffff;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --muted:          #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --brand:          #2d6a4f;
    --brand-soft:     #d8f3dc;
    --status-critico:  #d03b3b;
    --status-alto:     #ec835a;
    --status-moderado: #fab219;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page-plane:     #0d0d0d;
      --card-surface:   #202020;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --muted:          #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --border:         rgba(255,255,255,0.10);
      --brand:          #74c69d;
      --brand-soft:     #1b4332;
      --status-critico:  #e66767;
      --status-alto:     #ec835a;
      --status-moderado: #fab219;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --card-surface:   #202020;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --muted:          #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --brand:          #74c69d;
    --brand-soft:     #1b4332;
    --status-critico:  #e66767;
    --status-alto:     #ec835a;
    --status-moderado: #fab219;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--page-plane);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .report-root { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }

  header.report-header { margin-bottom: 28px; }
  header.report-header h1 { font-size: 24px; margin: 0 0 4px; }
  header.report-header p.subtitle { color: var(--text-secondary); margin: 0; font-size: 14px; }
  header.report-header .datebadge {
    display: inline-block; margin-top: 10px; font-size: 12px; color: var(--muted);
    border: 1px solid var(--border); border-radius: 999px; padding: 4px 12px;
  }

  .card {
    background: var(--card-surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 24px;
  }

  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .stat-tile {
    background: var(--card-surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 18px;
  }
  .stat-tile .label { font-size: 12.5px; color: var(--text-secondary); margin-bottom: 6px; }
  .stat-tile .value { font-size: 26px; font-weight: 600; }
  .stat-tile .value.brand { color: var(--brand); }
  @media (max-width: 720px) { .stats { grid-template-columns: repeat(2, 1fr); } }

  h2 { font-size: 16px; margin: 0 0 4px; }
  .section-sub { font-size: 13px; color: var(--text-secondary); margin: 0 0 16px; }

  .legend { display: flex; gap: 18px; margin-bottom: 14px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--text-secondary); }
  .legend-swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }

  #chart-wrap { position: relative; }
  #bar-chart { display: block; width: 100%; height: auto; overflow: visible; }
  .bar-row-label { font-size: 12px; fill: var(--text-secondary); }
  .bar-value-label { font-size: 12px; fill: var(--text-primary); font-weight: 600; }
  .axis-tick { font-size: 11px; fill: var(--muted); }
  .gridline { stroke: var(--gridline); stroke-width: 1; }
  .bar-mark { cursor: pointer; transition: filter .1s ease; }
  .bar-mark:hover, .bar-mark:focus { filter: brightness(1.12); outline: none; }
  .bar-hit { fill: transparent; cursor: pointer; }

  .tooltip {
    position: absolute; pointer-events: none; z-index: 5;
    background: var(--card-surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 12px; font-size: 12.5px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12); min-width: 180px;
    opacity: 0; transition: opacity .08s ease;
  }
  .tooltip.visible { opacity: 1; }
  .tooltip .t-name { color: var(--text-secondary); margin-bottom: 4px; }
  .tooltip .t-value { font-size: 16px; font-weight: 700; margin-bottom: 4px; }
  .tooltip .t-row { display: flex; justify-content: space-between; gap: 12px; color: var(--text-secondary); }
  .tooltip .t-row b { color: var(--text-primary); font-variant-numeric: tabular-nums; }

  table.alert-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.alert-table th {
    text-align: left; font-weight: 600; color: var(--text-secondary);
    font-size: 11.5px; text-transform: uppercase; letter-spacing: .02em;
    padding: 8px 10px; border-bottom: 1px solid var(--gridline);
  }
  table.alert-table td { padding: 10px; border-bottom: 1px solid var(--gridline); vertical-align: top; }
  table.alert-table tr:last-child td { border-bottom: none; }
  table.alert-table td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  table.alert-table td.motivo { color: var(--text-secondary); max-width: 280px; }
  table.alert-table td.nome { font-weight: 500; }

  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12.5px; font-weight: 600; color: var(--text-primary); white-space: nowrap;
  }
  .pill .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }

  .footnote { font-size: 12px; color: var(--muted); margin-top: 10px; }

  .methodology { font-size: 13px; color: var(--text-secondary); line-height: 1.6; }
  .methodology p { margin: 0 0 10px; }
  .methodology strong { color: var(--text-primary); }

  .table-scroll { overflow-x: auto; }
</style>
</head>
<body>
<div class="report-root">

  <header class="report-header">
    <h1>Lista de Alertas de Churn</h1>
    <p class="subtitle">Clientes priorizados para ação de retenção, ordenados por valor em risco</p>
    <span class="datebadge" id="generated-badge"></span>
  </header>

  <section class="stats" id="stats"></section>

  <section class="card">
    <h2>Top 15 — maior valor em risco</h2>
    <p class="section-sub">Barra = valor em risco (R$). Cor = nível de risco calibrado.</p>
    <div class="legend" id="chart-legend"></div>
    <div id="chart-wrap">
      <svg id="bar-chart"></svg>
      <div class="tooltip" id="tooltip"></div>
    </div>
  </section>

  <section class="card">
    <h2 id="table-title">Lista completa</h2>
    <p class="section-sub">Ordenada por valor em risco. "Motivo principal" resume os sinais que mais pesaram no score de cada cliente.</p>
    <div class="table-scroll">
      <table class="alert-table">
        <thead>
          <tr>
            <th>#</th><th>Empresa</th><th>Categoria</th><th>Risco</th>
            <th>Receita anual</th><th>Valor em risco</th><th>Motivo principal</th>
          </tr>
        </thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>
    <p class="footnote">† Histórico de compra curto ou parcial — o score deve ser interpretado com mais cautela (ver metodologia abaixo).</p>
  </section>

  <section class="card methodology">
    <h2>Como interpretar</h2>
    <p><strong>Como funciona:</strong> o modelo analisa o padrão de compra de cada cliente — frequência de pedidos, tempo sem comprar, tendência de queda no consumo, atrasos de pagamento — e estima a chance de o cliente parar de comprar nos próximos meses.</p>
    <p id="precisao-text"><strong>Precisão do modelo:</strong></p>
    <p id="montagem-text"><strong>Como esta lista foi montada:</strong></p>
    <p><strong>Sobre o "motivo principal":</strong> são os fatores que mais pesaram para o modelo considerar aquele cliente arriscado — não é uma causa comprovada, é o padrão de comportamento mais parecido com o de clientes que já pararam de comprar no passado.</p>
  </section>

</div>

<script id="report-data" type="application/json">
__DATA_JSON__
</script>
<script>
(function () {
  const payload = JSON.parse(document.getElementById("report-data").textContent);
  const rows = payload.rows;
  const summary = payload.summary;

  const TIER_COLOR = {
    critico:  "var(--status-critico)",
    alto:     "var(--status-alto)",
    moderado: "var(--status-moderado)",
  };
  const TIER_ORDER = ["critico", "alto", "moderado"];
  const TIER_LABEL = { critico: "Risco crítico (≥70%)", alto: "Risco alto (40–70%)", moderado: "Risco moderado (11–40%)" };

  function fmtBRL(v) {
    return "R$ " + Math.round(v).toLocaleString("pt-BR");
  }
  function fmtCompactBRL(v) {
    if (v >= 1000) return "R$ " + Math.round(v / 1000).toLocaleString("pt-BR") + " mil";
    return fmtBRL(v);
  }
  function truncate(s, n) {
    return s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s;
  }

  // ---- badge de geração + textos dinâmicos ----
  document.getElementById("generated-badge").textContent =
    "Gerado " + new Date().toLocaleDateString("pt-BR", { day: "2-digit", month: "long", year: "numeric" }) +
    " · modelo " + (summary.modelo_nome || "oficial") + " · AUC-ROC " + summary.auc_roc.toFixed(1) + "%";

  document.getElementById("table-title").textContent = "Lista completa — " + summary.n_clientes + " clientes";

  document.getElementById("precisao-text").innerHTML =
    "<strong>Precisão do modelo:</strong> em " + summary.auc_roc.toFixed(1).replace(".", ",") +
    "% das vezes, ao comparar dois clientes aleatórios, o modelo classifica corretamente qual dos dois " +
    "tem mais chance de sair (AUC-ROC). O percentual de risco de cada cliente já foi calibrado — ou seja, " +
    "se o modelo diz 80%, isso reflete de fato uma chance próxima de 80% com base no comportamento " +
    "histórico de clientes parecidos.";

  document.getElementById("montagem-text").innerHTML =
    "<strong>Como esta lista foi montada:</strong> dos " + summary.n_total + " clientes avaliados neste " +
    "ciclo, só entram no cálculo os que passaram do piso mínimo de risco (" + summary.piso_risco +
    "%). Todos são ordenados por <em>valor em risco</em> (risco × receita anual do próprio cliente — nunca " +
    "uma média de categoria) e os " + summary.n_capacidade + " primeiros aparecem aqui, de acordo com a " +
    "capacidade atual da equipe de retenção. Esse número é provisório e deve ser ajustado com o time comercial.";

  // ---- stat tiles ----
  const stats = [
    { label: "Clientes priorizados", value: String(summary.n_clientes), sub: "de " + summary.n_total + " avaliados este ciclo" },
    { label: "Valor total em risco", value: fmtCompactBRL(summary.valor_risco_total), sub: "soma de risco × receita", brand: true },
    { label: "Receita anual sob estes clientes", value: fmtCompactBRL(summary.receita_total), sub: "receita individual, não média de categoria" },
    { label: "Risco médio da lista", value: summary.risco_medio.toFixed(1).replace(".", ",") + "%", sub: "probabilidade calibrada" },
  ];
  const statsEl = document.getElementById("stats");
  stats.forEach(function (s) {
    const tile = document.createElement("div");
    tile.className = "stat-tile";
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = s.label;
    const value = document.createElement("div");
    value.className = "value" + (s.brand ? " brand" : "");
    value.textContent = s.value;
    const sub = document.createElement("div");
    sub.className = "label";
    sub.style.marginTop = "4px";
    sub.textContent = s.sub;
    tile.appendChild(label);
    tile.appendChild(value);
    tile.appendChild(sub);
    statsEl.appendChild(tile);
  });

  // ---- legenda do gráfico ----
  const legendEl = document.getElementById("chart-legend");
  TIER_ORDER.forEach(function (k) {
    const item = document.createElement("div");
    item.className = "legend-item";
    const sw = document.createElement("span");
    sw.className = "legend-swatch";
    sw.style.background = TIER_COLOR[k];
    const label = document.createElement("span");
    label.textContent = TIER_LABEL[k];
    item.appendChild(sw);
    item.appendChild(label);
    legendEl.appendChild(item);
  });

  // ---- gráfico de barras horizontais (top 15) ----
  const top15 = rows.slice(0, 15);
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("bar-chart");
  const tooltip = document.getElementById("tooltip");

  const labelW = 200, rightPad = 16, topPad = 26, rowH = 20, rowGap = 10;
  const chartW = 720 - labelW - rightPad;
  const plotH = top15.length * (rowH + rowGap);
  const totalW = 720, totalH = topPad + plotH + 6;
  svg.setAttribute("viewBox", "0 0 " + totalW + " " + totalH);

  const maxVal = Math.max.apply(null, top15.map(function (r) { return r.valor_em_risco; }));
  const niceMax = Math.ceil(maxVal / 10000) * 10000;
  const xScale = function (v) { return (v / niceMax) * chartW; };

  // gridlines + ticks
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const tv = (niceMax / ticks) * i;
    const x = labelW + xScale(tv);
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", x); line.setAttribute("x2", x);
    line.setAttribute("y1", topPad - 8); line.setAttribute("y2", topPad + plotH);
    line.setAttribute("class", "gridline");
    svg.appendChild(line);

    const text = document.createElementNS(svgNS, "text");
    text.setAttribute("x", x); text.setAttribute("y", topPad - 12);
    text.setAttribute("class", "axis-tick");
    text.setAttribute("text-anchor", i === 0 ? "start" : "middle");
    text.textContent = fmtCompactBRL(tv);
    svg.appendChild(text);
  }

  top15.forEach(function (r, i) {
    const y = topPad + i * (rowH + rowGap);
    const w = Math.max(xScale(r.valor_em_risco), 2);
    const color = TIER_COLOR[r.tier_key];

    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", labelW - 10);
    label.setAttribute("y", y + rowH * 0.72);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "bar-row-label");
    label.textContent = truncate(r.nome, 26);
    svg.appendChild(label);

    const bar = document.createElementNS(svgNS, "rect");
    bar.setAttribute("x", labelW);
    bar.setAttribute("y", y);
    bar.setAttribute("width", w);
    bar.setAttribute("height", rowH);
    bar.setAttribute("rx", 4);
    bar.setAttribute("fill", color);
    bar.setAttribute("class", "bar-mark");
    svg.appendChild(bar);

    if (w > 6) {
      const squareOff = document.createElementNS(svgNS, "rect");
      squareOff.setAttribute("x", labelW);
      squareOff.setAttribute("y", y);
      squareOff.setAttribute("width", 4);
      squareOff.setAttribute("height", rowH);
      squareOff.setAttribute("fill", color);
      svg.appendChild(squareOff);
    }

    if (i === 0) {
      const vlabel = document.createElementNS(svgNS, "text");
      vlabel.setAttribute("x", labelW + w + 8);
      vlabel.setAttribute("y", y + rowH * 0.72);
      vlabel.setAttribute("class", "bar-value-label");
      vlabel.textContent = fmtCompactBRL(r.valor_em_risco);
      svg.appendChild(vlabel);
    }

    const hit = document.createElementNS(svgNS, "rect");
    hit.setAttribute("x", labelW);
    hit.setAttribute("y", y - rowGap / 2);
    hit.setAttribute("width", chartW + rightPad);
    hit.setAttribute("height", rowH + rowGap);
    hit.setAttribute("class", "bar-hit");
    hit.setAttribute("tabindex", "0");
    hit.setAttribute("role", "button");
    hit.setAttribute("aria-label", r.nome + ", " + r.tier_label + ", valor em risco " + fmtBRL(r.valor_em_risco));

    function showTip(evt) {
      bar.style.filter = "brightness(1.12)";
      tooltip.innerHTML = "";
      const name = document.createElement("div");
      name.className = "t-name";
      name.textContent = r.nome;
      const val = document.createElement("div");
      val.className = "t-value";
      val.textContent = fmtBRL(r.valor_em_risco);
      const rowRisco = document.createElement("div");
      rowRisco.className = "t-row";
      const spanRisco = document.createElement("span"); spanRisco.textContent = "Risco calibrado";
      const b1 = document.createElement("b"); b1.textContent = r.p.toFixed(1).replace(".", ",") + "%";
      rowRisco.appendChild(spanRisco); rowRisco.appendChild(b1);
      const rowRec = document.createElement("div");
      rowRec.className = "t-row";
      const spanRec = document.createElement("span"); spanRec.textContent = "Receita anual";
      const b2 = document.createElement("b"); b2.textContent = fmtBRL(r.receita_anual);
      rowRec.appendChild(spanRec); rowRec.appendChild(b2);
      tooltip.appendChild(name);
      tooltip.appendChild(val);
      tooltip.appendChild(rowRisco);
      tooltip.appendChild(rowRec);
      tooltip.classList.add("visible");
      const rect = svg.getBoundingClientRect();
      const wrapRect = svg.parentElement.getBoundingClientRect();
      const px = evt && evt.clientX !== undefined ? evt.clientX - wrapRect.left : (rect.left - wrapRect.left) + labelW + w / 2;
      const py = (y / totalH) * rect.height;
      tooltip.style.left = Math.min(px + 12, wrapRect.width - 200) + "px";
      tooltip.style.top = py + "px";
    }
    function hideTip() {
      bar.style.filter = "";
      tooltip.classList.remove("visible");
    }
    hit.addEventListener("pointermove", showTip);
    hit.addEventListener("pointerenter", showTip);
    hit.addEventListener("pointerleave", hideTip);
    hit.addEventListener("focus", showTip);
    hit.addEventListener("blur", hideTip);
    svg.appendChild(hit);
  });

  // ---- tabela completa ----
  const tbody = document.getElementById("table-body");
  rows.forEach(function (r, i) {
    const tr = document.createElement("tr");

    const tdIdx = document.createElement("td"); tdIdx.textContent = String(i + 1);
    const tdNome = document.createElement("td"); tdNome.className = "nome";
    tdNome.textContent = r.nome + (r.historico_insuficiente ? " †" : "");
    const tdCat = document.createElement("td"); tdCat.textContent = r.categoria;
    const tdRisco = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = "pill";
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = TIER_COLOR[r.tier_key];
    const pctText = document.createElement("span");
    pctText.textContent = r.p.toFixed(1).replace(".", ",") + "%";
    pill.appendChild(dot);
    pill.appendChild(pctText);
    tdRisco.appendChild(pill);
    const tdRec = document.createElement("td"); tdRec.className = "num"; tdRec.textContent = fmtBRL(r.receita_anual);
    const tdVal = document.createElement("td"); tdVal.className = "num"; tdVal.textContent = fmtBRL(r.valor_em_risco);
    const tdMotivo = document.createElement("td"); tdMotivo.className = "motivo"; tdMotivo.textContent = r.motivo;

    tr.appendChild(tdIdx); tr.appendChild(tdNome); tr.appendChild(tdCat);
    tr.appendChild(tdRisco); tr.appendChild(tdRec); tr.appendChild(tdVal); tr.appendChild(tdMotivo);
    tbody.appendChild(tr);
  });
})();
</script>
</body>
</html>
"""


def gerar_relatorio():
    dados = montar_dados()
    data_json = json.dumps(dados, ensure_ascii=False)
    html = TEMPLATE.replace("__DATA_JSON__", data_json)

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Relatório gerado: {OUT_PATH}")
    print(f"  {dados['summary']['n_clientes']} clientes na lista, de {dados['summary']['n_total']} avaliados")
    print(f"  Modelo: {dados['summary']['modelo_nome']}  |  AUC-ROC: {dados['summary']['auc_roc']:.1f}%")


if __name__ == "__main__":
    gerar_relatorio()
