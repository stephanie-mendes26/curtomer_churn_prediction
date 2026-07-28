"""
Isola o subgrupo "cliente anual periódico" (pede ~1x/ano, em meses parecidos
entre os anos) dentro de categoria_pedido == baixo -- ver CORREÇÃO DE ESCOPO
em testar_ciclo_esterilizacao.py e CLAUDE.md, seção "Reunião jul/2026".

Critério: circstd (desvio padrão circular, trata dez/jan como vizinhos, não
extremos) do mês-do-pedido, calculado só sobre clientes com pedidos em pelo
menos 2 anos distintos na janela pré-CUTOFF_TREINO (2023-2024 = só 2 anos
disponíveis nessa janela, então o teste é necessariamente 2 pontos por
cliente -- registrar essa limitação de amostra).

Não faz parte do pipeline. Nada persistido em data/processed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import circstd, pearsonr, fisher_exact, norm
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists())
sys.path.insert(0, str(PROJECT_ROOT))

from src import db
from src.config import INICIO, CUTOFF_TREINO, PROCESSED_DIR

VALIDADE_MESES = {"VAPOR": 6, "OXIDO": 24}

# --- 1. Pedidos pré-cutoff ---
inicio_fmt = pd.Timestamp(INICIO).strftime("%Y%m%d")
cutoff_fmt = CUTOFF_TREINO.strftime("%Y%m%d")
sql = f"""
SELECT CLIENTE, DATA_ENTRADA, NRO_PED, TIPO_PED
FROM dbo.PBI_DESEMPENHO_MATERIAL
WHERE DATA_ENTRADA >= '{inicio_fmt}' AND DATA_ENTRADA <= '{cutoff_fmt}';
"""
df_ped = db.get_data(sql)
df_ped["DATA_ENTRADA"] = pd.to_datetime(df_ped["DATA_ENTRADA"])
df_ped["ano"] = df_ped["DATA_ENTRADA"].dt.year
df_ped["mes"] = df_ped["DATA_ENTRADA"].dt.month

# um "pedido" == um NRO_PED distinto, não uma linha de item
pedidos_unicos = df_ped.drop_duplicates(subset=["CLIENTE", "NRO_PED"])[["CLIENTE", "ano", "mes"]]
print(f"Pedidos únicos pré-cutoff: {len(pedidos_unicos):,}, {pedidos_unicos['CLIENTE'].nunique()} clientes")

# --- 2. Por cliente: nº de anos distintos com pedido, total de pedidos, circstd do mês ---
def resume_cliente(g):
    n_anos = g["ano"].nunique()
    n_pedidos = len(g)
    meses0 = (g["mes"] - 1).values  # 0-11 pra circstd
    cstd = circstd(meses0, high=12, low=0) if len(meses0) >= 2 else np.nan
    return pd.Series({"n_anos_distintos": n_anos, "n_pedidos_pre": n_pedidos, "circstd_mes": cstd})

resumo = pedidos_unicos.groupby("CLIENTE").apply(resume_cliente, include_groups=False).reset_index()

# --- 3. Junta com df_model_treino (churn, categoria_pedido) ---
df_treino = pd.read_parquet(PROCESSED_DIR / "df_model_treino.parquet")
df = df_treino[["CLIENTE", "churn", "categoria_pedido", "meses_sem_pedido_pre"]].merge(
    resumo, on="CLIENTE", how="left"
)

baixo = df[df["categoria_pedido"] == "baixo"].copy()
print(f"\nClientes categoria 'baixo' no treino: {len(baixo)}")
print(f"  com pedido em 2 anos distintos (pré-cutoff): {(baixo['n_anos_distintos'] == 2).sum()}")

elegiveis = baixo[baixo["n_anos_distintos"] == 2].copy()
print("\nDistribuição de circstd_mes (meses) entre os elegíveis:")
print(elegiveis["circstd_mes"].describe())

# --- 4. Flag periodico_anual: baixa variância circular no mês do pedido ---
for limiar in [1.0, 1.5, 2.0]:
    n = (elegiveis["circstd_mes"] <= limiar).sum()
    print(f"  circstd_mes <= {limiar} meses: {n} clientes")

# --- 4b. Sensibilidade do achado ao limiar (achado contra-intuitivo, jul/2026:
# periodico_anual CHURNA MAIS que o resto, oposto da hipótese de "falso positivo"
# por ciclo de validade) -- checar se é estável ou artefato de limiar/n pequeno.
# Adicionado depois (mesmo dia, a pedido do usuário): intervalo de confiança de
# Wilson por grupo + teste exato de Fisher pra saber se a diferença é
# estatisticamente significativa ou se pode ser só ruído de amostra pequena.
# Fisher em vez de qui-quadrado porque n é pequeno (18 no limiar mais estrito)
# -- a aproximação do qui-quadrado fica instável com células pequenas.
def wilson_ci(sucessos, n, conf=0.95):
    """Intervalo de confiança de Wilson para uma proporção -- mais estável que
    a aproximação normal quando n é pequeno ou p está perto de 0/1."""
    if n == 0:
        return (np.nan, np.nan)
    p = sucessos / n
    z = norm.ppf(1 - (1 - conf) / 2)
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margem = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0, centro - margem), min(1, centro + margem))

print("\nSensibilidade churn periodico_anual vs. resto, por limiar (com IC 95% de Wilson + Fisher exato):")
for limiar in [1.5, 2.0, 2.5, 3.0]:
    grp = elegiveis["circstd_mes"] <= limiar
    n_in, n_out = grp.sum(), (~grp).sum()
    churn_in_n = elegiveis.loc[grp, "churn"].sum()
    churn_out_n = elegiveis.loc[~grp, "churn"].sum()
    churn_in = churn_in_n / n_in
    churn_out = churn_out_n / n_out
    ci_in = wilson_ci(churn_in_n, n_in)
    ci_out = wilson_ci(churn_out_n, n_out)

    tabela_2x2 = [[churn_in_n, n_in - churn_in_n], [churn_out_n, n_out - churn_out_n]]
    odds_ratio, p_valor = fisher_exact(tabela_2x2)

    sig = "***" if p_valor < 0.01 else "**" if p_valor < 0.05 else "*" if p_valor < 0.10 else "ns"
    print(f"  limiar={limiar}:")
    print(f"    periodico: n={n_in:3d}  churn={churn_in:.1%}  IC95%=[{ci_in[0]:.1%}, {ci_in[1]:.1%}]")
    print(f"    resto:     n={n_out:3d}  churn={churn_out:.1%}  IC95%=[{ci_out[0]:.1%}, {ci_out[1]:.1%}]")
    print(f"    Fisher exato: odds_ratio={odds_ratio:.2f}  p={p_valor:.4f}  ({sig})")

LIMIAR = 1.5
elegiveis["periodico_anual"] = elegiveis["circstd_mes"] <= LIMIAR
print(f"\nUsando limiar={LIMIAR} meses -> periodico_anual: {elegiveis['periodico_anual'].sum()} clientes")

print("\nTaxa de churn:")
print(f"  periodico_anual=True:  {elegiveis.loc[elegiveis['periodico_anual'], 'churn'].mean():.1%} (n={elegiveis['periodico_anual'].sum()})")
print(f"  periodico_anual=False: {elegiveis.loc[~elegiveis['periodico_anual'], 'churn'].mean():.1%} (n={(~elegiveis['periodico_anual']).sum()})")
print(f"  baixo geral (ref):     {baixo['churn'].mean():.1%} (n={len(baixo)})")

# --- 5. Dentro do subgrupo periodico_anual, tipo_ped explica alguma coisa? ---
sub_periodico = elegiveis[elegiveis["periodico_anual"]].copy()
if len(sub_periodico) >= 15:
    tipo_norm = df_ped["TIPO_PED"].astype(str).str.upper()
    df_ped["tipo_esteril"] = np.select(
        [tipo_norm.str.contains("XIDO", na=False), tipo_norm.str.contains("VAPOR", na=False)],
        ["OXIDO", "VAPOR"], default=None,
    )
    feat_esteril = (
        df_ped.dropna(subset=["tipo_esteril"])
        .groupby("CLIENTE")["tipo_esteril"]
        .apply(lambda s: (s == "OXIDO").sum() / len(s))
        .rename("pct_itens_oxido")
        .reset_index()
    )
    sub_periodico = sub_periodico.merge(feat_esteril, on="CLIENTE", how="left")
    sub_periodico["validade_esperada_meses"] = 6 + 18 * sub_periodico["pct_itens_oxido"]
    sub_periodico["razao_recencia_validade"] = sub_periodico["meses_sem_pedido_pre"] / sub_periodico["validade_esperada_meses"]

    print(f"\n=== Pearson + MI contra churn, SÓ dentro de periodico_anual (n={len(sub_periodico)}) ===")
    for col in ["pct_itens_oxido", "razao_recencia_validade", "meses_sem_pedido_pre"]:
        s = sub_periodico.dropna(subset=[col])
        if len(s) < 10 or s["churn"].nunique() < 2:
            print(f"{col:28s}  amostra insuficiente (n={len(s)}, classes={s['churn'].nunique()})")
            continue
        r, _ = pearsonr(s[col], s["churn"])
        mi = mutual_info_classif(s[[col]], s["churn"], random_state=0)[0]
        print(f"{col:28s}  Pearson={r:+.3f}   MI={mi:.3f}   n={len(s)}")
else:
    print(f"\nSubgrupo periodico_anual pequeno demais (n={len(sub_periodico)}) pra um teste confiável.")
