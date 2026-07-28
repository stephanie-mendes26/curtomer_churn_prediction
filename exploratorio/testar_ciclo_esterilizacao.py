"""
Testa se o ciclo de validade do material esterilizado (achado da reunião de
jul/2026: VAPOR=6 meses, ÓXIDO=2 anos) ajuda a explicar churn.

Não faz parte do pipeline de treino/inferência — teste rápido de hipótese via
Pearson + MI contra o churn real, SÓ NO TREINO, features calculadas SÓ com
dado pré-CUTOFF_TREINO (evita vazamento). Nada é salvo em data/processed.
Mesma metodologia da simulação de razao_recencia/std_intervalo (decisão 59
do CLAUDE.md).

Resultado da 1ª rodada (jul/2026, recorte "categoria_pedido == baixo"):
sinal fraco/nulo (MI≈0 dentro de baixo) — ver CLAUDE.md, seção "Reunião
jul/2026". Ver detalhes em CLAUDE.md.

CORREÇÃO DE ESCOPO (mesma sessão): o usuário esclareceu que a hipótese da
empresa não é sobre TODO cliente categoria "baixo" — é especificamente sobre
o subconjunto de clientes que pedem ~1x/ano, de forma periódica, com datas
parecidas entre os anos (reposição por vencimento de validade, não sazonalidade
qualquer). O recorte "categoria_pedido == baixo" usado abaixo é mais largo que
isso e pode ter diluído o sinal. Falta ainda: um critério que isole
especificamente esse subgrupo periódico (ex.: baixa variância no mês-do-ano
do pedido entre anos, ou intervalo_medio/max_intervalo próximos de 12) antes
de re-testar o tipo_serv/tipo_ped só dentro dele.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists())
sys.path.insert(0, str(PROJECT_ROOT))

from src import db
from src.config import INICIO, CUTOFF_TREINO, PROCESSED_DIR

VALIDADE_MESES = {"VAPOR": 6, "OXIDO": 24}

# --- 1. Pedidos brutos pré-cutoff, só com TIPO_PED (nao esta em df_model_treino) ---
inicio_fmt = pd.Timestamp(INICIO).strftime("%Y%m%d")
cutoff_fmt = CUTOFF_TREINO.strftime("%Y%m%d")
sql = f"""
SELECT CLIENTE, DATA_ENTRADA, TIPO_PED
FROM dbo.PBI_DESEMPENHO_MATERIAL
WHERE DATA_ENTRADA >= '{inicio_fmt}' AND DATA_ENTRADA <= '{cutoff_fmt}';
"""
df_ped = db.get_data(sql)
print(f"Pedidos pré-cutoff carregados: {len(df_ped):,} linhas, {df_ped['CLIENTE'].nunique()} clientes")

# normaliza (encoding de acento pode variar entre driver/terminal)
tipo_norm = df_ped["TIPO_PED"].astype(str).str.upper()
df_ped["tipo_esteril"] = np.select(
    [tipo_norm.str.contains("XIDO", na=False), tipo_norm.str.contains("VAPOR", na=False)],
    ["OXIDO", "VAPOR"],
    default=None,
)
print("\nDistribuição tipo_esteril (pré-cutoff, bruto):")
print(df_ped["tipo_esteril"].value_counts(dropna=False))

# --- 2. Agrega por cliente: composição do portfólio + validade esperada ---
def agrega_cliente(g):
    n = len(g)
    pct_oxido = (g["tipo_esteril"] == "OXIDO").sum() / n
    validade_media = g["tipo_esteril"].map(VALIDADE_MESES).mean()
    return pd.Series({"pct_itens_oxido": pct_oxido, "validade_esperada_meses": validade_media})

feat_esteril = (
    df_ped.dropna(subset=["tipo_esteril"])
    .groupby("CLIENTE")
    .apply(agrega_cliente, include_groups=False)
    .reset_index()
)
print(f"\nFeature calculada pra {len(feat_esteril)} clientes")

# --- 3. Junta com df_model_treino (churn + meses_sem_pedido_pre já existem) ---
df_treino = pd.read_parquet(PROCESSED_DIR / "df_model_treino.parquet")
df_test = df_treino[["CLIENTE", "churn", "meses_sem_pedido_pre", "categoria_pedido"]].merge(
    feat_esteril, on="CLIENTE", how="left"
)

cobertura = df_test["pct_itens_oxido"].notna().mean()
print(f"Cobertura no treino: {cobertura:.1%} ({df_test['pct_itens_oxido'].notna().sum()}/{len(df_test)})")

# candidata extra: razão recência / validade esperada do próprio portfólio
# (em vez de comparar com intervalo_medio observado, compara com o ciclo
# esperado pelo tipo de material -- é a hipótese literal do cliente)
df_test["razao_recencia_validade"] = df_test["meses_sem_pedido_pre"] / df_test["validade_esperada_meses"]

candidatas = ["pct_itens_oxido", "validade_esperada_meses", "razao_recencia_validade"]

print("\n=== Pearson + MI contra churn (só clientes com cobertura, treino) ===")
sub = df_test.dropna(subset=candidatas)
print(f"n = {len(sub)} clientes com as 3 features calculáveis\n")
for col in candidatas:
    r, _ = pearsonr(sub[col], sub["churn"])
    mi = mutual_info_classif(sub[[col]], sub["churn"], random_state=0)[0]
    print(f"{col:28s}  Pearson={r:+.3f}   MI={mi:.3f}")

# referência: meses_sem_pedido_pre sozinha, no mesmo subconjunto (pra comparar maçã com maçã)
r_ref, _ = pearsonr(sub["meses_sem_pedido_pre"], sub["churn"])
mi_ref = mutual_info_classif(sub[["meses_sem_pedido_pre"]], sub["churn"], random_state=0)[0]
print(f"{'meses_sem_pedido_pre (ref)':28s}  Pearson={r_ref:+.3f}   MI={mi_ref:.3f}")

# --- 4. Recorte só na categoria "baixo" -- ver CORREÇÃO DE ESCOPO no topo do arquivo:
# esse recorte é mais largo que a hipótese real da empresa (clientes anuais
# periódicos), resultado abaixo deve ser lido com essa ressalva.
sub_baixo = sub[sub["categoria_pedido"] == "baixo"]
print(f"\n=== Mesmo teste, só categoria 'baixo' (n={len(sub_baixo)}) — recorte largo, ver ressalva ===")
if len(sub_baixo) >= 20:
    for col in candidatas:
        r, _ = pearsonr(sub_baixo[col], sub_baixo["churn"])
        mi = mutual_info_classif(sub_baixo[[col]], sub_baixo["churn"], random_state=0)[0]
        print(f"{col:28s}  Pearson={r:+.3f}   MI={mi:.3f}")
else:
    print("Poucos clientes pra um teste confiável nesse recorte.")
