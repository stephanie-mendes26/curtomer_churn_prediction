"""
Reconstrução do teste rápido (Pearson + MI) que motivou a decisão 59 do
CLAUDE.md — na 1ª rodada esse script foi rodado e descartado sem ser salvo;
recriado aqui pra ficar reprodutível, seguindo a convenção adotada depois
(guardar os testes de hipótese em exploratorio/, não descartar).

Não faz parte do pipeline. Nada persistido em data/processed. SÓ treino,
features calculadas SÓ com dado pré-CUTOFF_TREINO (evita vazamento) — mesma
metodologia de todos os outros scripts desta pasta.

Testa duas features candidatas contra `intervalo_medio` sozinha:
- razao_recencia = meses_sem_pedido_pre / intervalo_medio
- std_intervalo   = desvio padrão dos gaps entre meses ativos (não a média)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists())
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CUTOFF_TREINO, PROCESSED_DIR

CUTOFF_PERIOD = CUTOFF_TREINO.to_period("M")

# --- 1. std_intervalo: recalculado sobre cliente_mes, pré-cutoff, mesma lógica de src/features_fidelidade.py ---
cliente_mes = pd.read_parquet(PROCESSED_DIR / "cliente_mes.parquet")
cliente_mes["ANO_MES"] = pd.PeriodIndex(cliente_mes["ANO_MES"], freq="M")
cm_cut = cliente_mes[cliente_mes["ANO_MES"] <= CUTOFF_PERIOD]

def std_intervalo_cliente(g):
    meses = sorted(g["ANO_MES"].tolist())
    if len(meses) < 3:  # precisa de >= 2 intervalos pra ter desvio padrão definido
        return np.nan
    intervalos = [meses[i].ordinal - meses[i - 1].ordinal for i in range(1, len(meses))]
    return np.std(intervalos, ddof=1)

std_int = (
    cm_cut.sort_values(["CLIENTE", "ANO_MES"])
    .groupby("CLIENTE")
    .apply(std_intervalo_cliente, include_groups=False)
    .rename("std_intervalo")
    .reset_index()
)

# --- 2. Junta com df_model_treino (churn, meses_sem_pedido_pre, intervalo_medio já existem) ---
df_treino = pd.read_parquet(PROCESSED_DIR / "df_model_treino.parquet")
df = df_treino[["CLIENTE", "churn", "meses_sem_pedido_pre", "intervalo_medio"]].merge(
    std_int, on="CLIENTE", how="left"
)
df["razao_recencia"] = df["meses_sem_pedido_pre"] / df["intervalo_medio"]

print(f"n = {len(df)} clientes no treino")
print(f"Cobertura razao_recencia: {df['razao_recencia'].notna().mean():.1%}")
print(f"Cobertura std_intervalo:  {df['std_intervalo'].notna().mean():.1%}")

# --- 3. Pearson + MI contra churn ---
print("\n=== Pearson + MI contra churn ===")
for col in ["razao_recencia", "std_intervalo", "intervalo_medio", "meses_sem_pedido_pre"]:
    s = df.dropna(subset=[col])
    r, _ = pearsonr(s[col], s["churn"])
    mi = mutual_info_classif(s[[col]], s["churn"], random_state=0)[0]
    print(f"{col:24s}  Pearson={r:+.3f}   MI={mi:.3f}   n={len(s)}")
