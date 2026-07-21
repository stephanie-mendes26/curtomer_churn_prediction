# -*- coding: utf-8 -*-
"""
Validação ad-hoc: confere no banco, em tempo real, a situação atual dos clientes
que apareceram em clientes_alerta_modelo.xlsx (04_modelo, Seção 10D).

Não é parte do pipeline — script de teste pontual pedido pelo usuário para checar
se algum cliente da lista já aparece como desativado/inativo no cadastro, ou já
está sem comprar há mais tempo do que os dados usados no treino/scoring sugeriam.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists())
sys.path.insert(0, str(PROJECT_ROOT))

from src import db  # noqa: E402
from src.config import PROCESSED_DIR  # noqa: E402

scores = pd.read_parquet(PROCESSED_DIR / "scores_2026.parquet")
lista = scores[scores["entra_na_lista"]].sort_values("valor_em_risco", ascending=False).reset_index(drop=True)

clientes_ids = lista["CLIENTE"].tolist()
placeholders = ",".join(["?"] * len(clientes_ids))

sql = f"""
SELECT
    c.CODIGO                                                          AS CLIENTE,
    c.NOME,
    c.[STATUS]                                                        AS status_sistema,
    c.DIASINADIMPLENTE,
    MAX(p.DATA_ENTRADA)                                               AS ultima_compra,
    DATEDIFF(MONTH, MAX(p.DATA_ENTRADA), GETDATE())                   AS meses_sem_compra_hoje
FROM CONTROLLER.dbo.PBI_Clientes c
    LEFT JOIN CENTRAL.dbo.PBI_DESEMPENHO_MATERIAL p
        ON c.CODIGO = p.CLIENTE
WHERE c.CODIGO IN ({placeholders})
GROUP BY c.CODIGO, c.NOME, c.[STATUS], c.DIASINADIMPLENTE
"""

print("Consultando o banco em tempo real para", len(clientes_ids), "clientes...")
live = db.get_data(sql, params=clientes_ids)
live["NOME"] = live["NOME"].str.strip()

comp = lista.merge(live, on="CLIENTE", how="left", suffixes=("", "_live"))

print("\nStatus distintos encontrados no cadastro para esses clientes:")
print(comp["status_sistema"].value_counts(dropna=False).to_string())

# Heurística simples pra sinalizar status que pareçam inativo/desativado/cancelado —
# ajustar os termos abaixo depois de ver os valores reais impressos acima.
termos_inativo = ["inativ", "desativ", "cancel", "bloque", "encerrad"]
comp["parece_inativo"] = (
    comp["status_sistema"].astype(str).str.lower()
    .apply(lambda s: any(t in s for t in termos_inativo))
)

cols = [
    "CLIENTE", "NOME", "p_churn_calibrada", "valor_em_risco",
    "status_sistema", "parece_inativo", "ultima_compra", "meses_sem_compra_hoje",
]
print("\n=== Comparação modelo vs. banco em tempo real ===\n")
print(comp[cols].to_string(index=False, float_format="{:.2f}".format))

n_inativos = comp["parece_inativo"].sum()
print(f"\n{n_inativos} de {len(comp)} clientes da lista já aparecem com status potencialmente inativo no cadastro.")
if n_inativos > 0:
    print("Revisar antes de mandar a lista pra equipe comercial — não faz sentido alertar sobre quem já saiu.")
