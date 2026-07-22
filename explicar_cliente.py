# -*- coding: utf-8 -*-
"""
Validação ad-hoc: explica por que o modelo deu o score que deu pra um cliente
específico, via waterfall SHAP. Não é parte do pipeline — ferramenta de consulta.

Uso:
    python explicar_cliente.py <CLIENTE_ID>

Gera reports/waterfall_cliente_<CLIENTE_ID>.png com o gráfico.

IMPORTANTE: lê features_scoring_atual.parquet (dado AO VIVO, gerado por
scripts/scoring_mensal.py) — nunca df_model_teste.parquet, que é um snapshot
congelado em CUTOFF_TESTE. Usar o snapshot aqui já causou uma inconsistência
real: o mesmo cliente aparecia com meses_sem_pedido_pre=1 no waterfall (dado
antigo) mas meses_sem_pedido_pre=8 no relatório HTML (dado ao vivo) — o sinal
tinha direção oposta nos dois, sem ser bug de cálculo, só fonte de dado
diferente. Rode scoring_mensal.py antes de usar esta ferramenta, se ainda não
rodou neste ciclo.
"""
import sys
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # salva em arquivo — não depende de janela/display
import matplotlib.pyplot as plt
import pandas as pd
import shap

PROJECT_ROOT = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists())
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR  # noqa: E402
from src.model_prep import preparar_X  # noqa: E402

FEATURES_PATH = PROCESSED_DIR / "features_scoring_atual.parquet"


def explicar_cliente(cliente_id: int) -> Path:
    with open(PROCESSED_DIR / "modelo_churn.pkl", "rb") as f:
        artefatos = pickle.load(f)
    modelo      = artefatos["modelo"]
    modelo_nome = artefatos["modelo_nome"]
    calibrador  = artefatos["calibrador"]
    features    = artefatos["features"]

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} não existe ainda — rode scripts/scoring_mensal.py "
            "primeiro (ele gera esse arquivo com o dado ao vivo do ciclo atual)."
        )
    df = pd.read_parquet(FEATURES_PATH)
    row = df[df["CLIENTE"] == cliente_id]
    if row.empty:
        raise ValueError(
            f"Cliente {cliente_id} não encontrado em {FEATURES_PATH.name} — pode não "
            "ter pedido nenhum até o cutoff do ciclo atual, ou ID errado."
        )

    X = preparar_X(row, features)

    p_bruta     = modelo.predict_proba(X)[:, 1][0]
    p_calibrada = calibrador.predict_proba(X)[:, 1][0]
    nome        = row["NOME"].values[0].strip()

    print(f"Cliente {cliente_id} — {nome}")
    print(f"Modelo: {modelo_nome}")
    print(f"  p_churn_bruta     : {p_bruta:.3f}")
    print(f"  p_churn_calibrada : {p_calibrada:.3f}")

    explainer = shap.TreeExplainer(modelo)
    shap_vals = explainer(X)

    plt.figure()
    shap.plots.waterfall(shap_vals[0], show=False)
    plt.title(f"{nome} (cliente {cliente_id}) — p_calibrada={p_calibrada:.2f}", fontsize=10)

    out_path = PROJECT_ROOT / "reports" / f"waterfall_cliente_{cliente_id}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close()

    print(f"\nWaterfall salvo em: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python explicar_cliente.py <CLIENTE_ID>")
        sys.exit(1)
    explicar_cliente(int(sys.argv[1]))
