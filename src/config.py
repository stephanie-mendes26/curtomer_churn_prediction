import pandas as pd
from pathlib import Path

# === Caminhos ===
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
QUERIES_DIR   = PROJECT_ROOT / "queries"

# === Janela temporal ===
DATA_REF = pd.Timestamp("2026-05-01")   # fim dos dados disponíveis
INICIO   = "2023-01-01"                 # início da janela de análise

# === Splits para modelagem ===
CUTOFF_TREINO   = pd.Timestamp("2024-12-31")
CUTOFF_TESTE    = pd.Timestamp("2025-12-31")
HORIZONTE_MESES = 3                     # meses de janela de previsão

# === Thresholds — categoria_pedido (pedidos únicos por mês) ===
LIMITES_CATEGORIA = [
    ("baixo",   0,  2),
    ("medio",   3,  5),
    ("alto",    6, 19),
    ("premium", 20, float("inf")),
]

# === Thresholds — churn por categoria ===
THRESHOLD_CHURN = {
    "premium": 2,
    "alto":    2,
    "medio":   3,
    "baixo":   6,
}

# === Paleta de cores (do mais claro ao mais escuro) ===
VERDE = [
    "#d8f3dc", "#b7e4c7", "#95d5b2", "#74c69d",
    "#52b788", "#40916c", "#2d6a4f", "#1b4332",
]