"""
Script de exploração — lista valores distintos de AUTO_CLV, TIPO_SERV,
TIPO_PED e ADEQUAR em dbo.PBI_DESEMPENHO_MATERIAL, pra ver se alguma delas
indica tipo de esterilizacao (vapor vs oxido de etileno).

Achado (jul/2026): TIPO_PED já vem só com 2 valores -- VAPOR e ÓXIDO --
direto do banco, sem precisar extrair de texto do SERVICO. Ver CLAUDE.md,
seção "Reunião jul/2026 — Implementar Projeto".
"""
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists())
sys.path.insert(0, str(PROJECT_ROOT))

from src import db

for col in ["AUTO_CLV", "TIPO_SERV", "TIPO_PED", "ADEQUAR"]:
    sql = f"""
    SELECT [{col}], COUNT(*) AS n_linhas, COUNT(DISTINCT CLIENTE) AS n_clientes
    FROM dbo.PBI_DESEMPENHO_MATERIAL
    GROUP BY [{col}]
    ORDER BY n_linhas DESC;
    """
    df = db.get_data(sql)
    print(f"\n=== Valores distintos de {col} ({len(df)} valores) ===")
    print(df.head(30).to_string(index=False))
