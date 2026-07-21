import numpy as np
import pandas as pd


def tendencia_slope(series: pd.Series) -> float:
    """Inclinação da reta linear de quantidade ao longo dos meses. NaN se < 2 pontos."""
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    x = np.arange(len(s))
    slope, _ = np.polyfit(x, s.values, 1)
    return slope


def calc_features_itens_pre_cutoff(cliente_item_mes: pd.DataFrame, cutoff_period: pd.Period) -> pd.DataFrame:
    """
    Recalcula slope_portfolio_medio, pct_itens_queda e n_itens_portfolio usando
    apenas pedidos até cutoff_period (inclusive) — evita vazamento de meses
    futuros ao cutoff de treino/teste.

    Parâmetros
    ----------
    cliente_item_mes : DataFrame com colunas CLIENTE, SERVICO, ANO_MES, qtd
                        (granularidade cliente × item × mês, histórico completo)
    cutoff_period     : pd.Period ("M") — janela de features vai até aqui, inclusive

    Retorna
    -------
    DataFrame com uma linha por CLIENTE: slope_portfolio_medio, pct_itens_queda,
    n_itens_portfolio
    """
    cim_cut = cliente_item_mes[cliente_item_mes["ANO_MES"] <= cutoff_period].copy()
    cim_cut = cim_cut.sort_values(["CLIENTE", "SERVICO", "ANO_MES"])

    tendencias = (
        cim_cut.groupby(["CLIENTE", "SERVICO"])["qtd"]
        .apply(tendencia_slope)
        .reset_index(name="tendencia_slope")
    )

    features_itens = (
        tendencias.groupby("CLIENTE")
        .agg(
            slope_portfolio_medio=("tendencia_slope", "mean"),
            pct_itens_queda=("tendencia_slope", lambda x: (x < 0).sum() / len(x)),
        )
        .reset_index()
    )

    n_itens_portfolio = (
        cim_cut.groupby("CLIENTE")["SERVICO"]
        .nunique()
        .rename("n_itens_portfolio")
        .reset_index()
    )

    return features_itens.merge(n_itens_portfolio, on="CLIENTE", how="outer")
