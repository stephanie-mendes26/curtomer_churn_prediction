import pandas as pd


def meses_desde(serie_periodo: pd.Series, ref: pd.Period) -> pd.Series:
    """Meses inteiros entre cada valor da série e o período de referência."""
    return serie_periodo.apply(
        lambda x: (ref.year - x.year) * 12 + (ref.month - x.month)
    )


def calc_features_comportamento_pre_cutoff(
    cm_feat: pd.DataFrame, cutoff_period: pd.Period, inicio: str
) -> pd.DataFrame:
    """
    Agrega cliente_mes (já filtrado a ANO_MES <= cutoff_period) em features de
    comportamento por cliente: volume, ticket, recência, cadência.

    Parâmetros
    ----------
    cm_feat       : cliente_mes filtrado até o cutoff do split (treino ou teste)
    cutoff_period : pd.Period ("M") — data de corte do split
    inicio        : "YYYY-MM-DD" — início da janela de análise (INICIO do config)

    Retorna
    -------
    DataFrame com uma linha por CLIENTE: n_meses_ativos, total_pedidos,
    total_valor, media_pedidos_mes, ticket_medio, itens_por_pedido,
    categoria_pedido, cv_pedidos, meses_sem_pedido_pre, razao_atividade
    """
    janela_meses = (
        (cutoff_period.year - pd.Period(inicio, "M").year) * 12
        + (cutoff_period.month - pd.Period(inicio, "M").month)
    )

    features = (
        cm_feat.groupby("CLIENTE")
        .agg(
            n_meses_ativos    = ("ANO_MES",         "count"),
            ultimo_mes_pre    = ("ANO_MES",          "max"),
            total_pedidos     = ("total_pedidos",    "sum"),
            total_valor       = ("total_valor",      "sum"),
            media_pedidos_mes = ("total_pedidos",    "mean"),
            std_pedidos_mes   = ("total_pedidos",    "std"),
            ticket_medio      = ("ticket_medio",     "mean"),
            itens_por_pedido  = ("itens_por_pedido", "mean"),
            categoria_pedido  = ("categoria_pedido", lambda x: x.mode()[0]),
        )
        .reset_index()
    )

    features["cv_pedidos"] = features["std_pedidos_mes"] / features["media_pedidos_mes"]
    features["meses_sem_pedido_pre"] = meses_desde(features["ultimo_mes_pre"], cutoff_period)
    features["razao_atividade"] = features["n_meses_ativos"] / janela_meses

    return features.drop(columns=["ultimo_mes_pre", "std_pedidos_mes"])
