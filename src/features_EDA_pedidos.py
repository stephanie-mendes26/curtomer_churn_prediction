import pandas as pd
from src.config import LIMITES_CATEGORIA


def _classificar_categoria(total_pedidos: pd.Series) -> pd.Series:
    """Classifica pedidos/mês em baixo/medio/alto/premium."""
    cat = pd.Series("baixo", index=total_pedidos.index)
    for nome, minimo, maximo in LIMITES_CATEGORIA:
        mask = (total_pedidos >= minimo) & (total_pedidos <= maximo)
        cat[mask] = nome
    return cat


def make_cliente_mes(df, inicio: str, neg_policy: str = "clip0") -> pd.DataFrame:
    """
    Agrega pedidos brutos em cliente × mês.

    Parâmetros
    ----------
    df          : DataFrame com colunas CLIENTE, NRO_PED, DATA_ENTRADA, VL_SERVICO
    inicio      : data de corte inferior (ex: "2023-01-01")
    neg_policy  : "clip0" (negativos → 0) | "drop" (remove linha) | "keep" (mantém)

    Retorna
    -------
    DataFrame com colunas:
        CLIENTE, ANO_MES, total_pedidos, total_valor, qtd_itens,
        ticket_medio, itens_por_pedido, categoria_pedido
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["DATA_ENTRADA"]):
        df["DATA_ENTRADA"] = pd.to_datetime(df["DATA_ENTRADA"], errors="coerce")
    if "ANO_MES" not in df.columns:
        df["ANO_MES"] = df["DATA_ENTRADA"].dt.to_period("M")

    df = df[df["DATA_ENTRADA"] >= inicio].copy()

    df["VL_SERVICO_LIMPO"] = df["VL_SERVICO"]
    if neg_policy == "clip0":
        df["VL_SERVICO_LIMPO"] = df["VL_SERVICO_LIMPO"].clip(lower=0)
    elif neg_policy == "drop":
        df = df[df["VL_SERVICO_LIMPO"] >= 0].copy()
    elif neg_policy == "keep":
        pass
    else:
        raise ValueError("neg_policy deve ser: 'clip0', 'drop' ou 'keep'")

    cliente_mes = (
        df.groupby(["CLIENTE", "ANO_MES"], as_index=False)
          .agg(
              total_pedidos  = ("NRO_PED",         "nunique"),
              total_valor    = ("VL_SERVICO_LIMPO", "sum"),
              qtd_itens      = ("NRO_PED",          "count"),
          )
    )

    cliente_mes["ticket_medio"]      = cliente_mes["total_valor"] / cliente_mes["total_pedidos"]
    cliente_mes["itens_por_pedido"]  = cliente_mes["qtd_itens"]   / cliente_mes["total_pedidos"]
    cliente_mes["categoria_pedido"]  = _classificar_categoria(cliente_mes["total_pedidos"])

    return cliente_mes 