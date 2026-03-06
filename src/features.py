# Criar tabela agregada cliente mes 

import pandas as pd

def make_cliente_mes(df, inicio="2023-01-01", neg_policy="clip0"):
    """
    neg_policy:
      - "clip0": valores negativos viram 0 para cálculo de total_valor (recomendado)
      - "drop": remove linhas com VL_SERVICO < 0 (mais agressivo)
      - "keep": mantém como está (não recomendado no seu caso)
    """
    df = df.copy()

    # garante datetime + ANO_MES
    if not pd.api.types.is_datetime64_any_dtype(df["DATA_ENTRADA"]):
        df["DATA_ENTRADA"] = pd.to_datetime(df["DATA_ENTRADA"], errors="coerce")
    if "ANO_MES" not in df.columns:
        df["ANO_MES"] = df["DATA_ENTRADA"].dt.to_period("M")

    # filtro temporal
    df = df[df["DATA_ENTRADA"] >= inicio].copy()

    # saneamento de negativos (sem alterar a coluna original)
    df["VL_SERVICO_LIMPO"] = df["VL_SERVICO"]

    if neg_policy == "clip0":
        df["VL_SERVICO_LIMPO"] = df["VL_SERVICO_LIMPO"].clip(lower=0)
    elif neg_policy == "drop":
        df = df[df["VL_SERVICO_LIMPO"] >= 0].copy()
    elif neg_policy == "keep":
        pass
    else:
        raise ValueError("neg_policy deve ser: 'clip0', 'drop' ou 'keep'")

    # agrega
    cliente_mes = (
        df.groupby(["CLIENTE", "ANO_MES"], as_index=False)
          .agg(
              total_pedidos=("NRO_PED", "nunique"),
              total_valor=("VL_SERVICO_LIMPO", "sum"),
              qtd_itens=("NRO_PED", "count"),
              
          )
    )

    # métricas úteis
    cliente_mes["ticket_medio"] = cliente_mes["total_valor"] / cliente_mes["total_pedidos"]
    cliente_mes["itens_por_pedido"] = cliente_mes["qtd_itens"] / cliente_mes["total_pedidos"]

    return cliente_mes

# criar tabela agregada pedidos 


#intervalo entre compras , intervalo medio por cliente, juntar categoria e volume 