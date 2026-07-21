import numpy as np
import pandas as pd


def _calc_fidelidade_grupo(grupo: pd.DataFrame) -> pd.Series:
    meses = sorted(grupo["ANO_MES"].tolist())
    categoria = grupo["categoria_pedido"].mode().iloc[0]
    if len(meses) < 2:
        return pd.Series({
            "intervalo_medio":   np.nan,
            "max_intervalo":     np.nan,   # indefinido — não existe intervalo para 1 mês ativo
            "n_intervalos":      0,
            "categoria_cliente": categoria,
        })
    intervalos = [meses[i].ordinal - meses[i - 1].ordinal for i in range(1, len(meses))]
    return pd.Series({
        "intervalo_medio":   np.mean(intervalos),
        "max_intervalo":     max(intervalos),
        "n_intervalos":      len(intervalos),
        "categoria_cliente": categoria,
    })


def calc_features_fidelidade_pre_cutoff(cm_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula intervalo médio/máximo entre pedidos e a categoria predominante do
    cliente, usando só os meses já filtrados por cutoff em cm_feat (recebe o
    dataframe pronto — não recebe cutoff_period porque o filtro já deve ter
    sido aplicado antes, do mesmo jeito que calc_features_comportamento_pre_cutoff).

    Retorna
    -------
    DataFrame com uma linha por CLIENTE: intervalo_medio, max_intervalo,
    n_intervalos, categoria_cliente
    """
    return (
        cm_feat
        .sort_values(["CLIENTE", "ANO_MES"])
        .groupby("CLIENTE")
        .apply(_calc_fidelidade_grupo, include_groups=False)
        .reset_index()
    )
