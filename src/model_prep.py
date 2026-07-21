import pandas as pd

MAPA_CATEGORIA = {"baixo": 0, "medio": 1, "alto": 2, "premium": 3}
FILLNA_SEMANTICO = {"DIASINADIMPLENTE": 0, "n_itens_portfolio": 0}

IDENTIFICADORES = ["CLIENTE", "NOME"]
TARGET = "churn"


def _aplicar_transformacoes(X: pd.DataFrame) -> pd.DataFrame:
    for col in ["categoria_pedido", "categoria_cliente"]:
        if col in X.columns:
            X[col] = X[col].astype("object").map(MAPA_CATEGORIA)
    for col, val in FILLNA_SEMANTICO.items():
        if col in X.columns:
            X[col] = X[col].fillna(val)
    return X


def preparar_xy(df: pd.DataFrame):
    """
    Prepara X/y para TREINO a partir de df_model_treino/teste: mapeia
    categoria_pedido/categoria_cliente para número e aplica fillna semântico em
    DIASINADIMPLENTE/n_itens_portfolio. Usada por 04_modelo.ipynb (Seção 1) —
    espera que `df` tenha a coluna `churn` (target conhecido).
    """
    X = df.drop(columns=IDENTIFICADORES + [TARGET]).copy()
    X = _aplicar_transformacoes(X)
    y = df[TARGET]
    return X, y


def preparar_X(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Variante de preparar_xy() para INFERÊNCIA: usada por
    scripts/scoring_mensal.py, explicar_cliente.py e
    notebooks/explicar_cliente.ipynb — recebe uma lista fixa de `features` (a
    mesma persistida em modelo_churn.pkl) e não exige a coluna `churn` (dado
    novo/ao vivo não tem target conhecido ainda).
    """
    X = df[features].copy()
    return _aplicar_transformacoes(X)
