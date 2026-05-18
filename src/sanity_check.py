# Teste de sanidade generalizado para o dataset de pedidos. Verificações comuns incluem:

def sanity_check(df, value_col=None, date_col=None, id_cols=None):

    print("Shape:", df.shape)

    print("\nMissing:")
    print(df.isna().sum())

    print("\nDuplicados:")
    print(df.duplicated().sum())

    if value_col:
        cols = [value_col] if isinstance(value_col, str) else value_col
        negativos = {c: (df[c].dropna() < 0).sum() for c in cols}
        print("\nNegativos em", value_col, ":")
        for c, n in negativos.items():
            print(f"  {c}: {n}")

    if date_col:
        print("\nDatas:", df[date_col].min(), df[date_col].max())

    if id_cols:
        print("\nDuplicados por chave:")
        print(df.duplicated(id_cols).sum())