# -*- coding: utf-8 -*-
"""
Pipeline de inferência (scoring mensal) — Seções 10B/10C/10D que antes viviam em
04_modelo.ipynb, agora como script standalone.

Diferença chave para o notebook de treino (04_modelo.ipynb): este script NÃO treina
nada. Ele só carrega um modelo e um calibrador já treinados (persistidos em disco) e
aplica em cima de dados novos. Deve rodar sozinho, sem Jupyter, todo mês — depois que
o mês anterior fechar, para que `meses_sem_pedido_pre` reflita meses completos.

Passo 2 CONCLUÍDO (jul/2026): as features são recalculadas "ao vivo" a partir de uma
consulta fresca ao banco, usando o último mês fechado como cutoff — não depende mais
do snapshot congelado df_model_teste.parquet.

O modelo persistido em `modelo_churn.pkl` é o vencedor dinâmico da Seção 6 do
notebook (tunado ou não, decidido automaticamente) — não necessariamente LightGBM.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists())
sys.path.insert(0, str(PROJECT_ROOT))

from src import db  # noqa: E402
from src.config import PROCESSED_DIR, QUERIES_DIR, INICIO, PISO_RISCO, N_CAPACIDADE  # noqa: E402
from src.features_EDA_pedidos import make_cliente_mes  # noqa: E402
from src.features_comportamento import calc_features_comportamento_pre_cutoff  # noqa: E402
from src.features_fidelidade import calc_features_fidelidade_pre_cutoff  # noqa: E402
from src.features_itens import calc_features_itens_pre_cutoff  # noqa: E402
from src.model_prep import preparar_X  # noqa: E402

MODELO_PATH = PROCESSED_DIR / "modelo_churn.pkl"
IMPUTACAO_PATH = PROCESSED_DIR / "imputacao_treino.pkl"


def carregar_modelo():
    """Carrega o modelo vencedor (dinâmico) + calibrador já treinados (Passo 1)."""
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"{MODELO_PATH} não existe ainda — falta o Passo 1: treinar e persistir o "
            "modelo em 04_modelo.ipynb antes de rodar este script."
        )
    with open(MODELO_PATH, "rb") as f:
        artefatos = pickle.load(f)
    return artefatos["modelo"], artefatos["calibrador"], artefatos["features"]


def carregar_medianas_cadencia():
    """
    Carrega as medianas de cv_pedidos/intervalo_medio/max_intervalo calculadas
    sobre o TREINO em 03_base_master.ipynb — precisam ser as mesmas constantes
    usadas lá, nunca recalculadas aqui (evitaria a garantia anti-leakage de
    "estatística de imputação só vem do treino").
    """
    if not IMPUTACAO_PATH.exists():
        raise FileNotFoundError(
            f"{IMPUTACAO_PATH} não existe — rode 03_base_master.ipynb (célula de "
            "persistência, Passo 2) antes de usar este script."
        )
    with open(IMPUTACAO_PATH, "rb") as f:
        return pickle.load(f)["medianas_cadencia_treino"]


def determinar_cutoff() -> pd.Period:
    """
    Último mês FECHADO (decisão 47 do CLAUDE.md) — nunca o mês corrente, que
    ainda está incompleto e faria meses_sem_pedido_pre etc. tratarem "ainda não
    comprou este mês" como sinal de risco.
    """
    return pd.Period(pd.Timestamp.now(), "M") - 1


def puxar_pedidos_brutos() -> pd.DataFrame:
    """Mesma query usada em 01/02_*_eda.ipynb — puxa pedidos direto do banco."""
    sql = (QUERIES_DIR / "base_pedidos.sql").read_text()
    df = db.get_data(sql)
    df["DATA_ENTRADA"] = pd.to_datetime(df["DATA_ENTRADA"], errors="coerce")
    df["ANO_MES"] = df["DATA_ENTRADA"].dt.to_period("M")
    return df


def montar_cliente_item_mes(df_pedidos: pd.DataFrame) -> pd.DataFrame:
    """Mesma agregação de 02_itens_eda.ipynb — cliente × item × mês, quantidade bruta."""
    df = df_pedidos[df_pedidos["DATA_ENTRADA"] >= INICIO].copy()
    df["VL_SERVICO"] = df["VL_SERVICO"].clip(lower=0)
    return (
        df.groupby(["CLIENTE", "SERVICO", "ANO_MES"])
        .agg(qtd=("VL_SERVICO", "count"))
        .reset_index()
    )


def carregar_perfil_clientes() -> pd.DataFrame:
    """Mesma query/preparo de 03_base_master.ipynb (célula s1_perfil)."""
    sql = (QUERIES_DIR / "base_clientes.sql").read_text()
    df_clientes_raw = db.get_data(sql)
    return (
        df_clientes_raw[["CODIGO", "NOME", "DIASINADIMPLENTE"]]
        .drop_duplicates(subset="CODIGO")
        .rename(columns={"CODIGO": "CLIENTE"})
    )


def carregar_status_clientes(clientes_ids) -> pd.DataFrame:
    """
    STATUS ao vivo (Ativo/Inativo, único valores hoje no cadastro) do
    CONTROLLER.dbo.PBI_Clientes — mesma tabela/lógica de validar_lista_alertas.py.
    Usado pra nunca colocar na lista de alerta quem já está formalmente
    desativado: não faz sentido acionar retenção pra quem já saiu.
    """
    ids = list(clientes_ids)
    placeholders = ",".join(["?"] * len(ids))
    sql = f"""
    SELECT CODIGO AS CLIENTE, [STATUS] AS status_sistema
    FROM CONTROLLER.dbo.PBI_Clientes
    WHERE CODIGO IN ({placeholders})
    """
    return db.get_data(sql, params=ids)


def carregar_clientes_para_score(cutoff_period: pd.Period):
    """
    Recalcula as features pré-cutoff "ao vivo": puxa pedidos frescos do banco,
    monta cliente_mes/cliente_item_mes, filtra até o último mês fechado, e
    chama as mesmas funções de features usadas em 03_base_master.ipynb.

    Retorna (df, cliente_mes) — cliente_mes completo (sem filtrar por cutoff) é
    reaproveitado por calcular_receita_anual(), que olha os últimos 12 meses
    reais a partir de hoje, não do cutoff de features.
    """
    df_pedidos = puxar_pedidos_brutos()
    cm = make_cliente_mes(df_pedidos, inicio=INICIO)
    cim = montar_cliente_item_mes(df_pedidos)

    cm_feat = cm[cm["ANO_MES"] <= cutoff_period].copy()

    features_comportamento = calc_features_comportamento_pre_cutoff(cm_feat, cutoff_period, INICIO)
    features_fidelidade = calc_features_fidelidade_pre_cutoff(cm_feat)
    features_itens = calc_features_itens_pre_cutoff(cim, cutoff_period)
    df_perfil = carregar_perfil_clientes()

    clientes_elegiveis = cm_feat["CLIENTE"].unique()
    df = (
        pd.DataFrame({"CLIENTE": clientes_elegiveis})
        .merge(features_comportamento, on="CLIENTE", how="left")
        .merge(features_fidelidade,    on="CLIENTE", how="left")
        .merge(features_itens,         on="CLIENTE", how="left")
        .merge(df_perfil,              on="CLIENTE", how="left")
    )

    # ── Flags — criados ANTES de qualquer fillna (mesma ordem de 03_base_master) ──
    df["sem_historico_cadencia"] = (df["n_intervalos"] == 0).astype(int)
    df["sem_historico_itens"] = (
        df["pct_itens_queda"].isna() | df["slope_portfolio_medio"].isna()
    ).astype(int)

    # ── Imputação cadência — medianas do TREINO, sem recálculo (anti-leakage) ──
    medianas_cadencia_treino = carregar_medianas_cadencia()
    for feat, med in medianas_cadencia_treino.items():
        df[feat] = df[feat].fillna(med)

    # ── Imputação itens ────────────────────────────────────────────────────────
    df["slope_portfolio_medio"] = df["slope_portfolio_medio"].fillna(0)
    df["pct_itens_queda"] = df["pct_itens_queda"].fillna(0)

    return df, cm


def calcular_receita_anual(cliente_mes: pd.DataFrame, clientes_ids, data_referencia: pd.Timestamp):
    """Soma os últimos 12 meses reais de cliente_mes por cliente (decisão 39 do CLAUDE.md)."""
    inicio_12m = data_referencia - pd.DateOffset(months=12)
    cm_12m = cliente_mes[cliente_mes["ANO_MES"].dt.to_timestamp() > inicio_12m].copy()

    receita = (
        cm_12m.groupby("CLIENTE")
        .agg(receita_soma=("total_valor", "sum"), n_meses_obs=("ANO_MES", "nunique"))
        .reset_index()
    )
    receita["receita_anual"] = np.where(
        receita["n_meses_obs"] >= 12,
        receita["receita_soma"],
        (receita["receita_soma"] / receita["n_meses_obs"]) * 12,
    )
    receita["historico_parcial_receita"] = (receita["n_meses_obs"] < 12).astype(int)
    return receita[receita["CLIENTE"].isin(clientes_ids)]


def top_features_shap_positivas(modelo, X, feature_names, top_n=3):
    """Top-N features por contribuição SHAP POSITIVA (empurram para churn) — não |SHAP|."""
    import shap

    explainer = shap.TreeExplainer(modelo)
    shap_vals = explainer(X)

    resultado = []
    for row in shap_vals.values:
        idx_pos = np.where(row > 0)[0]
        idx_sorted = idx_pos[np.argsort(row[idx_pos])[::-1]]
        idx_top = idx_sorted[:top_n]
        resultado.append(", ".join(feature_names[idx_top]) if len(idx_top) else "")
    return resultado


def main():
    modelo, calibrador, feature_cols = carregar_modelo()

    hoje = pd.Timestamp.now()
    cutoff_period = determinar_cutoff()
    print(f"Hoje: {hoje.strftime('%Y-%m-%d')}  |  Cutoff de features (último mês fechado): {cutoff_period}")

    df, cliente_mes = carregar_clientes_para_score(cutoff_period)

    # Persistido pra scripts/gerar_relatorio.py montar o "motivo principal" por
    # cliente com valores reais — sem isso, o relatório teria que usar
    # df_model_teste.parquet (snapshot congelado) só pra montar o texto, o que
    # reintroduziria o mesmo problema de dado desatualizado que o Passo 2 corrigiu.
    df.to_parquet(PROCESSED_DIR / "features_scoring_atual.parquet", index=False)

    identificadores = df[["CLIENTE", "NOME", "categoria_cliente",
                           "sem_historico_cadencia", "sem_historico_itens"]].reset_index(drop=True)
    X = preparar_X(df, feature_cols).reset_index(drop=True)

    p_bruta = modelo.predict_proba(X)[:, 1]
    p_calibrada = calibrador.predict_proba(X)[:, 1]

    df_scores = identificadores.copy()
    df_scores["p_churn_bruta"] = p_bruta
    df_scores["p_churn_calibrada"] = p_calibrada

    receita = calcular_receita_anual(cliente_mes, df_scores["CLIENTE"].tolist(), hoje)
    df_scores = df_scores.merge(
        receita[["CLIENTE", "receita_anual", "historico_parcial_receita"]],
        on="CLIENTE", how="left",
    )
    df_scores["receita_anual"] = df_scores["receita_anual"].fillna(0)
    df_scores["historico_parcial_receita"] = df_scores["historico_parcial_receita"].fillna(0).astype(int)

    df_scores["valor_em_risco"] = df_scores["p_churn_calibrada"] * df_scores["receita_anual"]
    df_scores["decil"] = pd.qcut(df_scores["p_churn_calibrada"], 10, labels=False, duplicates="drop") + 1

    # Duas ressalvas DIFERENTES, que antes viviam misturadas num único flag
    # (historico_insuficiente) — achado do usuário (cliente 256: 29 meses de
    # histórico muito regular, mas caiu na mesma bandeira genérica de um
    # cliente com 1 mês de vida, tipo cliente 1641):
    #
    # - score_pouco_historico: o MODELO tem pouco comportamento pra aprender
    #   (cliente novo/esparso) — o score em si é incerto.
    # - receita_parcial: o CLIENTE tem histórico normal, mas parou de comprar
    #   há tempo suficiente pra a janela de receita (12 meses) não pegar 12
    #   meses reais — a receita_anual é extrapolada, mas o score continua
    #   bem fundamentado (às vezes é justamente esse silêncio que empurra o
    #   score pra cima).
    df_scores["score_pouco_historico"] = (
        (df_scores["sem_historico_cadencia"] == 1)
        | (df_scores["sem_historico_itens"] == 1)
    ).astype(int)
    df_scores["receita_parcial"] = df_scores["historico_parcial_receita"]
    df_scores["historico_insuficiente"] = (
        (df_scores["score_pouco_historico"] == 1) | (df_scores["receita_parcial"] == 1)
    ).astype(int)

    # STATUS ao vivo do cadastro — quem já está Inativo nunca entra na lista,
    # mas continua no parquet completo pra auditoria (nunca escondido, só
    # marcado como inelegível).
    status = carregar_status_clientes(df_scores["CLIENTE"].tolist())
    df_scores = df_scores.merge(status, on="CLIENTE", how="left")
    df_scores["status_sistema"] = df_scores["status_sistema"].fillna("Desconhecido")
    ja_inativo = df_scores["status_sistema"] == "Inativo"

    acima_piso = df_scores["p_churn_calibrada"] >= PISO_RISCO
    elegivel = acima_piso & ~ja_inativo
    ranking = df_scores[elegivel].sort_values("valor_em_risco", ascending=False)
    top_clientes = ranking.head(N_CAPACIDADE)["CLIENTE"]
    df_scores["entra_na_lista"] = df_scores["CLIENTE"].isin(top_clientes)

    df_scores["top_features_shap"] = top_features_shap_positivas(
        modelo, X, np.array(feature_cols)
    )

    colunas_output = [
        "CLIENTE", "NOME", "categoria_cliente",
        "p_churn_bruta", "p_churn_calibrada",
        "receita_anual", "valor_em_risco", "decil",
        "status_sistema", "entra_na_lista", "historico_insuficiente",
        "score_pouco_historico", "receita_parcial",
        "top_features_shap",
    ]
    scores_2026 = (
        df_scores[colunas_output]
        .sort_values("valor_em_risco", ascending=False)
        .reset_index(drop=True)
    )
    scores_2026.to_parquet(PROCESSED_DIR / "scores_2026.parquet", index=False)

    lista_alerta = scores_2026[scores_2026["entra_na_lista"]].copy()
    lista_alerta.to_excel(PROCESSED_DIR / "clientes_alerta_modelo.xlsx", index=False)

    print(f"PISO_RISCO = {PISO_RISCO}  ->  {acima_piso.sum()} / {len(df_scores)} clientes acima do piso")
    print(f"  dos quais {int((acima_piso & ja_inativo).sum())} já estão Inativo no cadastro — excluídos do ranking")
    print(f"N_CAPACIDADE = {N_CAPACIDADE}  ->  {df_scores['entra_na_lista'].sum()} clientes selecionados")
    print(f"scores_2026.parquet salvo: {len(scores_2026)} clientes")
    print(f"clientes_alerta_modelo.xlsx salvo: {len(lista_alerta)} clientes")

    # Relatório HTML sempre regenerado do zero a partir daqui — nunca editado
    # manualmente (foi editar esse arquivo à mão que corrompeu o relatório
    # antes, decisão 51 do CLAUDE.md).
    from gerar_relatorio import gerar_relatorio
    gerar_relatorio()


if __name__ == "__main__":
    main()
