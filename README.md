# Previsão de Churn de Clientes B2B

![Status](https://img.shields.io/badge/status-modelo%20aplicado-success?style=flat-square)
![Python](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![Modelo](https://img.shields.io/badge/modelo%20oficial-din%C3%A2mico%20(XGBoost)-2d6a4f?style=flat-square)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.893-brightgreen?style=flat-square)
![AUC-PR](https://img.shields.io/badge/AUC--PR-0.864-brightgreen?style=flat-square)
![Notebooks](https://img.shields.io/badge/notebooks-5%20de%206%20conclu%C3%ADdos-40916c?style=flat-square)
![Último commit](https://img.shields.io/github/last-commit/stephanie-mendes26/curtomer_churn_prediction?style=flat-square)

---

## Objetivo

Identificar, com antecedência de 3 meses, quais clientes B2B têm maior risco de parar de comprar — permitindo que a equipe comercial atue preventivamente antes que a perda aconteça.

O cliente é uma empresa distribuidora de materiais médico-hospitalares. Cada "cliente" é uma empresa (clínica, hospital, indústria) que faz pedidos recorrentes.

---

## Pipeline de Dados

```
Banco de dados
     │
     ├─── base_pedidos.sql ──────► 01_pedidos_eda.ipynb
     │                                    │
     │                                    ├── cliente_mes.parquet
     │                                    └── cliente_fidelidade.parquet
     │
     ├─── base_pedidos.sql ──────► 02_itens_eda.ipynb
     │                                    │
     │                                    ├── cliente_item_tendencia.parquet (EDA — histórico completo)
     │                                    └── cliente_item_mes.parquet (usado por 03_base_master p/ features pré-cutoff)
     │
     └─── base_clientes.sql ─────► 03_base_master.ipynb
                                          │
                                          ├── df_model_treino.parquet
                                          ├── df_model_teste.parquet
                                          └── imputacao_treino.pkl
                                                      │
                                                      └── 04_modelo.ipynb  (treino — roda raramente)
                                                                │
                                                                └── modelo_churn.pkl
                                                                          │
     banco de dados (ao vivo) ──► scripts/scoring_mensal.py  (inferência — roda todo mês)
                                                                          │
                                                                          ├── scores_2026.parquet
                                                                          ├── clientes_alerta_modelo.xlsx
                                                                          └── scripts/gerar_relatorio.py
                                                                                    │
                                                                                    └── reports/lista_alertas_churn.html
```

---

## Status dos Notebooks

| Notebook | Descrição | Status |
|---|---|---|
| `01_pedidos_eda.ipynb` | EDA de pedidos, criação de `cliente_mes` e `cliente_fidelidade` | ✅ Concluído |
| `02_clientes_eda.ipynb` | EDA de cadastro de clientes | ⏸️ Pausado — variáveis não confiáveis |
| `02_itens_eda.ipynb` | EDA de itens, criação de `cliente_item_tendencia` | ✅ Concluído |
| `03_base_master.ipynb` | Construção do dataset de modelagem com split temporal | ✅ Concluído |
| `calculo_receita.ipynb` | Receita média mensal por cliente | ✅ Concluído |
| `04_modelo.ipynb` | Modelagem, threshold, SHAP, calibração, scoring e lista de alertas | ✅ Concluído |

---

## Resultados do Modelo

O modelo oficial **não é fixo** — o notebook de treino compara 4 modelos e escolhe o
vencedor dinamicamente por AUC-ROC (mais tuning automático via Optuna, aceito só se
não piorar nenhuma métrica). Na execução mais recente (pós-correção de vazamento
de dado, ver abaixo), o vencedor foi **XGBoost**:

| Modelo | AUC-ROC | AUC-PR |
|---|---|---|
| **XGBoost (oficial)** | **0.8928** | **0.8639** |
| Regressão Logística | 0.8923 | 0.8662 |
| Random Forest | 0.8889 | 0.8527 |
| LightGBM | 0.8770 | 0.8386 |
| Dummy (baseline) | 0.5000 | 0.4903 |

> **Nota de correção:** uma versão anterior deste projeto reportava AUC-ROC ~0.918
> com LightGBM como vencedor fixo. Esse número estava inflado por vazamento de dado
> em 3 features de tendência de item (corrigido — ver Decisões Técnicas Chave). O
> desempenho real do modelo é a faixa 0.88–0.89 acima.

**Threshold:** F2-ótimo (recall pesa 2× — falso negativo é mais caro em B2B), recalculado a cada execução — ver output da Seção 7 do notebook para o valor exato desta versão.

**Lista de alertas atual:** `clientes_alerta_modelo.xlsx` — 25 clientes (capacidade operacional provisória), ranqueados por `valor_em_risco = p_churn_calibrada × receita_anual`, com piso de probabilidade de 0.11. Gerada com dado **ao vivo** do banco (não mais um snapshot fixo) — universo de clientes elegíveis muda a cada execução conforme quem está ativo no banco (776 na 1ª rodada ao vivo, jul/2026).

**Splits temporais:**
- Treino: jan/2023 – dez/2024 → 610 clientes, 36.2% churn
- Teste: jan/2023 – nov/2025 → 724 clientes, 49.0% churn

---

## Features Principais

Importância medida por SHAP (`|SHAP|` médio, Seção 9 do `04_modelo.ipynb`, modelo XGBoost):

| Feature | Descrição | \|SHAP\| médio |
|---|---|---|
| `meses_sem_pedido_pre` | Meses sem compra antes do cutoff | ⭐ 0.8502 — mais forte, disparado |
| `total_pedidos` | Total de pedidos na janela | 0.2003 |
| `n_meses_ativos` | Meses com pedido na janela de análise | 0.1972 |
| `itens_por_pedido` | Média de itens por pedido | 0.0923 |
| `ticket_medio` | Valor médio por pedido | 0.0833 |
| `intervalo_medio` | Intervalo médio entre compras (pré-cutoff) | 0.0756 |
| `DIASINADIMPLENTE` | Dias de inadimplência no cadastro | 0.0490 |

`categoria_pedido`, `sem_historico_itens`, `categoria_cliente`, `sem_historico_cadencia` (SHAP < 0.01) removidas do scoring final. Tabela completa de todas as 19 features na Seção 9 do notebook.

> **`n_itens_portfolio`, `slope_portfolio_medio` e `pct_itens_queda` caíram de importância** (de 0.11/0.09/0.07 para 0.01/0.02/0.04) depois da correção do vazamento — o sinal alto anterior era majoritariamente vazamento de dado futuro, não padrão real de comportamento.

---

## Estrutura do Projeto

```
central_eto/
├── CLAUDE.md                    # instruções do projeto para o assistente
├── README.md                    # este arquivo
├── requirements.txt
├── .env                         # credenciais (nunca commitar)
├── validar_lista_alertas.py     # ad-hoc: status Ativo/Inativo em tempo real dos clientes da lista
├── explicar_cliente.py          # ad-hoc: waterfall SHAP de um cliente específico
├── data/processed/              # outputs dos notebooks e scripts
├── docs/
│   ├── estrutura.txt            # árvore detalhada de todos os arquivos
│   └── variaveis.txt            # dicionário completo das features
├── notebooks/                   # análise e treino do modelo (roda raramente)
├── scripts/
│   ├── scoring_mensal.py        # pipeline de inferência — roda mensalmente, dado ao vivo
│   └── gerar_relatorio.py       # gera reports/lista_alertas_churn.html do zero
├── queries/                     # SQL de extração
├── reports/                     # relatório HTML gerado — nunca editar à mão
└── src/                         # módulos reutilizáveis
    ├── config.py                # constantes centralizadas
    ├── db.py                    # conexão com banco
    ├── features_EDA_pedidos.py  # make_cliente_mes()
    ├── features_comportamento.py # features de volume/ticket/recência pré-cutoff
    ├── features_fidelidade.py   # features de intervalo entre pedidos pré-cutoff
    ├── features_itens.py        # calc_features_itens_pre_cutoff() — tendência de item sem vazamento
    ├── model_prep.py            # preparar_xy()/preparar_X() — mapeamento e fillna, único lugar
    └── sanity_check.py          # validação de dataframes
```

> Veja [`docs/estrutura.txt`](docs/estrutura.txt) para a árvore completa com descrição de cada arquivo.

---

## Configuração do Ambiente

```bash
# Clonar o repositório
git clone https://github.com/stephanie-mendes26/curtomer_churn_prediction.git
cd curtomer_churn_prediction

# Ativar o ambiente virtual
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

# Instalar dependências
pip install -r requirements.txt

# Configurar credenciais do banco
cp .env.example .env            # preencher com os dados de conexão
```

---

## Decisões Técnicas Chave

- **Target:** `churn = 1` se o cliente não pediu dentro da janela de tolerância da sua categoria (2–6 meses após o cutoff), não um horizonte fixo de 3 meses para todos
- **Split:** estritamente temporal — sem data leakage entre treino e teste
- **Vazamento de dado corrigido (jul/2026):** as features de tendência de item (`slope_portfolio_medio`, `pct_itens_queda`, `n_itens_portfolio`) usavam histórico completo em vez de dado pré-cutoff — inflava a AUC-ROC de ~0.89 (real) para ~0.92. Corrigido recalculando essas features por split via `src/features_itens.py`. Ver `CLAUDE.md`, decisão 48.
- **Seleção de modelo é dinâmica:** o notebook escolhe o vencedor entre 4 modelos por AUC-ROC a cada execução — não fixado em código. Consequência direta da correção acima: o vencedor mudou de LightGBM para XGBoost.
- **Imputação de cadência (`cv_pedidos`, `intervalo_medio`, `max_intervalo`):** mediana do treino, calculada só sobre clientes com histórico real, reaplicada no teste sem recálculo (nunca sentinel artificial — ver `CLAUDE.md`, decisão 52)
- **`sem_historico_itens`:** feature binária criada antes do `fillna(0)` de `pct_itens_queda` — distingue clientes sem dados de itens de clientes com portfólio 100% em crescimento
- **Optuna com decisão automática tuned vs. default:** só substitui o modelo default se o tunado não piorar nenhuma métrica (AUC-ROC e AUC-PR) — regra de código, não julgamento manual a cada execução
- **Calibração de probabilidade:** boosted trees são mal-calibradas por padrão; `CalibratedClassifierCV` (isotonic) aplicado antes de multiplicar `p × receita`
- **Ranking por valor em risco, não por probabilidade pura:** `valor_em_risco = p_calibrada × receita_anual_individual` — nunca receita média da categoria (o máximo do segmento "baixo" já supera a mediana do "premium")
- **Treino separado de inferência:** `04_modelo.ipynb` só treina (roda raramente, com revisão humana); `scripts/scoring_mensal.py` recalcula as features **ao vivo** a partir do banco (não usa mais snapshot fixo), aplica o modelo já treinado (`data/processed/modelo_churn.pkl`) e gera o relatório — tudo numa única chamada, sem depender do Jupyter

---

## Como rodar

**Uso mensal (produção):**
```bash
.venv\Scripts\python.exe scripts\scoring_mensal.py
```
Puxa dado fresco do banco, aplica o modelo já treinado, e gera `scores_2026.parquet`, `clientes_alerta_modelo.xlsx` e `reports/lista_alertas_churn.html` — tudo numa chamada.

**Retreinar o modelo (raro):** rodar em sequência `01_pedidos_eda.ipynb` → `02_itens_eda.ipynb` → `03_base_master.ipynb` → `04_modelo.ipynb`, depois `scoring_mensal.py`. Ver `CLAUDE.md` para o runbook completo.

---

## Próximos Passos

- [ ] Validar `clientes_alerta_modelo.xlsx` com a equipe comercial — lista gerada com dado ao vivo e modelo corrigido (XGBoost)
- [ ] Cravar `PISO_RISCO` (atual: 0.11) e `N_CAPACIDADE` (atual: 25) com o cliente — hoje são placeholders provisórios
- [ ] Confirmar com o cliente o custo relativo FN/FP para fixar threshold em produção
- [ ] Investigar outlier de receita anualizada em clientes com histórico parcial (máximo observado R$ 3,52M — ver CLAUDE.md, decisão 42)
- [ ] Agendar `scripts/scoring_mensal.py` no Windows Task Scheduler pra rodar mensalmente, após o fechamento do mês
- [ ] Re-rodar `validar_lista_alertas.py` contra a lista atual (a validação anterior foi feita com os scores do modelo com vazamento)
