# Previsão de Churn de Clientes B2B

![Status](https://img.shields.io/badge/status-em%20andamento-yellow?style=flat-square)
![Python](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![Modelo](https://img.shields.io/badge/modelo%20oficial-LightGBM-2d6a4f?style=flat-square)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.914-brightgreen?style=flat-square)
![AUC-PR](https://img.shields.io/badge/AUC--PR-0.902-brightgreen?style=flat-square)
![Notebooks](https://img.shields.io/badge/notebooks-4%20de%205%20conclu%C3%ADdos-40916c?style=flat-square)
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
     │                                    └── cliente_item_tendencia.parquet
     │
     └─── base_clientes.sql ─────► 03_base_master.ipynb
                                          │
                                          ├── df_model_treino.parquet
                                          └── df_model_teste.parquet
                                                      │
                                                      └── 04_modelo.ipynb
                                                                │
                                                                └── clientes_alerta_modelo.xlsx
```

---

## Status dos Notebooks

| Notebook | Descrição | Status |
|---|---|---|
| `01_pedidos_eda.ipynb` | EDA de pedidos, criação de `cliente_mes` e `cliente_fidelidade` | ✅ Concluído |
| `02_clientes_eda.ipynb` | EDA de cadastro de clientes | ⏸️ Pausado — variáveis não confiáveis |
| `02_itens_eda.ipynb` | EDA de itens, criação de `cliente_item_tendencia` | ✅ Concluído |
| `03_base_master.ipynb` | Construção do dataset de modelagem com split temporal | ✅ Concluído |
| `04_modelo.ipynb` | Modelagem, threshold, SHAP, aplicação | 🔄 Em andamento (Seção 10 pendente) |

---

## Resultados do Modelo

| Modelo | AUC-ROC | AUC-PR |
|---|---|---|
| **LightGBM (oficial)** | **0.9140** | **0.9022** |
| XGBoost | 0.9108 | 0.9034 |
| Random Forest | 0.9071 | 0.8980 |
| Regressão Logística | 0.8789 | 0.8693 |
| Dummy (baseline) | 0.5095 | 0.4952 |

**Threshold:** F2-ótimo (recall pesa 2× — falso negativo é mais caro em B2B).

**Splits temporais:**
- Treino: jan/2023 – dez/2024 → 610 clientes, 36.2% churn
- Teste: jan/2023 – nov/2025 → 724 clientes, 49.0% churn

---

## Features Principais

| Feature | Descrição | Importância |
|---|---|---|
| `meses_sem_pedido_pre` | Meses sem compra antes do cutoff | ⭐ Mais forte (Pearson ~0.62) |
| `intervalo_medio` | Intervalo médio entre compras (pré-cutoff) | Alta |
| `n_meses_ativos` | Meses com pedido na janela de análise | Alta |
| `pct_itens_queda` | % de itens com tendência de queda | Média |
| `slope_portfolio_medio` | Inclinação média do portfólio (np.polyfit) | Média |
| `sem_historico_itens` | Flag: cliente sem histórico de itens suficiente | Correção de imputação |
| `DIASINADIMPLENTE` | Dias de inadimplência no cadastro | Baixa (Pearson ~0.12) |

---

## Estrutura do Projeto

```
central_eto/
├── CLAUDE.md                   # instruções do projeto para o assistente
├── README.md                   # este arquivo
├── requirements.txt
├── .env                        # credenciais (nunca commitar)
├── data/processed/             # outputs dos notebooks
├── docs/
│   ├── estrutura.txt           # árvore detalhada de todos os arquivos
│   ├── variaveis.txt           # dicionário completo das features
│   └── threshold_churn         # referência de thresholds
├── notebooks/                  # análise e modelagem
├── queries/                    # SQL de extração
├── reports/
│   ├── apresentacao_churn.pptx # slides para o cliente (não técnico)
│   └── gerar_apresentacao.py   # script para regenerar o .pptx
└── src/                        # módulos reutilizáveis
    ├── config.py               # constantes centralizadas
    ├── db.py                   # conexão com banco
    ├── features_EDA_pedidos.py # make_cliente_mes()
    └── sanity_check.py         # validação de dataframes
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
- **Imputação de `intervalo_medio`:** NaN → `2 × max_treino` (sinal de intervalo longo), aplicado com constante do treino para evitar leakage no teste
- **`sem_historico_itens`:** feature binária criada antes do `fillna(0)` de `pct_itens_queda` — distingue clientes sem dados de itens de clientes com portfólio 100% em crescimento
- **Optuna descartado:** ganho de +0.003 em AUC-PR com 610 exemplos de treino é ruído; threshold F2-ótimo do modelo tunado ficou em 0.059 (65% da base como alerta — inoperacionalizável)

---

## Próximos Passos

- [ ] Concluir Seção 10 do `04_modelo.ipynb` — aplicar modelo no snapshot atual e gerar `clientes_alerta_modelo.xlsx`
- [ ] Validar lista de alertas com a equipe comercial
- [ ] Confirmar com o cliente o custo relativo FN/FP para fixar threshold em produção
- [ ] Definir capacidade operacional de retenção (quantos contatos/mês) para calibrar volume de alertas
