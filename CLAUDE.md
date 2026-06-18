# Projeto — Análise e Previsão de Churn

## Objetivo
Desenvolver um modelo de previsão de churn para a base de clientes de uma empresa distribuidora. O cliente é B2B: cada "cliente" é uma empresa (ex.: clínica odontológica, clínica de cirurgia plástica) que faz pedidos recorrentes de materiais.

---

## Estrutura do Projeto

```
CENTRX/
├── CLAUDE.md
├── .env                        # credenciais de banco (nunca commitar)
├── pyvenv.cfg
├── estrutura.txt
├── variaveis.txt               # dicionário completo das features do modelo
├── threshold_churn             # referência de thresholds definidos
├── data/
│   └── processed/
│       ├── cliente_mes.parquet
│       ├── cliente_fidelidade.parquet
│       ├── cliente_item_tendencia.parquet
│       ├── clientes_alerta.xlsx
│       ├── df_master.parquet
│       ├── df_model.parquet
│       ├── df_model_treino.parquet
│       └── df_model_teste.parquet
├── notebooks/
│   ├── 01_pedidos_eda.ipynb    # CONCLUÍDO
│   ├── 02_clientes_eda.ipynb   # PAUSADO (variáveis não confiáveis)
│   ├── 02_itens_eda.ipynb      # CONCLUÍDO
│   ├── 03_base_master.ipynb    # CONCLUÍDO
│   └── 04_modelo.ipynb         # EM ANDAMENTO
├── queries/
│   ├── base_pedidos.sql
│   ├── base_clientes.sql
│   ├── base_clientes_enriquecida.sql
│   ├── profiling_pedidos.sql
│   └── profiling_clientes.sql
└── src/
    ├── __init__.py
    ├── config.py               # constantes centralizadas (DATA_REF, CUTOFFs, thresholds, paleta)
    ├── db.py                   # conexão com banco via db.get_data(sql) e db.test_connection()
    ├── features_EDA_pedidos.py # contém make_cliente_mes()
    ├── sanity_check.py         # contém sanity_check()
    └── utils.py
```

---

## Convenções do Projeto

- **Python environment:** sempre usar o `.venv` local (`pyvenv.cfg` na raiz)
- **Banco de dados:** acessado via `src/db.py` — usar `db.get_data(sql)` para queries, `db.test_connection()` para validar conexão
- **Constantes centralizadas em `src/config.py`:** importar de lá, nunca redefinir inline
- **Data de referência fixa:** `DATA_REF = pd.Timestamp("2026-05-01")` — fim dos dados disponíveis
- **Janela temporal:** dados a partir de `2023-01-01` (743 clientes ativos, maior qualidade)
- **Splits de modelagem:** `CUTOFF_TREINO = 2024-12-31`, `CUTOFF_TESTE = 2025-11-30`, `HORIZONTE_MESES = 3`
- **Paleta de cores:** verde `["#d8f3dc", "#b7e4c7", "#95d5b2", "#74c69d", "#52b788", "#40916c", "#2d6a4f", "#1b4332"]`
- **Valores negativos em VL_SERVICO:** tratados com `neg_policy="clip0"` (zerados, não removidos)
- **Parquet:** outputs salvos em `data/processed/` sempre com `index=False`
- **Nunca commitar `.env`**

---

## Tabelas Principais

### `df_base` (origem: banco via `queries/base_pedidos.sql`)
Granularidade: uma linha por item de pedido.

| Coluna | Descrição |
|---|---|
| `CLIENTE` | ID do cliente |
| `NRO_PED` | Número do pedido |
| `DATA_ENTRADA` | Data do pedido |
| `VL_SERVICO` | Valor do item |
| `SETOR` | Tipo de estabelecimento do cliente |

> Duplicatas esperadas: o sistema registra uma linha por unidade de item. Dois itens idênticos no mesmo pedido geram duas linhas — comportamento correto, não erro.

---

### `cliente_mes` (gerada por `src/features_EDA_pedidos.py → make_cliente_mes()`)
Granularidade: cliente × mês. Salva em `data/processed/cliente_mes.parquet`.

| Feature | Descrição |
|---|---|
| `CLIENTE` | ID do cliente |
| `ANO_MES` | Período (Period M) |
| `total_pedidos` | Pedidos únicos no mês (nunique de NRO_PED) |
| `total_valor` | Faturamento mensal (negativos zerados) |
| `qtd_itens` | Total de linhas/itens do mês |
| `ticket_medio` | total_valor / total_pedidos |
| `itens_por_pedido` | qtd_itens / total_pedidos |
| `categoria_pedido` | baixo / medio / alto / premium |

**Thresholds de categoria_pedido:**
- `baixo` → total_pedidos ≤ 2
- `medio` → 3–5
- `alto` → 6–19
- `premium` → ≥ 20 (7 clientes com > 108 pedidos/mês são tratados como extremos dentro do premium, sem categoria nova)

---

### `cliente_fidelidade` (gerada no `01_pedidos_eda.ipynb`)
Granularidade: um registro por cliente. Salva em `data/processed/cliente_fidelidade.parquet`.

| Feature | Descrição |
|---|---|
| `total_meses_ativos` | Meses com pelo menos 1 pedido |
| `intervalo_medio` | Média de meses entre pedidos consecutivos |
| `max_intervalo` | Maior intervalo registrado (em meses) |
| `total_valor_historico` | Faturamento acumulado |
| `ticket_medio_geral` | Ticket médio histórico |
| `valor_ultimo_mes` | Faturamento no último mês ativo |
| `meses_sumido` | Meses desde o último pedido até DATA_REF |
| `n_intervalos` | Quantidade de intervalos calculados |
| `threshold_churn` | Meses de silêncio tolerados por categoria |
| `categoria_cliente` | Moda de categoria_pedido — perfil geral |
| `RED_FLAG` | True se meses_sumido > threshold_churn AND n_intervalos ≥ 3 |
| `historico_confiavel` | True se n_intervalos ≥ 3 |

> `cliente_fidelidade` usa o histórico completo até DATA_REF. Para modelagem, as features de intervalo são **recalculadas sobre dados pré-CUTOFF** em `03_base_master.ipynb` para evitar leakage.

**Thresholds de churn por categoria:**
- `premium` → 2 meses
- `alto` → 2 meses
- `medio` → 3 meses
- `baixo` → 6 meses

---

### `cliente_item_tendencia` (gerada no `02_itens_eda.ipynb`)
Granularidade: cliente × item. Salva em `data/processed/cliente_item_tendencia.parquet`.

| Feature | Descrição |
|---|---|
| `CLIENTE` | ID do cliente |
| `SERVICO` | Nome do item |
| `tendencia_slope` | Inclinação da reta linear de quantidade ao longo dos meses (np.polyfit) |
| `var_pct_ultimo` | Variação percentual de quantidade no último mês ativo |
| `RED_FLAG` | Flag de risco herdada do cliente_fidelidade |

> `quantidade = count(linhas)` — o sistema registra uma linha por unidade, portanto count é a quantidade real pedida.

---

### `df_model_treino` / `df_model_teste` (geradas no `03_base_master.ipynb`)
Granularidade: um registro por cliente. Dataset final para modelagem.

| Coluna | Descrição |
|---|---|
| `CLIENTE`, `NOME` | Identificadores (fora do modelo) |
| `churn` | Target: 1 se o cliente não pediu nada nos 3 meses após o CUTOFF |
| `n_meses_ativos` | Meses com pedido na janela pré-CUTOFF |
| `total_pedidos` | Soma de pedidos na janela |
| `total_valor` | Faturamento total na janela |
| `media_pedidos_mes` | total_pedidos / n_meses_ativos |
| `ticket_medio` | Valor médio por pedido |
| `itens_por_pedido` | Média de itens por pedido |
| `cv_pedidos` | Coeficiente de variação dos pedidos mensais |
| `meses_sem_pedido_pre` | Recência — meses sem pedir antes do CUTOFF (feature mais forte, Pearson ~0.62) |
| `razao_atividade` | n_meses_ativos / 23 — candidata a remoção (redundante) |
| `categoria_pedido` | Moda de categoria_pedido — candidata a remoção (sinal fraco) |
| `intervalo_medio` | Média de meses entre pedidos (recalculada pré-CUTOFF) |
| `max_intervalo` | Maior intervalo histórico pré-CUTOFF |
| `n_intervalos` | Quantidade de intervalos pré-CUTOFF |
| `categoria_cliente` | Moda de categoria na janela pré-CUTOFF |
| `slope_portfolio_medio` | Média dos slopes de todos os itens do cliente |
| `pct_itens_queda` | % de itens com slope < 0 |
| `n_itens_portfolio` | Itens distintos comprados na janela |
| `DIASINADIMPLENTE` | Dias em atraso (sinal fraco, Pearson ~0.12) |

**Splits:**
- **Treino:** features calculadas até `CUTOFF_TREINO = 2024-12`, target = sem pedido em jan–mar/2025
- **Teste:** features calculadas até `CUTOFF_TESTE = 2025-11`, target = sem pedido em dez/2025–fev/2026

---

## Estado Atual por Notebook

### `01_pedidos_eda.ipynb` — CONCLUÍDO
- EDA da base de pedidos
- Criação de `cliente_mes` e `cliente_fidelidade`
- Sanity checks realizados
- Dashboard por cliente implementado (troca `CLIENTE_ID` para visualizar)
- Outputs salvos em `data/processed/`

### `02_clientes_eda.ipynb` — PAUSADO
- Tabela clientes com variáveis de confiabilidade insuficiente
- **Decisão do cliente: manter fora do escopo ativo**
- Não usar variáveis desta tabela no modelo até nova validação
- Exceção: `DIASINADIMPLENTE` incluído via `queries/base_clientes.sql` diretamente no `03_base_master.ipynb`

### `02_itens_eda.ipynb` — CONCLUÍDO
- EDA de itens: distribuição, itens mais pedidos, concentração de portfólio
- Teste de sazonalidade por Kruskal-Wallis (11.048 pares sem dados suficientes — sazonalidade descartada)
- Feature `n_itens_distintos` por cliente × mês
- Feature `tendencia_slope` via `np.polyfit` por (cliente, item)
- Feature `var_pct_ultimo` — variação percentual mês a mês
- Output: `cliente_item_tendencia.parquet`

### `03_base_master.ipynb` — CONCLUÍDO
- Construção do dataset de modelagem com split temporal
- Target `churn` definido: 1 = nenhum pedido nos 3 meses após CUTOFF
- Features recalculadas sobre dados pré-CUTOFF para evitar leakage
- Correção de leakage: `n_intervalos`, `intervalo_medio`, `max_intervalo` e `categoria_cliente` calculados sobre `cm_feat` (pré-CUTOFF), não sobre `cliente_fidelidade`
- Tratamento de NaN: `intervalo_medio` NaN → 2 × max observado no treino (constante salva para aplicar no teste)
- Outputs: `df_model_treino.parquet` e `df_model_teste.parquet`

### `04_modelo.ipynb` — EM ANDAMENTO

> **Atenção:** o `03_base_master.ipynb` foi corrigido (leakage no join do treino) mas os parquets ainda não foram re-gerados. **Re-executar o 03 antes de rodar o 04.**

**Seções escritas (código no notebook, aguardando re-execução após correção do master):**

- **Seção 1** — Setup, carga dos parquets, `preparar_xy()`
- **Seção 2** — EDA: Pearson + MI com target, boxplots top features, heatmap entre features
  - `HIPOTESES_BAIXO_SINAL`: features com Pearson < 0.10 E MI < 0.05 (hipótese, confirmada pelo SHAP)
  - `HIPOTESES_REDUNDANCIA`: pares com corr > 0.80, remove a de menor score combinado (hipótese)
- **Seção 3** — Análise exploratória de features
  - Remove `razao_atividade` agora (certeza — transformação linear de `n_meses_ativos`)
  - Gera `X_train_clean` / `X_test_clean`
- **Seção 4** — Distribuição de classes: 36.2% treino, 49.0% teste
  - Sem SMOTE, sem `scale_pos_weight` fixo
  - `scale_pos_weight` vai como hiperparâmetro no Optuna (Seção 8)
  - Shift 36%→49% é distribuição real (clientes novos de 2025), tratado por calibração de threshold no teste
- **Seção 5** — Baseline
  - 5.1 DummyClassifier (chão absoluto)
  - 5.2 VIF via sklearn (sem statsmodels): `1 / (1 - R²)`, gera `EXCLUIR_LOGISTICA`
  - 5.3 Logistic Regression com StandardScaler + `class_weight="balanced"` + plot de coeficientes
- **Seção 6** — Comparação de modelos tree-based (todos com parâmetros default + early stopping)
  - XGBoost, LightGBM, Random Forest
  - Tabela comparativa AUC-ROC + AUC-PR; vencedor segue para Optuna

**Seções ainda não escritas:**

- **Seção 7** — Threshold de decisão (curva PR no teste, F-beta, justificativa de negócio)
- **Seção 8** — Optuna: `scale_pos_weight` como hiperparâmetro, otimizar AUC-PR, StratifiedKFold(5)
- **Seção 9** — SHAP: beeswarm, waterfall de casos, dependence plot → decisão final sobre hipóteses
- **Seção 10** — Aplicação no snapshot atual → `clientes_alerta_modelo.xlsx`

---

## Decisões Técnicas Registradas

1. **Tabela clientes fora do escopo:** variáveis não confiáveis o suficiente para modelagem. Exceção: `DIASINADIMPLENTE` incluso diretamente via SQL.
2. **Janela 2023+:** dados anteriores têm menor qualidade. Usar `inicio="2023-01-01"` em `make_cliente_mes()`.
3. **neg_policy="clip0":** valores negativos de VL_SERVICO são zerados, não removidos — preserva o registro histórico.
4. **Categoria premium sem subdivisão adicional:** 7 clientes com >108 pedidos/mês não justificam nova categoria.
5. **RED_FLAG requer n_intervalos ≥ 3:** clientes com histórico curto não recebem flag — evita falso positivo em clientes novos.
6. **Target = horizonte temporal (não RED_FLAG):** `churn = 1` se o cliente não fez nenhum pedido nos 3 meses após o CUTOFF. Mais operacionalizável e livre de leakage.
7. **Split temporal estrito:** sem data leakage entre treino e teste. Features calculadas exclusivamente sobre dados anteriores ao CUTOFF de cada split.
8. **Leakage corrigido em features de intervalo:** `n_intervalos`, `intervalo_medio`, `max_intervalo` e `categoria_cliente` eram puxados de `cliente_fidelidade` (histórico até DATA_REF). Corrigido em `03_base_master.ipynb` para usar apenas dados pré-CUTOFF.
9. **Leakage corrigido no join do treino:** `df_target` contém colunas pós-CUTOFF (`primeiro_pedido_pos`, `outcome_end`, `categoria`, `threshold_meses`). O join usava `df_target` inteiro — corrigido para `df_target[["CLIENTE", "churn"]]` em `03_base_master.ipynb`. Parquets precisam ser re-gerados.
10. **`quantidade = count(linhas)` em itens:** o sistema registra uma linha por unidade — portanto `count` é a quantidade real, não uma proxy.
11. **Sazonalidade de itens descartada:** 11.048 pares (cliente, item) sem dados suficientes para teste de Kruskal; não incluída como feature.
12. **Constantes centralizadas em `src/config.py`:** DATA_REF, INICIO, CUTOFF_TREINO, CUTOFF_TESTE, HORIZONTE_MESES, LIMITES_CATEGORIA, THRESHOLD_CHURN, VERDE.
13. **Hipóteses de remoção de features são exploratórias:** listas geradas na Seção 2-3 do `04_modelo` (baixo sinal + alta correlação entre features) não são filtros definitivos para XGBoost. Decisão final vem do SHAP (Seção 9). Exceção: `razao_atividade` removida com certeza (transformação linear).
14. **VIF implementado via sklearn** (sem statsmodels): `1 / (1 - R²)` por regressão de cada feature contra as demais. Só relevante para a Regressão Logística.
15. **`scale_pos_weight` vai para Optuna:** a 36% de churn o imbalanceamento não é severo. `scale_pos_weight` entra como hiperparâmetro no Optuna — a CV decide se ajuda.

---

## Próximos Passos

### IMEDIATO — Completar `04_modelo.ipynb` (Seções 3–10)
Ver detalhamento completo no estado do notebook acima. Sequência: seleção de features → baseline → modelo principal → threshold → tuning → SHAP → aplicação.

### P2 — Subfragmentação da categoria "Baixo" (pendente)
Categoria atual "baixo" mistura clientes recorrentes de baixo volume com clientes sazonais (compra 1x/ano).
- Critério sugerido: `n_meses_ativos / janela_meses`
  - `baixo_recorrente` → razão ≥ 0.4
  - `baixo_sazonal` → razão < 0.4
- Threshold de churn para `baixo_sazonal` baseado em intervalo máximo histórico
- Validar critério com o cliente antes de implementar

### P3 — Categorização por tipo de cliente (SETOR) (pendente)
- Avaliar qualidade e cardinalidade da variável `SETOR` em `df_base`
- Agrupar em super-categorias se necessário (ex.: "saúde dental", "saúde estética")
- Usar como feature categórica ou critério de estratificação na avaliação

---

## Perguntas em Aberto

- [ ] Subfragmentação do "baixo": validar threshold `razao_atividade ≥ 0.4` com o cliente
- [ ] Qualidade da variável `SETOR`: verificar cardinalidade e cobertura antes de usar
- [ ] Custo assimétrico: qual o custo relativo de falso negativo vs falso positivo para definir threshold de decisão?
