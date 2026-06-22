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

- **Python environment:** sempre usar o `.venv` dentro de `c:\Users\carva\OneDrive\Área de Trabalho\central_eto\.venv` — este é o projeto ativo. Existe uma cópia antiga em `C:\Users\carva\central_eto` com `data/processed/` **vazia** — nunca usar esse kernel.
- **Banco de dados:** acessado via `src/db.py` — usar `db.get_data(sql)` para queries, `db.test_connection()` para validar conexão
- **Constantes centralizadas em `src/config.py`:** importar de lá, nunca redefinir inline
- **Data de referência fixa:** `DATA_REF = pd.Timestamp("2026-05-01")` — fim dos dados disponíveis
- **Janela temporal:** dados a partir de `2023-01-01` (743 clientes ativos, maior qualidade)
- **Splits de modelagem:** `CUTOFF_TREINO = 2024-12-31`, `CUTOFF_TESTE = 2025-11-30`, `HORIZONTE_MESES = 3`
- **Paleta de cores:** verde `["#d8f3dc", "#b7e4c7", "#95d5b2", "#74c69d", "#52b788", "#40916c", "#2d6a4f", "#1b4332"]`
- **Valores negativos em VL_SERVICO:** tratados com `neg_policy="clip0"` (zerados, não removidos)
- **Parquet:** outputs salvos em `data/processed/` sempre com `index=False`
- **Nunca commitar `.env`**
- **PROJECT_ROOT nos notebooks:** usar detecção dinâmica `next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists())` — nunca hardcodar caminho absoluto
- **`estrutura.txt` deve ser atualizado sempre** que um arquivo for criado, renomeado, removido ou mudar de status — é a referência de navegação do projeto

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
| `sem_historico_itens` | 1 se o cliente não tem nenhum dado em `cliente_item_tendencia` (NaN original em pct_itens_queda) |
| `DIASINADIMPLENTE` | Dias em atraso (sinal fraco, Pearson ~0.12) |

**Splits:**
- **Treino:** features calculadas até `CUTOFF_TREINO = 2024-12`, target = sem pedido em jan–mar/2025
- **Teste:** features calculadas até `CUTOFF_TESTE = 2025-11`, target = sem pedido em dez/2025–fev/2026

**Imputação de NaN (aplicada em `03_base_master.ipynb` e replicada em `04_modelo.ipynb` Seção 8):**
- `intervalo_medio` → `2 × max_treino` (sinal de intervalo muito longo; constante do treino aplicada no teste para evitar leakage)
- `pct_itens_queda` / `slope_portfolio_medio` → `0` após criação de `sem_historico_itens`

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
- **Correção aplicada:** `PROJECT_ROOT` era hardcoded para `C:\Users\carva\central_eto` (cópia antiga); substituído por detecção dinâmica

### `03_base_master.ipynb` — CONCLUÍDO (re-executar após adicionar `sem_historico_itens`)
- Construção do dataset de modelagem com split temporal
- Target `churn` definido com janela personalizada por categoria (não horizonte fixo de 3 meses para todos)
- Features recalculadas sobre dados pré-CUTOFF para evitar leakage
- Correção de leakage: join do treino usa `df_target[["CLIENTE", "churn"]]` apenas — sem colunas pós-CUTOFF
- Tratamento de NaN: `intervalo_medio` → 2 × max_treino; `pct_itens_queda` / `slope_portfolio_medio` → 0
- **Nova feature adicionada:** `sem_historico_itens` criada ANTES do fillna(0) em ambos os splits — distingue 0-real de 0-imputado
- Outputs: `df_model_treino.parquet` (610 clientes, 36.2% churn) e `df_model_teste.parquet` (724 clientes, 49.0% churn)

### `04_modelo.ipynb` — EM ANDAMENTO (Seção 9 em execução, Seção 10 pendente)

**Seções concluídas:**

- **Seção 1** — Setup, carga dos parquets, `preparar_xy()`
- **Seção 2** — EDA: Pearson + MI com target, boxplots top features, heatmap entre features
- **Seção 3** — Remove `razao_atividade` (certeza); gera `X_train_clean` / `X_test_clean`
- **Seção 4** — Distribuição de classes: 36.2% treino, 49.0% teste; justificativa para não usar SMOTE
- **Seção 5** — Baseline: DummyClassifier + VIF + Logistic Regression
- **Seção 6** — Comparação tree-based:

  | Modelo | AUC-ROC | AUC-PR |
  |---|---|---|
  | LightGBM | **0.9140** | 0.9022 |
  | XGBoost | 0.9108 | **0.9034** |
  | Random Forest | 0.9071 | 0.8980 |
  | Logistic Regression | 0.8789 | 0.8693 |

- **Seção 7** — Threshold via F2 (recall pesa 2× — FN mais caro em B2B); curva PR + tabela de sweep + matriz de confusão
- **Seção 8** — Optuna (100 trials, StratifiedKFold 5, otimiza AUC-PR): ganho de +0.0027 em AUC-PR, ROC -0.0029 — **modelo tunado descartado** (ganho irrelevante; threshold F2-ótimo em 0.059 gera 65% da base como alerta, inoperacionalizável)
- **Seção 9** — SHAP em execução: beeswarm + waterfall + dependence plot + decisão sobre hipóteses de features

**Modelo oficial: LightGBM default** com `y_prob_lgbm` e `THRESHOLD` definido na Seção 7.

> **Atenção:** `03_base_master.ipynb` foi alterado (adição de `sem_historico_itens`). Re-executar o 03 e depois rodar o 04 do início para que a nova feature entre no X_train/X_test.

**Seção pendente:**

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
9. **Leakage corrigido no join do treino:** `df_target` contém colunas pós-CUTOFF (`primeiro_pedido_pos`, `outcome_end`, `categoria`, `threshold_meses`). O join foi corrigido para `df_target[["CLIENTE", "churn"]]` em `03_base_master.ipynb`. Parquets re-gerados e modelo confirmado robusto após a correção.
10. **`quantidade = count(linhas)` em itens:** o sistema registra uma linha por unidade — portanto `count` é a quantidade real, não uma proxy.
11. **Sazonalidade de itens descartada:** 11.048 pares (cliente, item) sem dados suficientes para teste de Kruskal; não incluída como feature.
12. **Constantes centralizadas em `src/config.py`:** DATA_REF, INICIO, CUTOFF_TREINO, CUTOFF_TESTE, HORIZONTE_MESES, LIMITES_CATEGORIA, THRESHOLD_CHURN, VERDE.
13. **Hipóteses de remoção de features são exploratórias:** listas geradas na Seção 2-3 do `04_modelo` (baixo sinal + alta correlação entre features) não são filtros definitivos para XGBoost. Decisão final vem do SHAP (Seção 9). Exceção: `razao_atividade` removida com certeza (transformação linear).
14. **VIF implementado via sklearn** (sem statsmodels): `1 / (1 - R²)` por regressão de cada feature contra as demais. Só relevante para a Regressão Logística.
15. **Optuna descartado para o modelo oficial:** ganho de +0.0027 em AUC-PR é ruído para 610 exemplos de treino; `scale_pos_weight` alto deslocou scores → threshold F2-ótimo em 0.059 (65% da base como alerta, inoperacionalizável). Modelo oficial = LightGBM default. Optuna documentado na Seção 8 para referência futura.
16. **Threshold via F2 (beta=2):** FN mais caro que FP em B2B — perder um churner é pior que acionar retenção desnecessária. Confirmar custo relativo com o cliente antes de fixar em produção.
17. **`sem_historico_itens` adicionada para corrigir ambiguidade de imputação:** SHAP identificou comportamento espúrio — valores baixos (azul) de `pct_itens_queda` sendo empurrados para churn porque 0-imputado (sem histórico) era confundido com 0-real (portfólio 100% em crescimento). Feature binária criada ANTES do fillna(0) em `03_base_master.ipynb` nos dois splits e replicada na Seção 10 do `04_modelo.ipynb`.
18. **Cópia antiga do projeto em `C:\Users\carva\central_eto`:** pasta com `data/processed/` vazia e `.venv` desatualizado. Não usar — pode causar `FileNotFoundError` silencioso se o kernel apontar para lá.

---

## Próximos Passos

### IMEDIATO
1. Re-executar `03_base_master.ipynb` do início (nova feature `sem_historico_itens`)
2. Re-rodar `04_modelo.ipynb` do início com os parquets atualizados
3. Concluir Seção 9 — SHAP (coletar resultados e documentar decisão de features)
4. Escrever e rodar Seção 10 — aplicação no snapshot atual → `clientes_alerta_modelo.xlsx`

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
- [ ] Custo assimétrico: confirmar com o cliente a proporção FN/FP para fixar threshold em produção (atualmente usando F2 como proxy — FN pesa 2× FP)
- [ ] Capacidade operacional de retenção: quantos clientes a equipe consegue contatar por mês? Esse número pode definir o threshold diretamente via `n_alertas` na tabela de sweep (Seção 7)
