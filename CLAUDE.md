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
├── data/
│   └── processed/              # outputs dos notebooks
├── docs/
│   ├── estrutura.txt           # árvore detalhada do projeto
│   ├── variaveis.txt           # dicionário completo das features
│   └── threshold_churn         # referência de thresholds
├── notebooks/
│   ├── 01_pedidos_eda.ipynb    # CONCLUÍDO
│   ├── 02_clientes_eda.ipynb   # PAUSADO
│   ├── 02_itens_eda.ipynb      # CONCLUÍDO
│   ├── 03_base_master.ipynb    # CONCLUÍDO
│   └── 04_modelo.ipynb         # EM ANDAMENTO
├── queries/
├── reports/
│   ├── apresentacao_churn.pptx
│   └── gerar_apresentacao.py
└── src/
    ├── config.py               # constantes centralizadas
    ├── db.py                   # conexão com banco
    ├── features_EDA_pedidos.py
    └── sanity_check.py
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
- **`docs/estrutura.txt` deve ser atualizado sempre** que um arquivo for criado, renomeado, removido ou mudar de status — é a referência de navegação do projeto
- **`README.md` deve ser atualizado sempre** que: um notebook mudar de status, os resultados do modelo mudarem, uma decisão técnica relevante for tomada, ou uma seção nova for concluída

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
| `sem_historico_itens` | 1 se o cliente está ausente de `cliente_item_tendencia` OU presente mas com todos os slopes NaN (mean = NaN, pct_itens_queda = 0.0 por lambda) |
| `sem_historico_cadencia` | 1 se `n_intervalos == 0` (n_meses_ativos == 1) — cv_pedidos, intervalo_medio e max_intervalo são estruturalmente indefinidos para esses clientes |
| `DIASINADIMPLENTE` | Dias em atraso (sinal fraco, Pearson ~0.12) |

**Splits:**
- **Treino:** features calculadas até `CUTOFF_TREINO = 2024-12`, target = sem pedido em jan–mar/2025
- **Teste:** features calculadas até `CUTOFF_TESTE = 2025-11`, target = sem pedido em dez/2025–fev/2026

**Imputação de NaN (aplicada em `03_base_master.ipynb`, constantes do treino reusadas no teste):**
- Flags criadas **antes** de qualquer fillna em ambos os splits
- `intervalo_medio` → `intervalo_max_real × 2` onde `intervalo_max_real` é o max pré-fillna dos clientes com histórico real (~20 meses → fill ~40); mesma constante em treino e teste
- `max_intervalo` → mesmo sentinel que `intervalo_medio` — evita a inversão física max < media que corrompeu correlação para -0.33
- `cv_pedidos` → mediana do treino (clientes com n_meses_ativos ≥ 2); não mais 0, que conflata "cliente novo" com "cliente perfeitamente regular"
- `pct_itens_queda` / `slope_portfolio_medio` → `0` após criação de `sem_historico_itens` (expandido)

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

### `03_base_master.ipynb` — CONCLUÍDO (re-executar após auditoria de imputações jul/2026)
- Construção do dataset de modelagem com split temporal
- Target `churn` definido com janela personalizada por categoria (não horizonte fixo de 3 meses para todos)
- Features recalculadas sobre dados pré-CUTOFF para evitar leakage
- Correção de leakage: join do treino usa `df_target[["CLIENTE", "churn"]]` apenas — sem colunas pós-CUTOFF
- **Auditoria de imputações (jul/2026):** corrigidos 4 problemas que enviasavam coeficientes da logística:
  1. `max_intervalo` retornava `0` hardcoded para clientes de 1 mês → corrigido para `np.nan`
  2. `intervalo_max_real` calculado pós-fillna → corrigido para pré-fillna; treino e teste usam mesma constante (~40 meses)
  3. `cv_pedidos` preenchia com `0` (conflata novo com perfeito) → agora mediana do treino (clientes com n_meses_ativos ≥ 2)
  4. `sem_historico_itens` não cobria 162 clientes com slopes-todos-NaN → critério expandido para `pct_itens_queda.isna() | slope_portfolio_medio.isna()`
- **Novas features:** `sem_historico_cadencia` (n_intervalos == 0) e `sem_historico_itens` expandido — ambas criadas ANTES de qualquer fillna em ambos os splits
- Outputs: `df_model_treino.parquet` (610 clientes, 36.2% churn) e `df_model_teste.parquet` (724 clientes, 49.0% churn)

### `04_modelo.ipynb` — EM ANDAMENTO (Seção 9 pendente execução, Seção 10 pendente)

**Seções concluídas (código corrigido, re-executar do início):**

- **Seção 1** — Setup, carga dos parquets, `preparar_xy()` com fillna semântico para `DIASINADIMPLENTE` e `n_itens_portfolio`
- **Seção 2** — EDA: Pearson + MI com target, boxplots top features, heatmap entre features
- **Seção 3** — Remove `razao_atividade` (certeza); gera `X_train_clean` / `X_test_clean`
- **Seção 4** — Distribuição de classes: 36.2% treino, 49.0% teste; justificativa para não usar SMOTE
- **Seção 5** — Baseline: DummyClassifier (`strategy="prior"`) + VIF como diagnóstico pré-logística + Logistic Regression
- **Seção 6** — Comparação tree-based sob **protocolo limpo** (jul/2026: early stopping via CV no treino, sem X_*_xgb, sem test no fit). Números abaixo são pré-correção e serão atualizados após re-execução:

  | Modelo | AUC-ROC | AUC-PR |
  |---|---|---|
  | LightGBM | **0.9140** | 0.9022 |
  | XGBoost | 0.9108 | **0.9034** |
  | Random Forest | 0.9071 | 0.8980 |
  | Logistic Regression | 0.9018* | 0.8900* |

  _*pós-Caminho A; * números definitivos após re-execução completa_

- **Seção 7** — Threshold via F2 (recall pesa 2× — FN mais caro em B2B); curva PR + tabela de sweep + matriz de confusão
- **Seção 8** — Optuna (100 trials, StratifiedKFold 5, otimiza AUC-PR). CV objetivo limpo (usa folds do treino). Modelo tunado descartado (ganho irrelevante; threshold F2-ótimo 0.059 → 65% da base como alerta). `baf26ff9` corrigido: modelo final treinado sem eval_set no teste.
- **Seção 9** — SHAP: beeswarm + waterfall + dependence plot + decisão sobre hipóteses de features (pendente re-execução)

**Modelo oficial: LightGBM default** com `y_prob_lgbm` e `THRESHOLD` definido na Seção 7.

> **Para executar:** rodar `04_modelo.ipynb` do início com os parquets já corrigidos (`03_base_master` re-executado). Todas as correções do Caminho A e da auditoria de vazamento (jul/2026) estão no código — basta executar.

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
19. **Auditoria de imputações (jul/2026) — causa-raiz dos sinais contraintuitivos na logística:** 22% da base (136/610 clientes) tinha `n_intervalos == 0` recebendo valores sintéticos que misturavam duas populações. Sintomas: `cv_pedidos`, `pct_itens_queda`, `slope_portfolio_medio` com coeficiente negativo na direção errada; correlação `intervalo_medio ↔ max_intervalo = -0.33` (fisicamente impossível para clientes reais). Causa: `max_intervalo = 0` hardcoded + `intervalo_medio` imputado com 40 criavam inversão max < media nesses 136 clientes. Corrigido com flags + imputações consistentes.
20. **`sem_historico_cadencia` adicionada:** flag binária para `n_intervalos == 0` (n_meses_ativos == 1), criada ANTES do fillna em ambos os splits. Análogo ao `sem_historico_itens` para o eixo de cadência. Permite ao LightGBM distinguir "cliente novo" de "cliente estabelecido regular" sem descartar a observação.
21. **`max_intervalo` corrigido para `np.nan`:** `calc_fidelidade_pre_cutoff` retornava `0` quando `len(meses) < 2` — valor falso que criava inversão física `max < media` para os 136 clientes de 1 mês. Agora retorna `np.nan`; preenchido com mesmo sentinel que `intervalo_medio` (`intervalo_max_real × 2`), garantindo `max == media` para esses clientes em vez de inversão.
22. **`cv_pedidos` corrigido:** `.fillna(0)` substituído por mediana do treino (clientes com n_meses_ativos ≥ 2), calculada pré-fillna. O valor 0 conflata "cliente novo de 1 mês" com "cliente veterano perfeitamente regular" — populações opostas em risco de churn. Mediana é neutro e não cria sinal espúrio.
23. **`intervalo_max_real` calculado pré-fillna:** bug anterior calculava `intervalo_max_treino = df_model["intervalo_medio"].max()` depois do fillna (max pós-fill = 40), resultando em teste imputado com `40 × 2 = 80` vs treino com `20 × 2 = 40`. Corrigido: `intervalo_max_real` calculado antes do fillna (~20 meses reais); treino e teste ambos imputados com ~40.
24. **`sem_historico_itens` expandido:** critério anterior (`pct_itens_queda.isna()`) não detectava 162 clientes presentes em `cit` mas com todos os `tendencia_slope = NaN` — nesses casos `mean() = NaN` mas a lambda `(x < 0).sum() / len(x)` retornava `0.0`, deixando `pct_itens_queda` definido. Critério novo: `pct_itens_queda.isna() | slope_portfolio_medio.isna()`.
25. **Filtro `n_intervalos < 3` descartado do pipeline de modelagem:** a regra existe apenas no contexto do `RED_FLAG` rules-based de `cliente_fidelidade`. Aplicar como filtro de treino removeria ~30% de um dataset já pequeno (610 clientes) e criaria gap de cobertura no scoring 2026 (clientes novos aparecerão sem o modelo tê-los visto). Abordagem correta: `sem_historico_cadencia` + imputação com mediana — o LightGBM aprende a distinção via interações.
26. **DummyClassifier → `strategy="prior"`:** `stratified` amostra aleatoriamente e produz AUC-ROC ligeiramente diferente de 0.50 a cada execução. `prior` atribui probabilidade constante igual à prevalência do treino → AUC-ROC = 0.5000 deterministicamente. Correto para "chão absoluto" reproduzível.
27. **VIF repositionado na Seção 5 do `04_modelo`:** VIF não é baseline — é diagnóstico de multicolinearidade, pré-requisito para a regressão logística (VIF > 10 → coeficiente instável no modelo linear). Removido da tabela de comparação de baselines; descrito como etapa preparatória da Logistic Regression. Coeficientes comparáveis entre si porque o pipeline inclui `StandardScaler()` antes do modelo; VIF é invariante a escala (R² não muda com transformação linear).
28. **Auditoria de vazamento na Seção 6 e 8 (jul/2026):** três vazamentos identificados e corrigidos: (a) `X_*_xgb = X_*_clean.fillna(0)` sobrescrevia o Caminho A com fillna genérico em todos os modelos tree-based — removido; (b) XGBoost e LightGBM usavam `eval_set=[(X_test_xgb, y_test)]` com early stopping → test set selecionava o número de árvores (XGBoost parou em 11/300 por shift de distribuição 36%→49%); (c) Optuna (`baf26ff9`) treinava modelo final com `eval_set=[(X_test_opt, y_test)]` — test selecionava rounds do modelo tunado. O objetivo CV do Optuna era limpo (usa folds do treino); a decisão "default > tuned" não foi contaminada.
29. **Protocolo limpo de early stopping (jul/2026):** XGBoost e LightGBM agora selecionam `n_rounds` via `xgb.cv` / `lgb.cv` com `StratifiedKFold(k=5)` dentro do treino (`early_stopping_rounds=20`, métrica `aucpr`/`average_precision`). O modelo final é re-treinado no **treino completo** com `n_estimators` fixo, sem early stopping, sem tocar o teste. Test set usado uma única vez, na avaliação final. Optuna (`baf26ff9`) também corrigido: `lgbm_tuned.fit(X_train_clean, y_train)` sem eval_set.
30. **NaN semântico para `DIASINADIMPLENTE` e `n_itens_portfolio`:** após Caminho A, dois NaN remanescentes não cobertos pelo `03_base_master`. Tratados em `preparar_xy()` via `_FILLNA_SEMANTICO`: `DIASINADIMPLENTE → 0` (ausência no cadastro = 0 dias em atraso) e `n_itens_portfolio → 0` (coberto por `sem_historico_itens`). Nenhum fillna genérico nos blocos de modelo.
31. **Threshold calibrado no teste — ressalva para produção:** o `THRESHOLD` (F2-ótimo) foi encontrado via `argmax(f2_arr)` sobre `y_test`. Isso significa que o ponto de corte foi escolhido no mesmo conjunto usado para reportar o desempenho — grau leve de overfitting de threshold. Para produção, o correto é recalibrar o threshold num conjunto de validação separado do teste (ex.: reservar 20% do teste só para calibração) ou, mais pragmaticamente, substituir o F2-ótimo por um critério operacional direto (`n_alertas` na tabela de sweep da Seção 7 — "quantos clientes a equipe consegue contatar por mês"). O segundo caminho é mais defensável e mais útil para o cliente. **Ação necessária antes de ir a produção:** definir com o cliente a capacidade operacional de retenção e fixar o threshold por `n_alertas`, não por F2.

---

## Próximos Passos

### IMEDIATO
1. ~~Re-executar `03_base_master.ipynb`~~ — CONCLUÍDO (auditoria jul/2026)
2. Re-rodar `04_modelo.ipynb` do início com os parquets atualizados e protocolo limpo
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
- [ ] **Capacidade operacional de retenção (BLOQUEADOR PARA PRODUÇÃO):** quantos clientes a equipe consegue contatar por mês? Esse número define o threshold via `n_alertas` na tabela de sweep (Seção 7) — substitui o F2-ótimo calibrado no teste, que não deve ir para produção diretamente (ver decisão 31)
- [ ] **ROI do modelo:** coletar do cliente (a) ticket médio mensal por cliente e (b) custo de uma ação de retenção → `ROI = churners_capturados × ticket_médio - n_alertas × custo_retenção`. A tabela de sweep vira simulador de ROI por threshold.

---

## Guia de Comunicação com o Cliente

Respostas prontas para perguntas frequentes sobre o modelo:

**"Quantos clientes vão ser contatados por mês?"**
→ Abrir a tabela de sweep (Seção 7) e mostrar a linha com o `n_alertas` compatível com a capacidade da equipe. Ex.: *"Se conseguem contatar 60/mês, threshold = 0.70 → precisão 90%, recall 62%."*

**"Qual a precisão do modelo?"**
→ Não responder com um número único — depende do threshold. Explicar o trade-off e deixar o cliente escolher via `n_alertas`. Nunca usar accuracy (enganosa com classes desbalanceadas).

**"O modelo acerta quantos por cento?"**
→ Usar AUC-ROC: *"Em 91% das vezes o modelo ranqueia um churner acima de um cliente ativo ao comparar dois aleatórios. O chão seria 50%."*

**"A capacidade operacional afeta a precisão do modelo?"**
→ Não altera os scores do modelo (AUC-ROC = 0.91 fixo). Mas afeta o **recall realizável**: mandar 180 alertas para uma equipe que só age em 60 faz os 120 restantes serem ignorados — o modelo acertou, a retenção falhou por capacidade.

**"Qual o retorno financeiro?"**
→ `ROI = churners_capturados × ticket_médio - n_alertas × custo_retenção`. Precisa de dois números do cliente. Com eles, a tabela de sweep vira simulador de ROI — troca threshold e mostra como o lucro muda.
