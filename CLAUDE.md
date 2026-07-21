# Projeto — Análise e Previsão de Churn

## Objetivo
Desenvolver um modelo de previsão de churn para a base de clientes de uma empresa distribuidora. O cliente é B2B: cada "cliente" é uma empresa (ex.: clínica odontológica, clínica de cirurgia plástica) que faz pedidos recorrentes de materiais.

---

## Estrutura do Projeto

Visão de alto nível — árvore completa e sempre atualizada em `docs/estrutura.txt`:

```
central_eto/
├── CLAUDE.md
├── README.md
├── .env                          # credenciais de banco (nunca commitar)
├── validar_lista_alertas.py      # ad-hoc: status Ativo/Inativo em tempo real (banco) dos clientes da lista
├── explicar_cliente.py           # ad-hoc: waterfall SHAP de um cliente, salva PNG
├── data/processed/                # outputs dos notebooks e scripts — nunca editar à mão
├── docs/
│   ├── estrutura.txt             # árvore detalhada do projeto
│   ├── variaveis.txt             # dicionário completo das features
│   └── threshold_churn
├── notebooks/
│   ├── 01_pedidos_eda.ipynb      # CONCLUÍDO
│   ├── 02_clientes_eda.ipynb     # PAUSADO
│   ├── 02_itens_eda.ipynb        # CONCLUÍDO
│   ├── 03_base_master.ipynb      # CONCLUÍDO — pipeline de TREINO (parte 1: dataset)
│   ├── 04_modelo.ipynb           # CONCLUÍDO — pipeline de TREINO (parte 2: modelo)
│   ├── calculo_receita.ipynb     # CONCLUÍDO
│   └── explicar_cliente.ipynb    # ad-hoc: waterfall SHAP interativo
├── scripts/
│   ├── scoring_mensal.py         # pipeline de INFERÊNCIA — roda mensalmente, dado ao vivo
│   └── gerar_relatorio.py        # gera reports/lista_alertas_churn.html do zero
├── queries/
├── reports/
│   └── lista_alertas_churn.html  # gerado por scripts/gerar_relatorio.py — nunca editar à mão
└── src/
    ├── config.py                  # constantes centralizadas
    ├── db.py                      # conexão com banco
    ├── features_EDA_pedidos.py    # make_cliente_mes()
    ├── features_comportamento.py  # features de volume/ticket/recência pré-cutoff
    ├── features_fidelidade.py     # features de intervalo entre pedidos pré-cutoff
    ├── features_itens.py          # features de tendência de item pré-cutoff
    ├── model_prep.py              # preparar_xy() / preparar_X() — mapeamento e fillna, único lugar
    ├── sanity_check.py
    └── utils.py
```

---

## Convenções do Projeto

- **Python environment:** sempre usar o `.venv` dentro de `c:\Users\carva\OneDrive\Área de Trabalho\central_eto\.venv` — este é o projeto ativo. Existe uma cópia antiga em `C:\Users\carva\central_eto` com `data/processed/` **vazia** — nunca usar esse kernel.
- **Banco de dados:** acessado via `src/db.py` — usar `db.get_data(sql)` para queries, `db.test_connection()` para validar conexão
- **Constantes centralizadas em `src/config.py`:** importar de lá, nunca redefinir inline
- **Sempre invocar o Python via `.venv/Scripts/python.exe` explícito (ou o kernel Jupyter `central_eto`):** o `python`/`python -m jupyter` do PATH do sistema resolve para a cópia antiga em `C:\Users\carva\central_eto\.venv` (mesma pasta citada abaixo como proibida) — confirmado em jul/2026 quando `python -c "import pandas"` carregou o pacote de lá. Kernel Jupyter registrado corretamente aponta para este projeto: `central_eto` → `c:\Users\carva\OneDrive\Área de Trabalho\central_eto\.venv\Scripts\python.exe`.
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
| `churn` | Target: 1 se o cliente não pediu dentro da janela de tolerância da sua categoria (2–6 meses após o CUTOFF, dict `THRESHOLD_CHURN`) — **não** é horizonte fixo de 3 meses |
| `n_meses_ativos` | Meses com pedido na janela pré-CUTOFF |
| `total_pedidos` | Soma de pedidos na janela |
| `total_valor` | Faturamento total na janela |
| `media_pedidos_mes` | total_pedidos / n_meses_ativos |
| `ticket_medio` | Valor médio por pedido — mantida no modelo (SHAP derrubou a hipótese de sinal fraco) |
| `itens_por_pedido` | Média de itens por pedido |
| `cv_pedidos` | Coeficiente de variação dos pedidos mensais |
| `meses_sem_pedido_pre` | Recência — meses sem pedir antes do CUTOFF (feature mais forte, disparado — Pearson ~0.62, SHAP ~4x a 2ª colocada) |
| `razao_atividade` | n_meses_ativos / janela_meses do split — **REMOVIDA** do modelo (transformação linear, zero informação nova) |
| `categoria_pedido` | Moda de categoria_pedido — **REMOVIDA** do modelo (SHAP < 0.01, confirmado) |
| `intervalo_medio` | Média de meses entre pedidos (recalculada pré-CUTOFF) — ver limitação matemática na decisão 53 (soma telescópica, não mede regularidade) |
| `max_intervalo` | Maior intervalo histórico pré-CUTOFF |
| `n_intervalos` | Quantidade de intervalos pré-CUTOFF |
| `categoria_cliente` | Moda de categoria na janela pré-CUTOFF — **REMOVIDA** do modelo (SHAP = 0.0000) |
| `slope_portfolio_medio` | Média dos slopes de todos os itens do cliente (recalculada pré-cutoff, decisão 48) |
| `pct_itens_queda` | % de itens com slope < 0 (recalculada pré-cutoff, decisão 48) |
| `n_itens_portfolio` | Itens distintos comprados na janela (recalculada pré-cutoff, decisão 48) |
| `sem_historico_itens` | 1 se o cliente está ausente de `cliente_item_mes` no período OU com todos os slopes NaN — **REMOVIDA** do modelo (SHAP ≈ 0), mas segue calculada (compõe `historico_insuficiente` na Seção 10) |
| `sem_historico_cadencia` | 1 se `n_intervalos == 0` (n_meses_ativos == 1) — **REMOVIDA** do modelo (SHAP = 0.0000, mesma razão de `sem_historico_itens`) |
| `DIASINADIMPLENTE` | Dias em atraso — Pearson ~0.10 (fraco) mas MI ~0.125 (moderado, sinal não-linear); SHAP confirma relevância moderada (não é ruído puro) |

> Detalhe completo de cada feature (fórmula exata, imputação, Pearson/MI/SHAP atualizados): `docs/variaveis.txt`.

**Splits:**
- **Treino:** features calculadas até `CUTOFF_TREINO = 2024-12`, target = janela de tolerância por categoria após o cutoff
- **Teste:** features calculadas até `CUTOFF_TESTE = 2025-11`, target = janela de tolerância por categoria (travada em `DATA_REF` quando encurta o prazo)

**Imputação de NaN (aplicada em `03_base_master.ipynb`, constantes do treino reusadas no teste — mediana, não sentinel; ver decisão 52 sobre a correção desta seção):**
- Flags criadas **antes** de qualquer fillna em ambos os splits
- `cv_pedidos`, `intervalo_medio`, `max_intervalo` → mediana do treino, calculada só sobre clientes com histórico real (`sem_historico_cadencia == 0`), reaplicada no teste sem recálculo. Valores atuais: 0.3671 / 1.4000 / 3.0000. Persistidos em `data/processed/imputacao_treino.pkl` (decisão 54) para uso pelo scoring mensal.
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
- Output: `cliente_item_tendencia.parquet` (histórico completo — **uso correto é só EDA**, não usar direto como feature de modelo, ver decisão 48) e `cliente_item_mes.parquet` (grão cliente × item × mês, adicionado jul/2026 para permitir recálculo pré-cutoff em `03_base_master.ipynb`)
- **Correção aplicada:** `PROJECT_ROOT` era hardcoded para `C:\Users\carva\central_eto` (cópia antiga); substituído por detecção dinâmica
- **Correção de vazamento (jul/2026, decisão 48):** `cliente_item_tendencia.parquet` estava sendo reaproveitado por `03_base_master.ipynb` como feature de treino/teste sem respeitar cutoff — corrigido via `cliente_item_mes.parquet` + `src/features_itens.py`

### `calculo_receita.ipynb` — CONCLUÍDO
- Calcula `receita_media_mensal` por cliente (média de `total_valor` nos meses ativos)
- Agrupa por `categoria_cliente` (moda estável de `cliente_fidelidade`)
- Gráficos: boxplot geral com outliers, detalhe individual por categoria (2×2), média vs mediana
- Output: `data/processed/receita_cliente.parquet` — usado na Seção 10 de `04_modelo.ipynb` para ordenar alertas por receita
- **Motivação:** alerta de cliente com R$ 25k/mês tem impacto muito maior que R$ 154/mês com o mesmo score de churn

### `03_base_master.ipynb` — CONCLUÍDO (re-executado após auditoria de imputações jul/2026 E após correção de vazamento de item, mesmo mês)
- Construção do dataset de modelagem com split temporal
- Target `churn` definido com janela personalizada por categoria (não horizonte fixo de 3 meses para todos)
- Features recalculadas sobre dados pré-CUTOFF para evitar leakage
- Correção de leakage: join do treino usa `df_target[["CLIENTE", "churn"]]` apenas — sem colunas pós-CUTOFF
- **Correção de vazamento nas features de item (jul/2026, decisão 48):** `slope_portfolio_medio`/`pct_itens_queda`/`n_itens_portfolio` reaproveitavam `cliente_item_tendencia.parquet` (histórico completo) igual pro treino e pro teste, sem respeitar cutoff. Corrigido: `features_itens` agora vem de `calc_features_itens_pre_cutoff()` (`src/features_itens.py`), chamada separadamente com `cutoff_period` (treino) e `cutoff_period_teste` (teste) sobre `cliente_item_mes.parquet`.
- **Auditoria de imputações (jul/2026):** corrigidos 4 problemas que enviasavam coeficientes da logística:
  1. `max_intervalo` retornava `0` hardcoded para clientes de 1 mês → corrigido para `np.nan`
  2. `intervalo_max_real` calculado pós-fillna → corrigido para pré-fillna; treino e teste usam mesma constante (~40 meses)
  3. `cv_pedidos` preenchia com `0` (conflata novo com perfeito) → agora mediana do treino (clientes com n_meses_ativos ≥ 2)
  4. `sem_historico_itens` não cobria 162 clientes com slopes-todos-NaN → critério expandido para `pct_itens_queda.isna() | slope_portfolio_medio.isna()`
- **Novas features:** `sem_historico_cadencia` (n_intervalos == 0) e `sem_historico_itens` expandido — ambas criadas ANTES de qualquer fillna em ambos os splits
- Outputs: `df_model_treino.parquet` (610 clientes, 36.2% churn) e `df_model_teste.parquet` (724 clientes, 49.0% churn)

### `04_modelo.ipynb` — CONCLUÍDO (pipeline de TREINO — Seções 1–10A; ver decisão 45 para a separação treino/inferência)

**Seções 1–5:**

- **Seção 1** — Setup, carga dos parquets, `preparar_xy()` com fillna semântico para `DIASINADIMPLENTE` e `n_itens_portfolio`
- **Seção 2** — EDA: Pearson + MI com target, boxplots top features, heatmap entre features
- **Seção 3** — Remove `razao_atividade` (certeza); gera `X_train_clean` / `X_test_clean`
- **Seção 4** — Distribuição de classes: 36.2% treino, 49.0% teste; justificativa para não usar SMOTE
- **Seção 5** — Baseline: DummyClassifier (`strategy="prior"`, AUC-ROC 0.5000) + VIF como diagnóstico pré-logística + Logistic Regression

- **Seção 6** — Comparação tree-based sob **protocolo limpo**. Números **pós-correção de vazamento de item** (ver decisão 48 — os números anteriores a jul/2026 estão obsoletos e inflados):

  | Modelo | AUC-ROC | AUC-PR |
  |---|---|---|
  | **XGBoost** | **0.8928** | **0.8639** |
  | Logistic Regression | 0.8923 | 0.8662 |
  | Random Forest | 0.8889 | 0.8527 |
  | LightGBM | 0.8770 | 0.8386 |
  | Dummy (prior) | 0.5000 | 0.4903 |

  O vencedor **não é mais sempre LightGBM** — a partir de jul/2026 a Seção 6 monta um dict `MODELOS_TREINADOS` e escolhe `nome_vencedor`/`modelo_vencedor` dinamicamente por AUC-ROC. Todo o resto do notebook (Seções 7–10A) consulta essa escolha em vez de assumir um modelo fixo (ver decisão 49).

- **Seção 7** — Threshold via F2 (recall pesa 2× — FN mais caro em B2B), calculado sobre `y_prob_vencedor` (modelo da Seção 6, pré-tuning). Threshold e métricas exatas variam a cada re-treino — ver output da célula, não fixados aqui em prosa para não ficarem obsoletos de novo.
- **Seção 8** — Optuna (100 trials, StratifiedKFold 5, otimiza AUC-PR), implementado dinamicamente para **LightGBM ou XGBoost** (os dois únicos que já competiram pela 1ª posição); pulado automaticamente se o vencedor for Logistic Regression ou Random Forest. **Decisão tuned vs. default agora é automática:** só substitui o default se o tunado não piorar nem AUC-ROC nem AUC-PR — sem julgamento manual a cada execução. Na execução de jul/2026 (vencedor XGBoost): tuning piorou as duas métricas (ROC −0.0134, PR −0.0217) → modelo final = XGBoost default.
- **Seção 9** — SHAP: beeswarm + waterfall + dependence plot, com `shap.TreeExplainer(modelo_final)` (funciona para LightGBM/XGBoost/Random Forest; Logistic Regression cai no fallback dos coeficientes da Seção 5 — SHAP de pipeline com scaler não implementado). Importância SHAP média (|SHAP|) da execução de jul/2026 (modelo = XGBoost default), ordem decrescente: `meses_sem_pedido_pre` 0.8502, `total_pedidos` 0.2003, `n_meses_ativos` 0.1972, `itens_por_pedido` 0.0923, `ticket_medio` 0.0833, `media_pedidos_mes` 0.0830, `total_valor` 0.0827, `intervalo_medio` 0.0756, `n_intervalos` 0.0594, `DIASINADIMPLENTE` 0.0490, `pct_itens_queda` 0.0419, `cv_pedidos` 0.0408, `max_intervalo` 0.0274, `slope_portfolio_medio` 0.0190, `n_itens_portfolio` 0.0131, `categoria_pedido` 0.0027, `sem_historico_itens` 0.0015, `categoria_cliente` 0.0000, `sem_historico_cadencia` 0.0000. **Decisão final de features:**
  - REMOVER do `X_score` da Seção 10: `categoria_pedido`, `sem_historico_itens`, `categoria_cliente`, `sem_historico_cadencia` (todas com SHAP < 0.01)
  - MANTER: `total_valor`, `n_meses_ativos`, `n_intervalos`, `media_pedidos_mes`, `max_intervalo` (hipóteses de redundância derrubadas) e `ticket_medio` (hipótese de baixo sinal derrubada)
  - **Nota (jul/2026):** `n_itens_portfolio`, `slope_portfolio_medio` e `pct_itens_queda` — as três features afetadas pelo vazamento (decisão 48) — despencaram de importância após a correção (de 0.1098/0.0883/0.0674 para 0.0131/0.0190/0.0419). Confirma que o sinal anterior era majoritariamente vazamento, não padrão real de comportamento.
  - Esta decisão de features é específica da execução (depende de qual modelo venceu) — reavaliar sempre que o vencedor da Seção 6 mudar.

**Modelo oficial: dinâmico — vencedor da Seção 6, tunado ou não conforme a Seção 8 decidir automaticamente.** Execução de jul/2026: **XGBoost default**. Variáveis genéricas usadas a partir da Seção 7: `nome_vencedor`/`modelo_vencedor`/`y_prob_vencedor` (pré-tuning) e `nome_final`/`modelo_final`/`y_prob_final` (pós-decisão da Seção 8).

**Seção 10 — Calibração (parte do treino, permanece no notebook):**

- **10A — Calibração:** `CalibratedClassifierCV(cv=5)` sobre `modelo_final`. Execução de jul/2026: Brier isotonic 0.1422 (**escolhido**) — pior que os ~0.125 de antes da correção, consistente com o modelo ter menos sinal real para calibrar (AUC mais baixa e honesta). Fica no notebook porque `.fit()` roda sobre o treino — é decisão de treino, não de inferência.

> **Scoring (antigos 10B/10C/10D) foi extraído do notebook — ver `scripts/scoring_mensal.py` e decisão 45.** O notebook agora termina no modelo + calibrador treinados; a aplicação (receita anual, ranking por valor em risco, geração do Excel) é um script separado.

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
15. **Optuna descartado para o modelo oficial:** ganho de +0.0027 em AUC-PR é ruído para 610 exemplos de treino; `scale_pos_weight` alto deslocou scores → threshold F2-ótimo em 0.059 (65% da base como alerta, inoperacionalizável). Modelo oficial = LightGBM default. Optuna documentado na Seção 8 para referência futura. **Atualização jul/2026 (decisão 49):** essa comparação virou uma regra automática no código (`usar_tuned = ...`), reaplicada a cada execução independente de qual modelo vencer — não mais uma decisão manual específica do LightGBM.
16. **Threshold via F2 (beta=2):** FN mais caro que FP em B2B — perder um churner é pior que acionar retenção desnecessária. Confirmar custo relativo com o cliente antes de fixar em produção.
17. **`sem_historico_itens` adicionada para corrigir ambiguidade de imputação:** SHAP identificou comportamento espúrio — valores baixos (azul) de `pct_itens_queda` sendo empurrados para churn porque 0-imputado (sem histórico) era confundido com 0-real (portfólio 100% em crescimento). Feature binária criada ANTES do fillna(0) em `03_base_master.ipynb` nos dois splits e replicada na Seção 10 do `04_modelo.ipynb`.
18. **Cópia antiga do projeto em `C:\Users\carva\central_eto`:** pasta com `data/processed/` vazia e `.venv` desatualizado. Não usar — pode causar `FileNotFoundError` silencioso se o kernel apontar para lá.
19. **Auditoria de imputações (jul/2026) — causa-raiz dos sinais contraintuitivos na logística:** 22% da base (136/610 clientes) tinha `n_intervalos == 0` recebendo valores sintéticos que misturavam duas populações. Sintomas: `cv_pedidos`, `pct_itens_queda`, `slope_portfolio_medio` com coeficiente negativo na direção errada; correlação `intervalo_medio ↔ max_intervalo = -0.33` (fisicamente impossível para clientes reais). Causa: `max_intervalo = 0` hardcoded + `intervalo_medio` imputado com 40 criavam inversão max < media nesses 136 clientes. Corrigido com flags + imputações consistentes.
20. **`sem_historico_cadencia` adicionada:** flag binária para `n_intervalos == 0` (n_meses_ativos == 1), criada ANTES do fillna em ambos os splits. Análogo ao `sem_historico_itens` para o eixo de cadência. Permite ao LightGBM distinguir "cliente novo" de "cliente estabelecido regular" sem descartar a observação.
21. **`max_intervalo` corrigido para `np.nan`:** `calc_fidelidade_pre_cutoff` retornava `0` quando `len(meses) < 2` — valor falso que criava inversão física `max < media` para os 136 clientes de 1 mês. Agora retorna `np.nan`; ~~preenchido com mesmo sentinel que `intervalo_medio` (`intervalo_max_real × 2`)~~ **SUPERADO pela decisão 52** — hoje usa mediana do treino, não sentinel.
22. **`cv_pedidos` corrigido:** `.fillna(0)` substituído por mediana do treino (clientes com n_meses_ativos ≥ 2), calculada pré-fillna. O valor 0 conflata "cliente novo de 1 mês" com "cliente veterano perfeitamente regular" — populações opostas em risco de churn. Mediana é neutro e não cria sinal espúrio. **Ainda válida** — essa abordagem foi depois estendida a `intervalo_medio`/`max_intervalo` também (decisão 52).
23. **`intervalo_max_real` calculado pré-fillna:** bug anterior calculava `intervalo_max_treino = df_model["intervalo_medio"].max()` depois do fillna (max pós-fill = 40), resultando em teste imputado com `40 × 2 = 80` vs treino com `20 × 2 = 40`. Corrigido: `intervalo_max_real` calculado antes do fillna (~20 meses reais); treino e teste ambos imputados com ~40. **SUPERADO pela decisão 52** — o sentinel `intervalo_max_real × 2` foi substituído por mediana; esta decisão fica só como registro histórico do raciocínio da correção pré-fillna (que continua válido em espírito: calcular estatísticas de imputação sempre antes do fillna).
24. **`sem_historico_itens` expandido:** critério anterior (`pct_itens_queda.isna()`) não detectava 162 clientes presentes em `cit` mas com todos os `tendencia_slope = NaN` — nesses casos `mean() = NaN` mas a lambda `(x < 0).sum() / len(x)` retornava `0.0`, deixando `pct_itens_queda` definido. Critério novo: `pct_itens_queda.isna() | slope_portfolio_medio.isna()`.
25. **Filtro `n_intervalos < 3` descartado do pipeline de modelagem:** a regra existe apenas no contexto do `RED_FLAG` rules-based de `cliente_fidelidade`. Aplicar como filtro de treino removeria ~30% de um dataset já pequeno (610 clientes) e criaria gap de cobertura no scoring 2026 (clientes novos aparecerão sem o modelo tê-los visto). Abordagem correta: `sem_historico_cadencia` + imputação com mediana — o LightGBM aprende a distinção via interações.
26. **DummyClassifier → `strategy="prior"`:** `stratified` amostra aleatoriamente e produz AUC-ROC ligeiramente diferente de 0.50 a cada execução. `prior` atribui probabilidade constante igual à prevalência do treino → AUC-ROC = 0.5000 deterministicamente. Correto para "chão absoluto" reproduzível.
27. **VIF repositionado na Seção 5 do `04_modelo`:** VIF não é baseline — é diagnóstico de multicolinearidade, pré-requisito para a regressão logística (VIF > 10 → coeficiente instável no modelo linear). Removido da tabela de comparação de baselines; descrito como etapa preparatória da Logistic Regression. Coeficientes comparáveis entre si porque o pipeline inclui `StandardScaler()` antes do modelo; VIF é invariante a escala (R² não muda com transformação linear).
28. **Auditoria de vazamento na Seção 6 e 8 (jul/2026):** três vazamentos identificados e corrigidos: (a) `X_*_xgb = X_*_clean.fillna(0)` sobrescrevia o Caminho A com fillna genérico em todos os modelos tree-based — removido; (b) XGBoost e LightGBM usavam `eval_set=[(X_test_xgb, y_test)]` com early stopping → test set selecionava o número de árvores (XGBoost parou em 11/300 por shift de distribuição 36%→49%); (c) Optuna (`baf26ff9`) treinava modelo final com `eval_set=[(X_test_opt, y_test)]` — test selecionava rounds do modelo tunado. O objetivo CV do Optuna era limpo (usa folds do treino); a decisão "default > tuned" não foi contaminada.
29. **Protocolo limpo de early stopping (jul/2026):** XGBoost e LightGBM agora selecionam `n_rounds` via `xgb.cv` / `lgb.cv` com `StratifiedKFold(k=5)` dentro do treino (`early_stopping_rounds=20`, métrica `aucpr`/`average_precision`). O modelo final é re-treinado no **treino completo** com `n_estimators` fixo, sem early stopping, sem tocar o teste. Test set usado uma única vez, na avaliação final. Optuna (`baf26ff9`) também corrigido: `lgbm_tuned.fit(X_train_clean, y_train)` sem eval_set.
30. **NaN semântico para `DIASINADIMPLENTE` e `n_itens_portfolio`:** após Caminho A, dois NaN remanescentes não cobertos pelo `03_base_master`. Tratados em `preparar_xy()` via `_FILLNA_SEMANTICO`: `DIASINADIMPLENTE → 0` (ausência no cadastro = 0 dias em atraso) e `n_itens_portfolio → 0` (coberto por `sem_historico_itens`). Nenhum fillna genérico nos blocos de modelo.
32. **Método de entrega do alerta — valor em risco (ATA jul/2026):** alertas ordenados por `valor_em_risco = p_churn_calibrada × receita_anual_do_cliente`. Três guard-rails: (a) usar receita **individual** — nunca média/mediana da categoria; (b) aplicar `PISO_RISCO` (sugestão 0,11) antes do ranking para evitar que cliente valioso com risco mínimo domine a lista só pela receita; (c) cortar no `top-N_CAPACIDADE` — capacidade é o gargalo, não threshold. Referência: Provost & Fawcett, *Data Science for Business*, cap. 7–8.
33. **Calibração de probabilidade é obrigatória antes do ranking por valor:** quando se usa `p × receita`, a magnitude da probabilidade importa (não só o ranking). Boosted trees são sistematicamente mal-calibradas — distorção sigmoide (Niculescu-Mizil & Caruana, 2005). Usar `CalibratedClassifierCV` com método `isotonic` ou `sigmoid`, calibrado sobre CV no treino, nunca no teste. Sem calibrar, `valor_em_risco` está errado.
34. **Concentração de receita — dado real do projeto:** 17 clientes premium (2% da base) = 54% da receita total. Um único cliente fatura R$ 254k/mês ≈ 28% da receita. Máximo do segmento baixo = R$ 16k/mês (acima da mediana do premium = R$ 6,5k). Prova definitiva de que categoria de volume não é proxy de valor — usar receita individual sempre.
35. **N_CAPACIDADE como parâmetro configurável:** uma pessoa trabalha os alertas — volume de qualquer threshold supera a capacidade. Tratar como parâmetro no topo do código (placeholder `N_CAPACIDADE = 25`, marcado como provisório). Valor final a cravar com o cliente. Para cobrir a cauda (635 clientes baixo), a solução é aumentar o time + cota por segmento, nunca threshold por segmento.
36. **Custo assimétrico fora da equação:** cliente não soube mensurar custo de perder cliente vs. custo de retenção. Não implementar threshold ótimo por custo (Elkan 2001). O threshold governa volume — e quando capacidade é o gargalo, volume não é o problema. Custo fica como material ilustrativo de anexo.
38. **Decisão final de features (SHAP jul/2026):** remover `categoria_pedido` (SHAP 0.0043) e `categoria_cliente` (SHAP 0.0000). Manter todas as hipóteses de redundância das Seções 2/3 que o SHAP derrubou: `n_meses_ativos` (0.2245), `n_intervalos` (0.0532), `media_pedidos_mes` (0.0658), `max_intervalo` (0.0420), `total_valor` (0.0666), `ticket_medio` (0.0720). Feature mais importante: `meses_sem_pedido_pre` (SHAP 1.1237, Pearson ~0.62 confirmado). Retreino desnecessário — LightGBM já ignora features com SHAP=0 nos splits; remoção entra no `X_score` da Seção 10.
39. **Receita anual = soma dos últimos 12 meses reais de `cliente_mes`** — não `média × 12`. Respeita sazonalidade e clientes com histórico parcial. Para clientes com < 12 meses, anualizar pela média disponível e sinalizar com flag.
31. **Threshold calibrado no teste — ressalva para produção:** o `THRESHOLD` (F2-ótimo) foi encontrado via `argmax(f2_arr)` sobre `y_test`. Isso significa que o ponto de corte foi escolhido no mesmo conjunto usado para reportar o desempenho — grau leve de overfitting de threshold. Para produção, o correto é recalibrar o threshold num conjunto de validação separado do teste (ex.: reservar 20% do teste só para calibração) ou, mais pragmaticamente, substituir o F2-ótimo por um critério operacional direto (`n_alertas` na tabela de sweep da Seção 7 — "quantos clientes a equipe consegue contatar por mês"). O segundo caminho é mais defensável e mais útil para o cliente. **Ação necessária antes de ir a produção:** definir com o cliente a capacidade operacional de retenção e fixar o threshold por `n_alertas`, não por F2. Threshold F2 atual (re-execução jul/2026): **0.114**.
40. **Optuna não-determinístico entre execuções:** o `TPESampler` não recebeu seed fixa — o threshold F2-ótimo do modelo tunado variou entre execuções (0.059 documentado antes vs. 0.001 na re-execução de jul/2026). Irrelevante para a decisão (modelo tunado descartado em ambos os casos, AUC sempre pior que o default), mas registrar para não confundir números históricos com resultado reprodutível. Se o Optuna precisar ser reproduzido exatamente, fixar `seed` no `TPESampler`.
41. **Seção 10 (10A–10D) implementada e executada de ponta a ponta (jul/2026):** calibração isotonic escolhida (Brier 0.1228 vs. 0.1252 bruto), receita anual calculada sobre o snapshot atual (`DATA_REF`, não o cutoff do split de teste — Seção 10 aplica o modelo scoreado no presente, não reavalia o teste histórico), ranking por `valor_em_risco` com `PISO_RISCO=0.11` e `N_CAPACIDADE=25` (ambos provisórios), outputs gravados em `data/processed/scores_2026.parquet` (todos os 724 clientes de teste) e `data/processed/clientes_alerta_modelo.xlsx` (25 selecionados).
41.1. **Bug corrigido em `top_features_shap` (mesmo dia):** a primeira versão da Seção 10D escolhia os top-3 por `|SHAP|` (magnitude), sem checar o sinal — um cliente podia aparecer com um "motivo de risco" que na verdade reduzia o score (ex.: LIMED listada como "não faz pedidos há 0 meses", quando esse sinal reduz o risco). Corrigido para usar apenas contribuições **positivas** de SHAP (`shap_vals.values > 0`) por linha — os 3 fatores que de fato empurram o cliente na direção do churn. Notebook e `scores_2026.parquet`/`clientes_alerta_modelo.xlsx` já re-executados com a correção.
42. **Outlier de receita anualizada detectado, não corrigido:** clientes com poucos meses de histórico (`historico_parcial_receita=1`) têm a receita observada multiplicada para completar 12 meses — um cliente com pico de compra pontual em 1-2 meses observados pode ter `receita_anual` artificialmente inflada (máximo observado: R$ 3,52M, muito acima do p75 de R$ 6.588). Não afeta o ranking de forma sistemática (poucos casos), mas monitorar se aparecer no topo do `clientes_alerta_modelo.xlsx` — candidato a winsorização ou cap por percentil se recorrente.
43. **Pasta `reports/` recriada (jul/2026):** havia sido removida (o `.pptx` e o script antigos foram excluídos, decisão do usuário fora do escopo desta sessão). Recriada para abrigar `lista_alertas_churn.html` (decisão 44) — os outputs de dados (`clientes_alerta_modelo.xlsx`, `scores_2026.parquet`) continuam em `data/processed/`, só o relatório visual fica em `reports/`.
44. **Relatório visual `reports/lista_alertas_churn.html` criado para apresentação ao time comercial (jul/2026):** página HTML autocontida (sem dependências externas, abre localmente no navegador) com stat tiles, gráfico de barras dos top 15 por `valor_em_risco` (cor = tier de risco: crítico ≥70%, alto 40–70%, moderado 11–40%) e tabela completa dos 25 clientes com um "motivo principal" em linguagem natural por cliente (traduzido de `top_features_shap` + valores reais de `df_model_teste`, não jargão de SHAP). Não publicado como Artifact — contém nomes de empresas e receita reais, mantido só localmente por padrão. **Geração formalizada e automatizada depois — ver decisão 55 (`scripts/gerar_relatorio.py`), não precisa mais de passo manual.**
45. **Separação treino/inferência (jul/2026) — `04_modelo.ipynb` deixa de conter o scoring:** decisão de arquitetura para viabilizar produção. Dois pipelines diferentes, com frequências e naturezas diferentes:
    - **Treino** (`04_modelo.ipynb`, Seções 1–10A): roda raramente, precisa de revisão humana (gráficos, SHAP, decisão "esse modelo é bom o suficiente?") — continua notebook de propósito. Termina produzindo o modelo vencedor (dinâmico — ver decisão 49) + o calibrador escolhido (isotonic).
    - **Inferência** (`scripts/scoring_mensal.py`, novo): roda mensalmente, sem supervisão, precisa ser agendável — por isso é script `.py`, não notebook (notebook depende de kernel Jupyter, permite células fora de ordem, não agenda nativamente). Contém a lógica que antes era 10B/10C/10D: receita anual, ranking por `valor_em_risco`, geração de `scores_2026.parquet` e `clientes_alerta_modelo.xlsx`.
    - **Regra chave:** a calibração (`CalibratedClassifierCV.fit(...)`) é treino, não inferência — fica em `04_modelo.ipynb`, não no script. Só a *aplicação* do calibrador já ajustado (`.predict_proba`) entra no scoring mensal.
    - **Status:** `scripts/scoring_mensal.py` depende de dois passos:
      - **Passo 1 — CONCLUÍDO (jul/2026):** `04_modelo.ipynb` ganhou uma célula final que persiste `{"modelo": modelo_final, "modelo_nome": nome_final, "calibrador": calibrador, "features": list(X_train_clean.columns)}` via `pickle`. **Renomeado de `modelo_lgbm.pkl` para `data/processed/modelo_churn.pkl`** (mesmo dia, durante a correção de vazamento da decisão 48) — o nome antigo citava um modelo específico (LightGBM) que deixou de ser sempre o vencedor; `modelo_churn.pkl` é agnóstico a qual algoritmo venceu. Arquivo antigo removido. Script testado de ponta a ponta.
      - **Passo 2 — extração mecânica CONCLUÍDA (jul/2026, decisão 50); recálculo "ao vivo" ainda pendente.** Toda a lógica de features e de preparo do X foi extraída para `src/` (itens, comportamento, fidelidade, mapeamento categórico) e é hoje compartilhada por `03_base_master.ipynb`, `04_modelo.ipynb`, `scripts/scoring_mensal.py` e as ferramentas `explicar_cliente.*`. **O que falta:** `scoring_mensal.py` ainda lê `df_model_teste.parquet` (um snapshot fixo) em vez de puxar pedidos frescos do banco e recalcular essas mesmas funções sobre o mês mais recente fechado — ver Próximos Passos.
    - **Dois bugs encontrados e corrigidos ao testar o Passo 1:** (a) o script carregava `categoria_pedido`/`categoria_cliente` como texto ("baixo"/"medio"...), mas o modelo foi treinado com o mapeamento numérico de `preparar_xy()` (`{"baixo": 0, "medio": 1, "alto": 2, "premium": 3}`) — sem replicar esse mapeamento, `lgbm.predict_proba` falhava com `ValueError: pandas dtypes must be int, float or bool`; (b) ao corrigir (a), a coluna `categoria_cliente` usada para *exibição* no Excel também virou número — corrigido guardando `categoria_cliente_label` (texto original) antes do mapeamento, para exibição, mantendo a versão numérica só para o `X` do modelo. **Duplicação registrada como dívida técnica:** o mapeamento de `preparar_xy()` está copiado no script e no notebook — no Passo 2, extrair para `src/` junto com a lógica de features, para os dois nunca divergirem.
46. **Validação com dado ao vivo do banco encontrou 3 achados relevantes (jul/2026), via `validar_lista_alertas.py`:** (a) 6 dos 25 clientes da lista já aparecem como `STATUS = Inativo` no cadastro (`CONTROLLER.dbo.PBI_Clientes`) — incluindo a Pathos (maior valor em risco, 90% **na época** — ver decisão 48, esse número mudou), validação forte do modelo; (b) um cliente (Fundação Hospitalar, CLIENTE 9, 88% de risco **na época**) fez um pedido novo poucos dias antes da validação, contradizendo o score — possível falso positivo ou reengajamento de última hora, acompanhar; (c) o banco já tem pedidos mais recentes que `cliente_mes.parquet` (ex.: compras registradas em mai/jun/2026, depois do `DATA_REF` usado no scoring) — o parquet local está desatualizado em relação ao banco ao vivo, reforçando a necessidade do pipeline de inferência mensal (decisão 45) em vez de reusar o snapshot antigo indefinidamente. **Nota:** os percentuais de risco citados aqui são do modelo pré-correção de vazamento (decisão 48) — a validação qualitativa (banco mais atual que o snapshot, 6/25 já inativos) segue válida, mas os números exatos de probabilidade estão desatualizados. Recomenda-se re-rodar `validar_lista_alertas.py` contra a lista atual.
47. **Cadência de scoring: mensal, após o fechamento do mês — não diário.** `cliente_mes` agrega por mês, então `meses_sem_pedido_pre` (feature mais importante do modelo) só é confiável quando o mês corrente está fechado — pontuar no meio do mês trata "ainda não comprou este mês" como sinal de risco, quando o cliente pode simplesmente não ter comprado *ainda*. Atualizar o dado bruto pode ser frequente/barato, mas a lista de alertas em si só deve ser gerada com meses completos, e a operação (capacidade de ~25 contatos/mês) também não justificaria cadência mais rápida que mensal.
48. **BUG GRAVE — vazamento de dado nas features de tendência de item, corrigido jul/2026.** Descoberto durante a construção do Passo 2 (extrair features pré-cutoff pra `src/`): `slope_portfolio_medio`, `pct_itens_queda` e `n_itens_portfolio` vinham de `cliente_item_tendencia.parquet` (gerado por `02_itens_eda.ipynb`), que usa o **histórico completo** de pedidos (2023 até o presente), sem nenhum filtro por cutoff. O `03_base_master.ipynb` reaproveitava esse mesmo arquivo (sem recalcular) tanto pro treino (`cutoff=dez/2024`) quanto pro teste (`cutoff=nov/2025`) — ou seja, as features de tendência de item "viam" meses posteriores ao cutoff de cada split. Mesma categoria de bug já corrigida antes pras features de `cliente_fidelidade` (decisão 8), mas que não tinha sido pega na auditoria de jul/2026 (decisões 19–29) porque está numa fonte de dado diferente.
    - **Evidência de que o vazamento era real e grande:** após a correção, a importância SHAP dessas 3 features despencou (`n_itens_portfolio`: 0.1098 → ~0.01; `slope_portfolio_medio`: 0.0883 → ~0.02; `pct_itens_queda`: 0.0674 → ~0.04, valores variam ligeiramente por modelo). `n_itens_portfolio` chegou a ir a **SHAP=0.0000** com o LightGBM.
    - **Impacto no desempenho do modelo:** AUC-ROC do então-oficial LightGBM caiu de 0.9184 para 0.8770 — o número antigo era inflado pelo vazamento. **0.8770–0.8928 (a faixa dos 4 modelos reais pós-correção) é o desempenho honesto do modelo.**
    - **Correção implementada:**
      1. `02_itens_eda.ipynb` passa a salvar também `cliente_item_mes.parquet` (grão cliente × item × mês, sem nenhuma agregação de tendência — não é sensível a vazamento por si só).
      2. Nova função `src/features_itens.py::calc_features_itens_pre_cutoff(cliente_item_mes, cutoff_period)` — recalcula `tendencia_slope` (mesma lógica de `02_itens_eda`) filtrando só `ANO_MES <= cutoff_period`.
      3. `03_base_master.ipynb` chama essa função separadamente pro treino (`cutoff_period`) e pro teste (`cutoff_period_teste`) — antes reaproveitava a mesma tabela nos dois splits.
      4. Cascata completa re-executada: `02_itens_eda` → `03_base_master` → `04_modelo` → `scripts/scoring_mensal.py` → relatório HTML.
    - **Números antigos (pré-correção) em qualquer parte deste documento que mencionem AUC-ROC ~0.91-0.92, LightGBM como vencedor fixo, ou os SHAP antigos de `n_itens_portfolio`/`slope_portfolio_medio`/`pct_itens_queda` estão obsoletos.**
49. **Seleção de modelo tornada dinâmica (jul/2026) — consequência direta da decisão 48.** A correção do vazamento mudou o vencedor da Seção 6 de LightGBM para XGBoost — só que as Seções 7 (threshold), 8 (Optuna) e 9 (SHAP) e 10A (calibração) estavam **hardcoded pra usar `lgbm`**, ignorando o resultado real da comparação. Corrigido:
    - Seção 6 monta `MODELOS_TREINADOS` (dict nome → (modelo, y_prob)) e define `nome_vencedor`/`modelo_vencedor`/`y_prob_vencedor` a partir de `resultados.iloc[0]`.
    - Seção 7 (threshold) usa `y_prob_vencedor`.
    - Seção 8 (Optuna) branch por `nome_vencedor`: implementado para LightGBM e XGBoost (os dois hiperparam spaces são bem parecidos — `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha/lambda`, `scale_pos_weight`); pula tuning automaticamente se o vencedor for Logistic Regression ou Random Forest (nunca tiveram tuning implementado mesmo antes desta correção).
    - Decisão tuned vs. default **automática**: `usar_tuned = (auc_roc_tuned >= auc_roc_vencedor) and (auc_pr_tuned >= auc_pr_vencedor)` — sem julgamento manual a cada execução, mesma lógica que já vinha sendo aplicada manualmente desde a decisão 15.
    - Seção 9 (SHAP) usa `shap.TreeExplainer(modelo_final)` sem dataset de background (evita um `NotImplementedError` do XGBoost 3.x que aparece quando se passa background data pro despacho automático `shap.Explainer` — `TreeExplainer` puro usa `feature_perturbation="tree_path_dependent"` e não precisa de background). Logistic Regression cai num fallback (SHAP não implementado pro pipeline com scaler; usa os coeficientes da Seção 5).
    - Seção 10A (calibração) e a célula de persistência usam `modelo_final`/`y_prob_final`.
    - **Arquivo renomeado:** `modelo_lgbm.pkl` → `data/processed/modelo_churn.pkl` (nome agnóstico ao algoritmo). Arquivo antigo removido.
50. **Passo 2 (extração mecânica) concluído — jul/2026.** Toda a lógica de features pré-cutoff e de preparo do X, antes só dentro do `03_base_master.ipynb`/`04_modelo.ipynb` (ou duplicada em scripts avulsos), agora vive em `src/`:
    - `src/features_comportamento.py` — `meses_desde()` + `calc_features_comportamento_pre_cutoff()` (volume, ticket, recência, cadência)
    - `src/features_fidelidade.py` — `calc_features_fidelidade_pre_cutoff()` (intervalo entre pedidos, categoria predominante)
    - `src/features_itens.py` — já existia (decisão 48), tendência de item
    - `src/model_prep.py` — `preparar_xy()` (treino, exige coluna `churn`) e `preparar_X()` (inferência, só recebe a lista `features` do `.pkl`, não exige `churn`) — unifica o mapeamento categórico e o fillna semântico que estavam duplicados em `04_modelo.ipynb`, `scripts/scoring_mensal.py` e `explicar_cliente.py`/`.ipynb`
    - **Validado como refatoração pura (sem mudança de comportamento):** `df_model_treino/teste.parquet`, o modelo persistido e `scores_2026.parquet` foram comparados byte-a-byte antes/depois de cada refatoração — idênticos em todos os casos.
    - **Parte que faltava (features "ao vivo" no scoring) concluída depois, ver decisão 54.**
51. **BUG — `reports/lista_alertas_churn.html` corrompido ao atualizar os dados, corrigido (jul/2026).** Ao reexecutar o modelo (decisões 48/49) o JSON embutido no relatório precisou ser atualizado, e o script usado pra isso fazia `re.sub(r'(<script id="report-data"...>\n).*(\n</script>)', ...)` com `re.DOTALL` — o `.*` guloso não parou no primeiro `</script>`, e sim no **último** `</script>` do arquivo (o que fecha o segundo bloco, com toda a lógica JS que constrói os stat tiles, o gráfico e a tabela). Resultado: o bloco de JavaScript inteiro foi apagado, sobrando só o JSON solto — página abria em branco (usuário reportou "vazio no Microsoft Edge"). **Corrigido:** (a) JSON reatualizado com regex não-guloso (`.*?`) restrito à primeira ocorrência; (b) bloco de JS reconstruído e reinserido antes de `</body>`; (c) `build_report.py` passou a ler `modelo_nome` de `modelo_churn.pkl` em vez de hardcodar "LightGBM" no texto do relatório. **Lição registrada:** nunca usar regex guloso (`.*` com DOTALL) pra isolar um bloco quando existe mais de uma tag igual no arquivo (aqui, dois `<script>`) — usar `.*?` (não-guloso) ou, melhor, parsear o HTML de verdade em vez de regex.
52. **Documentação desatualizada encontrada e corrigida (jul/2026) — imputação de cadência real é mediana, não sentinel `2×max`.** Usuário questionou a confiabilidade matemática de `intervalo_medio` com base no `docs/variaveis.txt`, que descrevia o sentinel `2 × max_observado` (documentado nas decisões 19/21/23). Ao checar o código real de `03_base_master.ipynb` (célula `s6_impute`), a imputação **já não usa esse sentinel** — foi substituída em algum momento não documentado por medianas independentes por feature (`cv_pedidos`, `intervalo_medio`, `max_intervalo`), calculadas só sobre clientes com histórico real (`sem_historico_cadencia == 0`) no treino, reaplicadas no teste sem recálculo. Valores atuais: `cv_pedidos`=0.3671, `intervalo_medio`=1.4000, `max_intervalo`=3.0000. **As decisões 21 e 23 ficam superadas** (motivo do sentinel original preservado como registro histórico, mas o mecanismo não existe mais). `docs/variaveis.txt` reescrito por completo contra o código real (não contra a documentação anterior) — incluindo target (`churn` com janela por categoria, não `churn_h3`/3 meses fixos, que também estava errado), features de comportamento/intervalos/itens, e a lista final de remoção por SHAP (que tinha `ticket_medio` listado por engano — SHAP derrubou essa hipótese, feature mantida).
53. **Limitação matemática de `intervalo_medio` documentada, não corrigida (jul/2026).** `intervalo_medio = mean(diferenças consecutivas entre meses ativos)` — por soma telescópica, isso é algebricamente igual a `(último_mês - primeiro_mês) / n_intervalos`, sempre, independente de como os meses estão espaçados internamente. Ou seja, a feature mede taxa média sobre o período observado, não regularidade de espaçamento — um cliente de cadência perfeitamente regular e um cliente que comprava todo mês mas sumiu recentemente podem ter o mesmo `intervalo_medio`. **Não é bug** (é uma propriedade matemática do jeito que a feature foi definida), e o gate `n_intervalos >= 3` não tem o efeito de "estabilizar a média por amostragem" que teria numa média de variáveis i.i.d — é outro tipo de estimador. **Mitigação parcial já existente:** `max_intervalo` (feature separada) captura o maior gap, cobrindo parte do que `intervalo_medio` perde. **Não corrigido nesta sessão** — SHAP de `intervalo_medio` é moderado (0.0756, 8ª feature de 19), não prioritário frente ao Passo 2. Candidato registrado para o futuro: `std(diffs)`/IQR das diferenças (mede regularidade de verdade) e/ou razão `meses_sem_pedido_pre / intervalo_medio` (desvio do próprio padrão do cliente).
54. **Passo 2 concluído por completo (jul/2026) — `scripts/scoring_mensal.py` agora calcula features "ao vivo".** Antes lia `df_model_teste.parquet`, um snapshot congelado em `CUTOFF_TESTE` (nov/2025) — um cliente que voltou a comprar depois disso continuava aparecendo como "sumido há N meses" na lista de alertas (achado concreto: ON-HIGHWAY, cliente 799, tinha `meses_sem_pedido_pre=18` mesmo tendo comprado em mai/2026). Corrigido:
    - **Nova constante persistida:** `data/processed/imputacao_treino.pkl` — `medianas_cadencia_treino` (cv_pedidos, intervalo_medio, max_intervalo), calculada em `03_base_master.ipynb` (célula nova, logo após salvar `df_model_treino.parquet`) e consumida pelo scoring mensal. Sem isso, o script teria que recalcular medianas sobre dado ao vivo (quebraria a garantia "estatística de imputação só vem do treino").
    - **Cutoff dinâmico:** `determinar_cutoff()` = último mês FECHADO relativo a `pd.Timestamp.now()` (decisão 47) — nunca o mês corrente, que está incompleto.
    - **`DATA_REF` fixo abandonado no scoring:** a janela de 12 meses de `calcular_receita_anual()` agora usa `hoje` (timestamp real) em vez da constante congelada de `src/config.py` — essa constante continua correta para o *treino/backtest* (não mexi nela), só não faz mais sentido pra uma lista que se propõe "atual".
    - **Puxa pedidos e cadastro direto do banco** a cada execução: `puxar_pedidos_brutos()` (mesma query de `01/02_*_eda.ipynb`) alimenta `make_cliente_mes()` (reaproveitado de `src/features_EDA_pedidos.py`) e uma agregação local cliente×item×mês (mesma lógica de `02_itens_eda.ipynb`); `carregar_perfil_clientes()` (mesma query/preparo de `03_base_master.ipynb`) traz NOME/DIASINADIMPLENTE atualizados.
    - **Universo de clientes muda:** antes eram os 724 do split de teste (cohort histórico fixo); agora são todos os clientes com pedido até o cutoff atual — um conjunto diferente e maior (776 na 1ª execução ao vivo), incluindo clientes de alto valor (hospitais grandes) que não estavam no recorte de teste. Isso é o comportamento correto e esperado para uma lista de produção, não uma regressão.
    - **Testado contra o banco real, validado com o caso ON-HIGHWAY:** `meses_sem_pedido_pre` passou de 18 (snapshot nov/2025) para 1 (cutoff jun/2026, reflete o pedido de mai/2026) — confirma que a lacuna foi fechada.
    - ~~Pendência remanescente: relatório HTML não se atualiza sozinho~~ — **CONCLUÍDO na mesma sessão, ver decisão 55.**
55. **Relatório HTML automatizado — `scripts/gerar_relatorio.py` (jul/2026), resolve o bug da decisão 51 pela raiz.** O script antigo (`build_report.py`) vivia fora do repositório (pasta temporária da sessão) e exigia edição manual do HTML pra atualizar os dados — foi exatamente essa edição manual (um regex guloso) que corrompeu o relatório antes. O novo script:
    - **Gera o HTML inteiro do zero a cada execução** — nunca abre e edita o arquivo anterior. Um único placeholder (`__DATA_JSON__`) é substituído uma vez, na hora de escrever o arquivo — não há "segunda edição" possível.
    - **Lê `scores_2026.parquet` + `features_scoring_atual.parquet`** (novo, persistido por `scoring_mensal.py` — as features cruas do ciclo de scoring atual, necessárias pra montar o texto de "motivo principal" com valores reais). Nunca lê `df_model_teste.parquet` — usar o snapshot congelado aqui reintroduziria o mesmo problema que o Passo 2 (decisão 54) acabou de corrigir.
    - **AUC-ROC calculada na hora** (`calcular_auc_teste()`, carrega o modelo + `df_model_teste.parquet`) — não fica mais hardcoded no texto do relatório (era "91,8%" esquecido depois que o modelo mudou pra XGBoost com ~89,3%; agora nunca fica desatualizado).
    - **`n_total` dinâmico** — o rodapé metodológico dizia "dos 724 clientes avaliados" hardcoded (número do split de teste antigo); agora reflete o universo real de cada execução (776 na 1ª rodada ao vivo).
    - **Encadeado no fim de `scripts/scoring_mensal.py::main()`** — rodar `python scoring_mensal.py` já deixa `scores_2026.parquet`, `clientes_alerta_modelo.xlsx` e o relatório HTML todos atualizados numa única chamada. Pronto pra virar uma única entrada no Windows Task Scheduler (decisão 47).
    - **Script antigo (`build_report.py`, na pasta temporária da sessão) aposentado** — não faz mais parte do fluxo.

---

## Próximos Passos

### IMEDIATO
1. ~~Re-executar `03_base_master.ipynb`~~ — CONCLUÍDO (auditoria jul/2026)
2. ~~Re-rodar `04_modelo.ipynb` do início com os parquets atualizados e protocolo limpo~~ — CONCLUÍDO (jul/2026)
3. ~~Concluir Seção 9 — SHAP~~ — CONCLUÍDO (decisão de features documentada)
4. ~~Escrever e rodar Seção 10 — aplicação no snapshot atual~~ — CONCLUÍDO (`scores_2026.parquet` + `clientes_alerta_modelo.xlsx` gerados)
5. Validar `clientes_alerta_modelo.xlsx` (25 clientes) com a equipe comercial — lista gerada pelo modelo corrigido (XGBoost, pós-correção de vazamento — decisão 48)
6. Revisar o outlier de receita anualizada (decisão 42) caso apareça no top da lista de alertas
7. Cravar `PISO_RISCO` e `N_CAPACIDADE` com o cliente (atualmente 0.11 e 25, provisórios)
8. ~~Separar treino (notebook) de inferência (script)~~ — CONCLUÍDO (decisão 45): `04_modelo.ipynb` agora só treina; `scripts/scoring_mensal.py` criado como esqueleto
9. ~~Passo 1: persistir modelo + calibrador treinados em `data/processed/modelo_churn.pkl`~~ — CONCLUÍDO (jul/2026)
10. ~~Corrigir vazamento de dado nas features de item + tornar seleção de modelo dinâmica~~ — CONCLUÍDO (decisões 48 e 49, jul/2026): XGBoost agora é o modelo oficial, AUC-ROC honesta ~0.89 (era ~0.92 com vazamento)
11. ~~Passo 2 — extração mecânica: mover a lógica de comportamento/fidelidade/preparo do X pra `src/`~~ — CONCLUÍDO (decisão 50, jul/2026): `src/features_comportamento.py`, `src/features_fidelidade.py`, `src/model_prep.py`. Validado como refatoração pura (parquets/modelo/scores idênticos antes/depois).
12. ~~Passo 2 — parte que falta: features "ao vivo" no scoring mensal~~ — **CONCLUÍDO (decisão 54, jul/2026).**
13. Depois disso: agendar `scripts/scoring_mensal.py` (Windows Task Scheduler) pra rodar mensalmente, após o fechamento do mês (decisão 47)
14. Re-rodar `validar_lista_alertas.py` contra a lista atual (pós-correção) — a validação anterior foi feita com os scores do modelo com vazamento
15. **(baixa prioridade, registrado — decisão 53):** avaliar `std(diffs)`/IQR das diferenças entre pedidos e/ou razão `meses_sem_pedido_pre / intervalo_medio` como complemento à limitação matemática de `intervalo_medio`

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
- [ ] **`N_CAPACIDADE` (BLOQUEADOR DO CORTE FINAL):** quantos clientes/mês a pessoa que trabalha os alertas consegue contatar. Placeholder atual = 25. Cravar com o cliente antes de fixar o output final.

---

## Guia de Comunicação com o Cliente

Respostas prontas para perguntas frequentes sobre o modelo:

**"Quantos clientes vão ser contatados por mês?"**
→ Abrir a tabela de sweep (Seção 7) e mostrar a linha com o `n_alertas` compatível com a capacidade da equipe. Ex.: *"Se conseguem contatar 60/mês, threshold = 0.70 → precisão 90%, recall 62%."*

**"Qual a precisão do modelo?"**
→ Não responder com um número único — depende do threshold. Explicar o trade-off e deixar o cliente escolher via `n_alertas`. Nunca usar accuracy (enganosa com classes desbalanceadas).

**"O modelo acerta quantos por cento?"**
→ Usar AUC-ROC: *"Em X% das vezes o modelo ranqueia um churner acima de um cliente ativo ao comparar dois aleatórios. O chão seria 50%."* **X não é fixo** — o modelo oficial é escolhido dinamicamente a cada retreino (decisão 49) e o AUC muda conforme o vencedor; puxar o valor atual do `scripts/gerar_relatorio.py` (calcula na hora, decisão 55) ou da Seção 6 do `04_modelo.ipynb`. Última execução (XGBoost, jul/2026): 89,3%.

**"A capacidade operacional afeta a precisão do modelo?"**
→ Não altera os scores do modelo para uma dada versão treinada (o AUC-ROC não muda com quantos alertas a equipe consegue trabalhar). Mas afeta o **recall realizável**: mandar 180 alertas para uma equipe que só age em 60 faz os 120 restantes serem ignorados — o modelo acertou, a retenção falhou por capacidade.

**"Qual o retorno financeiro?"**
→ `ROI = churners_capturados × ticket_médio - n_alertas × custo_retenção`. Precisa de dois números do cliente. Com eles, a tabela de sweep vira simulador de ROI — troca threshold e mostra como o lucro muda.
