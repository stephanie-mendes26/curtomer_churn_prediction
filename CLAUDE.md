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
├── threshold_churn             # referência de thresholds definidos
├── data/
│   └── processed/
│       ├── cliente_mes.parquet
│       ├── cliente_fidelidade.parquet
│       ├── clientes_alerta.xlsx
│       └── df_master.parquet   # ainda não finalizado
├── notebooks/
│   ├── 01_pedidos_eda.ipynb    # CONCLUÍDO
│   ├── 02_clientes_eda.ipynb   # EM ANDAMENTO
│   └── 03_base_master.ipynb    # PENDENTE
├── queries/
│   ├── base_pedidos.sql
│   ├── base_clientes.sql
│   ├── base_clientes_enr.sql
│   ├── profiling_pedidos.sql
│   └── profiling_clientes.sql
└── src/
    ├── __init__.py
    ├── db.py                   # conexão com banco via db.get_data(sql) e db.test_connection()
    ├── features_EDA_pedidos.py # contém make_cliente_mes()
    ├── sanity_check.py         # contém sanity_check()
    └── utils.py
```

---

## Convenções do Projeto

- **Python environment:** sempre usar o `.venv` local (`pyvenv.cfg` na raiz)
- **Banco de dados:** acessado via `src/db.py` — usar `db.get_data(sql)` para queries, `db.test_connection()` para validar conexão
- **Data de referência fixa:** `DATA_REF = pd.Timestamp("2026-05-01")` — usar em todos os notebooks
- **Janela temporal:** dados a partir de `2023-01-01` (743 clientes ativos, maior qualidade)
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
| `intervalo_medio` | Média de dias entre pedidos consecutivos |
| `max_intervalo` | Maior intervalo registrado |
| `total_valor_historico` | Faturamento acumulado |
| `ticket_medio_geral` | Ticket médio histórico |
| `valor_ultimo_mes` | Faturamento no último mês ativo |
| `meses_sumido` | Meses desde o último pedido até DATA_REF |
| `n_intervalos` | Quantidade de intervalos calculados |
| `threshold_churn` | Meses de silêncio tolerados por categoria |
| `categoria_cliente` | Moda de categoria_pedido — perfil geral |
| `RED_FLAG` | True se meses_sumido > threshold_churn AND n_intervalos ≥ 3 |
| `historico_confiavel` | True se n_intervalos ≥ 3 |

**Thresholds de churn por categoria:**
- `premium` → 2 meses
- `alto` → 2 meses
- `medio` → 3 meses
- `baixo` → 6 meses

---

## Estado Atual por Notebook

### `01_pedidos_eda.ipynb` — CONCLUÍDO
- EDA da base de pedidos
- Criação de `cliente_mes` e `cliente_fidelidade`
- Sanity checks realizados
- Dashboard por cliente implementado (troca `CLIENTE_ID` para visualizar)
- Outputs salvos em `data/processed/`

### `02_clientes_eda.ipynb` — EM ANDAMENTO
- Tabela clientes com variáveis de confiabilidade insuficiente
- **Decisão do cliente: manter fora do escopo ativo por enquanto**
- Não usar variáveis desta tabela no modelo até nova validação

### `03_base_master.ipynb` — PENDENTE
- Será a consolidação de todas as features para o modelo

---

## Decisões Técnicas Registradas

1. **Tabela clientes fora do escopo:** variáveis não confiáveis o suficiente para modelagem. Retomar apenas após validação.
2. **Janela 2023+:** dados anteriores existem mas têm menor qualidade. Usar `inicio="2023-01-01"` em `make_cliente_mes()`.
3. **neg_policy="clip0":** valores negativos de VL_SERVICO são zerados, não removidos — preserva o registro histórico.
4. **Categoria premium sem subdivisão adicional:** 7 clientes com >108 pedidos/mês não justificam nova categoria.
5. **RED_FLAG requer n_intervalos ≥ 3:** clientes com histórico curto não recebem flag — evita falso positivo em clientes novos.

---

## Próximos Passos (priorizados)

### P1 — Análise de itens por pedido (`02_itens_eda.ipynb` — novo notebook)
Sinal de saída identificado pelo cliente: redução progressiva no volume de itens específicos pode indicar migração para concorrente.
- Granularidade: `cliente × item × mês`
- Features a construir:
  - Tendência linear por item (`np.polyfit` nos últimos N meses de quantidade)
  - `n_itens_distintos` por cliente por mês (concentração de portfólio)
  - Variação percentual mês a mês por item
- Output: tabela `cliente_item_tendencia.parquet` para join posterior com `cliente_mes`

### P2 — Subfragmentação da categoria "Baixo"
Categoria atual "baixo" mistura clientes recorrentes de baixo volume com clientes sazonais (compra 1x/ano).
- Critério sugerido: `n_meses_ativos / janela_meses`
  - `baixo_recorrente` → razão ≥ 0.4
  - `baixo_sazonal` → razão < 0.4
- Threshold de churn para `baixo_sazonal` baseado em intervalo máximo histórico, não meses corridos
- Atualizar `cliente_fidelidade` e `make_cliente_mes()`

### P3 — Categorização por tipo de cliente (SETOR)
- Avaliar qualidade e cardinalidade da variável `SETOR` em `df_base`
- Agrupar em super-categorias se necessário (ex.: "saúde dental", "saúde estética")
- Usar como feature categórica no modelo ou critério de estratificação na avaliação

### P4 — Modelo de previsão de churn (`03_base_master.ipynb`)
- Iniciar após P1 e P2 concluídos
- Base: `cliente_mes` + `cliente_fidelidade` + `cliente_item_tendencia`
- Target: a definir — `RED_FLAG` (classificação binária) ou churn em horizonte N meses
- **Pergunta em aberto com o cliente:** target é RED_FLAG ou horizonte temporal específico (ex.: "vai sumir nos próximos 3 meses")?

---

## Perguntas em Aberto

- [ ] Target do modelo: `RED_FLAG` atual ou previsão em horizonte N meses?
- [ ] Análise de itens: acréscimo no `01_pedidos_eda.ipynb` ou novo notebook `02_itens_eda.ipynb`?
- [ ] Critérios exatos de subfragmentação do "baixo": validar threshold com o cliente
- [ ] Qualidade da variável `SETOR`: verificar antes de usar no modelo
