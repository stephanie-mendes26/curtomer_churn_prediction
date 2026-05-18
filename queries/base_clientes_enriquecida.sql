-- Une o cadastro de clientes com o histórico de pedidos.
-- Clientes sem nenhum pedido aparecem com ultima_compra = NULL.
-- meses_sem_compra é calculado em relação à data atual do servidor.

SELECT
    c.CODIGO,
    c.NOME,
    c.CLASSE,
    c.ATIVIDADE,
    c.UFENTREGA,
    c.CIDADEENTREGA,
    c.CONDICAOPAGAMENTO,
    c.STATUS                                                          AS status_sistema,
    c.DIASINADIMPLENTE,

    MIN(p.DATA_ENTRADA)                                               AS primeira_compra,
    MAX(p.DATA_ENTRADA)                                               AS ultima_compra,
    COUNT(DISTINCT p.NRO_PED)                                         AS total_pedidos,
    SUM(CASE WHEN p.VL_SERVICO > 0 THEN p.VL_SERVICO ELSE 0 END)     AS total_valor,

    -- Recência: meses desde a última compra (NULL se nunca comprou)
    DATEDIFF(MONTH, MAX(p.DATA_ENTRADA), GETDATE())                   AS meses_sem_compra

FROM CONTROLLER.dbo.PBI_Clientes c
    LEFT JOIN CENTRAL.dbo.PBI_DESEMPENHO_MATERIAL p
    ON c.CODIGO = p.CLIENTE
        AND p.DATA_ENTRADA >= '2023-01-01'
-- janela de análise consistente com o EDA de pedidos

GROUP BY
    c.CODIGO, c.NOME, c.CLASSE, c.ATIVIDADE,
    c.UFENTREGA, c.CIDADEENTREGA, c.CONDICAOPAGAMENTO,
    c.STATUS, c.DIASINADIMPLENTE;
