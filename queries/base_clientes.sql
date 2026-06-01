-- QUERIE: TABELA BASE 2 
SELECT
    CODIGO, NOME, CLASSE, CIDADEENTREGA, UFENTREGA, ATIVIDADE, [STATUS], CODCONDICAOPAGAMENTO, CONDICAOPAGAMENTO, DIASINADIMPLENTE
FROM CONTROLLER.dbo.PBI_Clientes;

-- Colunas disponiveis 

-- Informacoes que podem me levar a entender o comportamento de compra dos clientes, como por exemplo:
-- 1. Localizacao: CIDADEENTREGA, UFENTREGA, existe algum comportamento de compra diferente entre clientes de diferentes regioes?
-- 2. Classe: existe alguma diferenca de comportamento entre clientes de classes diferentes?
        -- 
-- 3. Atividade: existe alguma atividade que e um pouco mais instavel que outras?
-- 4. condicao de pagamento: clientes com maior folga pra pagar churnam menos?
-- 5. dias inadimplente: clientes que ficam mais tempo inadimplentes tem maior risco de churnar? ele churna antes ou depois de ficar inadimplente