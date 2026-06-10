# 📃 ML Canvas: Churn

[cite_start]Neste documento apresentamos a formulação do problema focada nos conceitos de _"Business Understanding"_ e _"Data Understanding"_ para o Tech Challenge[cite: 1, 2, 41]. [cite_start]O modelo preditivo (Rede Neural MLP) estima a probabilidade de um cliente efetuar o cancelamento [cite: 11, 13][cite_start], mapeando-o em réguas de acionamento baseadas em risco e impacto financeiro direto na receita mensal (Monthly Charges)[cite: 71, 72].

##  Qual problema queremos resolver?
[cite_start]Desacelerar a perda de clientes e mitigar a evasão de receita recorrente mensal (MRR) na operadora de telecomunicações [cite: 10][cite_start], identificando clientes com alta probabilidade de cancelamento antes que encerrem seus contratos[cite: 11].

##  Métricas do negócio que consideraremos para a resolução do problema
Como o modelo estima a probabilidade de churn baseada no comportamento atual de faturamento e serviços, avaliaremos o impacto financeiro real utilizando:

- **MRR em Risco Protegido (MRR-RP):** O volume de faturamento mensal (`MonthlyCharges`) dos clientes classificados nas faixas de maior risco que permaneceram ativos após a janela de corte de 90 dias pós-intervenção.

$$\text{MRR-RP} = \sum_{i \in \text{Tratados Ativos}} \text{MonthlyCharges}_i$$

- **ARPU Preservation Rate (APR):** Taxa de controle para monitorar se os descontos concedidos pelo time de retenção não estão corroendo a Receita Média por Usuário (ARPU) de forma predatória.

##  O que queremos atingir
- Mapear e acionar proativamente os clientes que concentram a maior fatia de receita em risco, buscando reter pelo menos **80% do MRR total** do grupo identificado com propensão ao churn.
- Garantir que o custo total das ações de retenção (ex: concessão de descontos em faturas) não ultrapasse **15% do valor do MRR salvo** no trimestre subsequente.

##  O que atingimos
[cite_start]*(Espaço em branco reservado para os resultados pós-treinamento da MLP em PyTorch e validação do modelo)* [cite: 28, 48]

##  O que faremos com essa informação?
[cite_start]A camada de saída do modelo (Rede Neural MLP com ativação Sigmoide) retornará a **Probabilidade de Churn ($P(\text{churn})$)** de cada cliente[cite: 13, 28]. O pipeline categorizará os clientes em **Faixas de Risco** para que as equipes de CRM e Marketing possam direcionar ações diretamente no ecossistema de dados disponível:

**Fluxo de Risco e Ações Transicionais:**
- **$P(\text{churn}) \ge 80\%$ [Risco Crítico]:** Direcionamento para atendimento prioritário e ofertas agressivas de renovação de contrato (fidelização), focado em clientes com alto `MonthlyCharges`.
- **$50\% \le P(\text{churn}) < 80\%$ [Risco Alto]:** Disparo de campanhas automatizadas de engajamento via canais digitais (E-mail/SMS), oferecendo vantagens nos serviços de valor agregado que o cliente ainda não possui contratado.
- **$P(\text{churn}) < 50\%$ [Risco Monitorado]:** Clientes permanecem nas réguas de comunicação padrão da companhia.

##  Como atestar a assertividade das ações tomadas a partir do modelo?
As ações de retenção serão validadas através de um design de teste com grupo de controle isolado:
- **Grupo tratado:** Clientes com probabilidade de churn acima do threshold que **recebem** a oferta ou abordagem de retenção.
- **Grupo controle (Holdout):** Clientes com o mesmo perfil de risco elevado que **não recebem** nenhuma ação (mantidos no fluxo orgânico para avaliar a taxa de churn base).

A métrica de validação do experimento será o **Share de Churners**:

$$\text{Share de Churners} = \frac{\text{Churners no grupo}}{\text{Total do grupo}}$$

O modelo e a estratégia de negócio serão considerados eficazes se o Grupo Tratado contiver um Share de Churners significativamente menor e retiver mais receita (`TotalCharges`) do que o Grupo Controle.

##  Recursos necessários
- [cite_start]**Base de Dados:** Dataset estruturado contendo dados demográficos, contratuais e financeiros de clientes (IBM Telco Churn Dataset)[cite: 71, 72].
- [cite_start]**Infraestrutura em Nuvem:** Ambiente para execução dos pipelines de pré-processamento (Scikit-Learn), rastreamento de experimentos (MLflow) e exposição do modelo através de API (FastAPI)[cite: 29, 31, 32, 57].

##  Dados e variáveis relevantes (Data Understanding)
A análise e as predições serão baseadas exclusivamente nas features nativas do ecossistema do dataset:
- **`tenure`:** Quantidade de meses que o cliente está na base, refletindo a sua lealdade temporal.
- **`MonthlyCharges`:** O valor mensal atual cobrado do cliente, que dita o impacto direto no MRR em risco.
- **`TotalCharges`:** O montante total acumulado gasto pelo cliente na empresa.
- **`Contract`:** O tipo de contrato atual (Mensal, Anual, Bienal) – variável crítica para entender o risco de quebra de vínculo.
- **`PaymentMethod`:** Forma de pagamento elegida (Boleto eletrônico, Boleto enviado por correio, Débito automático, Cartão de crédito).
- **Aderência a Serviços Técnicos/Segurança:** Flags de contratação de serviços que aumentam o *lock-in* e reduzem o churn, como `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection` e `TechSupport`.
