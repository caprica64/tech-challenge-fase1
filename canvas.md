
# ML Canvas — Previsão de Churn em Telecom

Projeto de Machine Learning baseado nos princípios de Business Understanding e Data Understanding descritos no material da aula.

1. **Problema de Negócio**

A Calcom identificou uma alta taxa de cancelamento de clientes (churn) nos dois últimos trimestres impactando de maneira inesperada. As razões ainda são desconhecidas e para planejarem planos de ação e recuperação é necessário explorar a fundo o tem levado os clientes ao cancelamento.

O board determinou a criação de uma força tarefa inter-departamental para estudar quais razões levam aos cancelamentos e identificar oportunidades de melhorias.

Esta força tarefa é composta por representantes das áreas de atendimento ao cliente (inclui suporte técnico, instalação e retenção de clientes),infraestrutura, operações técnicas de campo, marketing, vendas e finanças.


2. **Objetivo de Negócio**

Reduzir o churn trimestral em **15%**, identificando clientes com alta probabilidade de cancelamento para ações preventivas.


3. **KPIs (Indicadores de Sucesso)**

**KPI Principal**

- Taxa de churn trimestral


4. **Objetivo Técnico (Data Science)**

- Construir um modelo de **classificação binária** para prever churn (0/1).
- Métricas técnicas mínimas:
	- [ ] AUC ≥ **0,85**
	- [ ] Recall ≥ **0,75**
	- [ ] Precisão ≥ **0,70**
- Entrega via API + dashboard explicativo.


5. **Stakeholders**

**Patrocinadores**

- CEO
- VP Marketing
- VP Vendas
- VP Operações
- VP Finanças

**Especialistas de Domínio**

- Vendas: Analisa métricas do CRM
- Atendimento ao cliente: fornecem suporte técnico, instalação e retenção de clientes
- Operações: gerencia a infraestrutura de operações internas (rede, data center) e externa (cabeamento subterraneo, estações rádio-base)

**Donos dos Dados**

- Engenharia de Dados
- Operações: TI, atendimento a clientes.
- Vendas: times de contas B2B, B2C

**Usuários Finais**

- Vendas
- Atendimento a clientes, incluindo retenção: responsável por escutar clientes, instruí-los no uso dos produtos.
- Operaçã: responsável pelas ações de melhoria de serviço.


6. **Requisitos do Projeto**

- Seeds fixados para reprodutibilidade **42**.
- Validação cruzada estratificada.
- Model card documentando limitações e viéses.
- Interpretabilidade obrigatória (ex.: SHAP).
- Testes automatizados (>= 3: smoke test, schema e API).
- Loggin estruturado sem print().
- Integração com sistemas internos via API.
- Linting com ruff sem erros.


7. **Restrições**

- Dados históricos incompletos para parte da base.
- Sistemas legados dificultam integração em tempo real.
- Orçamento limitado para dados externos.
- Equipe reduzida de engenharia.


8. **Pressupostos**

- A definição de churn permanecerá estável durante o projeto.
- A equipe de CRM fornecerá dados de campanhas anteriores.
- A infraestrutura atual suporta treinamento e inferência.
- Dados de cancelamento são confiáveis e atualizados.


9. **Dados Disponíveis**

**Fontes Internas**

- Histórico de faturamento
- Registros de suporte (tickets, chamadas)
- Uso de dados, voz e SMS
- Reclamações (SAC, Ouvidoria)
- Plano contratado
- Tempo como cliente
- Pagamentos em atraso

**Fontes Externas (opcionais)**

- Qualidade de cobertura por região
- Ofertas da concorrência
- Indicadores socioeconômicos por área geográfica (latitude e longitude)


10. **Variáveis Relevantes (Features)**

**Demográficas**

- Idade
- Região
- Sexo
- Parceiros
- Dependentes

**Comportamentais**

- Consume dados
- Consume voz
- Uso de backup na nuvem
- Uso de serviços de segurança
- Uso de streaming
- Mudanças recentes de plano 

**Históricas**

- Tempo como cliente
- Usou suporte técnico


**Contextuais**

- Cobertura na região
- Concorrência local

**Variáveis Derivadas**

- Score de insatisfação
- Variação do consumo mês a mês


11. **Variável Alvo (Target)**

- **Churn = 1** : cliente cancelou nos últimos 30 dias
- **Churn = 0** : cliente permaneceu ativo

12. **Riscos e Contingências**

**Riscos**

- Dados incompletos
- Baixa adoção pelo time de retenção
- Mudança na definição de churn
- Possível viés do modelo

**Contingências**

- Estratégias de imputação
- Treinamento dos usuários finais
- Revisão trimestral da definição de churn
- Auditoria de fairness


13. **Plano de Projeto (Macro)**

**Etapas (CRISP-DM)**

- 1	Business Understanding — concluído

- 2	Data Understanding — 

- 3	Data Preparation — 

- 4	Modeling — 

- 5	Evaluation — 

- 6	Deployment — 


14. **Entregáveis**

- Modelo preditivo de churn
- Dashboard com scores e explicações
- API interna
- Documentação técnica
- Relatório executivo
- Plano de monitoramento


15. **Critérios de Sucesso Final**

**Sucesso Técnico**

- AUC ≥ 0,85
- Recall ≥ 0,75

**Sucesso de Negócio**

- Redução real do churn ≥ 15%
- Aumento da retenção pós-campanha
- Adoção do modelo pelas equipes

