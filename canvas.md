
# ML Canvas — Previsão de Churn em Telecom

Projeto de Machine Learning baseado nos princípios de Business Understanding e Data Understanding descritos no material da aula.

1. **Problema de Negócio**

A FIAPMobile enfrenta uma alta taxa de cancelamento de clientes (churn), impactando receita recorrente, previsibilidade financeira e aumentando custos de aquisição.

2. **Objetivo de Negócio**

Reduzir o churn trimestral em **15%**, identificando clientes com alta probabilidade de cancelamento para ações preventivas.

3. **KPIs (Indicadores de Sucesso)**

**KPI Principal**

- Taxa de churn mensal/trimestral

**KPIs Secundários**

- Taxa de retenção pós-campanha
- Redução do CAC
- Aumento do LTV (Lifetime Value)
4. **Objetivo Técnico (Data Science)**
- Construir um modelo de **classificação binária** para prever churn (0/1).
- Métricas técnicas mínimas:
	- [ ] AUC ≥ **0,85**
	- [ ] Recall ≥ **0,75**
	- [ ] Precisão ≥ **0,70**
- Entrega via API + dashboard explicativo.
5. **Stakeholders**

**Patrocinadores**

- Diretoria de Marketing
- Diretoria de Operações

**Especialistas de Domínio**

- CRM
- Suporte técnico
- Atendimento ao cliente

**Donos dos Dados**

- Engenharia de Dados
- TI

**Usuários Finais**

- Time de retenção
- Time de campanhas personalizadas
6. **Requisitos do Projeto**
- Entrega do modelo em **90 dias**.
- Interpretabilidade obrigatória (ex.: SHAP).
- Conformidade com **LGPD**.
- Atualização semanal das previsões.
- Integração com sistemas internos via API.
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
- Indicadores socioeconômicos por CEP
10. **Variáveis Relevantes (Features)**

**Demográficas**

- Idade
- Região

**Comportamentais**

- Consumo de dados
- Consumo de voz
- Uso do app
- Mudanças recentes de plano

**Históricas**

- Tempo como cliente
- Histórico de pagamentos
- Reclamações recentes
- Número de tickets abertos

**Contextuais**

- Cobertura na região
- Concorrência local
- Sazonalidade

**Variáveis Derivadas**

- Score de insatisfação
- Variação do consumo mês a mês
- Frequência de atrasos
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

> 1	Business Understanding — concluído

> 2	Data Understanding — semanas 1–3

> 3	Data Preparation — semanas 3–6

> 4	Modeling — semanas 6–8

> 5	Evaluation — semanas 8–10

> 6	Deployment — semanas 10–12

**Pontos de Checagem**

- Semana 3: validação das fontes de dados
- Semana 8: apresentação dos primeiros modelos
- Semana 12: entrega final
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

Se quiser, posso gerar **uma versão visual**, **um PDF**, **um slide**, ou **um template reutilizável** desse canvas.

