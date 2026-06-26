# 🧾 Ficha Técnica (Model Card): Telecom Churn Probability Classifier

- **Nome** — Telecom Churn Probability Classifier (MLP)
- **Data de treinamento** — Junho/2026
- **Versão do modelo** — 1.0.0-alpha (Em Desenvolvimento)
- **Tipo de modelo** — Classificação binária probabilística com MLP em PyTorch
- **Pipeline** — Pré-processamento e transformações via Scikit-Learn unificados em módulo reutilizável
- **Proprietário** — Equipe FIAP MLET10 — Tech Challenge Fase 1

---

## 🎯 Uso Pretendido

- Predição da probabilidade individual de cancelamento (P(churn)) de clientes ativos de telecomunicações.
- Alimentação diária/periódica de réguas automatizadas de CRM divididas por faixas de risco e impacto de receita mensal (MRR).
- **Usuários finais**: Time de retenção, time de campanhas personalizadas, CRM.

### Cenários fora do escopo

- Não deve ser usado para decisões de crédito ou financeiras.
- Não substitui avaliação humana em casos de reclamação formal.
- Não generaliza para operadoras fora do estado da Califórnia sem retreinamento.

---

## 📊 Métricas de Avaliação

### Métricas técnicas

- **PR-AUC (Precision-Recall Area Under Curve):** Métrica primária de otimização. Em datasets desbalanceados, avalia a qualidade das probabilidades sem distorção do grande volume de verdadeiros negativos.
- **F2-Score:** Critério técnico para calibrar limiares de decisão, priorizando Recall sem degradar severamente a Precisão.
- **ROC-AUC:** Capacidade geral de discriminação entre classes.

### Métricas de negócio

- **MRR-RP (Monthly Recurring Revenue at Risk Protected):** Cruzamento das probabilidades com `Monthly Charges` para estimar receita recorrente passível de proteção pelas ações de retenção.
- **Taxa de retenção pós-campanha:** Percentual de churners previstos que permaneceram após intervenção.
- **Custo de churn evitado:** `FN × custo_perda_cliente` vs. `FP × custo_oferta_retenção`.

---

## 🏆 Métricas Principais (MLP vs. Baselines)

| Métrica | MLP PyTorch | Random Forest | Regressão Logística | DummyClassifier |
| :--- | :---: | :---: | :---: | :---: |
| **PR AUC** | **0.950** | 0.945 | 0.934 | 0.272 |
| **ROC AUC** | **0.979** | 0.977 | 0.974 | 0.516 |
| **Recall** | **0.947** | 0.874 | 0.858 | 0.291 |
| **Precision** | 0.778 | **0.872** | 0.829 | 0.289 |
| **F1** | 0.854 | **0.873** | 0.844 | 0.290 |
| **Accuracy** | 0.914 | **0.933** | 0.916 | 0.622 |

> **Análise:** A MLP obteve o melhor PR-AUC e Recall entre todos os modelos, priorizando a detecção de churners conforme estratégia de negócio. O Random Forest apresentou melhor equilíbrio geral (F1/Precision), sendo uma alternativa para cenários onde o custo de FP é mais relevante.

---

## ⚖️ Definição das Faixas de Risco (Trade-off de Custo)

**Threshold ótimo de custo:** 0.45 (minimiza custo total R$ 4.000)

| Faixa de Probabilidade | Classificação de Risco | Ação Direcionada no CRM | Custo estimado |
| :--- | :--- | :--- | :--- |
| **P(churn) ≥ 0.80** | Risco Crítico | Acionamento humano (Ouvidoria/Call Center) focado na preservação do MRR de alto valor. | R$ 7.680 |
| **0.45 ≤ P(churn) < 0.80** | Risco Alto | Disparo automatizado de ofertas de engajamento (upgrades, benefícios de valor agregado). | **R$ 4.000 (ótimo)** |
| **P(churn) < 0.45** | Risco Monitorado | Manutenção nas réguas tradicionais de comunicação e marketing. | — |

> O threshold 0.45 equilibra o custo assimétrico: FN (R$100/cliente perdido) vs FP (R$20/oferta desnecessária). Economia de ~R$500 em relação ao threshold padrão 0.5 no dataset de teste.

---

## 📐 Dados de Treinamento

- **Fonte:** IBM Telco Customer Churn dataset (adaptado para Calcomm)
- **Volume:** ~7.043 clientes
- **Features utilizadas:** Demographics, service subscriptions, billing, tenure, features engenheiradas (charge_rel, Engineered Monthly Charges)
- **Split:** 80% treino / 20% teste, estratificado pela variável target
- **Desbalanceamento:** ~73% não-churn / ~27% churn (ratio 0.36)

---

## ⚠️ Limitações Conhecidas

1. **Generalização restrita** — Treinado exclusivamente com dados da Califórnia (operadora única). Não deve ser aplicado diretamente em outras regiões ou operadoras sem retreinamento.
2. **Sazonalidade não capturada** — O dataset representa um corte temporal estático. Variações sazonais (férias, promoções da concorrência) não estão modeladas.
3. **Features estáticas** — Não captura variação de comportamento ao longo do tempo (séries temporais de uso).
4. **Calibração limitada** — As probabilidades geradas pela sigmoide podem não refletir frequências reais de churn sem calibração (Platt scaling ou isotonic regression).
5. **Desbalanceamento** — Apesar do tratamento com class_weight/SMOTE, falsos negativos em períodos atípicos podem aumentar.

---

## 🧭 Análise de Vieses

| Dimensão | Análise | Status |
| :--- | :--- | :--- |
| **Gênero** | Modelo não usa gênero como feature preditiva após OHE. Verificar disparidade de recall entre grupos. | ⚠️ Pendente auditoria |
| **Idade (Senior Citizen)** | Incluída como feature. Verificar se não penaliza desproporcionalmente clientes idosos. | ⚠️ Pendente auditoria |
| **Localidade** | Zip Code/coordenadas removidos do modelo final. Sem viés geográfico direto. | ✅ Mitigado |
| **Renda implícita** | `Monthly Charges` pode ser proxy de poder aquisitivo. Monitorar distribuição de FPs por faixa de cobrança. | ⚠️ Monitorar |

**Recomendação:** Executar auditoria de fairness (equalized odds, demographic parity) antes do deploy em produção.

---

## 🏗️ Arquitetura de Deploy

| Aspecto | Escolha | Justificativa |
| :--- | :--- | :--- |
| **Modalidade** | Real-time (API) | Permite integração direta com CRM para ações imediatas ao detectar risco. |
| **Infraestrutura** | AWS Lambda + API Gateway | Serverless, custo proporcional ao uso, sem gestão de servidores. |
| **Imagem** | Docker (ECR) | Reprodutibilidade garantida; CI/CD automatizado via GitHub Actions. |
| **Fallback** | Se Lambda timeout (>15s), degradar para batch diário via Step Functions. | Resiliência operacional. |

---

## 📡 Plano de Monitoramento

### Objetivo

Detectar degradação de performance do modelo em produção antes que impacte significativamente as métricas de negócio, disparando ações corretivas ou retreinamento.

---

### 1. Métricas Monitoradas

| Métrica | Tipo | Fonte | Frequência | Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **PR-AUC** | Técnica (offline) | Batch de avaliação semanal | Semanal | ≥ 0.70 |
| **ROC-AUC** | Técnica (offline) | Batch de avaliação semanal | Semanal | ≥ 0.85 |
| **Recall (classe churn)** | Técnica (offline) | Batch de avaliação semanal | Semanal | ≥ 0.75 |
| **Taxa de predições positivas** | Operacional (online) | Logs da API /predict | Diária | 20-35% |
| **Latência P95** | Operacional (online) | CloudWatch (Lambda) | Contínua | < 500ms |
| **Taxa de erro HTTP 5xx** | Operacional (online) | CloudWatch (API Gateway) | Contínua | < 1% |
| **Data drift (features numéricas)** | Dados | Comparação distribuição treino vs. produção | Semanal | PSI < 0.1 |
| **Concept drift (target)** | Dados | Taxa de churn real vs. predita | Mensal | Desvio < 5pp |

---

### 2. Alertas e Thresholds

| Condição | Severidade | Canal | Ação |
| :--- | :--- | :--- | :--- |
| ROC-AUC < 0.80 | 🔴 Crítico | Slack #ml-alerts + PagerDuty | Retreinamento imediato |
| PR-AUC cai > 0.05 vs. baseline | 🟠 Alto | Slack #ml-alerts | Investigar data drift em 24h |
| Taxa de predições positivas > 45% ou < 10% | 🟠 Alto | Slack #ml-alerts | Verificar mudança de distribuição nos dados de entrada |
| Latência P95 > 1000ms | 🟡 Médio | CloudWatch Alarm | Verificar concorrência/cold starts |
| Taxa 5xx > 5% em 5 min | 🔴 Crítico | PagerDuty | Rollback para versão anterior |
| PSI de qualquer feature > 0.2 | 🟠 Alto | Slack #ml-alerts | Iniciar retreinamento dentro de 1 semana |

---

### 3. Pipeline de Avaliação Semanal

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌────────────┐
│ Extrai dados│ ──▶ │ Gera predições   │ ──▶ │ Compara com    │ ──▶ │ Publica    │
│ da semana   │     │ com modelo atual │     │ churn real     │     │ métricas   │
└─────────────┘     └──────────────────┘     └────────────────┘     └────────────┘
                                                                           │
                                                                    ┌──────▼──────┐
                                                                    │ MLflow      │
                                                                    │ Dashboard   │
                                                                    └─────────────┘
```

- **Entrada:** Clientes que cancelaram na semana anterior (ground truth atrasado de 7-30 dias).
- **Execução:** AWS Step Functions (agendamento EventBridge, cron semanal).
- **Saída:** Métricas registradas no MLflow + alertas disparados se thresholds violados.

---

### 4. Estratégia de Retreinamento

| Gatilho | Tipo | Dados | Estimativa de tempo |
| :--- | :--- | :--- | :--- |
| Alerta crítico (ROC-AUC < 0.80) | Emergencial | Últimos 6 meses | < 24h |
| PSI > 0.2 por 2 semanas consecutivas | Proativo | Últimos 6 meses + dados novos | < 1 semana |
| Agendado (trimestral) | Preventivo | Últimos 12 meses | Planejado no sprint |

**Processo:**
1. Retreinar modelo com dados atualizados (mesmo pipeline `src/churn/model_trainer.py`).
2. Avaliar no holdout mais recente — modelo novo deve superar modelo atual nas métricas primárias.
3. Registrar no MLflow como novo modelo candidato.
4. Shadow deploy (dual-write) por 48h comparando predições do novo vs. atual.
5. Promover para produção se métricas forem iguais ou superiores.

---

### 5. Playbook de Resposta a Incidentes

| Cenário | Diagnóstico | Ação imediata | Ação de follow-up |
| :--- | :--- | :--- | :--- |
| Queda brusca de ROC-AUC | Verificar data drift + mudanças no catálogo de produtos | Ativar modelo anterior (rollback Lambda) | Retreinamento emergencial |
| Spike de latência | Cold starts + tamanho da imagem Docker | Provisioned Concurrency no Lambda | Otimizar dependências no Docker |
| 100% de predições = 0 (ou = 1) | Feature constante na entrada / bug no preprocessamento | Rollback + alerta ao time de dados | Corrigir pipeline de dados |
| Churn real sobe sem que o modelo detecte | Concept drift (novos motivos de saída) | Comunicar stakeholders + análise exploratória | Adicionar novas features + retreinar |

---

### 6. Dashboard de Monitoramento

**Ferramentas:** MLflow (métricas de modelo) + CloudWatch (métricas operacionais)

Painéis recomendados:
1. **Performance do modelo** — ROC-AUC, PR-AUC, Recall semanal (linha temporal)
2. **Distribuição de scores** — Histograma de P(churn) em produção vs. treino
3. **Operacional** — Latência P50/P95/P99, taxa de erros, invocações/min
4. **Negócio** — MRR at risk, taxa de retenção pós-campanha, conversão de ofertas

---

## 📝 Histórico de Versões

| Versão | Data | Mudanças |
| :--- | :--- | :--- |
| 1.0.0-alpha | Jun/2026 | Versão inicial — MLP PyTorch + baselines sklearn |

---

## 📚 Referências

- Mitchell, M. et al. (2019). "Model Cards for Model Reporting." *ACM FAccT*.
- FIAP MLET10 — Material das disciplinas de Ciclo de Vida de ML, Fundamentos, Engenharia de Software e APIs.
- IBM Telco Customer Churn Dataset.
