# 🧾 Ficha Técnica (Model Card): Time-to-Churn Probability Model

- **Nome** - Telecom Churn Probability Classifier (MLP)
- **Data de treinamento** - Junho/2026
- **Versão do modelo** - 1.0.0-alpha (Em Desenvolvimento)
- **Tipo de modelo** - Classificação binária probabilística com MLP em PyTorch
- **Pipeline** - Pré-processamento e transformações via Scikit-Learn unificados em módulo reutilizável

- **Para quais casos o modelo foi projetado**
    - Predição da probabilidade individual de cancelamento ($P(\text{churn})$) de clientes ativos de telecomunicações.
    - Alimentação diária/periódica de réguas automatizadas de CRM divididas por faixas de risco e impacto de receita mensal (MRR).

## 🎯 Saída

- `churn_probability`: Valor contínuo entre $0.0$ e $1.0$ (Ativação Sigmoide na última camada da rede).
- `risk_tier`: Categoria transicional de risco baseada na probabilidade calculada para direcionamento de ações.

---

# 📊 Métricas de Avaliação

Para a avaliação do modelo durante os experimentos e validação cruzada estratificada, adotamos métricas técnicas focadas no comportamento da classe minoritária (churners) e na estabilidade da probabilidade gerada.

- **PR-AUC (Precision-Recall Area Under Curve):** Nossa métrica técnica primária de otimização. Em datasets desbalanceados onde a classe positiva é minoritária, ela avalia a qualidade das probabilidades sem a distorção causada pelo grande volume de verdadeiros negativos.
- **F2-Score:** Utilizado como critério técnico para calibrar e desempatar limiares de decisão, priorizando o Recall (capturar o maior número possível de clientes em risco) sem degradar severamente a Precisão.
- **Métricas de Negócio Simuladas (MRR-RP):** Cruzamento das probabilidades geradas com o campo `MonthlyCharges` para estimar o volume de receita recorrente passível de proteção pelas ações de retenção.

## 🏆 Métricas Principais (MLP vs. Baselines)
*(Os valores abaixo serão preenchidos após a execução do loop de treinamento e registro no MLflow)*

| Métrica | MLP PyTorch (Proposto) | Regressão Logística (Baseline) | Diferença |
| :--- | :---: | :---: | :---: |
| **PR AUC** | *[Preencher]* | *[Preencher]* | *[Preencher]* |
| **F2-Score** | *[Preencher]* | *[Preencher]* | *[Preencher]* |
| **Recall** | *[Preencher]* | *[Preencher]* | *[Preencher]* |
| **Precision** | *[Preencher]* | *[Preencher]* | *[Preencher]* |
| **ROC AUC** | *[Preencher]* | *[Preencher]* | *[Preencher]* |
| **Accuracy** | *[Preencher]* | *[Preencher]* | *[Preencher]* |

---

## ⚖️ Definição das Faixas de Risco (Trade-off de Custo)

Diferente de um corte binário duro fixo, a probabilidade gerada pela ativação Sigmoide será mapeada em faixas de atuação para equilibrar o custo operacional das ofertas de retenção em Telecomunicações:

| Faixa de Probabilidade | Classificação de Risco | Ação Direcionada no CRM |
| :--- | :--- | :--- |
| **$P(\text{churn}) \ge 0.80$** | **Risco Crítico** | Acionamento de contingência/humano (Ouvidoria/Call Center) focado na preservação do MRR de alto valor. |
| **$0.50 \le P(\text{churn}) < 0.80$** | **Risco Alto** | Disparo automatizado de ofertas de engajamento digitais (Upgrades, benefícios em serviços de valor agregado). |
| **$P(\text{churn}) < 0.50$** | **Risco Monitorado** | Manutenção nas réguas tradicionais de comunicação e marketing da companhia. |

*(Espaço reservado para a tabela de variação de limiares técnicos pós-validação cruzada)*
