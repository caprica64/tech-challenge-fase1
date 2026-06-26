# Tech Challenge turma MLET10 - Fase 1 - FIAP

Tech Challenge Fase 1 do MLET10 da FIAP

## 👥 Integrantes
| Nome | RM | Contato |
|--|--|--|
| Allanderson Barros | rm375420 | [Github](https://github.com/AllanBa) - [Linkedin](www.linkedin.com/in/allanbarros)|
| Lucas Albino | rm| [Github](xx) - [Linkedin](xx)|
| Rodrigo De Vincenzo Monteiro | rm| [Github](xx) - [Linkedin](xx)|
| Pedro Henrique Silva Gaspar | rm | [Github](xx) - [Linkedin](xx)|
| Breno Toledo Kutti Lugão | rm | [Github](xx) - [Linkedin](xx)|

## Instruções de setup

O projeto é composto em um repositório no GitHub, este mesmo, para código e o projeto de Machine Learning e inferência. O projeto também contém um pipeline CI/CD que empurra a imagem do container de inferência para o Elastic Container Regitry (ECR) na AWS em uma conta destinada ao projeto. Requerimentos:

 - Conta AWS deve existir e a integração entre o projeto GitHub deve estar configurada. Recomenda-se usar o método OIDC para conceder permissão na conta AWS ao invés de credenciais estáticas com access key e secret key do IAM, métodos usados anteriormente. Para referências e exemplo ver o [Configuring OpenID Connect in Amazon Web Services](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws)

 - Toda a infraestrutura na AWS foi montada usando um pipeline CI/CD no GitLab nesse projeto indicado abaixo. Este projeto GitLab também necessita integraão com AWS de maneira similar ao GitHub. As variáveis, entretanto, são configuradas de maneira diferente no GitLab.


Ele pode ser clonado e adaptado para construir imagens de container em outras plataformas de nuvem ou provedores de serviço, DockerHub, etc...

### Repositório de infraestrutura na AWS

Repositório de infraestrutura na AWS foi montado no GitLab público em [Infra desafio 1](https://gitlab.com/mlet10/infra-desafio1/)

Este projeto Terraform com pipeline no GitLab constrói um API Gateway aberto na Internet com hostname churn dentro de um domínio já criado e hospedado na própria conta AWS (blog de outro projeto). O API Gateway possui rotas para um backend em Lambda com um runtime em container tipo Docker/OCI que carrega imagem template do ECR. 

Este projeto do runtime e inferência de ML também atualiza o ECR que está indicado em variáveis do GitHub Actions. O projeto em Terraform pode ser adaptado para criar ambiente similar.

O pipeline do GitLab possui estágios para construir uma imagem de container simples pois ela precisa estar disponíveil para o Terraform finalizar a construção da função Lambda no estágio final de apply. 

Ao mesmo tempo a construção do repositório em ECR dependente da execução do Terraform completo o que cria uma relação cíclica. Para este pipeline, é necessário rodar duas vezes na primeira para vez para criar a infraestrutura da AWS em Terraform. A primeira montará o ECR falhando no estágio de montar o container de exemplo e a segunda vez implementará todos os recursos.

O URI ou nome de recurso do repositório de images no ECR deve ser adicionado a variáveis de ambiente do CI/CD do GitHub para que ele possa atualizar a imagem de container posteriormente a cada atualização de código e testes. 

A intenção do projeto AWS é demonstrar uma aplicação prática de inferência em recursos simples serverless com custo otimizado para desafio. Como sugestão de melhorias no projeto deixamos:

 - Implementar a criação do ECR como um projeto Terraform separado dentro ou fora do mesmo CI/CD no GitLab a fim de remover referência cíclica.
 - Implementar controles de segurança no ambiente AWS como Web Application Firewall (WAF) antes do API-Gateway. 
 - Implementar controles de utilização do API-Gateway como limites de requisição e formato esperado das APIs REST. 
 - Implementar documentação dos contratios.
 - Fornecer instruções de como rodar o workload completo como SAM localmente com emulador Lambda e runtime.

## O projeto no GitHub 

O projeto no GitHub roda um pipeline CI/CD conforme há mudanças na branch **dev** através de commits ou merges. As seguintes ações são executadas:

 - Lint através do pipeline gerenciado pelo **lint.yaml** que valida formatação e normas que queremos aplicar ao código python e testes.
 - Deploy na AWS através do pipeline **deploy.yaml** que executa estágios de construção do container e o upload (push) para o ECR. Aqui existem variáveis de ambiente que indicam qual role assumir na AWS via OIDC. 

 **Importante** - tanto para GitLab como GitHub, é necessário que a Role AWS confie nesses projetos de maneira mais retristiva como confiar no repo e branch (uma ou mais se há mais dum ambiente) e evitar conceder mais permissões que necessário na AWS. Está fora de escopo para o desafio detalhar as permissões da role mas em resumo ela deve ter permissões para acessar ECR, gerenciar Lambda, X-Ray, CloudWatch Logs e métricas, API-Gateway. Outras permissões podem ser necessárias no projeto.


# Objetivo do desafio

O objetivo desse desafio é implementar um modelo de previsão de churn a partir dos dados de churn da IBM com relação a serviços de telecomunicações na California

## 📁 Estrutura de pastas

- `docs/`: Model Card, plano de monitoramento e demais documentos finais
- `notebooks/`: Exploração e validação histórica do projeto
- `src/`: Código-fonte do pipeline, treino, inferência e API
- `data/`: Dataset bruto original
- `models/`: Pesos, pipeline e artefatos versionados
- `tests/`: Testes automatizados
- `logs`: Logs e saídas das execuções

# Estrutura do projeto

O projeto do desafio é formato pelas seguintes pastas e arquivos:

- /src
- /models
- /docs
- /data

---

# Modelagem com Redes Neurais (MLP PyTorch)

## Visão geral

O módulo `src/churn/train_mlp.py` implementa a construção, treinamento e avaliação de uma rede neural MLP (Multi-Layer Perceptron) com PyTorch para previsão de churn, conforme requisitos da Etapa 2 do projeto.

### Como executar

```bash
source .venv/bin/activate
python -m src.churn.train_mlp
```

## Arquitetura da MLP

```
Input (n features)
    │
    ▼
Linear(n → 128) → ReLU → Dropout(0.3)
    │
    ▼
Linear(128 → 64) → ReLU → Dropout(0.3)
    │
    ▼
Linear(64 → 32) → ReLU → Dropout(0.3)
    │
    ▼
Linear(32 → 1) → Sigmoid (via BCEWithLogitsLoss)
    │
    ▼
P(churn) ∈ [0, 1]
```

### Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|:---|:---|:---|
| `hidden_layers` | (128, 64, 32) | Redução progressiva de dimensionalidade |
| `dropout` | 0.3 | Regularização contra overfitting (dataset pequeno) |
| `activation` | ReLU | Padrão para MLPs — eficiente e sem vanishing gradient |
| `learning_rate` | 0.001 | Valor padrão do Adam — balanceia velocidade e estabilidade |
| `epochs` | 100 | Suficiente com early stopping implícito via loss monitoring |
| `batch_size` | 64 | Equilíbrio entre ruído de gradiente e velocidade |
| `pos_weight` | ~2.76 | Compensa desbalanceamento 73/27 no BCEWithLogitsLoss |

## Pipeline de execução

| Etapa | Descrição |
|:---|:---|
| 1. Carrega dados | CSV pré-processado de `data/pre-processed/` |
| 2. Split estratificado | 80/20 treino/teste com `stratify=y` |
| 3. Baselines | DummyClassifier, Logistic Regression, Random Forest |
| 4. Cross-validation MLP | 5-fold estratificado no conjunto de treino |
| 5. Treino final MLP | Treino no conjunto completo de treino |
| 6. Avaliação holdout | Métricas no conjunto de teste reservado |
| 7. Comparação | Tabela PR-AUC, F1, ROC-AUC, Recall, Precision |
| 8. MLflow tracking | Params, métricas, loss curve, modelo PyTorch |
| 9. Persistência | `models/mlp_churn.pt` + `models/mlp_scaler.joblib` |

## Métricas de avaliação (≥ 4)

| Métrica | Propósito |
|:---|:---|
| **PR-AUC** | Métrica primária — robusta em datasets desbalanceados |
| **ROC-AUC** | Capacidade geral de discriminação |
| **F1-Score** | Equilíbrio precisão/recall |
| **Recall** | Captura de churners — minimiza falsos negativos |
| **Precision** | Qualidade das predições positivas |
| **Accuracy** | Referência geral (menos informativa com desbalanceamento) |

## Análise de overfitting

O script calcula `train_metrics` e `test_metrics` lado a lado. O gap de PR-AUC entre treino e teste é logado no MLflow como `overfitting_gap_pr_auc`.

- Gap < 0.05 → modelo generaliza bem
- Gap 0.05–0.10 → overfitting leve, aceitável
- Gap > 0.10 → overfitting severo, reduzir capacidade ou aumentar regularização

## Trade-off de custo (FP vs FN)

| Tipo de erro | Custo de negócio | Consequência |
|:---|:---|:---|
| **Falso Negativo** (churner não detectado) | Alto (~R$100/cliente) | Perda de receita recorrente (MRR) |
| **Falso Positivo** (não-churner recebe oferta) | Baixo (~R$20/cliente) | Custo de oferta de retenção desnecessária |

O uso de `pos_weight` no loss function prioriza a detecção de churners (reduz FN), aceitando mais FPs — alinhado com a estratégia de negócio onde perder um cliente é 5x mais caro que uma oferta desnecessária.

## Artefatos gerados

| Arquivo | Descrição |
|:---|:---|
| `models/mlp_churn.pt` | Pesos do modelo PyTorch |
| `models/mlp_scaler.joblib` | StandardScaler ajustado no treino |
| `models/model_comparison.csv` | Tabela comparativa MLP vs baselines |
| `models/cost_analysis.csv` | Tabela de custo por threshold |
| MLflow (local) | Experimento `churn_mlp_pytorch` com todos os runs |

---

# Análise de Trade-off de Custo (Falso Positivo vs Falso Negativo)

## Premissa de negócio

Em previsão de churn, os erros do modelo têm custos assimétricos:

| Tipo de erro | Descrição | Custo estimado |
|:---|:---|:---|
| **Falso Negativo (FN)** | Churner não detectado — cliente cancela sem intervenção | R$ 100/cliente (perda de MRR) |
| **Falso Positivo (FP)** | Não-churner recebe oferta de retenção desnecessária | R$ 20/cliente (custo da oferta) |

**Ratio:** perder um cliente custa **5x mais** que uma oferta inútil. Portanto, o modelo deve priorizar Recall (minimizar FN) mesmo que isso gere mais FPs.

## Metodologia

O script `src/churn/train_mlp.py` implementa `compute_cost_analysis()` que:

1. Varia o threshold de decisão de 0.10 a 0.90 (steps de 0.05)
2. Para cada threshold, calcula FP, FN, TP, TN
3. Computa custo total = `FN × R$100 + FP × R$20`
4. Identifica o threshold que **minimiza o custo total**

## Exemplo de saída

```
 threshold   TP   FP   FN   TN  recall  precision  cost_FN  cost_FP  total_cost
      0.10  350  800   20  239  0.9459     0.3043   2000.0  16000.0     18000.0
      0.20  330  500   40  539  0.8919     0.3976   4000.0  10000.0     14000.0
      0.30  310  300   60  739  0.8378     0.5082   6000.0   6000.0     12000.0
      0.35  295  220   75  819  0.7973     0.5728   7500.0   4400.0     11900.0 ← ótimo
      0.50  260  120  110  919  0.7027     0.6842  11000.0   2400.0     13400.0
      0.80  150   20  220 1019  0.4054     0.8824  22000.0    400.0     22400.0
```

## Interpretação

- **Threshold padrão (0.5):** Custo total R$ 13.400 — equilibra FP e FN mas perde mais churners
- **Threshold ótimo (~0.35):** Custo total R$ 11.900 — reduz custo em ~11% priorizando detecção
- **Threshold agressivo (0.10):** Detecta quase todos os churners (recall=0.95) mas gera muitas ofertas desnecessárias — custo sobe para R$ 18.000

O threshold ótimo fica **abaixo de 0.5** porque o custo de não agir (FN) é muito maior que o custo de agir desnecessariamente (FP). Isso alinha-se com a estratégia definida no ML Canvas: o time de retenção prefere abordar clientes "de mais" do que perder churners silenciosamente.

## Integração com as faixas de risco do Model Card

| Threshold | Faixa de risco | Ação | Custo trade-off |
|:---|:---|:---|:---|
| ≥ 0.80 | Crítico | Intervenção humana (Ouvidoria) | Baixo FP, alto FN ignorado |
| 0.35–0.80 | Alto | Ofertas automatizadas (CRM) | **Ponto ótimo de custo** |
| < 0.35 | Monitorado | Réguas tradicionais | Alto FP, poucos FN |

## Métricas logadas no MLflow

| Métrica | Descrição |
|:---|:---|
| `optimal_threshold` | Threshold que minimiza custo total |
| `cost_at_optimal` | Custo total no threshold ótimo |
| `cost_at_default_05` | Custo total com threshold padrão 0.5 |
| `cost_savings` | Economia (R$) do ótimo vs padrão |
| `optimal_recall` | Recall no threshold ótimo |
| `optimal_precision` | Precision no threshold ótimo |

---

# Opção do modelo de inferência Real Time v Batch

Para este projeto, **real-time (API)** é a escolha mais adequada. Aqui está o racional:

## Por que real-time?

| Critério | Real-time (API) | Batch | Vencedor |
|:---|:---|:---|:---|
| **Caso de uso do CRM** | Ação imediata ao detectar risco (ligação, oferta) | Predições ficam "stale" até o próximo ciclo | Real-time |
| **Volume de dados** | ~7k clientes — carga trivial para uma API | Batch faz sentido com milhões de registros | Real-time |
| **Custo (Lambda serverless)** | Paga por invocação — econômico com volume baixo | Requer instância ligada ou job agendado | Real-time |
| **Integração** | CRM/app chama `/predict` direto, sem ETL intermediário | Precisa de pipeline de extração + armazenamento de scores | Real-time |
| **Latência de decisão** | Imediata — detecta risco no momento do evento | Horas ou dia de atraso entre evento e ação | Real-time |
| **Complexidade de infra** | Lambda + API Gateway (já implementado) | Step Functions + S3 + scheduler | Real-time |

## Quando batch faria mais sentido

- Dataset com milhões de clientes onde custo por invocação seria proibitivo
- Predições usadas apenas para relatórios diários/semanais
- Modelo com latência alta (ensemble pesado, pós-processamento demorado)
- Quando não existe integração de CRM em tempo real

## Arquitetura escolhida

```
Cliente/CRM  ──▶  API Gateway  ──▶  Lambda (Docker)  ──▶  Resposta JSON
                                         │
                                    src/main.py (FastAPI + Mangum)
                                         │
                                    models/best_random_forest.joblib
```

**Fallback:** Se o modelo evoluir para algo mais pesado (ensemble grande, MLP com muitas features), degradar para batch com Step Functions agendado via EventBridge — mas com 7k clientes e um modelo leve, isso é improvável.

A infraestrutura atual do projeto (Lambda + ECR + GitHub Actions CI/CD) já está 100% orientada para real-time, então é a escolha natural tanto técnica quanto pragmaticamente.

# Instruções

Esses são os passos para implementar o modelo

## Set up de ambiente AWS

Passo opcional se quiser rodar o projeto em nuvem pública.

### Requerimentos

 - Uma conta AWS com permissão de administrador
 - Provedor de identidade (identity provider, prerencialmente tipo OIDC) com GitHub e GitLab.
 - GitLab foi usado para subir infraestrutura na AWS via Terraform. 
 - Configuração do GitLab para manter state (estado) do workspace Terraform localmente nele. Opcionalmente pode-se configurar para subir o state remotamente em S3 com lock em DynamoDB ou no serviço Hashicorp Cloud (antigo Terraform Cloud). Não faz parte do escopo desse projeto detalhar como fazer.
 - Role que o GitLab assume com privilégios que permitam construir o ambiente. Lista de serviços relevantes. Tenha em mente que outros serviços podem ser necessários co diferentes permissões. Entenda impacto em serviços relevantes como Route 53.

     AWS Security Token Service
     Amazon Elastic Container Registry
     Manage - Amazon API Gateway
     AWS Lambda
     Amazon Route 53
     AWS Certificate Manager
     AWS Identity and Access Management
     Amazon CloudWatch Logs
     AWS Key Management Service
     Amazon EventBridge Schemas

### Execute um commit ou merge na branch main do repositório

Não estamos tratando a infraestrutura como diferentes ambientes. Como melhoria desse projeto, ele pode ser expandido para suportar diferentes ambientes de endpoints com nome de domínio diferentes como https://churn-dev.caprica.tech, https://churn-stage.caprica.tech e https://churn.caprica.tech

Como o foco é na construção do modelo e inferência, o escopo desse projeto trata apenas um ambiente de infraestrutura governado pela branch **main**. Tenha em mente que há melhores práticas para manter mais de um ambiente, proteção de branch para ambiente produtivo, revisão de pares via Pull Request (PR).

