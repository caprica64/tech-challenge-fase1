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

## Repositório de infraestrutura na AWS

Repositório de infraestrutura na AWS foi montado no GitLab público em [Infra desafio 1](https://gitlab.com/mlet10/infra-desafio1/)

Este projeto Terraform com pipeline no GitLab constrói um API Gateway aberto na Internet com hostname churn dentro de um domínio já criado e hospedado na própria conta AWS (blog de outro projeto). O API Gateway possui rotas para um backend em Lambda com um runtime em container tipo Docker/OCI que carrega imagem template do ECR. 

Este projeto do runtime e inferência de ML também atualiza o ECR que está indicado em variáveis do GitHub Actions. O projeto em Terraform pode ser adaptado para criar ambiente similar.


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

Esses são os passos para implementar o modelo...
