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
