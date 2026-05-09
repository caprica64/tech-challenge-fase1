# Tech Challenge turma MLET10 - Fase 1 - FIAP

Tech Challenge Fase 1 do MLET10 da FIAP


# Objetivo do desafio

O objetivo desse desafio é implementar um modelo de previsão de churn a partir dos dados XYZ da IBM com relação ao XPTO...

# Estrutura do projeto

O projeto do desafio é formato pelas seguintes pastas e arquivos:

- /src
- /models
(...)

# Instruções

Esses são os passos para implementar o modelo...

# To-Do

Passo a Passo Resumido

• [Etapa 1] EDA + ML Canvas + Baselines → MLflow tracking.

• [Etapa 2] MLP PyTorch + comparação de modelos + análise de custo.

• [Etapa 3] Refatoração + API FastAPI + testes + Makefile.

• [Etapa 4] Model Card + README + vídeo STAR + (opcional) deploy em nuvem.


## Repositório GitHub criado

- [ ] Estrutura organizada: src/, data/, models/, tests/, notebooks/, docs/
- [X] README.md completo descrição do projeto
- [ ] pyproject.toml como single source of truth (dependências, linting, pytest)
- [ ] Histórico de commits limpo e significativo (não 1 commit gigante)    
- [ ] .gitignore adequado para projetos de ML

## Video & Deploy

- [ ] Video de 5 minutos
- [ ] Deploy em ambiente de produção

## Bibliotecas Requeridas

- [ ] PyTorch — construção e treinamento da rede neural (MLP).
- [ ] Scikit-Learn — pipelines de pré-processamento e modelos baseline.
- [ ] MLflow — tracking de experimentos (parâmetros, métricas, artefatos).
- [ ] FastAPI — API de inferência do modelo.

## Boas Práticas Obrigatórias

- [ ] Seeds fixados para reprodutibilidade.
- [ ] Validação cruzada estratificada.
- [ ] Model Card documentando limitações e vieses.
- [ ] Testes automatizados (≥ 3: smoke test, schema, API).
- [ ] Logging estruturado (sem print()).
- [ ] Linting com ruff sem erros.

## Etapa 1 — Entendimento e Preparação (Disciplinas 01 e 02)

**Foco**: formulação do problema, exploração de dados e construção de baselines.

- [ ] Preencher ML Canvas (stakeholders, métricas de negócio, SLOs) [Ciclo de Vida, Aula 01]
- [ ] EDA completa: volume, qualidade, distribuição, data readiness [Ciclo de Vida, Aula 01]
- [ ] Definir métrica técnica (AUC-ROC, PR-AUC, F1) e métrica de negócio (custo de churn evitado) [Fundamentos, Aula 05]
- [ ] Treinar baseline com DummyClassifier e Regressão Logística (Scikit-Learn) [Fundamentos, Aulas 01–02]
- [ ] Registrar experimentos no MLflow (parâmetros, métricas, dataset version) [Ciclo de Vida, Aula 02]

**Entregável**: notebook de EDA + baselines registrados no MLflow.


## Etapa 2 — Modelagem com Redes Neurais (Disciplina 02)

**Foco**: Construção, treinamento e avaliação de MLP com PyTorch.

- [ ] Construir MLP em PyTorch: definir arquitetura, função de ativação, loss function. [Fundamentos, Aula 04]
- [ ] Implementar loop de treinamento com early stopping e batching. [Fundamentos, Aula 04]
- [ ] Comparar MLP vs. baselines (lineares + árvores) usando ≥ 4 métricas. [Fundamentos, Aula 05]
- [ ] Analisar trade-off de custo (falso positivo vs. negativo). [Fundamentos, Aula 05]
- [ ] Registrar todos os experimentos (MLP e ensembles) no MLflow. [Ciclo de Vida, Aula 02]

**Entregável**: tabela comparativa de modelos + MLP treinado + artefatos no MLflow.

## Etapa 3 — Engenharia e API (Disciplinas 03, 04 e 05)

**Foco**: refatoração profissional, API de inferência e pacote reutilizável.

- [ ] Refatorar código em módulos (src/) com estrutura limpa. [Eng. Software, Aula 01]
- [ ] Criar pipeline reprodutível (sklearn + transformadores custom). [Eng. Software, Aula 01; Bibliotecas, Aula 02]
- [ ] Escrever testes (pytest): unitários, schema (pandera), smoke test. [Eng. Software, Aula 03]
C- [ ] onstruir API FastAPI: /predict, /health, validação Pydantic. [APIs, Aulas 01–03]
- [ ] Adicionar logging estruturado e middleware de latência. [APIs, Aula 04]
C- [ ] onfigurar pyproject.toml, ruff, Makefile (lint, test, run). [Eng. Software, Aulas 04–05]

**Entregável**: repositório refatorado + API funcional + testes passando.

## Etapa 4 — Documentação e Entrega Final (Todas as disciplinas)

**Foco**: consolidação, documentação e vídeo de apresentação.

- [ ] Gerar Model Card completo (performance, limitações, vieses, cenários de falha). [Ciclo de Vida, Aula 03]
- [ ] Documentar arquitetura de deploy escolhida (batch vs. real-time) + justificativa. [Ciclo de Vida, Aula 04]
- [ ] Criar plano de monitoramento (métricas, alertas, playbook de resposta). [Ciclo de Vida, Aula 05]
- [ ] Finalizar README com instruções de setup + execução + arquitetura. [Eng. Software / APIs]

**Entregável**: repositório final + vídeo STAR + (opcional) URL do deploy em nuvem.

