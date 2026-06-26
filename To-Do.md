# To-Do

Passo a Passo Resumido

• [Etapa 1] EDA + ML Canvas + Baselines → MLflow tracking.

• [Etapa 2] MLP PyTorch + comparação de modelos + análise de custo.

• [Etapa 3] Refatoração + API FastAPI + testes + Makefile.

• [Etapa 4] Model Card + README + vídeo STAR + (opcional) deploy em nuvem.


## Repositório GitHub criado

- [X] Estrutura organizada: src/, data/, models/, tests/, notebooks/, docs/
- [X] README.md completo descrição do projeto
- [X] pyproject.toml como single source of truth (dependências, linting, pytest)
- [X] Histórico de commits limpo e significativo (não 1 commit gigante)    
- [X] .gitignore adequado para projetos de ML

## Video & Deploy

- [ ] Video de 5 minutos
- [X] Deploy em ambiente de produção

## Bibliotecas Requeridas

- [X] PyTorch — construção e treinamento da rede neural (MLP).
- [X] Scikit-Learn — pipelines de pré-processamento e modelos baseline.
- [X] MLflow — tracking de experimentos (parâmetros, métricas, artefatos).
- [X] FastAPI — API de inferência do modelo.

## Boas Práticas Obrigatórias

- [X] Seeds fixados para reprodutibilidade.
- [ ] Validação cruzada estratificada. <Albino>
- [O] Model Card documentando limitações e vieses. <Rodrigo>
- [ ] Testes automatizados (≥ 3: smoke test, schema, API). <Allan>
- [X] Logging estruturado (sem print()). <Albino>
- [O] Linting com ruff sem erros. <Rodrigo>

## Etapa 1 — Entendimento e Preparação (Disciplinas 01 e 02)

**Foco**: formulação do problema, exploração de dados e construção de baselines.

- [O] Preencher ML Canvas (stakeholders, métricas de negócio, SLOs) [Ciclo de Vida, Aula 01] <Rodrigo>
- [X] EDA completa: volume, qualidade, distribuição, data readiness [Ciclo de Vida, Aula 01]
- [ ] Definir métrica técnica (AUC-ROC, PR-AUC, F1) e métrica de negócio (custo de churn evitado) [Fundamentos, Aula 05]
  - [ ] PR-AUC <<EXPLICAR RACIONAL>> Target 1 <Albino>
- [X] Treinar baseline com DummyClassifier e Regressão Logística (Scikit-Learn) [Fundamentos, Aulas 01–02]
- [X] Registrar experimentos no MLflow (parâmetros, métricas, dataset version) [Ciclo de Vida, Aula 02] <Albino>

**Entregável**: notebook de EDA + baselines registrados no MLflow.


## Etapa 2 — Modelagem com Redes Neurais (Disciplina 02)

**Foco**: Construção, treinamento e avaliação de MLP com PyTorch.

- [X] Construir MLP em PyTorch: definir arquitetura, função de ativação, loss function. [Fundamentos, Aula 04]
- [X] Implementar loop de treinamento com early stopping e batching. [Fundamentos, Aula 04]
- [X] Comparar MLP vs. baselines (lineares + árvores) usando ≥ 4 métricas. [Fundamentos, Aula 05]
- [X] Analisar trade-off de custo (falso positivo vs. negativo). [Fundamentos, Aula 05]
- [X] Registrar todos os experimentos (MLP e ensembles) no MLflow. [Ciclo de Vida, Aula 02]

**Entregável**: tabela comparativa de modelos + MLP treinado + artefatos no MLflow.

## Etapa 3 — Engenharia e API (Disciplinas 03, 04 e 05)

**Foco**: refatoração profissional, API de inferência e pacote reutilizável.

- [X] Refatorar código em módulos (src/) com estrutura limpa. [Eng. Software, Aula 01]
- [X] Criar pipeline reprodutível (sklearn + transformadores custom). [Eng. Software, Aula 01; Bibliotecas, Aula 02]
- [X] Escrever testes (pytest): unitários, schema (pandera), smoke test. [Eng. Software, Aula 03]
- [X] Construir API FastAPI: /predict, /health, validação Pydantic. [APIs, Aulas 01–03]
- [X] Adicionar logging estruturado e middleware de latência. [APIs, Aula 04]
- [X] Configurar pyproject.toml, ruff, Makefile (lint, test, run). [Eng. Software, Aulas 04–05]

**Entregável**: repositório refatorado + API funcional + testes passando.

## Etapa 4 — Documentação e Entrega Final (Todas as disciplinas)

**Foco**: consolidação, documentação e vídeo de apresentação.

- [X] Gerar Model Card completo (performance, limitações, vieses, cenários de falha). [Ciclo de Vida, Aula 03]
- [X] Documentar arquitetura de deploy escolhida (batch vs. real-time) + justificativa. [Ciclo de Vida, Aula 04]
- [ ] Criar plano de monitoramento (métricas, alertas, playbook de resposta). [Ciclo de Vida, Aula 05]
- [X] Finalizar README com instruções de setup + execução + arquitetura. [Eng. Software / APIs] <Rodrigo>

**Entregável**: repositório final + vídeo STAR + (opcional) URL do deploy em nuvem.

