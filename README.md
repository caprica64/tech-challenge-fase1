# fiap-mlet10-fase1
Tech Challenge Fase 1 do MLET10 da FIAP


# Objetivo do projeto

# Estrutura do projeto

# Instruções


# To-Do

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


