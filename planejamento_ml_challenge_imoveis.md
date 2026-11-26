# ML CHALLENGE --- PREVISÃO DE PREÇOS DE IMÓVEIS

**Estudante:** Murilo Coelho\
**Área:** Análise e Desenvolvimento de Sistemas\
**Disciplina:** Machine Learning\
**Tipo de projeto:** Regressão\
**Linguagem:** Python\
**Interface:** Streamlit\
**Entrega:** 25/11/2026

------------------------------------------------------------------------

## 1. Apresentação do projeto

O projeto consiste no desenvolvimento de uma aplicação de Machine
Learning capaz de estimar o preço de venda de imóveis a partir de suas
características.

A solução utilizará uma base de dados real para treinar algoritmos de
regressão, identificar padrões e aprender relações entre as
características dos imóveis e seus respectivos valores de venda.

Ao final, será desenvolvida uma aplicação interativa utilizando
Streamlit, permitindo que o usuário informe as características de um
imóvel e receba uma estimativa de preço gerada pelo modelo treinado.

O projeto terá finalidade educacional e demonstrativa, aplicando
conceitos de Ciência de Dados e Inteligência Artificial.

## 2. Objetivo geral

Desenvolver uma solução completa de Machine Learning para previsão de
preços de imóveis, contemplando coleta e análise de dados,
pré-processamento, treinamento e comparação de algoritmos, avaliação de
resultados e disponibilização de uma interface interativa.

### 2.1 Objetivos específicos

-   Obter e documentar uma base de dados imobiliários real.
-   Realizar análise exploratória dos dados.
-   Identificar padrões, distribuições, valores ausentes e possíveis
    inconsistências.
-   Preparar os dados para utilização em algoritmos de Machine Learning.
-   Implementar pelo menos dois modelos de regressão.
-   Comparar os modelos utilizando métricas apropriadas.
-   Selecionar e salvar o modelo com melhor desempenho.
-   Desenvolver uma interface para realizar previsões.
-   Documentar o funcionamento e publicar o projeto no GitHub.

## 3. Tecnologias e ferramentas

  Tecnologia     Finalidade
  -------------- --------------------------------------
  Python 3       Desenvolvimento
  Pandas         Manipulação de dados
  NumPy          Operações numéricas
  Scikit-learn   Pré-processamento e Machine Learning
  Matplotlib     Gráficos
  Seaborn        Análise visual
  Joblib         Salvamento do modelo
  Streamlit      Interface interativa
  Git            Versionamento
  GitHub         Hospedagem do código

O desenvolvimento será realizado no Visual Studio Code, utilizando
ambiente virtual Python (`venv`).

------------------------------------------------------------------------

# 4. Fases de desenvolvimento

## FASE 01 --- Planejamento e configuração do ambiente

**Objetivo:** Preparar o ambiente de desenvolvimento e organizar a
estrutura inicial do projeto.

Atividades:

1.  Verificar a instalação do Python.
2.  Instalar as extensões Python e Jupyter no VS Code, caso necessário.
3.  Criar a pasta principal do projeto.
4.  Criar e ativar o ambiente virtual `.venv`.
5.  Atualizar o gerenciador pip.
6.  Instalar as bibliotecas necessárias.
7.  Criar a estrutura inicial de arquivos e diretórios.

**Entregáveis:** - Ambiente Python configurado. - Ambiente virtual
funcionando. - Dependências instaladas. - Estrutura inicial do projeto.

## FASE 02 --- Seleção e obtenção do dataset

**Objetivo:** Encontrar uma base de dados imobiliários adequada para
treinar os modelos.

Atividades:

1.  Pesquisar bases de dados públicas.
2.  Avaliar a qualidade e a quantidade de registros.
3.  Verificar a licença e as condições de utilização.
4.  Identificar a variável-alvo (preço de venda).
5.  Identificar as variáveis explicativas, como área, quartos, banheiros
    e idade do imóvel.
6.  Documentar a origem, autoria, URL e descrição da base.
7.  Obter e armazenar o dataset no projeto.

**Dataset candidato:** Ames Housing, sujeito à validação da fonte,
licença e compatibilidade com os requisitos.

**Entregáveis:** - Dataset selecionado e validado. - Documentação da
fonte dos dados. - Identificação das colunas e variável-alvo.

## FASE 03 --- Análise exploratória dos dados (EDA)

**Objetivo:** Compreender os dados antes de iniciar o treinamento dos
modelos.

Atividades:

1.  Carregar o dataset utilizando Pandas.
2.  Verificar quantidade de linhas e colunas.
3.  Identificar os tipos de dados.
4.  Analisar estatísticas descritivas.
5.  Identificar valores ausentes e duplicados.
6.  Investigar possíveis valores extremos (outliers).
7.  Analisar a distribuição dos preços.
8.  Investigar relações entre as características dos imóveis e seus
    preços.
9.  Construir pelo menos dois gráficos relevantes.
10. Registrar pelo menos três conclusões objetivas.

**Entregáveis:** - Análise exploratória documentada. - Dois ou mais
gráficos. - Três ou mais descobertas relevantes.

## FASE 04 --- Pré-processamento dos dados

**Objetivo:** Preparar os dados para o treinamento dos algoritmos.

Atividades:

1.  Separar as variáveis explicativas (X) da variável-alvo (y).
2.  Remover duplicatas quando justificável.
3.  Tratar valores ausentes.
4.  Identificar e tratar inconsistências.
5.  Separar os dados em conjuntos de treinamento e teste.
6.  Codificar variáveis categóricas.
7.  Preparar variáveis numéricas.
8.  Implementar Pipeline e ColumnTransformer quando necessário.
9.  Garantir que as transformações sejam ajustadas somente com os dados
    de treinamento.

**Estratégia inicial:** Separação de 80% para treinamento e 20% para
teste, com `random_state=42`.

**Entregáveis:** - Dados preparados. - Processo de pré-processamento
reproduzível. - Prevenção de vazamento de informações (Data Leakage).

## FASE 05 --- Treinamento dos modelos de Machine Learning

**Objetivo:** Implementar e comparar pelo menos dois algoritmos de
regressão.

Modelos inicialmente propostos:

**Modelo 1 --- Regressão Linear Regularizada (Ridge)**

Será utilizado como modelo de referência, permitindo avaliar relações
entre as características dos imóveis e seus preços.

**Modelo 2 --- Random Forest Regressor**

Será utilizado para identificar relações mais complexas e não lineares
entre as variáveis.

Atividades:

1.  Implementar os algoritmos.
2.  Integrar cada algoritmo ao processo de pré-processamento.
3.  Treinar os modelos com os dados de treinamento.
4.  Realizar previsões utilizando os dados de teste.
5.  Comparar os resultados obtidos.
6.  Justificar tecnicamente a escolha do modelo final.

**Entregáveis:** - Dois modelos treinados. - Resultados de avaliação. -
Comparação dos algoritmos. - Modelo selecionado e salvo em arquivo.

## FASE 06 --- Avaliação dos resultados

**Objetivo:** Avaliar a capacidade dos modelos de estimar corretamente
os preços dos imóveis.

Métricas obrigatórias:

-   **MAE:** Erro absoluto médio, indicando quanto o modelo erra em
    média.
-   **RMSE:** Raiz do erro quadrático médio, penalizando mais os erros
    elevados.
-   **R²:** Coeficiente de determinação, indicando quanto da variação
    dos preços é explicada pelo modelo.

Atividades:

1.  Calcular as três métricas para cada algoritmo.
2.  Construir um quadro comparativo.
3.  Interpretar os resultados.
4.  Identificar vantagens e limitações dos modelos.
5.  Selecionar o modelo final considerando o desempenho e a coerência
    técnica.

**Entregáveis:** - Métricas reais. - Comparação documentada. - Arquivo
`metricas.json`. - Justificativa da escolha final.

## FASE 07 --- Desenvolvimento da aplicação com Streamlit

**Objetivo:** Criar uma interface para que qualquer pessoa consiga
utilizar o modelo treinado.

A aplicação deverá permitir que o usuário informe características de um
imóvel e receba uma estimativa de preço.

Campos previstos, conforme a disponibilidade no dataset:

-   Área construída.
-   Quantidade de quartos.
-   Quantidade de banheiros.
-   Ano de construção.
-   Qualidade geral do imóvel.
-   Características da garagem.
-   Outras variáveis relevantes.

Atividades:

1.  Criar a interface com Streamlit.
2.  Desenvolver os campos de entrada.
3.  Implementar validações básicas.
4.  Carregar o modelo salvo utilizando Joblib.
5.  Implementar o botão de previsão.
6.  Exibir o preço estimado na moeda correspondente à base,
    identificando eventuais conversões.
7.  Informar qual algoritmo está sendo utilizado.
8.  Apresentar pelo menos uma métrica de avaliação.
9.  Adicionar aviso de finalidade educacional.
10. Tratar situações como ausência do arquivo do modelo e entradas
    inválidas.

**Entregáveis:** - Aplicação funcional. - Previsões geradas pelo modelo
treinado. - Interface testada com pelo menos três conjuntos diferentes
de entrada.

## FASE 08 --- Testes e validação

**Objetivo:** Garantir que todos os componentes funcionem corretamente.

Atividades:

1.  Executar o treinamento pelo terminal.
2.  Verificar se os modelos são treinados sem erros.
3.  Conferir as métricas produzidas.
4.  Verificar se o modelo é salvo corretamente.
5.  Executar a aplicação Streamlit.
6.  Realizar previsões com diferentes características de imóveis.
7.  Verificar se as alterações nos campos produzem previsões
    correspondentes.
8.  Conferir os caminhos dos arquivos e as dependências.
9.  Corrigir eventuais problemas.

**Entregáveis:** - Projeto testado. - Aplicação funcional. - Evidências
de execução. - Captura de tela da interface.

## FASE 09 --- Documentação e organização do GitHub

**Objetivo:** Organizar o projeto para que qualquer pessoa consiga
compreender, instalar e executar a solução.

Arquivos obrigatórios:

-   `README.md`
-   `requirements.txt`
-   `.gitignore`
-   `train.py`
-   `app.py`

Estrutura de diretórios:

-   `data/`: documentação e instruções para obtenção do dataset.
-   `models/`: modelo treinado.
-   `reports/`: métricas e gráficos.
-   `screenshots/`: capturas de tela da aplicação.

O README deverá apresentar:

1.  Nome e descrição do projeto.
2.  Objetivo e problema abordado.
3.  Fonte e licença do dataset.
4.  Tecnologias utilizadas.
5.  Processo de pré-processamento.
6.  Algoritmos implementados.
7.  Comparação das métricas.
8.  Justificativa do modelo escolhido.
9.  Instruções de instalação e execução.
10. Exemplos de previsões.
11. Limitações e melhorias futuras.
12. Declaração de utilização de ferramentas de IA, se aplicável.

**Entregáveis:** - Repositório público no GitHub. - Código-fonte
organizado. - Documentação completa. - Dependências listadas. -
Evidências da aplicação.

## FASE 10 --- Apresentação e entrega

**Objetivo:** Preparar a demonstração do funcionamento do projeto e
realizar a submissão.

Atividades:

1.  Preparar uma explicação de até um minuto sobre o problema.
2.  Apresentar a origem dos dados.
3.  Demonstrar os gráficos e os principais achados.
4.  Explicar os dois algoritmos utilizados.
5.  Apresentar as métricas reais.
6.  Executar pelo menos três previsões diferentes.
7.  Explicar pelo menos duas limitações e uma melhoria futura.
8.  Conferir se o repositório está público.
9.  Enviar o link do GitHub ao professor.

**Entregável final:** Link público do repositório GitHub.

------------------------------------------------------------------------

# 5. Estrutura final do projeto

``` text
ml-challenge-imoveis/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── README.md
│
├── models/
│   └── model.joblib
│
├── reports/
│   ├── metricas.json
│   └── graficos/
│
└── screenshots/
    └── interface.png
```

Os arquivos e diretórios poderão ser ajustados conforme as necessidades
identificadas durante o desenvolvimento.

# 6. Critérios de conclusão

-   [ ] O dataset real estará documentado.
-   [ ] A análise exploratória apresentará pelo menos três conclusões e
    dois gráficos.
-   [ ] O pré-processamento estará implementado corretamente.
-   [ ] Dois algoritmos estarão treinados e avaliados.
-   [ ] As métricas estarão documentadas.
-   [ ] O modelo escolhido estará salvo.
-   [ ] A aplicação Streamlit estará funcionando.
-   [ ] Três previsões diferentes terão sido testadas.
-   [ ] O README e o requirements.txt estarão completos.
-   [ ] O repositório estará público e acessível.
-   [ ] A demonstração estará preparada.

------------------------------------------------------------------------

**Observação:** Este documento é o planejamento técnico inicial do ML
Challenge. As decisões específicas sobre dataset, atributos, algoritmos
definitivos e versões das bibliotecas serão confirmadas durante as
respectivas fases, conforme os dados reais e os testes realizados.

O desenvolvimento seguirá uma abordagem incremental: cada fase será
concluída e validada antes do início da próxima.
