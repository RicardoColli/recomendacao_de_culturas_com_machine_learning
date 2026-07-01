# Explicação do backend

## O que fica no backend

O backend é responsável por toda a lógica de dados e modelo do projeto:

- carregar o dataset a partir de Crop_recommendation.csv
- preparar e treinar os modelos de classificação
- converter a variável alvo em números para o treinamento
- gerar previsões para novas entradas do usuário
- calcular valores ideais por cultura para explicar a recomendação
- validar os dados digitados na interface

## Fluxo principal

1. O arquivo de dados é lido e transformado em um DataFrame do pandas.
2. As colunas de entrada são separadas da coluna alvo (label).
3. Os modelos são treinados com os dados de treino e avaliados com os dados de teste.
4. O Random Forest é escolhido para fazer as previsões da aplicação.
5. A interface envia os valores inseridos pelo usuário para o backend.
6. O backend retorna um ranking das culturas mais prováveis.

## Funções principais

- carregar_dados(): lê o CSV.
- preparar_modelos(): treina os classificadores e calcula métricas.
- prever_cultura(): devolve as 3 culturas mais prováveis.
- calcular_valores_ideais(): cria uma referência por cultura para comparação.
- validar_valores(): garante que os campos estejam preenchidos corretamente.

## Por que essa separação é útil

Separar o backend da interface deixa o código mais organizado, facilita manutenção e permite reutilizar a lógica de machine learning em outras telas ou APIs no futuro.
