# Recomendação de Culturas com Machine Learning

Projeto desenvolvido para a disciplina de Mineração de Dados.

## Objetivo

Desenvolver modelos de Machine Learning capazes de recomendar a cultura agrícola mais adequada com base em características do solo e condições climáticas.

## Dataset

Crop Recommendation Dataset

Atributos utilizados:

- Nitrogênio (N)
- Fósforo (P)
- Potássio (K)
- Temperatura
- Umidade
- pH
- Chuva

Variável alvo:

- Cultura recomendada

## Algoritmos avaliados

- Decision Tree
- Random Forest
- KNN
- Logistic Regression

## Melhor modelo

Random Forest

Accuracy obtida:

99,55%

## Estrutura do Projeto

comparacao_modelos.py
importancia_random_forest.py
recomendador_cultura.py
Crop_recommendation.csv
requirements.txt

## Como executar

Instalar dependências:

pip install -r requirements.txt

Executar comparação de modelos:

python comparacao_modelos.py

Executar recomendador:

python recomendador_cultura.py
