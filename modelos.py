from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Métricas de avaliação
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# Divisão treino/teste
from sklearn.model_selection import train_test_split

# Pipeline e pré-processamento
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


# --------------------------------------------------------------------
# ETAPA 1 - CARREGAMENTO DOS DADOS

def carregar_dados():
    """
    Carrega o dataset Crop Recommendation.
    """

    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "Crop_recommendation.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {csv_path}"
        )

    return pd.read_csv(csv_path)


# --------------------------------------------------------------------
# ETAPA 2 - PREPARAÇÃO DOS DADOS E TREINAMENTO

def preparar_modelos(df):
    """
    Realiza:
    - Separação dos atributos e da variável alvo
    - Codificação das classes
    - Divisão treino/teste
    - Treinamento dos modelos
    - Avaliação dos modelos
    - Cálculo da importância das variáveis
    """

    # -----------------------------------
    # X = variáveis de entrada
    # y = variável alvo (cultura)
    # -----------------------------------
    X = df.drop("label", axis=1)
    y = df["label"]

    # -----------------------------------
    # Transforma culturas em números
    # Exemplo:
    # arroz -> 0
    # milho -> 1
    # banana -> 2
    # -----------------------------------
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # -----------------------------------
    # DIVISÃO TREINO E TESTE
    #
    # 80% dos dados -> treinamento
    # 20% dos dados -> teste
    # -----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    # ======================================================
    # MODELOS UTILIZADOS
    # ======================================================

    modelos = {

        # Árvore de decisão
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        # Random Forest
        # Conjunto de 200 árvores de decisão
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        # KNN
        # Analisa os 5 vizinhos mais próximos
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        # Regressão Logística
        # Recebe padronização dos dados antes
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=10000,
                solver="lbfgs"
            ),
        ),
    }

    # ------------------------------------------------------
    # TREINAMENTO E TESTE DOS MODELOS


    resultados = {}

    for nome, modelo in modelos.items():

        # -------------------------------
        # TREINAMENTO
        #
        # O modelo aprende padrões
        # usando os dados de treino
        # -------------------------------
        modelo.fit(X_train, y_train)

        # -------------------------------
        # TESTE
        #
        # Faz previsões em dados
        # que o modelo nunca viu
        # -------------------------------
        y_pred = modelo.predict(X_test)

        # -------------------------------
        # MÉTRICAS DE AVALIAÇÃO

        resultados[nome] = {

            "model": modelo,

            # Taxa geral de acertos
            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            # Precisão das classificações
            "precision": precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            # Capacidade de encontrar
            # corretamente cada cultura
            "recall": recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            # Média harmônica entre
            # Precision e Recall
            "f1": f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),
        }

    # ======================================================
    # IMPORTÂNCIA DAS VARIÁVEIS
    # ======================================================

    # Seleciona o Random Forest
    rf_model = modelos["Random Forest"]

    # Calcula a importância de cada atributo
    importances = rf_model.feature_importances_

    # Converte para porcentagem
    importance_pct = (
        100 * importances / importances.sum()
    )

    # Tabela ordenada por importância
    importancia_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": importance_pct
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    # Dicionário auxiliar
    importancias_por_variavel = dict(
        zip(
            X.columns,
            importances / importances.sum()
        )
    )

    # Calcula médias das variáveis
    # para cada cultura
    valores_ideais = calcular_valores_ideais(
        df,
        encoder
    )

    # Retorna os objetos necessários
    return (
        encoder,
        modelos["Random Forest"],
        resultados,
        importancia_df,
        X,
        valores_ideais,
        importancias_por_variavel
    )


# -----------------------------------------------------------
# SISTEMA DE RECOMENDAÇÃO

def prever_cultura(modelo, encoder, valores):
    """
    Recebe os dados informados pelo usuário
    e retorna as 3 culturas mais prováveis.
    """

    # Probabilidade de cada cultura
    probabilidades = modelo.predict_proba(
        [valores]
    )[0]

    culturas = encoder.classes_

    ranking = list(
        zip(culturas, probabilidades)
    )

    # Ordena da maior para a menor
    ranking.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return ranking[:3]


# ==========================================================
# EXPLICAÇÃO DAS RECOMENDAÇÕES

def calcular_valores_ideais(df, encoder):
    """
    Calcula os valores médios de N, P, K,
    temperatura, umidade, pH e chuva
    para cada cultura do dataset.
    """

    X = df.drop("label", axis=1)
    y = df["label"]

    valores_ideais = {}

    for cultura in encoder.classes_:

        mask = y == cultura

        valores_ideais[cultura] = (
            X[mask]
            .mean()
            .to_dict()
        )

    return valores_ideais


# ==========================================================
# VALIDAÇÃO DOS DADOS DIGITADOS

def validar_valores(entrada):
    """
    Verifica se todos os campos foram
    preenchidos corretamente.
    """

    valores = {}

    for chave, texto in entrada.items():

        if texto.strip() == "":
            raise ValueError(
                f"O campo '{chave}' não pode ficar vazio."
            )

        try:
            valores[chave] = float(texto)

        except ValueError as exc:
            raise ValueError(
                f"Valor inválido para '{chave}'."
            ) from exc

    return valores


# ==========================================================
# ANÁLISE EXPLORATÓRIA

def criar_stats_text(df):
    """
    Gera estatísticas descritivas
    para exibição na interface.
    """

    # Apenas colunas numéricas
    numeric_df = df.select_dtypes(
        include="number"
    )

    # Média, desvio padrão,
    # mínimo e máximo
    stats = numeric_df.describe().T

    # Quantidade por cultura
    counts = df["label"].value_counts()

    # Percentual por cultura
    percentages = (
        df["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
    )

    # MATRIZ DE CORRELAÇÃO
    corr = numeric_df.corr()

    texto = ["=== Estatísticas Descritivas ===\n"]

    for col in stats.index:

        linha = stats.loc[col]

        texto.append(
            f"{col}: média={linha['mean']:.2f}, "
            f"desvio padrão={linha['std']:.2f}, "
            f"mínimo={linha['min']:.2f}, "
            f"máximo={linha['max']:.2f}"
        )

    texto.append(
        "\n=== Distribuição de Culturas ==="
    )

    for cultura, contagem in counts.items():
        texto.append(
            f"{cultura}: {contagem} exemplos"
        )

    texto.append(
        "\n=== Correlação entre Variáveis ==="
    )

    texto.append(
        corr.round(2).to_string()
    )

    return "\n".join(texto)