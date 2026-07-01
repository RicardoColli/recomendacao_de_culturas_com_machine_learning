from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def carregar_dados():
    """Carrega o dataset de culturas a partir do arquivo CSV localizado na pasta do projeto."""
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "Crop_recommendation.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {csv_path}")

    return pd.read_csv(csv_path)


def preparar_modelos(df):
    """Treina os modelos de classificação e prepara os dados de análise e explicação."""
    X = df.drop("label", axis=1)
    y = df["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    modelos = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=10000, solver="lbfgs"),
        ),
    }

    resultados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        resultados[nome] = {
            "model": modelo,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

    rf_model = modelos["Random Forest"]
    importances = rf_model.feature_importances_
    importance_pct = 100 * importances / importances.sum()
    importancia_df = pd.DataFrame(
        {"feature": X.columns, "importance": importance_pct}
    ).sort_values("importance", ascending=False)

    importancias_por_variavel = dict(zip(X.columns, importances / importances.sum()))
    valores_ideais = calcular_valores_ideais(df, encoder)

    return encoder, modelos["Random Forest"], resultados, importancia_df, X, valores_ideais, importancias_por_variavel


def prever_cultura(modelo, encoder, valores):
    """Retorna o ranking das 3 culturas mais prováveis junto com as probabilidades."""
    probabilidades = modelo.predict_proba([valores])[0]
    culturas = encoder.classes_

    ranking = list(zip(culturas, probabilidades))
    ranking.sort(key=lambda item: item[1], reverse=True)
    return ranking[:3]


def calcular_valores_ideais(df, encoder):
    """Calcula os valores médios de cada variável para cada cultura, usados na explicação do sistema."""
    X = df.drop("label", axis=1)
    y = df["label"]

    valores_ideais = {}
    for cultura in encoder.classes_:
        mask = y == cultura
        valores_ideais[cultura] = X[mask].mean().to_dict()

    return valores_ideais


def validar_valores(entrada):
    """Valida e converte os valores informados pelo usuário em números decimais."""
    valores = {}
    for chave, texto in entrada.items():
        if texto.strip() == "":
            raise ValueError(f"O campo '{chave}' não pode ficar vazio.")
        try:
            valores[chave] = float(texto)
        except ValueError as exc:
            raise ValueError(f"Valor inválido para '{chave}': use apenas números.") from exc
    return valores


def criar_stats_text(df):
    """Gera um texto resumido com estatísticas descritivas e distribuição das culturas."""
    numeric_df = df.select_dtypes(include="number")
    stats = numeric_df.describe().T
    counts = df["label"].value_counts()
    percentages = df["label"].value_counts(normalize=True).mul(100).round(1)
    corr = numeric_df.corr()

    texto = ["=== Estatísticas Descritivas ===\n"]
    for col in stats.index:
        linha = stats.loc[col]
        texto.append(
            f"{col}: média={linha['mean']:.2f}, std={linha['std']:.2f}, min={linha['min']:.2f}, "
            f"25%={linha['25%']:.2f}, 50%={linha['50%']:.2f}, 75%={linha['75%']:.2f}, max={linha['max']:.2f}"
        )

    texto.append("\n=== Distribuição de Culturas ===")
    for cultura, contagem in counts.items():
        texto.append(f"{cultura}: {contagem} exemplos")

    texto.append("\n=== Percentual de Cada Cultura ===")
    for cultura, pct in percentages.items():
        texto.append(f"{cultura}: {pct:.1f}%")

    texto.append("\n=== Correlação entre Variáveis ===")
    texto.append(corr.round(2).to_string())
    return "\n".join(texto)
