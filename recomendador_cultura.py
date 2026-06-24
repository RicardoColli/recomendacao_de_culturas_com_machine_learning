from pathlib import Path
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def carregar_dados():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "Crop_recommendation.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def preparar_modelos(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    modelos = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=10000, solver="lbfgs")
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
    importancia_df = pd.DataFrame({"feature": X.columns, "importance": importance_pct}).sort_values("importance", ascending=False)
    
    # Criar dicionário de importâncias por variável (escala 0-1)
    importancias_por_variavel = dict(zip(X.columns, importances / importances.sum()))
    
    # Calcular valores ideais
    valores_ideais = calcular_valores_ideais(df, encoder)

    return encoder, modelos["Random Forest"], resultados, importancia_df, X, valores_ideais, importancias_por_variavel


def prever_cultura(modelo, encoder, valores):
    """Retorna ranking das 3 culturas mais prováveis com suas probabilidades."""
    probabilidades = modelo.predict_proba([valores])[0]
    culturas = encoder.classes_
    
    # Criar ranking com culturas e probabilidades
    ranking = list(zip(culturas, probabilidades))
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    # Retornar top 3
    return ranking[:3]


def calcular_valores_ideais(df, encoder):
    """Calcula os valores ideais (média) para cada cultura baseado nos dados."""
    X = df.drop("label", axis=1)
    y = df["label"]
    
    valores_ideais = {}
    for cultura in encoder.classes_:
        mask = y == cultura
        valores_ideais[cultura] = X[mask].mean().to_dict()
    
    return valores_ideais


def validar_valores(entrada):
    valores = {}
    for chave, texto in entrada.items():
        if texto.strip() == "":
            raise ValueError(f"O campo '{chave}' não pode ficar vazio.")
        try:
            valores[chave] = float(texto)
        except ValueError:
            raise ValueError(f"Valor inválido para '{chave}': use apenas números.")
    return valores


def criar_stats_text(df):
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


def criar_grafico_distribuicao(fig, df):
    ax1 = fig.add_subplot(121)
    counts = df["label"].value_counts()
    counts.plot.bar(ax=ax1, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab"])
    ax1.set_title("Contagem por Cultura")
    ax1.set_ylabel("Quantidade")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = fig.add_subplot(122)
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    heat = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_title("Matriz de Correlação")
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax2.set_yticklabels(corr.columns)
    fig.colorbar(heat, ax=ax2, fraction=0.046, pad=0.04)


def criar_grafico_comparacao(fig, resultados):
    ax = fig.add_subplot(111)
    nomes = list(resultados.keys())
    accuracies = [resultados[n]["accuracy"] for n in nomes]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    ax.bar(nomes, accuracies, color=colors)
    ax.set_title("Comparação de Accuracy entre Modelos")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")


def criar_grafico_importancia(fig, importancia_df):
    ax = fig.add_subplot(111)
    ax.barh(importancia_df["feature"], importancia_df["importance"], color="#4e79a7")
    ax.set_title("Importância dos Parâmetros (%)")
    ax.set_xlabel("Importância (%)")
    ax.invert_yaxis()
    for i, valor in enumerate(importancia_df["importance"]):
        ax.text(valor + 0.5, i, f"{valor:.1f}%", va="center")


def criar_interface(df, modelo, encoder, resultados, importancia_df, valores_ideais, importancias_por_variavel):
    root = tk.Tk()
    root.title("Recomendador de Culturas")
    root.geometry("920x680")
    root.configure(bg="#e9eff5")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TLabel", background="#e9eff5", font=("Segoe UI", 10), foreground="#1f2937")
    style.configure("Header.TLabel", background="#e9eff5", font=("Segoe UI", 16, "bold"), foreground="#111827")
    style.configure("Title.TLabel", background="#e9eff5", font=("Segoe UI", 11, "bold"), foreground="#1f2937")
    style.configure("Result.TLabel", background="#e9eff5", font=("Segoe UI", 12, "bold"), foreground="#0d6efd")
    style.configure("Info.TLabel", background="#e9eff5", font=("Segoe UI", 9), foreground="#4b5563")
    style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=10)
    style.configure("Card.TFrame", background="#ffffff", relief="flat")
    style.configure("TNotebook", background="#e9eff5")
    style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 8])
    style.configure("Treeview", font=("Segoe UI", 10), background="#ffffff", fieldbackground="#ffffff")
    style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    header = ttk.Frame(root, style="Card.TFrame", padding=(20, 16))
    header.pack(fill="x", padx=16, pady=(16, 0))
    ttk.Label(header, text="Sistema de Recomendação de Culturas", style="Header.TLabel").pack(anchor="w")
    ttk.Label(header, text="Modelos treinados: Decision Tree, KNN, Random Forest e Regressão Logística", style="Info.TLabel").pack(anchor="w", pady=(6, 0))

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=16, pady=16)

    tab_recomendacao = ttk.Frame(notebook, style="Card.TFrame")
    tab_eda = ttk.Frame(notebook, style="Card.TFrame")
    tab_comparacao = ttk.Frame(notebook, style="Card.TFrame")
    tab_importancia = ttk.Frame(notebook, style="Card.TFrame")

    notebook.add(tab_recomendacao, text="Recomendação")
    notebook.add(tab_eda, text="Análise Exploratória")
    notebook.add(tab_comparacao, text="Comparação de Modelos")
    notebook.add(tab_importancia, text="Importância dos Parâmetros")

    criar_tab_recomendacao(tab_recomendacao, modelo, encoder, valores_ideais, importancias_por_variavel)
    criar_tab_eda(tab_eda, df)
    criar_tab_comparacao(tab_comparacao, resultados)
    criar_tab_importancia(tab_importancia, importancia_df)

    return root


def criar_tab_recomendacao(parent, modelo, encoder, valores_ideais, importancias_por_variavel):
    # Frame principal com padding
    main_frame = ttk.Frame(parent, style="Card.TFrame", padding=16)
    main_frame.pack(fill="both", expand=True, padx=12, pady=12)

    # ===== SEÇÃO DE INPUTS =====
    input_frame = ttk.Frame(main_frame, style="Card.TFrame", padding=(12, 0))
    input_frame.pack(fill="x", pady=(0, 16))

    campos = [
        ("Nitrogênio (N, kg/ha)", "N"),
        ("Fósforo (P, kg/ha)", "P"),
        ("Potássio (K, kg/ha)", "K"),
        ("Temperatura (°C)", "temperature"),
        ("Umidade (%)", "humidity"),
        ("pH", "ph"),
        ("Chuva (mm)", "rainfall"),
    ]

    entradas = {}
    # Colocar campos em 2 colunas
    for idx, (rotulo, chave) in enumerate(campos):
        col = idx % 2
        row = idx // 2
        
        # Coluna esquerda
        if col == 0:
            ttk.Label(input_frame, text=rotulo, style="Title.TLabel").grid(row=row, column=0, sticky="w", pady=6, padx=(0, 12))
            var = tk.StringVar()
            ttk.Entry(input_frame, width=18, textvariable=var).grid(row=row, column=1, sticky="ew", pady=6)
            entradas[chave] = var
        # Coluna direita
        else:
            ttk.Label(input_frame, text=rotulo, style="Title.TLabel").grid(row=row, column=2, sticky="w", pady=6, padx=(24, 12))
            var = tk.StringVar()
            ttk.Entry(input_frame, width=18, textvariable=var).grid(row=row, column=3, sticky="ew", pady=6)
            entradas[chave] = var
    
    input_frame.columnconfigure(1, weight=1)
    input_frame.columnconfigure(3, weight=1)

    # ===== SEÇÃO DE RESULTADO COM SCROLL =====
    resultado_frame = ttk.Frame(main_frame, style="Card.TFrame")
    resultado_frame.pack(fill="both", expand=True, pady=(0, 16))

    scrolled = ScrolledText(resultado_frame, wrap="word", height=12, font=("Segoe UI", 10), background="#ffffff")
    scrolled.pack(fill="both", expand=True)
    scrolled.insert("1.0", "Insira os dados e clique em Recomendar.")
    scrolled.configure(state="disabled")

    def gerar_explicacao(cultura_top, valores_input):
        """Gera explicação baseada na importância das variáveis e desvio do ideal."""
        valores_ideais_cultura = valores_ideais[cultura_top]
        nomes_campos = {
            "N": "Nitrogênio (kg/ha)",
            "P": "Fósforo (kg/ha)",
            "K": "Potássio (kg/ha)",
            "temperature": "Temperatura (°C)",
            "humidity": "Umidade (%)",
            "ph": "pH",
            "rainfall": "Chuva (mm)",
        }
        
        # Calcular impacto para cada campo
        impactos = []
        for chave, nome in nomes_campos.items():
            valor_ideal = valores_ideais_cultura[chave]
            valor_input = valores_input[chave]
            
            # Calcular desvio percentual em relação ao ideal
            if valor_ideal != 0:
                desvio_percentual = abs(valor_ideal - valor_input) / valor_ideal * 100
            else:
                desvio_percentual = abs(valor_ideal - valor_input) * 100
            
            # Importância da variável no modelo (valor entre 0 e 1)
            importancia = importancias_por_variavel.get(chave, 0.01)
            importancia_pct = importancia * 100
            
            # Score = desvio * importância (normalizado)
            # Quanto maior o desvio E maior a importância, maior o impacto
            score = desvio_percentual * importancia
            
            impactos.append((nome, valor_ideal, valor_input, desvio_percentual, importancia_pct, score))
        
        # Ordenar por MAIOR score de impacto
        impactos.sort(key=lambda x: x[5], reverse=True)
        
        # Pegar os 3 maiores impactos
        top_3 = impactos[:3]
        
        explicacao = f"As condições estão alinhadas aos valores ideais para {cultura_top}.\n"
        explicacao += f"Fatores mais impactantes:\n"
        
        for idx, (nome, ideal, entrada, desvio, importancia, score) in enumerate(top_3, 1):
            explicacao += f"\n{idx}. {nome}\n"
            explicacao += f"   • Ideal: {ideal:.1f} | Sua entrada: {entrada:.1f}\n"
            explicacao += f"   • Desvio: {desvio:.1f}% | Importância: {importancia:.1f}%"
        
        return explicacao

    def recomendar():
        try:
            entrada = {chave: var.get() for chave, var in entradas.items()}
            valores = validar_valores(entrada)
            vetor = [
                valores["N"],
                valores["P"],
                valores["K"],
                valores["temperature"],
                valores["humidity"],
                valores["ph"],
                valores["rainfall"],
            ]
            ranking = prever_cultura(modelo, encoder, vetor)
            
            # Formatar resultado com ranking
            resultado_texto = "🌱 RANKING DE CULTURAS RECOMENDADAS:\n\n"
            for idx, (cultura, prob) in enumerate(ranking, 1):
                percentual = prob * 100
                resultado_texto += f"{idx}. {cultura.upper()} ({percentual:.1f}%)\n"
            
            # Adicionar valores ideais da cultura top
            cultura_top = ranking[0][0]
            valores_ideais_cultura = valores_ideais[cultura_top]
            
            resultado_texto += f"\n📊 VALORES IDEAIS PARA {cultura_top.upper()}:\n"
            resultado_texto += f"  • Nitrogênio: {valores_ideais_cultura['N']:.1f} kg/ha\n"
            resultado_texto += f"  • Fósforo: {valores_ideais_cultura['P']:.1f} kg/ha\n"
            resultado_texto += f"  • Potássio: {valores_ideais_cultura['K']:.1f} kg/ha\n"
            resultado_texto += f"  • Temperatura: {valores_ideais_cultura['temperature']:.1f} °C\n"
            resultado_texto += f"  • Umidade: {valores_ideais_cultura['humidity']:.1f} %\n"
            resultado_texto += f"  • pH: {valores_ideais_cultura['ph']:.1f}\n"
            resultado_texto += f"  • Chuva: {valores_ideais_cultura['rainfall']:.1f} mm\n"
            
            # Adicionar explicação
            resultado_texto += f"\n💡 POR QUÊ?\n"
            resultado_texto += gerar_explicacao(cultura_top, valores)
            
            scrolled.configure(state="normal")
            scrolled.delete("1.0", tk.END)
            scrolled.insert("1.0", resultado_texto)
            scrolled.configure(state="disabled")
        except ValueError as erro:
            messagebox.showerror("Erro de validação", str(erro))
        except Exception as erro:
            messagebox.showerror("Erro", f"Ocorreu um problema ao gerar a recomendação.\n{erro}")

    def limpar():
        for var in entradas.values():
            var.set("")
        scrolled.configure(state="normal")
        scrolled.delete("1.0", tk.END)
        scrolled.insert("1.0", "Insira os dados e clique em Recomendar.")
        scrolled.configure(state="disabled")

    # ===== SEÇÃO DE BOTÕES (sempre visível) =====
    botoes_frame = ttk.Frame(main_frame, style="Card.TFrame")
    botoes_frame.pack(fill="x")
    botoes_frame.columnconfigure(0, weight=1)
    botoes_frame.columnconfigure(1, weight=1)

    ttk.Button(botoes_frame, text="Recomendar", command=recomendar).grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=8)
    ttk.Button(botoes_frame, text="Limpar", command=limpar).grid(row=0, column=1, sticky="ew", pady=8)

    ttk.Label(main_frame, text="Modelo: Random Forest", style="Info.TLabel").pack(pady=(8, 0))


def criar_tab_eda(parent, df):
    card = ttk.Frame(parent, style="Card.TFrame", padding=16)
    card.pack(fill="both", expand=True, padx=12, pady=12)

    texto = criar_stats_text(df)
    scrolled = ScrolledText(card, wrap="word", height=16, font=("Segoe UI", 10))
    scrolled.insert("1.0", texto)
    scrolled.configure(state="disabled")
    scrolled.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=4)

    fig = Figure(figsize=(5.8, 4.8), dpi=100)
    criar_grafico_distribuicao(fig, df)
    canvas = FigureCanvasTkAgg(fig, master=card)
    canvas.get_tk_widget().pack(side="right", fill="both", expand=True)
    canvas.draw()


def criar_tab_comparacao(parent, resultados):
    card = ttk.Frame(parent, style="Card.TFrame", padding=16)
    card.pack(fill="both", expand=True, padx=12, pady=12)

    tree = ttk.Treeview(card, columns=("model", "accuracy", "precision", "recall", "f1"), show="headings", height=6)
    tree.heading("model", text="Modelo")
    tree.heading("accuracy", text="Accuracy")
    tree.heading("precision", text="Precision")
    tree.heading("recall", text="Recall")
    tree.heading("f1", text="F1-Score")
    tree.column("model", anchor="w", width=210)
    tree.column("accuracy", anchor="center", width=110)
    tree.column("precision", anchor="center", width=110)
    tree.column("recall", anchor="center", width=110)
    tree.column("f1", anchor="center", width=110)
    tree.pack(fill="x", pady=(0, 16))

    modelos_ordenados = sorted(resultados.items(), key=lambda item: item[1]["accuracy"], reverse=True)
    for nome, dados in modelos_ordenados:
        tree.insert("", "end", values=(
            nome,
            f"{dados['accuracy']:.4f}",
            f"{dados['precision']:.4f}",
            f"{dados['recall']:.4f}",
            f"{dados['f1']:.4f}"
        ))

    fig = Figure(figsize=(8.5, 3.8), dpi=100)
    criar_grafico_comparacao(fig, resultados)
    canvas = FigureCanvasTkAgg(fig, master=card)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()


def criar_tab_importancia(parent, importancia_df):
    card = ttk.Frame(parent, style="Card.TFrame", padding=16)
    card.pack(fill="both", expand=True, padx=12, pady=12)

    tree = ttk.Treeview(card, columns=("importance",), show="headings", height=8)
    tree.heading("importance", text="Parâmetro e Importância")
    tree.column("importance", anchor="w", width=260)
    tree.pack(side="left", fill="y", pady=(0, 12), padx=(0, 12))

    for _, linha in importancia_df.iterrows():
        tree.insert("", "end", values=(f"{linha['feature']}: {linha['importance']:.1f}%",))

    fig = Figure(figsize=(6.5, 4.6), dpi=100)
    criar_grafico_importancia(fig, importancia_df)
    canvas = FigureCanvasTkAgg(fig, master=card)
    canvas.get_tk_widget().pack(side="right", fill="both", expand=True)
    canvas.draw()


if __name__ == "__main__":
    df = carregar_dados()
    encoder, modelo, resultados, importancia_df, _, valores_ideais, importancias_por_variavel = preparar_modelos(df)
    app = criar_interface(df, modelo, encoder, resultados, importancia_df, valores_ideais, importancias_por_variavel)
    app.mainloop()
