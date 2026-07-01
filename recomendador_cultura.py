from modelos import carregar_dados, preparar_modelos
from interface import criar_interface


if __name__ == "__main__":
    df = carregar_dados()
    encoder, modelo, resultados, importancia_df, _, valores_ideais, importancias_por_variavel = preparar_modelos(df)
    app = criar_interface(df, modelo, encoder, resultados, importancia_df, valores_ideais, importancias_por_variavel)
    app.mainloop()
