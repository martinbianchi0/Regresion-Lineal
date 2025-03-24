from models import LinearRegression
from metrics import mse, mae, rmse
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def save_results():
    return

def load_model(x_train, y_train, method, L1=0, L2=0):
    """Compara los métodos de aprendizaje y sus errores en train y validation."""
    modelo = LinearRegression(x_train, y_train, L1, L2)
    if method == "gd":
        modelo.gradiente_descendiente()
    elif method == "pinv":
        modelo.pseudoinversa()
    return modelo

def data_info(df_dev, df_test):
    df = pd.concat([df_dev, df_test])

    print("Fragmento aleatorio\n", df.sample(7))

    print("\nRango de valores de cada columna\n", df.describe().loc[['min', 'max']])

    print("\nCategorías con valores faltantes\n", df.isna().sum()[df.isna().sum() > 0].to_string())

    print("\nFilas duplicadas:", df.duplicated().sum())

def data_analisis(df_dev, df_test):
    df = pd.concat([df_dev, df_test], ignore_index=True)
    # Definir variables
    num_vars = ['area', 'price', 'rooms', 'age']
    bin_vars = ['is_house', 'has_pool']

    # Pairplot con color según la nueva categoría
    sns.pairplot(df, vars=num_vars, hue="zona", palette="coolwarm")
    plt.show()

    # Crear figura 4x4 para boxplots
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    # Graficar cada numérica contra cada binaria
    for ax, (num_var, bin_var) in zip(axes.flat, [(x, y) for x in num_vars for y in bin_vars]):
        sns.boxplot(x=df[bin_var], y=df[num_var], ax=ax)
        ax.set_title(f'{num_var} vs {bin_var}')

    # Ocultar gráficos vacíos si sobran espacios
    for ax in axes.flat[len(num_vars) * len(bin_vars):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Comparación de zona con is_house
    sns.countplot(x=df["zona"], hue=df["is_house"], palette="coolwarm", ax=axes[0])
    axes[0].set_title("Distribución de is_house según zona")

    # Comparación de zona con has_pool
    sns.countplot(x=df["zona"], hue=df["has_pool"], palette="coolwarm", ax=axes[1])
    axes[1].set_title("Distribución de has_pool según zona")

    # Comparación de is_house con has_pool
    sns.countplot(x=df["is_house"], hue=df["has_pool"], palette="coolwarm", ax=axes[2])
    axes[2].set_title("Distribución de has_pool según is_house")

    plt.tight_layout()
    plt.show()



def model_onefeature(x_train, y_train, X_val, y_val):
    modelos = {
        "Gradiente Descendiente": load_model(x_train, y_train, "gd"),
        "Pseudoinversa": load_model(x_train, y_train, "pinv")
    }

    print("\n--- Comparación de métodos ---")
    
    for nombre, modelo in modelos.items():
        coef = modelo.obtener_coeficientes()
        print(f"\n{nombre}:")
        print("MSE Train:", mse(x_train, y_train, coef), "| MSE Validation:", mse(X_val, y_val, coef))

        # Graficar regresión
        X_plot = np.column_stack((np.ones(x_train.shape[0]), x_train))
        y_pred = X_plot @ coef

        plt.scatter(x_train.iloc[:, 0], y_train, color="blue", label="Datos")
        plt.plot(np.sort(x_train.iloc[:, 0]), np.sort(y_pred), color='red', lw=2, label="Recta de Regresión")
        plt.xlabel("Área")
        plt.ylabel("Precio")
        plt.legend()
        plt.title(f"Regresión Lineal: Precio vs Área ({nombre})")
        plt.show()


def feature_engineering(df):
    df["area_per_room"] = df["area"] / df["rooms"]
    df["luxury"] = df["is_house"] * df["has_pool"] * df["rooms"]
    #luxury (house, pool, rooms or area)
    df["log_area"] = np.log1p(df["area"])
    #log(1+area) esto ni idea
    #is house * age, a ver si las casas modernas son mas caras
    df["is_house_age"] = df["is_house"] * df["age"]
    df["zona_area"] = df["zona"] * df["area"]

def metricas(x_train, y_train, X_val, y_val, test=None, L1=0, L2=0):
    """Calcula y muestra las métricas de entrenamiento y validación para distintos métodos."""

    modelo_gd = load_model(x_train, y_train, "gd", L1, L2)
    modelo_pinv = load_model(x_train, y_train, "pinv", L1, L2)

    # Obtener coeficientes entrenados
    b_gd = modelo_gd.obtener_coeficientes()
    b_pinv = modelo_pinv.obtener_coeficientes()

    # Calcular métricas
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    }

    print("\n--- Comparación de métodos ---")
    for nombre, modelo, b in [("Gradiente descendiente", modelo_gd, b_gd),
                               ("Pseudoinversa", modelo_pinv, b_pinv)]:
        print(f"\n{nombre}:")

        for metrica, funcion in metrics.items():
            train_score = funcion(x_train, y_train, b)
            val_score = funcion(X_val, y_val, b)
            print(f"{metrica} Train: {train_score:.4f} | {metrica} Validation: {val_score:.4f}")

        if isinstance(test, pd.DataFrame):
            print("Predicción de price:", modelo.predict(test))

def graficar_coef_vs_l(x_train, y_train, reg):
    """Grafica la evolución de los coeficientes en función del coeficiente de regularización L1 o L2."""
    
    lambdas = np.logspace(-4, 1, 20)  # Valores de regularización entre 10⁻⁴ y 10²
    coefs_pinv, coefs_gd = [], []

    for L in lambdas:
        if reg == "L2":
            # Modelo con pseudoinversa
            modelo_pinv = LinearRegression(x_train, y_train, L2=L, L1=0)
            modelo_pinv.pseudoinversa()
            coefs_pinv.append(modelo_pinv.obtener_coeficientes())
            # Modelo con gradiente descendiente
            modelo_gd = LinearRegression(x_train, y_train, L2=L, L1=0)
            modelo_gd.gradiente_descendiente()
            coefs_gd.append(modelo_gd.obtener_coeficientes())

        else:  # Para L1, solo gradiente descendiente
            modelo = LinearRegression(x_train, y_train, L1=L, L2=0)
            modelo.gradiente_descendiente()
            coefs_gd.append(modelo.obtener_coeficientes())

    if reg == "L2":
        coefs_pinv = np.array(coefs_pinv)
        coefs_gd = np.array(coefs_gd)

        # Gráfico para pseudoinversa
        plt.figure(figsize=(10, 6))
        for i in range(coefs_pinv.shape[1]):
            plt.plot(lambdas, coefs_pinv[:, i], label=f"Coef {i}")
        plt.xscale("log")
        plt.xlabel(f"Regularización {reg} (λ)")
        plt.ylabel("Valor de los coeficientes")
        plt.title(f"Coeficientes vs Regularización {reg} (Pseudoinversa)")
        plt.legend()
        plt.show()

        # Gráfico para gradiente descendiente
        plt.figure(figsize=(10, 6))
        for i in range(coefs_gd.shape[1]):
            plt.plot(lambdas, coefs_gd[:, i], label=f"Coef {i}")
        plt.xscale("log")
        plt.xlabel(f"Regularización {reg} (λ)")
        plt.ylabel("Valor de los coeficientes")
        plt.title(f"Coeficientes vs Regularización {reg} (Gradiente Descendiente)")
        plt.legend()
        plt.show()

    else:
        coefs_gd = np.array(coefs_gd)
        # Gráfico para L1 (solo GD)
        plt.figure(figsize=(10, 6))
        for i in range(coefs_gd.shape[1]):
            plt.plot(lambdas, coefs_gd[:, i], label=f"Coef {i}")
        plt.xscale("log")
        plt.xlabel(f"Regularización {reg} (λ)")
        plt.ylabel("Valor de los coeficientes")
        plt.title(f"Coeficientes vs Regularización {reg} (Gradiente Descendiente)")
        plt.legend()
        plt.show()