"""ML Challenge — Previsão de preços de imóveis (Ames Housing).

Executa todo o fluxo do projeto:
    1. Carrega (ou baixa) o dataset;
    2. Faz a análise exploratória e salva os gráficos em reports/graficos/;
    3. Limpa e prepara os dados;
    4. Treina e ajusta Ridge e Random Forest com validação cruzada;
    5. Avalia com MAE, RMSE e R², escolhe o melhor e salva:
         - models/model.joblib
         - reports/metricas.json

Uso:
    python train.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend sem interface gráfica (necessário para salvar arquivos)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

# Garante acentuação correta no console do Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# O OpenML marca a versão 1 desta base como "inactive" por causa da curadoria dos
# rótulos (ver data/README.md). Os dados conferem com a publicação original.
warnings.filterwarnings("ignore", message=".*is inactive.*")

# --------------------------------------------------------------------------- #
# Caminhos e constantes
# --------------------------------------------------------------------------- #

RAIZ = Path(__file__).resolve().parent
DIR_DADOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "models"
DIR_RELATORIOS = RAIZ / "reports"
DIR_GRAFICOS = DIR_RELATORIOS / "graficos"

ARQUIVO_CSV = DIR_DADOS / "ames_housing.csv"
ARQUIVO_MODELO = DIR_MODELOS / "model.joblib"
ARQUIVO_METRICAS = DIR_RELATORIOS / "metricas.json"

OPENML_ID = 41211
URL_PARQUET = "https://data.openml.org/datasets/0004/41211/dataset_41211.pq"

ALVO = "Sale_Price"
RANDOM_STATE = 42
TAMANHO_TESTE = 0.20

NUMERICAS = [
    "Gr_Liv_Area",
    "Total_Bsmt_SF",
    "First_Flr_SF",
    "Lot_Area",
    "Year_Built",
    "Year_Remod_Add",
    "Overall_Qual",
    "Overall_Cond",
    "Garage_Cars",
    "Garage_Area",
    "Full_Bath",
    "Bedroom_AbvGr",
    "TotRms_AbvGrd",
    "Fireplaces",
]

CATEGORICAS = [
    "Neighborhood",
    "Bldg_Type",
    "House_Style",
    "Exter_Qual",
    "Kitchen_Qual",
    "Bsmt_Qual",
    "Central_Air",
    "Garage_Type",
    "Sale_Condition",
]

# As variáveis ordinais vêm como texto nesta versão do OpenML e são convertidas
# para a escala numérica original (1 a 10).
MAPA_ORDINAL = {
    "Very_Poor": 1,
    "Poor": 2,
    "Fair": 3,
    "Below_Average": 4,
    "Average": 5,
    "Above_Average": 6,
    "Good": 7,
    "Very_Good": 8,
    "Excellent": 9,
    "Very_Excellent": 10,
}

VARS_ORDINAIS = ["Overall_Qual", "Overall_Cond"]

sns.set_theme(style="whitegrid", palette="viridis")


def formatar_moeda(valor: float) -> str:
    """Formata um número no padrão brasileiro: US$ 180.796,06."""
    texto = f"{valor:,.2f}"
    texto = texto.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"US$ {texto}"


# --------------------------------------------------------------------------- #
# 1. Carregamento
# --------------------------------------------------------------------------- #


def _baixar_dataset() -> pd.DataFrame:
    """Tenta baixar o dataset do OpenML por dois caminhos diferentes."""
    erros: list[str] = []

    try:
        from sklearn.datasets import fetch_openml

        print("  [1/2] Tentando fetch_openml...")
        ds = fetch_openml(data_id=OPENML_ID, as_frame=True)
        return ds.frame
    except Exception as exc:  # noqa: BLE001 - queremos registrar qualquer falha
        erros.append(f"fetch_openml: {exc}")
        print(f"  Falhou: {exc}")

    try:
        print("  [2/2] Tentando download do Parquet oficial...")
        return pd.read_parquet(URL_PARQUET)
    except Exception as exc:  # noqa: BLE001
        erros.append(f"parquet: {exc}")
        print(f"  Falhou: {exc}")

    detalhes = "\n".join(f"  - {e}" for e in erros)
    raise RuntimeError(
        "Não foi possível baixar o dataset Ames Housing.\n"
        f"Tentativas:\n{detalhes}\n\n"
        "Solução manual: baixe o arquivo em https://www.openml.org/d/41211 "
        "(aba 'Download') e salve como data/ames_housing.csv"
    )


def carregar_dados() -> pd.DataFrame:
    """Lê o CSV local; se não existir, baixa do OpenML e salva em data/."""
    if ARQUIVO_CSV.exists():
        return pd.read_csv(ARQUIVO_CSV)

    print("Dataset não encontrado. Baixando do OpenML...")
    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    df = _baixar_dataset()
    df.to_csv(ARQUIVO_CSV, index=False)
    print(f"Dataset salvo em {ARQUIVO_CSV}")
    return df


def resumo_inicial(df: pd.DataFrame) -> None:
    """Imprime as verificações básicas da base carregada."""
    print("\n=== RESUMO INICIAL ===")
    print(f"Registros: {df.shape[0]} | Colunas: {df.shape[1]}")
    print(f"Linhas duplicadas: {df.duplicated().sum()}")
    print(f"Valores ausentes: {df.isna().sum().sum()}")

    print("\n--- Variável-alvo (Sale_Price) ---")
    print(df[ALVO].describe().round(2).to_string())

    print("\n--- Colunas com valores ausentes ---")
    nulos = df.isna().sum().sort_values(ascending=False)
    nulos = nulos[nulos > 0]
    print("Nenhuma." if nulos.empty else nulos.to_string())


# --------------------------------------------------------------------------- #
# 2. Análise exploratória
# --------------------------------------------------------------------------- #


def analise_exploratoria(df: pd.DataFrame) -> list[str]:
    """Gera os gráficos da EDA e devolve as conclusões objetivas."""
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    print("\n=== ANÁLISE EXPLORATÓRIA ===")

    # --- Gráfico 1: distribuição do preço -----------------------------------
    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[ALVO], bins=50, kde=True, ax=eixos[0], color="#2a6f97")
    eixos[0].set_title("Distribuição dos preços de venda")
    eixos[0].set_xlabel("Preço (US$)")
    eixos[0].set_ylabel("Quantidade de imóveis")
    eixos[0].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k")
    )

    sns.boxplot(x=df[ALVO], ax=eixos[1], color="#61a5c2")
    eixos[1].set_title("Boxplot dos preços (detecção de outliers)")
    eixos[1].set_xlabel("Preço (US$)")
    eixos[1].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k")
    )
    fig.tight_layout()
    fig.savefig(DIR_GRAFICOS / "01_distribuicao_preco.png", dpi=130)
    plt.close(fig)

    # --- Gráfico 2: área construída x preço ---------------------------------
    fig, eixo = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(
        data=df,
        x="Gr_Liv_Area",
        y=ALVO,
        hue="Sale_Condition",
        alpha=0.65,
        s=28,
        ax=eixo,
    )
    eixo.set_title("Área construída × preço de venda")
    eixo.set_xlabel("Área construída acima do solo (pés²)")
    eixo.set_ylabel("Preço (US$)")
    eixo.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y / 1000:.0f}k")
    )
    eixo.legend(title="Condição da venda", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_GRAFICOS / "02_area_vs_preco.png", dpi=130)
    plt.close(fig)

    # --- Gráfico 3: mapa de correlação --------------------------------------
    numericas = [c for c in NUMERICAS if c not in VARS_ORDINAIS] + [ALVO]
    fig, eixo = plt.subplots(figsize=(9.5, 7.5))
    sns.heatmap(
        df[numericas].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        annot_kws={"size": 7},
        ax=eixo,
    )
    eixo.set_title("Correlação entre variáveis numéricas e o preço")
    fig.tight_layout()
    fig.savefig(DIR_GRAFICOS / "03_mapa_correlacao.png", dpi=130)
    plt.close(fig)

    # --- Gráfico 4: qualidade geral x preço ---------------------------------
    ordem = [rotulo for rotulo, _ in sorted(MAPA_ORDINAL.items(), key=lambda x: x[1])]
    presentes = [rotulo for rotulo in ordem if rotulo in set(df["Overall_Qual"])]
    fig, eixo = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(
        data=df,
        x="Overall_Qual",
        y=ALVO,
        order=presentes,
        ax=eixo,
        color="#40916c",
    )
    eixo.set_title("Qualidade geral do imóvel × preço de venda")
    eixo.set_xlabel("Qualidade geral (1 = muito ruim, 10 = muito excelente)")
    eixo.set_ylabel("Preço (US$)")
    eixo.set_xticks(range(len(presentes)))
    eixo.set_xticklabels([str(MAPA_ORDINAL[r]) for r in presentes])
    eixo.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y / 1000:.0f}k")
    )
    fig.tight_layout()
    fig.savefig(DIR_GRAFICOS / "04_qualidade_vs_preco.png", dpi=130)
    plt.close(fig)

    # --- Conclusões ---------------------------------------------------------
    assimetria = df[ALVO].skew()
    correlacoes = (
        df[numericas]
        .corr()[ALVO]
        .drop(ALVO)
        .sort_values(ascending=False)
    )
    top_2 = correlacoes.head(2)
    medias_qualidade = df.groupby("Overall_Qual")[ALVO].mean()
    menor_qual, maior_qual = "Very_Poor", "Very_Excellent"
    razao = (
        medias_qualidade[maior_qual] / medias_qualidade[menor_qual]
        if menor_qual in medias_qualidade and maior_qual in medias_qualidade
        else float("nan")
    )

    conclusoes = [
        f"O preço é assimétrico à direita (assimetria = {assimetria:.2f}): "
        f"média de {formatar_moeda(df[ALVO].mean())} contra mediana de "
        f"{formatar_moeda(df[ALVO].median())}, com cauda de imóveis de alto valor.",
        f"As variáveis mais correlacionadas com o preço são "
        f"{top_2.index[0]} ({top_2.iloc[0]:.2f}) e {top_2.index[1]} "
        f"({top_2.iloc[1]:.2f}), indicando que o porte do imóvel é o principal sinal "
        f"de valor.",
        f"Imóveis com qualidade geral máxima valem cerca de {razao:.1f}× mais que os de "
        f"qualidade mínima ({formatar_moeda(medias_qualidade.get(maior_qual, np.nan))} "
        f"contra {formatar_moeda(medias_qualidade.get(menor_qual, np.nan))}).",
        f"Foram encontrados {len(df) - len(limpar_dados(df))} registros inconsistentes "
        f"(vendas parciais de imóveis acima de 4.000 pés²), removidos antes do "
        f"treinamento.",
    ]

    print("\n--- CONCLUSÕES DA EDA ---")
    for numero, texto in enumerate(conclusoes, start=1):
        print(f"{numero}. {texto}")

    print(f"\nGráficos salvos em {DIR_GRAFICOS}")
    return conclusoes


# --------------------------------------------------------------------------- #
# 3. Pré-processamento
# --------------------------------------------------------------------------- #


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas e vendas parciais atípicas de imóveis grandes."""
    df = df.drop_duplicates()
    mascara = (df["Sale_Condition"] == "Partial") & (df["Gr_Liv_Area"] > 4000)
    return df.loc[~mascara].copy()


def preparar_dados(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X e y, remove inconsistências e converte as ordinais para número."""
    df = limpar_dados(df)

    X = df[NUMERICAS + CATEGORICAS].copy()
    for coluna in VARS_ORDINAIS:
        X[coluna] = X[coluna].map(MAPA_ORDINAL)

    y = df[ALVO].copy()
    return X, y


def construir_preprocessador() -> ColumnTransformer:
    """Monta o ColumnTransformer aplicado dentro de cada pipeline."""
    pipeline_numericas = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="median")),
            ("escala", StandardScaler()),
        ]
    )
    pipeline_categoricas = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
            ("codificador", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numericas", pipeline_numericas, NUMERICAS),
            ("categoricas", pipeline_categoricas, CATEGORICAS),
        ]
    )


# --------------------------------------------------------------------------- #
# 4. Treinamento e avaliação
# --------------------------------------------------------------------------- #


def construir_modelos() -> dict[str, tuple[TransformedTargetRegressor, dict]]:
    """Define os dois modelos e as grades de hiperparâmetros.

    O alvo é transformado com log1p durante o treino e revertido com expm1 na
    previsão, o que reduz o efeito da assimetria dos preços.
    """
    modelos = {}

    ridge = Pipeline(
        [
            ("preprocessamento", construir_preprocessador()),
            ("modelo", Ridge(random_state=RANDOM_STATE)),
        ]
    )
    modelos["Ridge"] = (
        TransformedTargetRegressor(regressor=ridge, func=np.log1p, inverse_func=np.expm1),
        {"regressor__modelo__alpha": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
    )

    floresta = Pipeline(
        [
            ("preprocessamento", construir_preprocessador()),
            (
                "modelo",
                RandomForestRegressor(
                    n_estimators=300, random_state=RANDOM_STATE, n_jobs=1
                ),
            ),
        ]
    )
    modelos["RandomForest"] = (
        TransformedTargetRegressor(
            regressor=floresta, func=np.log1p, inverse_func=np.expm1
        ),
        {
            "regressor__modelo__max_depth": [None, 20],
            "regressor__modelo__min_samples_leaf": [1, 2, 4],
        },
    )

    return modelos


def avaliar(modelo, X_teste: pd.DataFrame, y_teste: pd.Series) -> dict[str, float]:
    """Calcula MAE, RMSE e R² no conjunto de teste."""
    previsoes = modelo.predict(X_teste)
    return {
        "MAE": float(mean_absolute_error(y_teste, previsoes)),
        "RMSE": float(np.sqrt(mean_squared_error(y_teste, previsoes))),
        "R2": float(r2_score(y_teste, previsoes)),
    }


def treinar_modelos(
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
) -> dict[str, dict]:
    """Treina os dois modelos com GridSearchCV e avalia cada um no teste."""
    validacao = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    resultados: dict[str, dict] = {}

    for nome, (modelo, grade) in construir_modelos().items():
        print(f"\n--- Treinando {nome} ---")
        inicio = time.time()

        busca = GridSearchCV(
            estimator=modelo,
            param_grid=grade,
            cv=validacao,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,
        )
        busca.fit(X_treino, y_treino)

        metricas = avaliar(busca.best_estimator_, X_teste, y_teste)
        duracao = time.time() - inicio

        resultados[nome] = {
            "modelo": busca.best_estimator_,
            "metricas": metricas,
            "parametros": busca.best_params_,
            "rmse_cv": float(-busca.best_score_),
            "duracao": duracao,
        }

        print(f"Melhores parâmetros: {busca.best_params_}")
        print(f"RMSE médio na validação cruzada: {formatar_moeda(-busca.best_score_)}")
        print(
            f"Teste -> MAE: {formatar_moeda(metricas['MAE'])} | "
            f"RMSE: {formatar_moeda(metricas['RMSE'])} | R²: {metricas['R2']:.4f}"
        )
        print(f"Tempo: {duracao:.1f}s")

    return resultados


def gerar_grafico_previsoes(
    modelo, X_teste: pd.DataFrame, y_teste: pd.Series, nome: str
) -> None:
    """Salva o gráfico de valores reais × previstos do modelo escolhido."""
    previsoes = modelo.predict(X_teste)

    fig, eixos = plt.subplots(1, 2, figsize=(12.5, 5))
    eixos[0].scatter(y_teste, previsoes, alpha=0.55, s=26, color="#1d3557")
    limite = [y_teste.min(), y_teste.max()]
    eixos[0].plot(limite, limite, "r--", linewidth=1.4, label="Previsão perfeita")
    eixos[0].set_title(f"{nome}: valores reais × previstos")
    eixos[0].set_xlabel("Preço real (US$)")
    eixos[0].set_ylabel("Preço previsto (US$)")
    eixos[0].legend()

    residuos = y_teste - previsoes
    sns.histplot(residuos, bins=40, kde=True, ax=eixos[1], color="#e76f51")
    eixos[1].axvline(0, color="black", linestyle="--", linewidth=1)
    eixos[1].set_title("Distribuição dos resíduos")
    eixos[1].set_xlabel("Erro (real − previsto), em US$")

    fig.tight_layout()
    fig.savefig(DIR_GRAFICOS / "05_real_vs_previsto.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Salvamento dos artefatos
# --------------------------------------------------------------------------- #


def salvar_artefatos(
    resultados: dict[str, dict],
    df: pd.DataFrame,
    conclusoes: list[str],
) -> str:
    """Escolhe o melhor modelo e grava model.joblib e metricas.json."""
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)

    melhor_nome = min(resultados, key=lambda n: resultados[n]["metricas"]["RMSE"])
    melhor = resultados[melhor_nome]
    metricas = melhor["metricas"]
    outro_nome = next(n for n in resultados if n != melhor_nome)
    outro = resultados[outro_nome]["metricas"]

    justificativa = (
        f"O {melhor_nome} obteve o menor RMSE no conjunto de teste "
        f"({formatar_moeda(metricas['RMSE'])}) e o maior R² ({metricas['R2']:.4f}), "
        f"contra {formatar_moeda(outro['RMSE'])} e {outro['R2']:.4f} do {outro_nome}. "
    )
    if melhor_nome == "RandomForest":
        justificativa += (
            "Como o Random Forest captura relações não lineares e interações entre "
            "variáveis, teve vantagem sobre o modelo linear, que assume uma relação "
            "aditiva e proporcional entre atributos e preço."
        )
    else:
        justificativa += (
            "Apesar de o Random Forest capturar relações não lineares, o modelo "
            "regularizado generalizou melhor com o volume de dados disponível, "
            "evitando o sobreajuste típico de florestas em bases pequenas."
        )

    joblib.dump(
        {
            "pipeline": melhor["modelo"],
            "algoritmo": melhor_nome,
            "metricas": metricas,
            "metricas_comparativo": {
                nome: resultado["metricas"] for nome, resultado in resultados.items()
            },
            "features": {"numericas": NUMERICAS, "categoricas": CATEGORICAS},
            "mapa_ordinal": MAPA_ORDINAL,
            "alvo": ALVO,
            "moeda": "USD",
            "versao": "1.0",
            "treinado_em": datetime.now().isoformat(timespec="seconds"),
        },
        ARQUIVO_MODELO,
    )
    print(f"\nModelo salvo em {ARQUIVO_MODELO}")

    relatorio = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "nome": "Ames Housing",
            "fonte": "OpenML data_id=41211",
            "registros_originais": int(df.shape[0]),
            "registros_usados": int(len(limpar_dados(df))),
            "features_utilizadas": len(NUMERICAS) + len(CATEGORICAS),
            "fonte_original": "De Cock (2011), Journal of Statistics Education 19(3)",
        },
        "alvo": "Sale_Price (USD)",
        "transformacao_alvo": "log1p (revertida com expm1 na previsão)",
        "divisao": {"treino": 0.80, "teste": 0.20, "random_state": RANDOM_STATE},
        "modelos": {
            nome: {
                "MAE": round(resultado["metricas"]["MAE"], 2),
                "RMSE": round(resultado["metricas"]["RMSE"], 2),
                "R2": round(resultado["metricas"]["R2"], 4),
                "rmse_validacao_cruzada": round(resultado["rmse_cv"], 2),
                "melhores_parametros": resultado["parametros"],
            }
            for nome, resultado in resultados.items()
        },
        "melhor_modelo": melhor_nome,
        "justificativa": justificativa,
        "conclusoes_eda": conclusoes,
    }

    with open(ARQUIVO_METRICAS, "w", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)
    print(f"Métricas salvas em {ARQUIVO_METRICAS}")

    return melhor_nome


# --------------------------------------------------------------------------- #
# 6. Execução principal
# --------------------------------------------------------------------------- #


def main() -> None:
    inicio = time.time()

    df = carregar_dados()
    resumo_inicial(df)
    conclusoes = analise_exploratoria(df)

    print("\n=== PREPARAÇÃO DOS DADOS ===")
    X, y = preparar_dados(df)
    print(f"Registros usados: {len(X)} | Variáveis: {X.shape[1]}")
    print(f"Variáveis numéricas: {len(NUMERICAS)} | categóricas: {len(CATEGORICAS)}")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=TAMANHO_TESTE, random_state=RANDOM_STATE
    )
    print(f"Treino: {len(X_treino)} registros | Teste: {len(X_teste)} registros")

    resultados = treinar_modelos(X_treino, y_treino, X_teste, y_teste)
    melhor_nome = salvar_artefatos(resultados, df, conclusoes)
    gerar_grafico_previsoes(
        resultados[melhor_nome]["modelo"], X_teste, y_teste, melhor_nome
    )

    print(f"\n=== CONCLUÍDO em {time.time() - inicio:.1f}s ===")
    print(f"Modelo escolhido: {melhor_nome}")


if __name__ == "__main__":
    main()
