"""ML Challenge — Previsão de preços de imóveis (Ames Housing).

Uso:
    python train.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

# Garante acentuação correta no console do Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# O OpenML marca a versão 1 desta base como "inactive" por causa da curadoria dos
# rótulos (ver data/README.md). Os dados conferem com a publicação original.
warnings.filterwarnings("ignore", message=".*is inactive.*")

RAIZ = Path(__file__).resolve().parent
DIR_DADOS = RAIZ / "data"
ARQUIVO_CSV = DIR_DADOS / "ames_housing.csv"

OPENML_ID = 41211
URL_PARQUET = "https://data.openml.org/datasets/0004/41211/dataset_41211.pq"

ALVO = "Sale_Price"


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
    if nulos.empty:
        print("Nenhuma.")
    else:
        print(nulos.to_string())


def main() -> None:
    df = carregar_dados()
    resumo_inicial(df)


if __name__ == "__main__":
    main()
