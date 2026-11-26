"""Aplicação Streamlit — Previsão de preços de imóveis (Ames Housing).

Uso:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
ARQUIVO_MODELO = RAIZ / "models" / "model.joblib"
ARQUIVO_METRICAS = RAIZ / "reports" / "metricas.json"

MOEDA = "US$"

# Rótulos em português para as variáveis ordinais (a base usa texto em inglês).
QUALIDADE = {
    1: "1 — Muito ruim",
    2: "2 — Ruim",
    3: "3 — Regular",
    4: "4 — Abaixo da média",
    5: "5 — Média",
    6: "6 — Acima da média",
    7: "7 — Boa",
    8: "8 — Muito boa",
    9: "9 — Excelente",
    10: "10 — Muito excelente",
}

BAIRROS = [
    "Bloomington_Heights",
    "Blueste",
    "Briardale",
    "Brookside",
    "Clear_Creek",
    "College_Creek",
    "Crawford",
    "Edwards",
    "Gilbert",
    "Green_Hills",
    "Greens",
    "Iowa_DOT_and_Rail_Road",
    "Landmark",
    "Meadow_Village",
    "Mitchell",
    "North_Ames",
    "Northpark_Villa",
    "Northridge",
    "Northridge_Heights",
    "Northwest_Ames",
    "Old_Town",
    "Sawyer",
    "Sawyer_West",
    "Somerset",
    "South_and_West_of_Iowa_State_University",
    "Stone_Brook",
    "Timberland",
    "Veenker",
]

TIPOS_EDIFICACAO = ["OneFam", "TwoFmCon", "Duplex", "Twnhs", "TwnhsE"]
ESTILOS = [
    "One_Story",
    "Two_Story",
    "One_and_Half_Fin",
    "One_and_Half_Unf",
    "Two_and_Half_Fin",
    "Two_and_Half_Unf",
    "SFoyer",
    "SLvl",
]
QUALIDADES_CATEGORICAS = ["Excellent", "Good", "Typical", "Fair", "Poor"]
QUALIDADES_PORAO = ["Excellent", "Good", "Typical", "Fair", "Poor", "No_Basement"]
TIPOS_GARAGEM = [
    "Attchd",
    "Detchd",
    "BuiltIn",
    "Basment",
    "CarPort",
    "More_Than_Two_Types",
    "No_Garage",
]
CONDICOES_VENDA = ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"]


def formatar_moeda(valor: float) -> str:
    """Formata no padrão brasileiro: US$ 180.796."""
    texto = f"{valor:,.0f}"
    return f"{MOEDA} {texto.replace(',', '.')}"


@st.cache_resource
def carregar_modelo() -> dict:
    return joblib.load(ARQUIVO_MODELO)


@st.cache_data
def carregar_metricas() -> dict:
    with open(ARQUIVO_METRICAS, encoding="utf-8") as arquivo:
        return json.load(arquivo)


def validar_entradas(dados: dict) -> list[str]:
    """Validações que os campos numéricos sozinhos não cobrem."""
    erros: list[str] = []
    if dados["Gr_Liv_Area"] <= 0:
        erros.append("A área construída deve ser maior que zero.")
    if dados["Year_Remod_Add"] < dados["Year_Built"]:
        erros.append("O ano da reforma não pode ser anterior ao ano de construção.")
    if dados["Garage_Cars"] == 0 and dados["Garage_Area"] > 0:
        erros.append(
            "Informe ao menos uma vaga na garagem quando a área da garagem for maior "
            "que zero."
        )
    return erros


def main() -> None:
    st.set_page_config(
        page_title="Previsão de Preços de Imóveis",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 Previsão de Preços de Imóveis")
    st.caption(
        "ML Challenge — regressão sobre o Ames Housing Dataset (Ames, Iowa, EUA). "
        "Valores em dólares americanos (USD), sem conversão cambial."
    )

    st.info(
        "⚠️ **Finalidade educacional.** As estimativas são geradas por um modelo de "
        "Machine Learning treinado com dados de vendas de 2006 a 2010 e não substituem "
        "uma avaliação imobiliária profissional."
    )

    if not ARQUIVO_MODELO.exists():
        st.error(
            "O arquivo `models/model.joblib` não foi encontrado. "
            "Treine o modelo antes de usar a aplicação:"
        )
        st.code("python train.py", language="bash")
        st.stop()

    modelo_info = carregar_modelo()
    pipeline = modelo_info["pipeline"]
    algoritmo = modelo_info["algoritmo"]

    with st.form("formulario_imovel"):
        st.subheader("Características do imóvel")

        coluna_1, coluna_2, coluna_3 = st.columns(3)

        with coluna_1:
            area_construida = st.number_input(
                "Área construída (pés²)",
                min_value=300,
                max_value=6000,
                value=1500,
                step=50,
                help="Área habitável acima do solo (Gr_Liv_Area).",
            )
            quartos = st.number_input(
                "Quartos", min_value=0, max_value=8, value=3, step=1
            )
            banheiros = st.number_input(
                "Banheiros completos", min_value=0, max_value=5, value=2, step=1
            )

        with coluna_2:
            ano_construcao = st.number_input(
                "Ano de construção",
                min_value=1870,
                max_value=2010,
                value=1990,
                step=1,
            )
            qualidade_geral = st.select_slider(
                "Qualidade geral do imóvel",
                options=list(QUALIDADE.keys()),
                value=6,
                format_func=lambda valor: QUALIDADE[valor],
            )
            vagas_garagem = st.number_input(
                "Vagas na garagem", min_value=0, max_value=4, value=2, step=1
            )

        with coluna_3:
            area_terreno = st.number_input(
                "Área do terreno (pés²)",
                min_value=1000,
                max_value=25000,
                value=9000,
                step=500,
            )
            area_porao = st.number_input(
                "Área do porão (pés²)",
                min_value=0,
                max_value=3500,
                value=1000,
                step=100,
            )
            bairro = st.selectbox("Bairro", BAIRROS, index=BAIRROS.index("North_Ames"))

        with st.expander("Parâmetros avançados"):
            avancado_1, avancado_2, avancado_3 = st.columns(3)

            with avancado_1:
                ano_reforma = st.number_input(
                    "Ano da última reforma",
                    min_value=1870,
                    max_value=2010,
                    value=1990,
                    step=1,
                )
                conservacao = st.select_slider(
                    "Estado de conservação",
                    options=list(QUALIDADE.keys()),
                    value=5,
                    format_func=lambda valor: QUALIDADE[valor],
                )
                area_primeiro_pavimento = st.number_input(
                    "Área do 1º pavimento (pés²)",
                    min_value=300,
                    max_value=4000,
                    value=1200,
                    step=50,
                )

            with avancado_2:
                total_comodos = st.number_input(
                    "Total de cômodos (exclui banheiros)",
                    min_value=2,
                    max_value=15,
                    value=7,
                    step=1,
                )
                lareiras = st.number_input(
                    "Lareiras", min_value=0, max_value=4, value=0, step=1
                )
                area_garagem = st.number_input(
                    "Área da garagem (pés²)",
                    min_value=0,
                    max_value=1500,
                    value=480,
                    step=20,
                )

            with avancado_3:
                tipo_edificacao = st.selectbox("Tipo de edificação", TIPOS_EDIFICACAO)
                estilo = st.selectbox("Estilo da casa", ESTILOS)
                qualidade_cozinha = st.selectbox(
                    "Qualidade da cozinha", QUALIDADES_CATEGORICAS, index=2
                )

            avancado_4, avancado_5, avancado_6 = st.columns(3)

            with avancado_4:
                qualidade_externa = st.selectbox(
                    "Qualidade do revestimento externo", QUALIDADES_CATEGORICAS, index=2
                )
                qualidade_porao = st.selectbox(
                    "Qualidade do porão", QUALIDADES_PORAO, index=2
                )

            with avancado_5:
                ar_condicionado = st.selectbox(
                    "Ar-condicionado central", ["Y", "N"], index=0
                )
                tipo_garagem = st.selectbox("Tipo de garagem", TIPOS_GARAGEM)

            with avancado_6:
                condicao_venda = st.selectbox("Condição da venda", CONDICOES_VENDA)

        enviado = st.form_submit_button(
            "Estimar preço", type="primary", width="stretch"
        )

    if not enviado:
        st.caption(
            "Preencha as características e clique em **Estimar preço** para ver o "
            "resultado."
        )
        return

    dados = {
        "Gr_Liv_Area": area_construida,
        "Total_Bsmt_SF": area_porao,
        "First_Flr_SF": area_primeiro_pavimento,
        "Lot_Area": area_terreno,
        "Year_Built": ano_construcao,
        "Year_Remod_Add": ano_reforma,
        "Overall_Qual": qualidade_geral,
        "Overall_Cond": conservacao,
        "Garage_Cars": vagas_garagem,
        "Garage_Area": area_garagem,
        "Full_Bath": banheiros,
        "Bedroom_AbvGr": quartos,
        "TotRms_AbvGrd": total_comodos,
        "Fireplaces": lareiras,
        "Neighborhood": bairro,
        "Bldg_Type": tipo_edificacao,
        "House_Style": estilo,
        "Exter_Qual": qualidade_externa,
        "Kitchen_Qual": qualidade_cozinha,
        "Bsmt_Qual": qualidade_porao,
        "Central_Air": ar_condicionado,
        "Garage_Type": tipo_garagem,
        "Sale_Condition": condicao_venda,
    }

    erros = validar_entradas(dados)
    if erros:
        for erro in erros:
            st.error(erro)
        return

    entrada = pd.DataFrame([dados])
    preco = float(pipeline.predict(entrada)[0])

    st.divider()
    st.subheader("Resultado da estimativa")

    coluna_preco, coluna_modelo, coluna_metricas = st.columns([1.4, 1, 1])

    with coluna_preco:
        st.metric("Preço estimado de venda", formatar_moeda(preco))
        st.caption(
            f"Faixa de referência do erro médio: "
            f"{formatar_moeda(preco - modelo_info['metricas']['MAE'])} a "
            f"{formatar_moeda(preco + modelo_info['metricas']['MAE'])}"
        )

    with coluna_modelo:
        st.metric("Algoritmo utilizado", algoritmo)
        st.caption("Modelo selecionado na comparação de métricas.")

    with coluna_metricas:
        st.metric("R² no conjunto de teste", f"{modelo_info['metricas']['R2']:.4f}")
        st.caption(f"MAE: {formatar_moeda(modelo_info['metricas']['MAE'])}")

    st.subheader("Comparação dos modelos treinados")
    comparativo = modelo_info.get("metricas_comparativo", {})
    if comparativo:
        tabela = pd.DataFrame(
            [
                {
                    "Modelo": nome,
                    "MAE": formatar_moeda(valores["MAE"]),
                    "RMSE": formatar_moeda(valores["RMSE"]),
                    "R²": f"{valores['R2']:.4f}",
                }
                for nome, valores in comparativo.items()
            ]
        )
        st.dataframe(tabela, hide_index=True, width="stretch")
        st.caption(
            "MAE e RMSE medem o erro em dólares (quanto menor, melhor); "
            "R² indica a proporção da variação dos preços explicada pelo modelo "
            "(quanto mais próximo de 1, melhor)."
        )
    else:
        st.write("Arquivo de métricas comparativas não encontrado.")

    with st.expander("Ver características informadas"):
        resumo = pd.DataFrame([dados]).T.rename(columns={0: "Valor"})
        resumo["Valor"] = resumo["Valor"].astype(str)
        st.dataframe(resumo, width="stretch")

    st.caption(
        "Projeto educacional — Murilo Coelho · Machine Learning · "
        "Dataset: Ames Housing (De Cock, 2011), via OpenML."
    )


if __name__ == "__main__":
    main()
