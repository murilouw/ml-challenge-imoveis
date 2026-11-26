# Dataset — Ames Housing

## Identificação

| Item | Informação |
|---|---|
| **Nome** | Ames Housing Dataset |
| **Autor** | Dean De Cock |
| **Publicação** | De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*. Journal of Statistics Education, 19(3). |
| **Fonte utilizada** | OpenML — `data_id = 41211` — <https://www.openml.org/d/41211> |
| **Licença declarada** | *Public* (OpenML) — uso livre para fins educacionais e de pesquisa |
| **Localidade** | Ames, Iowa, Estados Unidos |
| **Período das vendas** | 2006 a 2010 |
| **Registros** | 2.930 imóveis |
| **Colunas** | 81 (80 atributos + variável-alvo) |
| **Variável-alvo** | `Sale_Price` — preço de venda em **dólares americanos (USD)** |
| **Arquivo local** | `data/ames_housing.csv` (UTF-8, separador vírgula, ~1 MB) |

> Os valores monetários estão em dólares da época das vendas (2006–2010). O projeto
> **não realiza conversão cambial**; todas as previsões são apresentadas em USD.

## Citação sugerida

> De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of
> Semester Regression Project. *Journal of Statistics Education*, 19(3).

## Como o arquivo é obtido

O CSV é baixado automaticamente pelo `train.py` na primeira execução:

```bash
python train.py
```

A função `carregar_dados()` tenta, nesta ordem:

1. Ler `data/ames_housing.csv`, se já existir;
2. Baixar via `sklearn.datasets.fetch_openml(data_id=41211)`;
3. Baixar o arquivo Parquet oficial do OpenML
   (`https://data.openml.org/datasets/0004/41211/dataset_41211.pq`);
4. Falhar com mensagem explicando como obter o arquivo manualmente.

O CSV fica **versionado no repositório** para que o projeto funcione offline e de forma
reprodutível, sem depender da disponibilidade da rede.

## Particularidades desta versão

A versão do OpenML difere do arquivo original de De Cock em dois pontos, e ambos foram
verificados antes do início do projeto:

1. **Rótulos decodificados.** As variáveis categóricas vêm escritas por extenso em vez
   dos códigos originais: `Overall_Qual` traz `Very_Poor … Very_Excellent` (em vez de
   `1 … 10`), `Bsmt_Qual` traz `No_Basement` (em vez de `NA`), e assim por diante.
   O pré-processamento converte as variáveis ordinais de volta para escala numérica
   1–10 antes de qualquer cálculo.
2. **Valores ausentes já tratados na origem.** A ausência de porão, garagem, piscina ou
   lareira foi convertida em categoria explícita (`No_Basement`, `No_Garage`,
   `No_Pool`, `No_Fireplace`). Restam valores nulos apenas em `Misc_Feature` (2.824) e
   `Mas_Vnr_Type` (1.775), colunas que **não são utilizadas** no modelo. Ainda assim, o
   pipeline mantém etapas de imputação por robustez.

O OpenML marca esta versão como *inactive* por causa dessa curadoria, mas a contagem de
registros (2.930) e a variável-alvo conferem com a publicação original.

## Dicionário das variáveis utilizadas

O projeto usa um subconjunto de 23 atributos. Os nomes originais (em inglês) são
mantidos no código para evitar erros de renomeação; a tradução aparece na documentação
e na interface.

### Numéricas

| Coluna | Significado | Unidade |
|---|---|---|
| `Gr_Liv_Area` | Área construída acima do solo | pés² |
| `Total_Bsmt_SF` | Área total do porão | pés² |
| `First_Flr_SF` | Área do primeiro pavimento | pés² |
| `Lot_Area` | Área do terreno | pés² |
| `Year_Built` | Ano de construção | ano |
| `Year_Remod_Add` | Ano da última reforma (igual à construção se nunca reformado) | ano |
| `Overall_Qual` | Qualidade geral dos materiais e acabamento | 1 a 10 (convertido de rótulo) |
| `Overall_Cond` | Estado geral de conservação | 1 a 10 (convertido de rótulo) |
| `Garage_Cars` | Capacidade da garagem | nº de carros |
| `Garage_Area` | Área da garagem | pés² |
| `Full_Bath` | Banheiros completos acima do solo | unidade |
| `Bedroom_AbvGr` | Quartos acima do solo | unidade |
| `TotRms_AbvGrd` | Total de cômodos acima do solo (exclui banheiros) | unidade |
| `Fireplaces` | Número de lareiras | unidade |

### Categóricas

| Coluna | Significado | Categorias |
|---|---|---|
| `Neighborhood` | Bairro dentro de Ames | 28 bairros |
| `Bldg_Type` | Tipo de edificação | OneFam, TwoFmCon, Duplex, Twnhs, TwnhsE |
| `House_Style` | Estilo da casa | One_Story, Two_Story, One_and_Half_Fin, One_and_Half_Unf, Two_and_Half_Fin, Two_and_Half_Unf, SFoyer, SLvl |
| `Exter_Qual` | Qualidade do revestimento externo | Excellent, Good, Typical, Fair |
| `Kitchen_Qual` | Qualidade da cozinha | Excellent, Good, Typical, Fair, Poor |
| `Bsmt_Qual` | Altura/qualidade do porão (`No_Basement` = sem porão) | Excellent, Good, Typical, Fair, Poor, No_Basement |
| `Central_Air` | Ar-condicionado central | Y, N |
| `Garage_Type` | Tipo de garagem (`No_Garage` = sem garagem) | Attchd, BuiltIn, Basment, Detchd, CarPort, More_Than_Two_Types, No_Garage |
| `Sale_Condition` | Condição da venda | Normal, Abnorml, AdjLand, Alloca, Family, Partial |

### Conversão das variáveis ordinais

| Rótulo | Valor |
|---|---|
| `Very_Poor` | 1 |
| `Poor` | 2 |
| `Fair` | 3 |
| `Below_Average` | 4 |
| `Average` | 5 |
| `Above_Average` | 6 |
| `Good` | 7 |
| `Very_Good` | 8 |
| `Excellent` | 9 |
| `Very_Excellent` | 10 |

## Inconsistências identificadas

- **Vendas parciais:** 3 imóveis com `Gr_Liv_Area > 4000` pés² vendidos muito abaixo do
  padrão de mercado, todos com `Sale_Condition = Partial` (a casa não estava concluída
  no momento da venda). São removidos no pré-processamento.
- **Variáveis fora do escopo:** `Latitude`, `Longitude`, `Order`/índices e os
  identificadores não são utilizados no modelo.
