# Problemática

Os dados foram extraídos do conjunto de dados hospedado no site da kaggle, eles podem ser obtidos através do link a seguir
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Companhias que disponibilizam cartões de crédito precisam lidar com a tentativa de fraude, logo é comum o uso de modelos de inteligência artificial para tentar diminuir o problema. O dataset disponibilizado foi obtido de transações feitas em setembro de 2009 por donos de cartão de crédito na europa.

O conjunto de dados é desbalanceado, há 492 fraudes de um total de 284807 transações, ou seja, a classe positiva equivale a 0.172% dos dados. Os dados originais dos clientes não foram disponibilizados, o que foi disponibilizado é resultado de uma transformação de PCA.

# Tecnicas de Resample

### TomekLinks

Tomek Links toma a distância entre um dado da classe minoritária e o seu par mais próxima da classe majoritária. Existe uma distância de threshold de modo que se a distância entre esses pares for menos, o dado da classe majoritária é eliminado, por padrão, também é possível modificar para apagar os dados da classe minoritária ou o par.

### SMOTETomek

SMOTE é o nome da técnica que cria dados artificiais da classe minoritária introduzindo ruído aos dados. SOMOTETomek aplica SMOTE e depois Tomek.

# Resultados
## Modelos de Machine Learning
# Tree
O modelo de Tree foi treinado com o uso do GridSearchCV para obtermos os melhores hiperparâmetros, a matriz de confusão a seguir foi obtida
