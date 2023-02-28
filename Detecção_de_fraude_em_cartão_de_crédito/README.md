# Problemática

Os dados foram extraídos do conjunto de dados hospedado no site da kaggle, eles podem ser obtidos através do link a seguir
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Companhias que disponibilizam cartões de crédito precisam lidar com a tentativa de fraude, logo é comum o uso de modelos de inteligência artificial para tentar diminuir o problema. O dataset disponibilizado foi obtido de transações feitas em setembro de 2009 por donos de cartão de crédito na europa.

O conjunto de dados é desbalanceado, há 492 fraudes de um total de 284807 transações, ou seja, a classe positiva equivale a 0.172% dos dados. Os dados originais dos clientes não foram disponibilizados, o que foi disponibilizado é resultado de uma transformação de PCA.

# Técnicas de Resample

### TomekLinks

Tomek Links toma a distância entre um dado da classe minoritária e o seu par mais próxima da classe majoritária. Existe uma distância de threshold de modo que se a distância entre esses pares for menos, o dado da classe majoritária é eliminado, por padrão, também é possível modificar para apagar os dados da classe minoritária ou o par.

### SMOTETomek

SMOTE é o nome da técnica que cria dados artificiais da classe minoritária introduzindo ruído aos dados. SOMOTETomek aplica SMOTE e depois Tomek.

# Resultados
## Modelos de Machine Learning
### Tree
O modelo de Tree foi treinado com o uso do GridSearchCV para obtermos os melhores hiperparâmetros, a matriz de confusão a seguir foi obtida

![alt text](https://github.com/MMartins-ai/Portfolio_de_Ciencia_de_Dados/blob/main/Detec%C3%A7%C3%A3o_de_fraude_em_cart%C3%A3o_de_cr%C3%A9dito/imagens/Matriz_confusao_tree.png)

A primeira matriz de confusão foi obtida do modelo treinado com os dados que passaram po SMOTETomek, o modelo apresenta excelente acurácia, isso é comum em conjuntos desbalanceados, o modelo é muito bom em prever a classe majoritária, falha, porém, em prever quando há fraude. O recall obtido nesse primeiro modelo foi de 81,44%. Neste projeto tomaremos como desafio melhorar o recall, pois outras etapas podem ser aplicadas depois para filtrar falsos verdadeiros.

A segunda matriz corresponde ao modelo treinado no conjunto de dados que passaram por Tomek apenas. Ela obteve uma alta precisão, mas perdeu em recall.


*   Precisão: 83,33%
*   Recall: 72,16%

Como queremos melhorar o recall, o primeiro modelo de árvore é melhor.

### KNN

O modelo de KNeighbors também foi treinado com o uso do GridSearchCV para obtermos os melhores hiperparâmetros, a matriz de confusão a seguir foi obtida

![alt text](https://github.com/MMartins-ai/Portfolio_de_Ciencia_de_Dados/blob/main/Detec%C3%A7%C3%A3o_de_fraude_em_cart%C3%A3o_de_cr%C3%A9dito/imagens/matriz_confusaoo_knn.png)

Os dois fits do modelo foram superiores a árvore de decisão. O modelo para SMOTETomek foi muito superior a sua contraparte do modelo tree, o número de falsos positivos caiu de 460 para 37. O recall foi o mesmo para os dois modelos, porém o salto de qualidade na precisão do modelo torna o modelo de Kneighbors melhor no conjunto de dados SMOTETomek.

* Precisão: 68,10%
* Recall: 81,44%

O uso de TomekLinks ajudou o modelo a ter uma melhor precisão, dessa vez errou apenas 6 vezes quando disse que houve fraude. No geral o modelo performa bem, mas vamos tentar obter melhores recalls com redes neurais.

* Precisão: 92,3%
* Recall: 72,2%

## Modelos de Redes Neurais

### 1° Modelo
O primeiro modelo é uma rede neural simples com apenas 1 hidden layer e 16 neurônios, a seguri vamos analisar as métricas e matriz de confusão 

![alt text](https://github.com/MMartins-ai/Portfolio_de_Ciencia_de_Dados/blob/main/Detec%C3%A7%C3%A3o_de_fraude_em_cart%C3%A3o_de_cr%C3%A9dito/imagens/metricas_1_modelo.png)

Como podemos ver a loss diminui com o passar das épocas, comportamento normal já que queremos minimizar a loss. O importante a notar nesse gráfico é que a precisão do conjunto de dados de treino está muito próxima do 100%, enquanto no conjunto de validação está muito baixa, isso indica que, apesar do uso de camadas de dropout, o modelo sofreu com overfitting. Desse modo, seria adequado usar mais técnicas de regularização para melhorar o modelo.

![alt text](https://github.com/MMartins-ai/Portfolio_de_Ciencia_de_Dados/blob/main/Detec%C3%A7%C3%A3o_de_fraude_em_cart%C3%A3o_de_cr%C3%A9dito/imagens/matriz_confusao_1_modelo.png)

Pela matriz de confusão acima, a acurácia ficou alta, aproximadamente 99.91%, comportamento normal em conjunto de dados desbalanceados, portanto não deve ser considerado como uma boa métrica para medir o desempenho do modelo. A precisão e o recall estão quase empatados,  74.71% e 78.31%, respectivamente. Tentaremos melhorar o recall, em troca disso espera-se uma queda na precisão.

### 2° Modelo

O segundo modelo tem as mesmas especificações de camadas e neurônios, porém modificamos os pesos de cada classe, desse modo a classe que aparece menos tem um peso maior. A seguir estão as imagens geradas

![alt text](https://github.com/MMartins-ai/Portfolio_de_Ciencia_de_Dados/blob/main/Detec%C3%A7%C3%A3o_de_fraude_em_cart%C3%A3o_de_cr%C3%A9dito/imagens/metricas_2_modelo.png)

A loss está muito alta se comparada ao modelo anterior. O fato mais notório é a queda da precisão em ambos os conjuntos de dados, como modificamos os pesos do modelo, era esperado uma precisão baixa em troca do aumento do recall.

![alt text](https://github.com/MMartins-ai/Portfolio_de_Ciencia_de_Dados/blob/main/Detec%C3%A7%C3%A3o_de_fraude_em_cart%C3%A3o_de_cr%C3%A9dito/imagens/matriz_confusao_2_modelo.png)

Aqui tivemos um salto no recall em comparação com o modelo anterior de 78.31% para 87.95%, uma melhora considerável. Em troca dessa melhora a precisão despencou para 12.54%, mas não nós preucuparemos com isso, pois outras técnicas podem ser aplicadas para detectar outliers em quem foi detectado como classe 1, assim podemos separar quem realmente é fraude e quem não é.

### 3° Modelo

O terceiro modelo tem mais profundidade e mais neurônios 
  
```python   
def make_model(metrics = metricas, output_bias=None):
  if output_bias is not None:
    output_bias = keras.initializer.Constant(output_bias)

  model = keras.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(100, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(100, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(100, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(100, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(100, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid', bias_initializer = output_bias)
  ])
  ```
  
  A seguir estão as métricas e matriz de confusão do modelo
  
  ![alt text](https://github.com/MMartins-ai/Portfolio_de_Ciencia_de_Dados/blob/main/Detec%C3%A7%C3%A3o_de_fraude_em_cart%C3%A3o_de_cr%C3%A9dito/imagens/metricas_3_modo.png)
  
  O uso do earliestopping fez o modelo parar antes de completar as 100 épocas, pelo gráfico da loss o modelo não convergiu para uma loss menor ao longo das épocas. Esse não é um modelo confiável, não poderemos usá-lo como o melhor modelo. Seguirimos, porém, com a análise de outras métricas.
  
  ![alt text](https://github.com/MMartins-ai/Portfolio_de_Ciencia_de_Dados/blob/main/Detec%C3%A7%C3%A3o_de_fraude_em_cart%C3%A3o_de_cr%C3%A9dito/imagens/matriz_confusao_3_modelo.png)
  
  Esse foi o melhor modelo quando consideramos apenas o recall, porém essa rede não foi muito treinada, provavelmente essa é a rede com os pesos após o fim da primeira época. Ela apresenta baixa precisão, com o uso de outras técnicas, esse modelo poderia ser útil. Seria necessário buscar melhores formas de convergência para termos um modelo confiável.
