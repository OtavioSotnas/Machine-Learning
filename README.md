## ``Sumário``

- [Modificando Variáveis Categóricas]()
- [K-Nearest Kneighbors]()
- [Bernoulli Naive Bayes]()
- [DecisionTreeClassifier]()

### ``Modificando Variáveis Categóricas``

**Manualmente**
´´´python
traducao_dic = {'Sim' : 1, 'Nao: 0}

dadosmodificados = dados.replace(traducao_dic)
´´´

**Automatizada**
´´´python
dadosmodificados = pd.get_dummies(dados, axis=1)
´´´
