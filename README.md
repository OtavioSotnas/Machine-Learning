## ``Sumário``

- [1. Modificando Variáveis Categóricas]()
- [2. K-Nearest Kneighbors]()
- [3. Bernoulli Naive Bayes]()
- [4. DecisionTreeClassifier]()

Base de dados pode ser encontrado [aqui](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets).

### ``1. Modificando Variáveis Categóricas``

- **Manualmente**

```python
traducao_dic = {'Sim' : 1, 'Nao: 0}

dadosmodificados = dados.replace(traducao_dic)
```

- **Automatizada**

```python
dadosmodificados = pd.get_dummies(dados, axis=1)
```
