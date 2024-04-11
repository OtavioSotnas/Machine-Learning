## ``Sumário``

- [1. Modificando Variáveis Categóricas]()
- [2. K-Nearest Kneighbors]()
- [3. Bernoulli Naive Bayes]()
- [4. DecisionTreeClassifier]()

Base de dados pode ser encontrado [aqui](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets).

## ``1. Modificando Variáveis Categóricas``

- **Manualmente**

```python
traducao_dic = {'Sim' : 1, 'Nao: 0}

dados = dados.replace(traducao_dic)
```

- **Automatizada**

```python
dados = pd.get_dummies(dados, axis=1)
```
### 1.2 Balanceando dados

- **Verificando balanceamento dos dados**
```python
ax = sns.countplot(x='Churn', data=dados_final)
```
![image](https://github.com/OtavioSotnas/Machine-Learning/assets/142911747/83f776ae-a122-4632-bcf2-9a672804f988)

- **Over Sampling com SMOTE**
  
Ele cria observações intermediárias entre os dados próximos
```python
from imblearn.over_sampling import SMOTE

#dividindo os dados em caracteristicas e target
X = dados.drop('Churn', axis = 1)
y = dados['Churn']

smt = SMOTE(random_state=123)
X, y = smt.fit_resample(X, y)

#junção dos dados balanceados
dados = pd.concat([X, y], axis=1)
```

- **Verificando novamente**
```python
ax = sns.countplot(x='Churn', data=dados)
```
![image](https://github.com/OtavioSotnas/Machine-Learning/assets/142911747/330a4e19-5af8-4317-a4c8-019809997d86)
