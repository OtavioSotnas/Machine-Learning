## ``Sumário``

- [1. Modificando Variáveis Categóricas]()
- [2. K-Nearest Kneighbors]()
- [3. Bernoulli Naive Bayes]()
- [4. DecisionTreeClassifier]()

Base de dados pode ser encontrado [aqui](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets).

## ``1. Primeiros Passos``

- **Devemos transformar todos valores categóricos em binários:**

```python
# Manualmente
traducao_dic = {'Sim' : 1, 'Nao: 0}
dados = dados.replace(traducao_dic)

# Automatizado
dados = pd.get_dummies(dados, axis=1)
```

- **Tratamento para dados desbalanceados**
```python
ax = sns.countplot(x='Churn', data=dados_final)
```
![image](https://github.com/OtavioSotnas/Machine-Learning/assets/142911747/83f776ae-a122-4632-bcf2-9a672804f988)

- **Over Sampling com SMOTE**
  
```python
# Cria observações intermediárias entre os dados próximos
from imblearn.over_sampling import SMOTE

# Dividindo os dados em caracteristicas e target
X = dados.drop('Churn', axis = 1)
y = dados['Churn']

smt = SMOTE(random_state=123)
X, y = smt.fit_resample(X, y)

# Junção dos dados balanceados
dados = pd.concat([X, y], axis=1)
ax = sns.countplot(x='Churn', data=dados)
```
![image](https://github.com/OtavioSotnas/Machine-Learning/assets/142911747/330a4e19-5af8-4317-a4c8-019809997d86)
