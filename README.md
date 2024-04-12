## ``Sumário``

- [1. Primeiros Passos]()
  - [1.1 Transformar valores categóricos em binários]()
  - [1.2 Tratamento para dados desbalanceados]()
  - [1.3 Padronização dos Dados]()
  - [1.4 Divisão em treino e teste]()
    
- [2. K-Nearest Kneighbors]().  
- [3. Bernoulli Naive Bayes]()
- [4. Decision Tree Classifier]()
- [5. Validação dos modelos]()

Base de dados pode ser encontrado [aqui](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets).

## ``1. Primeiros Passos``

- **1.1 Transformar valores categóricos em binários:**

```python
# Manualmente
traducao_dic = {'Sim' : 1, 'Nao: 0}
dados = dados.replace(traducao_dic)

# Automatizado
dados = pd.get_dummies(dados, axis=1)
```

- **1.2 Tratamento para dados desbalanceados**
```python
ax = sns.countplot(x='Churn', data=dados_final)
```
![image](https://github.com/OtavioSotnas/Machine-Learning/assets/142911747/83f776ae-a122-4632-bcf2-9a672804f988)

**Over Sampling com SMOTE**
  
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

- **1.3 Padronização dos Dados**
```python
# Devemos deixar todos valores na mesma escala
from sklearn.preprocessing import StandardScaler

norm = StandardScaler()
X_normalizado = norm.fit_transform(X)
```

- **1.4 Divisão em treino e teste**
```python
# Biblioteca para divisão dos dados
from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(X_normalizado, y, test_size=0.3, random_state=123)
```

## ``2. K-Nearest Kneighbors``
![image](https://github.com/OtavioSotnas/Machine-Learning/assets/142911747/45437577-b5b6-4721-b6c8-c85b93d5c76a)

```python
# Biblioteca para criarmos o modelo de machine learning
from sklearn.neighbors import KNeighborsClassifier

# Instanciar o modelo (criamos o modelo) - por padrão são 5 vizinhos  
knn = KNeighborsClassifier(metric='euclidean')

# Treinando o modelo com os dados de treino
knn.fit(X_treino, y_treino)

# Testando o modelo com os dados de teste
predicao_knn = knn.predict(X_teste)
```
