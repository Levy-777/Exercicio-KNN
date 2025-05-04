import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

""""
exercicio: gerar 5 pontos fixos e através do KNN criar 10 pontos aleatorios que ficam com a mesma cor que os pontos fixos mais proximos
"""""

dados = {
    'A': [100, 900, 500, 100, 900],  
    'B': [100, 100, 500, 900, 900],
    'Y' : [1,2,3,4,5] 
}

fixos = pd.DataFrame(dados)


rand = {
    'A' : np.random.randint(0, 1000,30),
    'B' : np.random.randint(0, 1000,30)
}

aleatorios = pd.DataFrame(rand)


KNN = KNeighborsClassifier(n_neighbors=1)  
KNN.fit(fixos[['A', 'B']], fixos['Y'])

aleatorios['Y_pred'] = KNN.predict(aleatorios[['A', 'B']])
print(aleatorios[['A', 'B', 'Y_pred']])


plt.figure(figsize=(12, 5))


scatter1 = plt.scatter(fixos.A, fixos.B, c=fixos.Y, s=100, cmap='RdYlBu', edgecolors='k', label="Pontos Fixos")


scatter2 = plt.scatter(aleatorios.A, aleatorios.B, c=aleatorios.Y_pred, s=100, marker='s', cmap='RdYlBu', edgecolors='k', label="Pontos Aleatórios")

plt.legend()
plt.show()