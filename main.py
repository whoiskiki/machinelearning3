import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


#Чтение базы данных о покупателях  интернет магазина N

table = pd.read_csv('data.csv', index_col='DEPTH_MD')
table.dropna(inplace=True)

print(table.head())
#x=table.values[:,1:]

#Стандартизуем данные с мат.ожиданиим 0 и сред. отклонением 1
scaler = StandardScaler()
print(table.describe())

table[['RHOB_T', 'NPHI_T', 'GR_T', 'PEF_T', 'DTC_T']] = scaler.fit_transform(table[['RHOB', 'NPHI', 'GR', 'PEF', 'DTC']])
print(table.head())


#Elbow method - метод для подсчета оптимального числа кластеров
def optimal_k(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(data)
        means.append(k)
        inertias.append(kmeans.inertia_)

    fig = plt.subplots(figsize=(10,5))
    plt.plot(means, inertias,'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


#Метод к-средних!
optimal_k(table[['RHOB_T', 'NPHI_T']], 10)
kmeans = KMeans(init='k-means++', n_clusters=3)
kmeans.fit(table[['NPHI_T', 'RHOB_T']])

labels = kmeans.labels_
print(labels)

table['clusters'] = labels
print(table.head(10))

#Визуализация результатов
#area = np.pi * (x[:,1])**2
#,  x[:, 0], x[:, 3], s=area, c=labels.astype(float), alpha=0.5
plt.scatter(x=table['NPHI'], y=table['RHOB'], c=table['clusters'])
plt.xlabel('NPHI')
plt.ylabel('RHOB')
plt.xlim(-0.1,1)
plt.ylim(3, 1.5)
plt.show()

#fig = plt.figure(1, figsize=(8, 6))
#plt.clf()
#ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

#ax.set_xlabel('Age')
#ax.set_ylabel('Income')
#ax.set_zlabel('Education')

#ax.scatter()
