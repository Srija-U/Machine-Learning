import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
data = pd.read_csv('Iris.csv')
x = data.iloc[:,].values
wcss = []


for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()
kmeans = KMeans(n_clusters = 2)
kmeans.fit(x)
pred=[[3.1,2.0,1.4,0.4]]
res = kmeans.predict(pred)
print("Result of a new input is:",res)