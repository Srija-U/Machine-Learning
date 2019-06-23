from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
data = pd.read_csv('data.csv')
X = data.iloc[:, :-1].values
y=data['Outcome'].values
pred=[[61,0.7,0.2,145,53,41,5.8,2.7,0.87]]
#data[[0,1,2,3,4,5,6,7,8]] = data[[0,1,2,3,4,5,6,7,8]].replace({0:np.NaN})
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=data['Outcome'], random_state=66)
kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: diabetic or Not diabetic patient
kmeans.fit(X)
res=kmeans.predict(pred)
print("Output of given input is ")
print(res)
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
plt.figure(figsize=(8,6))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()