import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
#Number of times pregnant (preg)
#Plasma glucose concentration a 2 hours in an oral glucose tolerance test (plas)
#Diastolic blood pressure in mm Hg (pres)
#Triceps skin fold thickness in mm (skin)
#2-Hour serum insulin in mu U/ml (insu)
#Body mass index measured as weight in kg/(height in m)^2 (mass)
#Diabetes pedigree function (pedi)
#Age in years (age)
diabetes = pd.read_csv('diabetes.csv')
data = pd.read_csv('diabetes1.csv',header=None)
data[[0,1,2,3,4,5,6,7]] = data[[0,1,2,3,4,5,6,7]].replace({0:np.NaN})
X = data.iloc[:, :-1].values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)
print(diabetes.head())
print(diabetes.groupby('Outcome').size())
pred=[[7,107,74,0,0,29.6,0.254,31]]
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'],test_size=0.2, stratify=diabetes['Outcome'], random_state=66)
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.title("KNN GRAPH")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')
res=knn.predict(pred)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
print("The Predicted Result of K-NN classifier on new input: ",res)

#Logistic Regression
logreg = LogisticRegression().fit(X_train, y_train)
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
res=logreg100.predict(pred)
print("Accuracy of Logistic Regression on training set: {:.2f}".format(logreg.score(X_train, y_train)))
print("Accuracy of Logistic Regression on testing set: {:.2f}".format(logreg.score(X_test, y_test)))
print("The Predicted Result of Logistic Regression on new input: ",res)
diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
plt.figure(figsize=(8,6))
plt.title("LOGISTIC REGRESSION GRAPH")
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(diabetes.shape[1]), diabetes_features, rotation=90)
plt.hlines(0, 0, diabetes.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.savefig('log_coef')

#Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)
res=tree.predict(pred)
print("Accuracy of Decision tree on training set: {:.2f}".format(tree.score(X_train, y_train)))
print("Accuracy of Decision tree on testing set: {:.2f}".format(tree.score(X_test, y_test)))
print("The Predicted Result of Decision tree on new input: ",res)
def plot_feature_importances_diabetes(model,tit):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(tit)
    plt.ylim(-1, n_features)
plot_feature_importances_diabetes(tree,"DECISION TREE GRAPH")
plt.savefig('feature_importance_DT')

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
res=rf.predict(pred)
print("Accuracy of Random Forest on training set: {:.2f}".format(rf.score(X_train, y_train)))
print("Accuracy of Random Forest on test set: {:.2f}".format(rf.score(X_test, y_test)))
print("The Predicted Result of Random Forest on new input: ",res)
plot_feature_importances_diabetes(rf,"RANDOM FOREST GRAPH")
plt.savefig('feature_importance_RF')

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)
res=gb.predict(pred)
print("Accuracy of Gradient Boosting on training set: {:.2f}".format(gb.score(X_train, y_train)))
print("Accuracy of Gradient Boosting on test set: {:.2f}".format(gb.score(X_test, y_test)))
print("The Predicted Result of Gradient Boosting on new input: ",res)
plot_feature_importances_diabetes(gb,"GRADIENT BOOSTING GRAPH")
plt.savefig('feature_importance_GB')

#Kmeans clustering
kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: diabetic or Not diabetic patient
kmeans.fit(X)
res=kmeans.predict(pred)
print("The Predicted Result of Kmeans clustering on new input: ",res)
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