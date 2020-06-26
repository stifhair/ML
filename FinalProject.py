from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
import pandas as pd
wine = pd.read_csv('./train.csv')

print('<< train>>')
#pre_processing
def pre(dataset):
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    X=dataset.drop(['label'],axis=1)
    y=dataset['label']
    X=X.values
    y = labelencoder.fit_transform(y)
    return X,y
def split(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train , X_test , y_train, y_test


X,y =pre(wine)


from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import numpy as np
import time
class PCA_preprocessing:
    def __init__(self ,X, n):
        self.n=n
        self.X=X
        # PCA
        self.pca = PCA(n_components=n)
        self.PCA_time = time.time()
        self.X_PCA = self.pca.fit_transform(X)
        self.PCA_time = time.time() - self.PCA_time
        print('{0} PCA VAR {1}: '.format(n,self.pca.explained_variance_ratio_))



        # IncrementalPCA
        self.n_batches = 2
        self.inc_pca = IncrementalPCA(n_components=n)
        self.IPCA_time = time.time()
        for X_batch in np.array_split(X, self.n_batches):
            self.inc_pca.partial_fit(X_batch)
        self.X_IPCA = self.inc_pca.transform(X)
        self.IPCA_time = time.time() - self.IPCA_time
        print('{0} IPCA VAR {1}: '.format(n,self.pca.explained_variance_ratio_))

        # Randomized PCA
        self.rnd_pca = PCA(n_components=n, svd_solver='randomized')
        self.RPCA_time = time.time()
        self.X_RPCA = self.rnd_pca.fit_transform(X)
        self.RPCA_time = time.time() - self.RPCA_time
        print('{0} RPCA VAR {1}: '.format(n,self.pca.explained_variance_ratio_))

    def getTime(self):
        print('PCA fit_transform time : ', self.PCA_time)
        print('IPCA fit_transform time : ', self.IPCA_time)
        print('IPCA fit_transform time : ', self.RPCA_time)
        print('PCA VAR : ', self.PCA_var)
        print('IPCA VAR : ', self.IPCA_var)
        print('IPCA VAR : ', self.RPCA_var)


    def getX(self):
        return self.X_PCA , self.X_IPCA , self.X_RPCA


class model:
    def __init__(self,X_train,X_test,y_train,y_test):
        import time
        self.X = X_train
        self.X_test = X_test
        self.y = y_train
        self.y_test = y_test

        self.sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42, loss='log')
        self.sgd_time = time.time()
        self.sgd_clf.fit(self.X, self.y)
        self.sgd_time = time.time() - self.sgd_time


        self.knn_clf = KNeighborsClassifier(n_neighbors=2)
        self.knn_time = time.time()
        self.knn_clf.fit(self.X, self.y)
        self.knn_time = time.time() - self.knn_time


        self.tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        self.tree_time = time.time()
        self.tree_clf.fit(self.X, self.y)
        self.tree_time = time.time() - self.tree_time


        self.mlp_clf = MLPClassifier()
        self.mlp_time = time.time()
        self.mlp_clf.fit(self.X, self.y)
        self.mlp_time = time.time() - self.mlp_time

    def getScore(self,model):
        from sklearn.metrics import accuracy_score
        y_score = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_score)
        return accuracy

    def printScore(self):
        print('SGD classifier Accuracy : {}'.format(self.getScore(self.sgd_clf)))
        print('KNN classifier Accuracy : {}'.format(self.getScore(self.knn_clf)))
        print('Decision Tree classifier Accuracy : {}'.format(self.getScore(self.tree_clf)))
        print('MLP classifier Accuracy : {}'.format(self.getScore(self.mlp_clf)))
        print()

    def letTest(self):
        print('<Test data>')
        print('input data : ', self.X_test[13])
        print('output data : ', self.y_test[13])
        print('<Prediction>')
        print('SGD Prediction : ',self.sgd_clf.predict([self.X_test[13]]))
        print('KNN Prediction : ',self.knn_clf.predict([self.X_test[13]]))
        print('Decision Tree Prediction : ',self.tree_clf.predict([self.X_test[13]]))
        print('MLP Prediction : ', self.mlp_clf.predict([self.X_test[13]]))
        print()

    def getTime(self):
        print('<Training Time>')
        print('SGD Classifier : {}'.format(self.sgd_time))
        print('knn Classifier : {}'.format(self.knn_time))
        print('tree Classifier : {}'.format(self.tree_time))
        #print('SVC Classifier : {}'.format(self.svc_time))
        print('MLP Classifier : {}'.format(self.mlp_time))
        print()


def printAll(X,y):
    X_train, X_test, y_train, y_test = split(X, y)
    p_model = model(X_train, X_test, y_train, y_test)
    p_model.printScore()
    p_model.letTest()
    p_model.getTime()
    print('=====================================================================')
    print()


def PCA_print(X,y,n):
    pca = PCA_preprocessing(X, n)
    X_PCA, X_IPCA, X_RPCA = pca.getX()
    print('<PCA n_components = {}>'.format(n))
    printAll(X_PCA,y)
    print('<IPCA n_components = {}>'.format(n))
    printAll(X_IPCA, y)
    print('<RPCA n_components = {}>'.format(n))
    printAll(X_RPCA, y)

#To use grid search
def grid_kernelPCA(model,n):
    clf = Pipeline([
        ('kpca', KernelPCA(n_components=n)),
        ('reg',model)
    ])

    param = [{
        'kpca__gamma' : np.linspace(0.03,0.05,10),
        'kpca__kernel' : ['rbf','sigmoid']
    }]
    grid_search = GridSearchCV(clf, param_grid=param, cv=5)
    grid_search.fit(X,y)
    return grid_search

#original data
print('<< model : Original data >>')
printAll(X,y)

#n_components = 2
PCA_print(X,y,2)
#n_components = 4
PCA_print(X,y,4)


from sklearn.metrics import mean_squared_error
def getMSE(pca):
    X_reduced = pca.fit_transform(X)
    X_preimage = pca.inverse_transform(X_reduced)
    print(mean_squared_error(X,X_preimage))
