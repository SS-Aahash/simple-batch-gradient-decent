from sklearn.datasets import load_diabetes
x,y = load_diabetes(return_X_y=True)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


class GDRegressor:
    
    def __init__(self,learning_rate = 0.01,epochs = 100):
        self.coef = None
        self.intercept = None
        self.lr = learning_rate
        self.epochs = epochs 
    
    def fit(self,x_train,y_train):
        self.intercept = 0
        self.coef = np.ones(x_train.shape[1])

        for i in range(self.epochs):
            
            # updating intercept
            y_hat = np.dot(x_train,self.coef)+self.intercept
            intercept_der = -2*np.mean(y_train-y_hat)
            self.intercept = self.intercept-(self.lr*intercept_der)
            
            # updating coef
            coef_der = -2 * np.dot((y_train-y_hat),x_train)/x_train.shape[0]
            self.coef = self.coef-(self.lr*coef_der)
        
    def predict(self,x_test):
        return self.intercept+np.dot(x_test,self.coef)

grd = GDRegressor(epochs=100000,learning_rate=0.4)
grd.fit(x_train,y_train)

y_pred = grd.predict(x_test)
print(r2_score(y_test,y_pred))