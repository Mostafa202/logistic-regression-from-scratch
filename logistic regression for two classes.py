
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("logistic.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


from sklearn.preprocessing import *

#en=LabelEncoder()
#x[:,1]=en.fit_transform(x[:,1])
m=MinMaxScaler()
x=m.fit_transform(x)

def sigmoid(theta,x):
    return 1.0/(1+np.exp(-np.dot(x,theta.T)))

def calc_val(theta,x,y):
    val=sigmoid(theta,x)
    first=val-y.reshape(len(y),1)
    second=np.dot(first.T,x)
    return second


def cost_func(theta,x,y):
    log_func_v = sigmoid(theta, x) 
    y = np.squeeze(y) 
    y=y.reshape(len(y),1)
    step1 = y*np.log(log_func_v)
    step2 = (1 - y) * np.log(1 - log_func_v) 
    final = -step1 - step2 
    

    return np.mean(final)


def grad(theta,x,y):
    change_cost=1
    cost=cost_func(theta,x,y)
    learning_rate=0.01
    i=1
    while(change_cost>0.00001 ):
        old_cost=cost
        val=calc_val(theta,x,y)
        theta=theta-(learning_rate*val)
        cost=cost_func(theta,x,y)
        change_cost=old_cost-cost
        i+=1
        
    return theta,i

X=np.append(np.ones((x.shape[0],1)),x,axis=1)
from sklearn.model_selection import *
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=0)

theta=np.random.rand(1,X.shape[1])

theta,num=grad(theta,train_x,train_y)
#           
get_val=np.dot(test_x,theta.T)
y_pred=np.where(get_val>=0.5,1,0)    



x0=test_x[np.where(test_y==0)]
x1=test_x[np.where(test_y==1)]

plt.scatter([x1[:,1]],[x1[:,2]],color="G")
plt.scatter([x0[:,1]],[x0[:,2]],color="B")


x1 = np.arange(0, 1, 0.1) 
x2 = -(theta[0,0] + theta[0,1]*x1)/theta[0,2]
plt.plot(x1, x2, c='k', label='reg line') 
plt.xlabel('input values')
plt.ylabel('predicted values')
plt.title('classification using logistic regression')
plt.show()

accuracy=np.sum(test_y.reshape(-1,1)==y_pred)/len(y_pred)*100
print('Accuracy: ',accuracy,' %')

