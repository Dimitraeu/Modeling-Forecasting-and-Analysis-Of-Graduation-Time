# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 10:23:16 2018

@author: Dimitra-Spuridoula
"""
#SVR
#Linear, OLS      

import pandas as pd

#einai gia to cross  validation
from kfold import *
#mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

# montela
from sklearn.linear_model import LinearRegression
# SVC,SVR,LinearRegression, BayesianRidge,Ridge,RidgeCV,Lasso,Perceptron
from sklearn.svm import SVC,SVR
from numpy.polynomial import polynomial 
from sklearn.preprocessing import PolynomialFeatures

import numpy as np

#gia to plot
import matplotlib.pyplot as plt 
from pandas.tools.plotting import scatter_matrix
#from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

from datetime import datetime 


import seaborn as sb



'''
kanonikopoihsh twn dedοmenwn
Vazoume panw  thn synarthsh,giati ta dedomena pou tha parei theloume 
na treksoun apo thn arxh kanonikopoihmena
'''

#dhlwseis metavlitwn
a=0;
b=1;
#synarthsh kanonikopoihshs
def Transform(X,a,b):
    nx=a+(b-a)*(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0)) #prostethike to axis=0 
    return nx
    
  

#Diavase ta dedomena 
df=pd.read_excel("C:\\Users\\Dimitra-Spuridoula\\Desktop\\ptyxiakh\\new_file_for_data\\data\\data_corrected_dimitra.xlsx", sheet_name = 'data',header=1,sep=' ')

#edw parnoume tis grammes pou den einai kenes
#df = df.dropna()
#df=df.reshape(3,5)


#h ejodos einai y1 kai oxi y

df=df[ ['x1','x2','x3','x4','x5','x6','x7','x9','x10','y1'] ]
#metonomazoyme thn sthlh y1 se y
df.rename(columns={'y1':'y'}, inplace=True)
#check the rename
print(df.columns)

#edw sthn sthlh y leme oti theloume apo 4 ews 10
df = df.query('y <=10 and y >=4')

a = 0;
b = 1;

lst=['x5']

#value mas dinei thn timh
#epilegoume mono tis times X poy yparxoyn sti lista
X = df[lst].values

#X.shape=(-1,1) # if one variable is considered

#to .loc pairnei apo kapoio eurhthrio mia etiketa mono
Y = df.loc[:,'y'].values

#normalize inputs only
# to axis=0 xreiazetai gia den ypologizei to min,max


x=Transform(X,a,b)
y=Y #uetoyme os y to Y
 


nfolds=5
kf = mykfold(x, nfolds)

#exei dhmiourghthei mia kenh lista gia na mpoun ta apotelesmata
cvscore=[]

for i in range(nfolds):
    x_trn,y_trn,x_tst,y_tst=kf.get_data_fold(i, x, y)
    
    
    
    
    
    #Xrhsiopoieitai to montelo polynomial  
   # model=PolynomialFeatures(degree=2)
    #poly_features=model.fit_transform(x_trn)
    #poly_regression=LinearRegression()
    #poly_regression.fit(poly_features,y_trn)
    #yyhat=PolynomialFeatures.transform(x_tst,y_tst)
    
    #model=LinearRegression()
    #poly=PolynomialFeatures(2)
    #x_transform=poly.fit_transform(x_trn)
    #model.fit(x_transform,y_trn)
   # y_preds=model.predict(x_tst)
   
   
   
   # poly=PolynomialFeatures(degree=2)
    #x_new=poly.fit_transform(x_tst)
    #model=LinearRegression()
    #model.fit(x_new,y_tst)
    #yhat=model.predict(x_new)
    ''' 
    lin_regressor=LinearRegression()
    poly=PolynomialFeatures(degree=2)
    
    x_transform=poly.fit_transform(x_trn)
    lin_regressor.fit(x_transform,y_trn)
    yyhat=PolynomialFeatures.transform(x_tst)
    yhat=lin_regressor.predict(yyhat)
    '''



# to alpha meiwnei thn diakymansh twn ektimhsewn
#edw xrhsimopoiountai ta montela ridge,lasso,linearRegression,SVR
    #gia svr kernel=poly,linear,rbf
    
    model=SVR(C=2.0,kernel='rbf',gamma=10.0)
    #model=LinearRegression()
    #model=Ridge(alpha=0.001)
    #model=Lasso(alpha=0.001)
    model.fit(x_trn,y_trn)
   


    
    
    #upologizei tin eksodo tou montelou gia ta dedomena elegxou
    yhat=model.predict(x_tst)
   # yyhat=PolynomialFeatures.transform(poly_features,x_tst)
    #yhat=poly_regression.predict(x_tst)
    


#edw ypologizetai to meso apolyto sfalma(mae),meso tetragwniko sfallma(mse),(rmse)  
    
    mae=mean_absolute_error(yhat,y_tst)
    #rmse = sqrt(mse)
    cvscore.append(mae)



print( 'crosval score: ', cvscore )

e=np.array(cvscore)
#emfanizei to sfalmata
print('Error(Years): ','min: ',e.min(),'max: ',e.max(),'mean: ',e.mean(),u"\u00B1",0.5*e.std())

#print ("se posa exw pesei eksw",e.sum())
#emfanise to meso oro sfamatos
print(lst,'Error(Years): -->',e.mean())


