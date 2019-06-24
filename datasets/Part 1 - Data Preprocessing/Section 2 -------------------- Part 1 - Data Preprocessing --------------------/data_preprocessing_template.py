#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 11-17, 2019
@author: LMGS
"""
#--------------------------------------------------------#
#-------------- Plantilla de Pre Procesado---------------#
#--------------------------------------------------------#

#1.---------------Importamos las librerias
#Importamos las librerías basicas
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

#2.---------------Importar el data set
dataset = pd.read_csv('Data.csv')
#Generamos una matriz de las variables independientes
X = dataset.iloc[:, :-1].values

#3.---------------Tratamos los datos, los limpiamos
#Importamos la libreria para el tratamiento de datos NaN
#Generamos un vector de las variables dependientes
y = dataset.iloc[:, 3].values
#[fila , columnas] -> [1:2,2:4]

from sklearn.preprocessing import Imputer
#En python axis = 0 se refiere a la columna, axis = 1 se refiere a la fila
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

#fit se usa para aplicar una funcion a un objeto
#limite superior en python no se toma
imputer = imputer.fit(X[:,1:3])
#Sobreescribimos los valores de nuestra matriz X 
X[:,1:3] = imputer.transform(X[:,1:3])

#4.---------------Tratamos los datos, codificar datos categoricos
#Los datos categoricos a diferencia de los ordinales no son comparables.
#VariableDummy o OneHotVector es decir, puramente categorica
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y =  LabelEncoder()
y = labelencoder_y.fit_transform(y)

#5.------Dividir el data set en conjunto de entrenamiento y conjunto de testing
#OVERFITTING. Nuestro algoritmo se aprende los datos de memoria y es un 
#problema que debemos tratar de evitar
from sklearn.model_selection import train_test_split
#test_size define el porcentaje del dataset que sera usado para el testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#6.---------------Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)