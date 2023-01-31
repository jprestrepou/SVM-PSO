# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:39:56 2019

@author: andresorozco
"""
import numpy as np
import numpy.matlib as ll
from sklearn import svm
from math import ceil
from time import time
from sklearn.preprocessing import Normalizer
from random import randrange
from PSO_CLASS import *

names = ["monk-2.dat"]
Typ = ['Mirror']    #'Mirror'


for data in names:  
    print('Se esta analizanzo la base de datos ' + data) 
    DATA= data  
    X,Y = readDataBase(DATA)
    scaler=Normalizer().fit(X)
    X=scaler.transform(X)
    m,n = X.shape
    k_folds=10
    Index=ll.repmat(np.asarray(range(k_folds)),1,ceil(m/k_folds))
    Sorteo=np.random.permutation(m)
    Index=Index[:,Sorteo]
    Adicional={'X':X,'Y':Y,'Index':Index.T}
    Lim=np.array([[0.00001,1000],[0.0001,10]])
    
    for Type in Typ:
        tiempo_inicial = time()
        print('Se esta analizanzo la base de datos ' + data + ' mediante ' + Type)
        Modelo_1=PSO(30,Lim,2,2,0.95,svmOpFun,Data=Adicional,Topology=(3,3,3))
        Pos=np.empty((10,2))
        Fun=np.empty((10,1))
        for i in range(10):
            Pos[i,:],Fun[i,0] = Modelo_1.runPSO(500,1e-5,0.6,tipo=Type)  
            
        #print(Pos,Fun)
        tiempo_final = time() 
        tiempo_ejecucion = tiempo_final - tiempo_inicial
         
        print ('El tiempo de ejecucion fue: ' , tiempo_ejecucion)
        
        directorio=('C:/Users/andresorozco/OneDrive - Instituto Tecnológico Metropolitano/Juan Pablo/150')
        np.savez(directorio + 'accuracy_150_newman'+ '_' + Type +'_' + DATA[0:-4], Pos = Pos, Fun = Fun, tiempo_ejecucion = tiempo_ejecucion)
        print('SE ALMACENARON LOS DATOS')
        print ('--------------------------------------------------')
        
        tiempo_inicial = time()
        Modelo_2=PSO(30,Lim,2,2,0.95,svmOpFun,Data=Adicional)
        Pos=np.empty((10,2))
        Fun=np.empty((10,1))
        for i in range(10):
            Pos[i,:],Fun[i,0] = Modelo_2.runPSO(500,1e-5,0.6,tipo=Type)  
            
        #print(Pos,Fun)
        tiempo_final = time() 
        tiempo_ejecucion = tiempo_final - tiempo_inicial
         
        print ('El tiempo de ejecucion fue: ' , tiempo_ejecucion)
        
        directorio=('C:/Users/andresorozco/OneDrive - Instituto Tecnológico Metropolitano/Juan Pablo/150')
        np.savez(directorio + 'accuracy_150_'+ '_' + '500_' + Type +'_' + DATA[0:-4], Pos = Pos, Fun = Fun, tiempo_ejecucion = tiempo_ejecucion)
        print('SE ALMACENARON LOS DATOS')
        print ('--------------------------------------------------')

"""   
npzfile = np.load('Aritculo J-J-Jaccuracy _Penalization_Dynamicbupa.npz')
npzfile.files
X=npzfile['Pos']
Y=npzfile['Fun']
time=npzfile['tiempo_ejecucion']
"""
