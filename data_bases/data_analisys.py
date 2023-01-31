# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:31:19 2023

@author: jupa_
"""
import numpy as np
import numpy.matlib as ll
from sklearn import svm
from math import ceil
from time import time
from sklearn.preprocessing import Normalizer
from random import randrange

def readDataBase(fileName):
    file = open(fileName,"r")
    cont = 0
    for line in file:
        if line[0]!='@':
            line=line.split(',')
            data = np.asarray(line[:-1],dtype=np.float32)
            clase = np.asarray(line[-1])
            try:
                X = np.vstack((X,data))
            except:
                X = data
            try:
                Y = np.vstack((Y,clase))
            except:
                Y = clase
    file.close()
    unicos,Y=np.unique(Y, return_inverse=True)
    m,n=X.shape
    Y.resize((m,1))
    return X,Y



X,Y = readDataBase('titanic.dat')

sum(Y)