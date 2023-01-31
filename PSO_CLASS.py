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


class PSO:
    
    def __init__(self,n,lim,alpha,beta,theta,funObj,Topology=None,Data=None):
        self.dimension = np.size(lim,axis=0)
        
        self.funObj = funObj
        self.alpha = alpha
        self.beta  = beta
        self.lim = lim
        self.Data=Data
        if Topology is None:
            self.Ady=None
            self.numParticles = n
        else:
            self.genAdyMat(Topology)
            self.numParticles = Topology[0]*Topology[1]*Topology[2]
            

    def correctPos(self,Pos,Vel,tipo='Border'):
        n,d = Pos.shape
        Penalization = np.zeros((n,1))
        if tipo=='Border':
            for i in range(n):
                for j in range(d):
                    if Pos[i,j] > self.lim[j,1]:
                        Pos[i,j]= self.lim[j,1]
                        Vel[i,j]=0
                    elif Pos[i,j] < self.lim[j,0]:
                        Pos[i,j]= self.lim[j,0]
                        Vel[i,j]=0
                        
        elif tipo=='Mirror':
            for i in range(n):
                for j in range(d):
                    while True:
                        if Pos[i,j] > self.lim[j,1]:
                            Pos[i,j]= 2*self.lim[j,1]-Pos[i,j]
                            Vel[i,j]=0
                        elif Pos[i,j] < self.lim[j,0]:
                            Pos[i,j]= 2*self.lim[j,0]-Pos[i,j]
                            Vel[i,j]=0
                        else:
                            break
                        
        elif tipo=='Dynamic':
            for i in range(n):
                for j in range(d):
                    if Pos[i,j] > self.lim[j,1]:
                        Penalization[i,0]=np.exp(2*abs(Pos[i,j]-self.lim[j,1]))
                        Pos[i,j]= self.lim[j,1]
                        Vel[i,j]=0
                    elif Pos[i,j] < self.lim[j,0]:
                        Penalization[i,0]=np.exp(2*abs(Pos[i,j]-self.lim[j,0]))
                        Pos[i,j]= self.lim[j,0]
                        Vel[i,j]=0
            
        elif tipo=='Penalization':
            for i in range(n):
                for j in range(d):
                    if Pos[i,j] > self.lim[j,1]:
                        Vel[i,j]=0
                        Penalization[i,0]=float('inf')
                    elif Pos[i,j] < self.lim[j,0]:
                        Penalization[i,0]=float('inf')
                        Vel[i,j]=0
            
        return Pos,Vel,Penalization
    
    def genAdyMat(self,topSize):
        a=topSize[0]*topSize[1]*topSize[2]
        Ady=np.zeros((a,a))
        for i in range(topSize[0]):
            for j in range(topSize[1]):
                for k in range(topSize[2]):
                    linearInd = np.ravel_multi_index((i,j,k),topSize)
                    if (i-1) < 0:
                        Vec1=np.ravel_multi_index((topSize[0]-1,j,k),topSize)
                    else:
                        Vec1=np.ravel_multi_index((i-1,j,k),topSize)
                    if (j-1) < 0:
                        Vec2=np.ravel_multi_index((i,topSize[1]-1,k),topSize)
                    else:
                        Vec2=np.ravel_multi_index((i,j-1,k),topSize)
                    if (k-1) < 0:
                        Vec3=np.ravel_multi_index((i,j,topSize[2]-1),topSize)
                    else:
                        Vec3=np.ravel_multi_index((i,j,k-1),topSize)
                    Ady[linearInd,Vec1]=1
                    Ady[linearInd,Vec2]=1
                    Ady[linearInd,Vec3]=1
        self.Ady=(Ady+np.eye((a)))               
        
    def runPSO(self,maxIter,m,p,tipo='Border'):
        a=self.alpha
        b=self.beta
        particles2eval=round(p*self.numParticles)
        Pos=np.random.rand(self.numParticles,self.dimension)*(self.lim[:,1]-self.lim[:,0])+self.lim[:,0]
        Vel=np.random.rand(self.numParticles,self.dimension)
        funBest=np.zeros((self.numParticles,1))
        for i in range(self.numParticles):
            funBest[i]=self.funObj(Pos[i,:],Data=self.Data)
        posBest = Pos
        
        if self.Ady is None:
            gold = np.argmin(funBest)
            posGold = posBest[gold,:]
            funGold = funBest[gold]
            Iter = 1
            funVal = np.zeros((self.numParticles,1))
            while True:
                R=np.random.rand(2)
                Vel = Vel + b*R[1]*(posBest-Pos) + a*R[0]*(posGold-Pos) 
                nPos = Pos + Vel
                nPos,Vel,Penalization = self.correctPos(nPos,Vel,tipo=tipo)
                # Evaluate Change
                Pos = nPos            
                for i in range(self.numParticles):
                    if Penalization[i,0] == float('inf'):
                        funVal[i] = float('inf')
                    else:
                        funVal[i] = self.funObj(Pos[i,:],Data=self.Data)+Penalization[i,0]
                    
                    if funVal[i] < funGold or (funVal[i] == funGold and randrange(10) < 5):
                        funGold = funVal[i]
                        posGold = Pos[i,:]
                    if funVal[i] < funBest[i] or (funVal[i] == funBest[i] and randrange(10) < 5):
                        funBest[i] = funVal[i]
                        posBest[i,:] = Pos[i,:]                
                Iter += 1
                #if Iter%10 == 0:
                print(Iter)
    
                sortParticles = np.argsort(funVal)
                maxDist=-float('inf')
                for i in range(particles2eval):
                    dist=np.linalg.norm(Pos[sortParticles[i],:]-posGold)
                    if dist>maxDist:
                        maxDist=dist            
                if maxIter<=Iter :
                    print('Detenido por iteraciones')
                    break      
                elif maxDist<m  and funGold < 0:
                    print('Detenido por distancia')
                    break
            return posGold,funGold 
        else:
            gold = np.argmin(funBest*self.Ady,axis=0)
            posGold = posBest[gold,:]
            funGold = funBest[gold]
            Iter = 1
            funVal = np.zeros((self.numParticles,1))
            while True:
                R=np.random.rand(2)
                Vel = Vel + b*R[1]*(posBest-Pos) + a*R[0]*(posGold-Pos) 
                nPos = Pos + Vel
                nPos,Vel,Penalization = self.correctPos(nPos,Vel,tipo=tipo)
                # Evaluate Change
                Pos = nPos            
                for i in range(self.numParticles):
                    if Penalization[i,0] == float('inf'):
                        funVal[i] = float('inf')
                    else:
                        funVal[i] = self.funObj(Pos[i,:],Data=self.Data)+Penalization[i,0]
                    
                    if funVal[i] < funBest[i] or (funVal[i] == funBest[i] and randrange(10) < 5):
                        funBest[i] = funVal[i]
                        posBest[i,:] = Pos[i,:]  
                        
                gold = np.argmin(funBest*self.Ady,axis=0)
                for i in range(self.numParticles):
                    if funBest[gold[i]] < funGold[i] or (funBest[gold[i]] == funGold[i] and randrange(10) < 5):
                        funGold[i] = funBest[gold[i]]
                        posGold[i,:] = posBest[gold[i],:]
                                  
                Iter += 1
                #if Iter%10 == 0:
                print(Iter)
    
                sortParticles = np.argsort(funVal)
                gold=np.argmin(funGold,axis=0)
                maxDist=-float('inf')
                for i in range(particles2eval):
                    dist=np.linalg.norm(Pos[sortParticles[i],:]-posGold[gold,:])
                    if dist>maxDist:
                        maxDist=dist            
                if maxIter<=Iter :
                    print('Detenido por iteraciones')
                    break      
                elif maxDist<m  and funGold[gold] < 0:
                    print('Detenido por distancia')
                    break                
            return posGold[gold,:],funGold[gold]       

def svmOpFun(N,Data):
    X=Data['X']
    Y=Data['Y']
    Index = Data['Index']    
    Modelo = svm.SVC(C=N[0],kernel='rbf',gamma=N[1])
    Acc = 0
    Cont=0
    for i in np.unique(Index):
        Xtrain = X[Index[:,0]!=i,:]
        Ytrain = Y[Index[:,0]!=i,:]
        Xtest = X[Index[:,0]==i,:]
        Ytest = Y[Index[:,0]==i,:]
        Modelo.fit(Xtrain,np.ravel(Ytrain))
        Yes=Modelo.predict(Xtest)
        Acc += np.sum(Yes==np.ravel(Ytest))/Yes.size
        Cont += 1
    return -Acc/Cont



        









