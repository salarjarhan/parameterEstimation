import ast
import math
import numpy as np
from simpleModel import calculate_model
from numpy import linalg as LA

def parse_str(s):
   try:
      return ast.literal_eval(str(s))
   except:
      return

ob = [69.52, 71.44, 72.99, 73.86, 58.73, 50.57, 54.31, 58.06, 56.31, 52.32, 46.35, 29.01, 57.24, 54.24, 39.48, 48.47]

#Parameters Definiterion
lb= [3, 3, 3]
ub= [15, 15, 15]
pn=3

maxiter=100          #Maximum Number of iterations
npop=20              #Number of Fireflies
L=1
gamma=1/math.sqrt(L)         #Light Absorption Coefficient
beta0=2;                     #Attraction Coefficient Base Value
alpha=0.02                   #Mutation Coefficient
alpha_RF=0.95                #Radius Reduction Factor

#Lists
pop=[]
er=[]
rij=np.zeros((npop,pn))
beta=np.zeros((npop,pn))
E=np.zeros((npop,pn))
newpop=np.zeros((npop,pn))
BEST=np.zeros((maxiter,1))
MEAN=np.zeros((maxiter,pn))

#Create Random Pop
for i in range(npop):
    pop.append(np.random.uniform(lb,ub))
    z1_hk=pop[i][0]
    z2_hk=pop[i][1]
    z3_hk=pop[i][2]
    result = calculate_model(z1_hk, z2_hk, z3_hk)
    r=parse_str(result)
    sum = 0;
    for j in range(len(ob)):
        sum += (ob[j]-r[j]) ** 2
    er.append(sum/len(ob))

#Main Loop
for iter in range(maxiter):
    for i in range(npop):
        for j in range(npop):
            if er[j]<er[i]:
                
                rij[i][0]=LA.norm(pop[i][0]-pop[j][0])
                rij[i][1]=LA.norm(pop[i][1]-pop[j][1])
                rij[i][2]=LA.norm(pop[i][2]-pop[j][2])
                
                beta[i][0]=beta0*math.exp(-gamma*(rij[i][0])**2)
                beta[i][1]=beta0*math.exp(-gamma*(rij[i][0])**2)
                beta[i][2]=beta0*math.exp(-gamma*(rij[i][0])**2)
                
                E[i][0]=alpha*np.random.uniform(-1,1)*(ub[0]-lb[0])
                E[i][1]=alpha*np.random.uniform(-1,1)*(ub[1]-lb[1])
                E[i][2]=alpha*np.random.uniform(-1,1)*(ub[2]-lb[2])
                
                newpop[i][0]=pop[i][0]+beta[i][0]*(pop[j][0]-pop[i][0])+E[i][0]
                newpop[i][1]=pop[i][1]+beta[i][0]*(pop[j][1]-pop[i][0])+E[i][1]
                newpop[i][2]=pop[i][2]+beta[i][0]*(pop[j][2]-pop[i][0])+E[i][2]
                
                if lb[0]<=newpop[i][0]<=ub[0] and lb[1]<=newpop[i][1]<=ub[1] and lb[2]<=newpop[i][2]<=ub[2]:
                    z1_hk=newpop[i][0]
                    z2_hk=newpop[i][1]
                    z3_hk=newpop[i][2]
                    result = calculate_model(z1_hk, z2_hk, z3_hk)
                    r=parse_str(result)
                    pop.append(newpop[i])
                    sum = 0;
                    for j in range(len(ob)):
                        sum += (ob[j]-r[j]) ** 2
                    er.append(sum/len(ob))
                
    er=np.asarray(er)
    pop=np.asarray(pop)
    ind = er.argsort()
    er=er[ind]
    pop=pop[ind]
    er=er[0:npop]    
    pop=pop[0:npop,:]    
    BEST[iter]=er[0]
    MEAN[iter]=pop[0]
    er=er.tolist()
    pop=pop.tolist()
    #Reduction Mutation Coefficient
    alpha=alpha*alpha_RF;
    
ind2 = BEST.argsort()
BEST=BEST[ind2]
MEAN=MEAN[ind2]

#Result
answer=MEAN[0]
MSE=BEST[0]
print(MSE)
print(answer)
