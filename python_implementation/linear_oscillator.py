# -*- coding: utf-8 -*-
"""


Created on Sun Oct 21 09:48:58 2018

@author: David O'Gara --Electrical and Systems Engineering
         Washington U. in St. Louis
"""

#try to replicate the pool data function from Brunton et al.

from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from utility_functions import sparsifyDynamics, poolData



    


if __name__ == '__main__':
    
    #set up some dummy data
    dummy_data = 1
    if dummy_data:
        #set up the linear oscillator example
        
        
        def rhs(x,t):
            A = np.array([(-0.1, 2.0),
                    (-2.0, -0.1)])
            #A = [-.1 2; -2 -.1]
            
            #sol = list(np.matmul(A,x))
            sol = np.matmul(A,x)
            #print (sol)
            return sol
        A = np.array([(-0.1, 2),
                    (-2, -0.1)])
        rhs2 = lambda x,t: np.matmul(A,x)
        x0 =np.array((2,0))
        
        #0 =[2,0]
        t_span = np.linspace(0,25,2500)
        X_s = odeint(rhs2,x0,t_span)
        x1 = X_s[:,0]
        x2=X_s[:,1]
        

        #%% now run poolData
        #generate dx
        dx = []
        for x in X_s:
            dx.append(rhs2(x,0)) #0 is a placeholder
        dx_true = pd.DataFrame(np.array(dx))
        
        noise = 0.1*np.random.normal(loc = 0.0, scale = 1.0, size = dx_true.shape)
        noise = pd.DataFrame(noise)
        dx = dx_true + noise
        n,nVars = X_s.shape
        cols = ['x1','x2']
        dot_labels = pd.Index([s + 'dot' for s in cols])
        yin = pd.DataFrame(X_s, columns = cols)
        #use a dict to label the dataframe?
        polyorder = 3
        usesine = 0
        
        Theta = poolData(yin,nVars, polyorder, usesine)
        lam = 0.001
        
        Xhat_df = sparsifyDynamics(Theta,dx,lam)
        Xhat_df = Xhat_df.set_index(dot_labels)
        
        plt.title("Phase Plane Plot")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.plot(x1,x2)
        plt.show()
        plt.close()
        
       
        
        
        
        