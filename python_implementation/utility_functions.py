# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:25:58 2018

@author: David O'Gara -- Electrical and Systems Engineering
         Washington U. in St. Louis
"""
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np

def poolData(yin, nVars, polyorder, usesine):
    #%% need to decide if we want to use numpy or pandas
    cols = yin.columns
    df = pd.DataFrame()
    #just concat along each column
    d = set()
    n = len(yin)
    #polyorder 0
    yout = pd.Series(np.ones(n))
    df = pd.concat([df,yout])
    
    #poly order 1
    for i in cols:
        yout = yin.loc[:,i]
        label = i
        if label not in d:
            d.add(label)
            yout = yout.rename(label)
            df = pd.concat([df,yout], axis = 1)   
    if polyorder>=2:
        for i in cols:
            for j in cols:
                label = tuple(sorted([i,j]))
                if label not in d:
                    d.add(label)
                    yout = yin.loc[:,i]*yin.loc[:,j]
                    yout = yout.rename(label)
                    df = pd.concat([df,yout], axis = 1)                
    if polyorder>=3:
        for i in cols:
            for j in cols:
                for k in cols:
                    label = tuple(sorted([i,j,k]))
                    if label not in d:
                        d.add(label)
                        yout = yin.loc[:,i]*yin.loc[:,j]*yin.loc[:,k]
                        yout = yout.rename(label)
                        df = pd.concat([df,yout], axis = 1)     
    
    
    if polyorder>=4:
        for i in cols:
            for j in cols:
                for k in cols:
                    for l in cols:
                        label = tuple(sorted([i,j,k,l]))
                        if label not in d:
                            d.add(label)
                            yout = yin.loc[:,i]*yin.loc[:,j]*yin.loc[:,k]
                            yout = yout.rename(label)
                            df = pd.concat([df,yout], axis = 1)  
    
    #drop duplicate columns
    #print('Polyorder', polyorder)
    return df


def sparsifyDynamics(Theta, dx, lam, max_iter=10000):
    '''
    This script runs a Lasso procedure to return a sparse solution to a set of differential equations
    '''
    #%% for now, just set up the optimization problem and use Lasso 
    
    clf  = Lasso(alpha = lam, normalize = True, max_iter=max_iter)
    Xhat = clf.fit(Theta,dx)

    Xhat_df = pd.DataFrame(Xhat.coef_, columns=Theta.columns)
    '''

    Xhat = []
    for i in dx.columns:
        clf = Lasso(alpha = lam, normalize = True)
        xhat = clf.fit(Theta,dx[i]).coef_
        Xhat.append(xhat)
    Xhat_df = pd.DataFrame(Xhat, columns = Theta.columns)
    return Xhat_df
    '''
    return Xhat_df