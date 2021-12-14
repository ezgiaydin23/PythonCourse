#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 

def linRegCalc(df):
    #drop missing values
    df.dropna(axis=0,inplace=True)
    
    #create x, y arrays based on df input
    x = df.iloc[:, 0:2]
    x = np.hstack([np.ones((len(x),1)), x])
    y = np.array(df.iloc[:,2:3].values)
    
    
    #beta = (X'X)^-1X'y
    beta = np.linalg.inv(x.T@x)@x.T@y
    
    #error term 
     #rows
    n = len(y)
     #columns
    k = len(x[0]) 
    
    e = y-(x@beta) 
    
    sigmasqr = (e.T@e)/(n-k-1)
    
    #var = sigma^2 (X'X)^-1
    variance = sigmasqr * np.linalg.inv(x.T@x) 
    StdError = np.sqrt(np.diag(variance)).reshape(3,1)   
    
    #confidence interval
    #t stats --- two-tailed alpha= 0.05
    t = stats.t.ppf(.975, n-k-1) 
    lowerCI = beta - StdError*t 
    upperCI = beta + StdError*t
   
    CI = np.hstack([lowerCI, upperCI]) 
    
    #results 
    results = pd.DataFrame(np.hstack([beta,StdError,CI,]), 
                          columns=['Coefficient', 'Std. Err.', 'Lower Confidance Interval','Upper Confidance Interval'])
    results.rename(index={1: "at least bachelors degree", 2: "gov.expenditure on education"}, inplace=True)
    
    xval = np.array(df.iloc[:, 1:2].values)
    yval = np.array(df.iloc[:,2:3].values)
    cval = np.array(df.iloc[:, 0:1].values)
    
    #plotting
    plt.plot(xval,yval,'ob')
    plt.plot(cval,yval,'-r')
    plt.xlabel("at least bachelors degree")
    plt.ylabel("Female share of employment in senior and middle management")
    plt.savefig('regression_plot.jpg')
    plt.show()
   
    
    return(results)


# In[ ]:





# In[ ]:





# In[ ]:




