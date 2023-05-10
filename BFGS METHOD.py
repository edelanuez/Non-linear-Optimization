#!/usr/bin/env python
# coding: utf-8

# In[8]:


#import packages
import cvxpy as cp 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import triang
import os
from scipy import optimize


# In[9]:


#load in matrices
def load_matrices(input_sheet, input_tab):
    df = pd.read_excel(input_sheet, sheet_name=input_tab)
    if "sec_id" not in df.columns:
         raise ValueError("The dataframe must have a column named 'sec_id'")

    df = df.sort_values(by="sec_id")
    sec_id = df["sec_id"].astype("category")
    sectors = df[[col for col in df.columns if col.startswith("s_")]].to_numpy()
    weight = df[["weight"]].to_numpy()
    dts = df[["dts"]].to_numpy()
    dts = np.nan_to_num(dts)
    c_dts=df[["c_dts"]].to_numpy()
    quality = df[[col for col in df.columns if col.startswith("q_")]].to_numpy()
    return (sectors, weight, dts, quality, sec_id, c_dts)

input_sheet = "/Users/emariedelanuez/Downloads/clean_data.xlsx"
sectors_M, weight_M, dts_M, quality_M, sec_id_M, c_dts_M = load_matrices(input_sheet, "model")
#print(c_dts_M)
sectors_T, weight_T, dts_T, quality_T, sec_id_T, c_dts_T = load_matrices(input_sheet, "target1")
#print(c_dts_T)
assert abs(sum(weight_M) - 1) < 1e-6
assert list(sec_id_M) == list(sec_id_T)


# In[114]:


# model
#sectors_M
#weight_M
# dts_M
#quality_M
# target data
#sectors_T
# weight_T
#dts_T
#quality_T


# In[10]:


C_1 = sectors_M.T @ weight_M
C_2 = quality_M.T @ weight_M
C_3 = dts_M.T @ weight_M


# In[110]:


#old objective function

n = sec_id_M.shape[0]
w = cp.Variable((n, 1))
lamba1 = 0.2
lamba2 = 0.2
lamba3 = 0.6
objective = cp.Minimize(
 lamba3 * cp.norm(C_3 - dts_T.T @ w, 2)
 + lamba2 * cp.norm(C_2 - quality_T.T @ w, 2)
 + lamba1 * cp.norm(C_1 - sectors_T.T @ w, 2)
)
constraints = [w >= 0, w <= 0.03, cp.sum(w) == 1]
prob = cp.Problem(objective, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
#print("optimal var", w.value)
w_opt = w.value


# In[11]:


##generate random r
c_vibes = [.2, .4, 0,0,.05]
c_vector = pd.Series(c_vibes)

#d_vector
d_vibes = [.4, .5, .1, .18,0]
d_vector = pd.Series(d_vibes)                                                                                                       
#turn c and d into numpy
c_vector = c_vector.values
d_vector = d_vector.values                                                                                                      #generate random vector r
r = np.random.uniform(low=d_vector, high=c_vector)
#normalize r so that entries sum up to one
#r /= np.sum(r)
r = r.reshape((-1, 1))  # reshape to column vector
print(r)


# In[12]:


##irreducible rank matrix

n = 5  # rank of the matrix

# create the standard basis vectors
basis = np.eye(5)[:n]

# create the matrix
matrix = pd.DataFrame(basis)

# convert the DataFrame to a NumPy matrix
A = matrix.to_numpy()

# print the NumPy matrix
print(A)


# In[13]:


w = cp.Variable((n,1))

def objective(w, r, A):
    # Define the objective function
    obj = cp.Minimize(cp.norm(A @ w -r, 2))

    # Define the constraints
    constraints = [cp.sum(w) == 1, w >= 0]
####for tasissa the .03 constraint doesnt work
    # Create the problem and solve it
    problem = cp.Problem(obj, constraints)
    problem.solve()

    # Return the optimal value of w
    return w.value

optimal_w = objective(w, r, A)

# Print the optimal value of w
print("Optimal value of w: ", optimal_w)
#print(quality_T.T@w_opt)


# In[17]:


bounds = np.array([0.34706337, 0.46593029, 0.08290541, 0.10159413, 0.00250681])


# In[ ]:





# In[18]:


w = cp.Variable((564, 1))
objective = cp.Minimize(cp.norm(C_3 - dts_T.T@ w,2))
constraints = [w >= 0, w <= 0.03, cp.sum(w) == 1, quality_T.T@w<=[bounds]]
prob = cp.Problem(objective, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
w_opt = w.value
#print("optimal var", w.value)
# print(Q_t.T @ w_opt)
#print(quality_T.T @ w_opt)


# In[143]:


m=564
n =1

# Generate the random vector using np.random.uniform()
v = np.random.uniform(size=(m, n))
#print (v)


#print(dts_T.T@v)
dts_T.T.shape
#print(quality_T.T@v)


# In[ ]:


bounds = np.array([0.34706337, 0.46593029, 0.08290541, 0.10159413, 0.00250681])


# In[79]:


###scipy optimize BFGS ###BFGS NOT WORKING 
alpha=-0.0005
beta=-0.75
quartz=-.4
def objective(w):
    return  np.linalg.norm(C_3 - dts_T.T @ w, 2) + alpha * np.minimum(w - 0.03, 0).sum() + beta * np.maximum(w, 0).sum() + quartz * np.minimum(quality_T.T @ w - [0.34706337, 0.46593029, 0.08290541, 0.10159413, 0.00250681], 0).sum()
w0 = np.ones((564, 1)) / 564  # initial guess
w0 = np.ravel(w0)
result = optimize.minimize(objective, w0, method='BFGS')
w_opt = result.x
print("status:", result.message)
print("optimal value:", result.fun)
#print("w_opt:", w_opt)

w_opt.shape
w_opt= w_opt.reshape((564, 1))
print(w_opt)


# In[80]:


#check to see if the r vector is the same 
print(quality_T.T@w_opt)


# In[30]:


#recall that r:[[0.35244743],[0.47131435],[0.08828947],[0.10697819],[0.00789083]]
##so our r when solving with the BFGS optimal w is not the same r
### also r is not inbetween our original c and d constraints. 

#c_vibes = [.2, .4, 0,0,.05]
#c_vector = pd.Series(c_vibes)

#d_vector
#d_vibes = [.4, .5, .1, .18,0]
#d_vector = pd.Series(d_vibes)  
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




