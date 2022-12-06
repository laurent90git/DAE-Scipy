#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:09:37 2022

Test the dense output in the case of a DAE system.

We test this on a simplified DAE system of the form

y'=-k*(y-cos(t))
z=y

which is of index one

@author: lfrancoi
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
from radauDAE import RadauDAE


#%% Define the DAE system
mass     = np.array([[1.,0.],[0.,0.]])
k = 1e2
modelfun_DAE = lambda t,y: np.array([ -k*(y[0]-np.cos(t)),
                                      (y[0]-y[1])
                                    ])
# jac_dae = lambda t,y: np.array([[-k, 0.],
#                                 [1., -1.]])
jac_dae = None

# Integration parameters
y0 = np.array([3., 3.]) # initial condition
rtol=1e-4; atol=1e-6 # absolute and relative tolerances for time step adaptation
tf = 2*np.pi # final time

# solve the DAE formulation
sol_dae = solve_ivp(fun=modelfun_DAE, t_span=(0., tf), y0=y0, max_step=np.inf,
                    rtol=rtol, atol=atol, jac=jac_dae, jac_sparsity=None,
                    method=RadauDAE, vectorized=False, first_step=1e-8, dense_output=True,
                    mass_matrix=mass, bPrint=False)
assert sol_dae.success
print("DAE solved in {} time steps, {} fev, {} jev, {} LUdec".format(
  sol_dae.t.size, sol_dae.nfev, sol_dae.njev, sol_dae.nlu))

t,y,z = sol_dae.t, sol_dae.y[0,:], sol_dae.y[1,:]

#%% Plot the solution and some useful statistics
plt.figure()
plt.plot(t, y, label='y')
plt.plot(t, z, label='z')
plt.grid()
plt.legend()

#%% Error in z
plt.figure()
plt.semilogy( t, abs(z-y))

#%% dense output
# create time vector with substeps
t_eval = []
for i in range(t.size-1):
    t_eval.extend(  np.linspace(t[i], t[i+1], 5).tolist() )
t_eval.append(t[-1])
t_eval = np.array(t_eval)

dense_sol = sol_dae.sol(t_eval)

plt.figure()

p,=plt.plot(t, y, label='y')
plt.plot(t_eval, dense_sol[0,:], label='y (dense)', linestyle='--', color=p.get_color())

p,=plt.plot(t, z, label='z')
plt.plot(t_eval, dense_sol[1,:], label='z (dense)', linestyle='--', color=p.get_color())

plt.grid()
plt.legend()
