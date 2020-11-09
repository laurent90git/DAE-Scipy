#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:54:33 2020

The idea is to test the "mass" option with a simple vector ODE
on the array y=(y0,y1,y2,...,yn), which reads:
  
  d(y[k])/dt = eigvals[k]*y, with eigvals[k] the k-th eigenvalue of the system
  
  <=> dy/dt = diag(eigvals)*y
  
This simple ODE (no mass matrix) can also be reformulated as a:
  
  M*dy/dt = y,  with M=diag(1/eigvals).
  
This allows to verify that the mass matrix implementation is coherent. 

@author: laurent
"""
import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)

## Choice of integration method
from radauDAE import RadauDAE
method=RadauDAE
rtol=1e-10; atol=1e-10

n=7 # number of equations
eigvals  = np.array([float(i)*(-1)**(i) for i in range(1,n+1)]) # eigenvalues of the reference ODE
mass     = diags(diagonals=(1/eigvals,) ,offsets=(0,), format='csc') # corresponding mass-matrix for the alternative ODE
mass_inv = np.linalg.inv(mass.toarray()) # diags(eigvals)
modelfun_without_mass = lambda t,x: mass_inv.dot(x) # or simply eigvals*x
modelfun_with_mass    = lambda t,x: x

# solve both ODEs
tf = 1.0 # final time
y0 = np.ones((n,))
true_solution = y0*np.exp(eigvals*tf) # theoretical solution
sol_with_mass = solve_ivp(fun=modelfun_with_mass, t_span=(0., tf), y0=y0, max_step=np.inf,
                rtol=rtol, atol=atol, jac=None, jac_sparsity=None, method=method,
                vectorized=False, first_step=None, mass=mass)
sol_without_mass = solve_ivp(fun=modelfun_without_mass, t_span=(0., tf), y0=y0, max_step=np.inf,
                rtol=rtol, atol=atol, jac=None, jac_sparsity=None, method=method,
                vectorized=False, first_step=None, mass=None)



# print("ODE with mass matrix    solved in {} time steps, {} fev, {} jev, {} LUdec, {} LU solves".format(
#   sol_with_mass.t.size, sol_with_mass.nfev, sol_with_mass.njev,
#   sol_with_mass.nlu, sol_with_mass.solver.nlusove))
# print("ODE without mass matrix solved in {} time steps, {} fev, {} jev, {} LUdec, {} LU solves".format(
#   sol_without_mass.t.size, sol_without_mass.nfev, sol_without_mass.njev,
#   sol_without_mass.nlu, sol_without_mass.solver.nlusove))
# Alternative print statements if "nlusolve" is not available
print("ODE with mass matrix    solved in {} time steps, {} fev, {} jev, {} LUdec".format(
  sol_with_mass.t.size, sol_with_mass.nfev, sol_with_mass.njev,
  sol_with_mass.nlu))
print("ODE without mass matrix solved in {} time steps, {} fev, {} jev, {} LUdec".format(
  sol_without_mass.t.size, sol_without_mass.nfev, sol_without_mass.njev,
  sol_without_mass.nlu))

print('true solution: ', true_solution)
print('no mass solution: ', sol_without_mass.y[:,-1])
print('with mass solution: ', sol_with_mass.y[:,-1])

#%% Tests
assert_(sol_without_mass.success, msg=f'solver {method} failed without mass matrix')
assert_(sol_with_mass.success,    msg=f'solver {method} failed with mass matrix')
assert_allclose(sol_without_mass.y[:,-1], true_solution, rtol=10*max((atol,rtol)), err_msg='the solution without mass-matrix is not coherent with the theoretical solution')
assert_allclose(sol_with_mass.y[:,-1],    true_solution, rtol=10*max((atol,rtol)), err_msg='the solution with the mass-matrix is not coherent with the theoretical solution')
assert_allclose(sol_without_mass.t.size,  sol_with_mass.t.size, rtol=0.05, err_msg='the mass-matrix option should not affect the number of steps so much')

#%% Plots
import matplotlib.pyplot as plt
plt.figure()
plt.semilogy(sol_without_mass.t, sol_without_mass.y.T, color='r', linestyle='-', label=None)
plt.semilogy(sol_with_mass.t, sol_with_mass.y.T, color='b', linestyle=':', label=None)
# legend hack to avoid repeated labels
plt.plot(np.nan, np.nan, color='r', linestyle='-', label='without mass')
plt.plot(np.nan, np.nan, color='b', linestyle='-', label='with mass')
plt.legend()
plt.grid()
plt.xlabel('t (s)')
plt.ylabel('y')

plt.figure()
plt.plot(sol_without_mass.t[:-1], np.diff(sol_without_mass.t), color='r', linestyle='-', label=None)
plt.plot(sol_with_mass.t[:-1], np.diff(sol_with_mass.t), color='b', linestyle=':', label=None)
# legend hack to avoid repeated labels
plt.plot(np.nan, np.nan, color='r', linestyle='-', label='without mass')
plt.plot(np.nan, np.nan, color='b', linestyle='-', label='with mass')
plt.legend()
plt.grid()
plt.xlabel('t (s)')
plt.ylabel('dt (s)')

