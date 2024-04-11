#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:54:33 2020

@author: laurent
"""
import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
from scipyDAE.radauDAE import RadauDAE

method=RadauDAE

""" Transistor amplifier as describer in [1], page 377. The system is of the form:

  M y' = f(t,y)

  with a singular masss matrix M. The resulting DAE is of index 1.

  The cicuirt has the tension Ue as input (sinusoid). The output of the
  circuit is U5, which is an amplification of Ue.

References:
  [1] Hairer, Ernst & Wanner, Gerhard. (1996)
        Solving Ordinary Differential Equations II:
        Stiff and Differential-Algebraic Problems
  """

## Setup the model
C1=1e-6; C2=2e-6; C3=3e-6
R0=1E3
R1=R2=R3=R4=R5=9E3
Ub=6

mass = -np.array([[-C1,C1,0,0,0],
                 [C1,-C1,0,0,0],
                 [0,0,-C2,0,0],
                 [0,0,0,-C3,C3],
                 [0,0,0,C3,-C3]])

y0 = np.array([0., Ub*R1/(R1+R2), Ub*R1/(R1+R2), Ub, 0.]) # consistent initial conditions
# y0 = np.array([1., 2., 3., 4., 5.]) # inconsistent initial conditions
Ue = lambda t: 0.4*np.sin(200*np.pi*t) # input tension
def modelfun_DAE(t,y):
  U1,U2,U3,U4,U5=y[0],y[1],y[2],y[3],y[4]
  f = 1e-6*(np.exp((U2-U3)/0.026)-1)
  return np.array([(Ue(t)-U1)/R0,
                   Ub/R2 - U2*(1/R1+1/R2) - 0.01*f,
                   f - U3/R3,
                   Ub/R4 - U4/R4 - 0.99*f,
                   -U5/R5
                   ])

# The jacobian is determined via finite-differences
import scipy.optimize._numdiff
jac_dae = lambda t,x: scipy.optimize._numdiff.approx_derivative(fun=lambda x: modelfun_DAE(t,x),
                                                                x0=x, method='cs',
                                                                rel_step=1e-50, f0=None,
                                                                bounds=(-np.inf, np.inf), sparsity=None,
                                                                as_linear_operator=False, args=(), kwargs={})

# jac_dae = None # the solver detemrines the Jacobian with its own routines


# 1 - solve the ODE
tf = 1. # final time
rtol=1e-4; atol=1e-4

sol = solve_ivp(fun=modelfun_DAE, t_span=(0., tf), y0=y0, max_step=np.inf,
                    rtol=rtol, atol=atol, jac=jac_dae, jac_sparsity=None,
                    method=method, vectorized=False, first_step=1e-8, dense_output=True,
                    mass_matrix=mass,
                    var_index=None,
                    # newton_tol=1e-1,
                    scale_residuals = False,
                    scale_newton_norm = False,
                    scale_error = True,
                    zero_algebraic_error = False,
                    max_bad_ite=0)

#%% Tests
assert_(sol.success, msg='DAE solution failed with solver {method}')

#%% Plot the solution and some useful statistics
# print("DAE solved in {} time steps, {} fev, {} jev, {} LUdec, {} LU solves".format(
#   sol.t.size, sol.nfev, sol.njev, sol.nlu, sol.solver.nlusove))
print("DAE solved in {} time steps, {} fev, {} jev, {} LUdec".format(
  sol.t.size, sol.nfev, sol.njev, sol.nlu))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(3,1,sharex=True, dpi=300)
pltfun = ax[0].plot
# pltfun = ax[0].semilogx

for i in range(y0.size):
  pltfun(sol.t, sol.y[i,:], linestyle='-', linewidth=2, label=f'U{i+1}')
ax[0].legend(ncol=2, frameon=True, framealpha=0.5)
ax[0].grid()
ax[0].set_ylabel('solution\ncomponents')

ax[1].plot(sol.t[:-1], np.diff(sol.t), color='tab:orange', linestyle='-', linewidth=1, label='DAE')
ax[1].grid()
ax[1].legend(frameon=False)
ax[1].set_ylabel('dt (s)')

ax[2].plot(sol.t, range(len(sol.t)), color='tab:orange', linestyle='-', linewidth=1, label='DAE')
ax[2].legend(frameon=False)
ax[2].grid()
ax[2].set_ylabel('number\nof steps')

ax[-1].set_xlabel('t (s)')

plt.figure()
plt.plot(sol.t, sol.y[-1,:], label='U5', color='b')
plt.plot(sol.t, Ue(sol.t), label='Ue', color='r')
plt.grid()
plt.legend()
plt.xlabel('t (s)')
plt.ylabel('U5 (V)')
plt.title('I/O of the circuit')
