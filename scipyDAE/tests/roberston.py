#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:54:33 2020

The idea is to test the "mass" option with the simple Roberston system
(see [1]). This is a stiff ODE system which reads:

   dy[0]/dt = -0.04*y[0] + 1e4*y[1]*y[2]
   dy[1]/dt =  0.04*y[0] - 1e4*y[1]*y[2] - 3e7*(y[1]**2)
   dy[2]/dt =  3e7*(y[1]**2)

with initial conditions y(t0)=[1,0,0].

By summing the three time derivatives, we see that the system has the following invariant:

   y[0] + y[1] + y[2] = 1

This allows simple ODE (no mass matrix) can also be reformulated as a DAE
(semi-explicit index-1,  also called Hessenberg index-1):

   dy[0]/dt = -0.04*y[0] + 1e4*y[1]*y[2]
   dy[1]/dt =  0.04*y[0] - 1e4*y[1]*y[2] - 3e7*(y[1]**2)
   0        =  y[0] + y[1] + y[2] - 1

where y[2] has become an algebraic variable.

It is easy to provide this DAE system to a standard implicit ODE solver by
introducing a singular mass matrix:

  M * Y = f

  with M=[[1,0,0],
          [0,1,0],
          [0,0,0]]

  and f = [ -0.04*y[0] + 1e4*y[1]*y[2],
             0.04*y[0] - 1e4*y[1]*y[2] - 3e7*(y[1]**2),
             y[0] + y[1] + y[2] - 1 ]

We can then verify that the DAE is solved correctly by comparing its solution
to the one of the original ODE system.

References:
  [1] Robertson, H.H. The solution of a set of reaction rate equations.
      In Numerical Analysis: An Introduction;Academic Press: London, UK, 1966

@author: laurent
"""
import numpy as np
from scipy.sparse import diags
# from scipy.integrate import solve_ivp
from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
from scipyDAE.radauDAE import solve_ivp_custom as solve_ivp
import time as pytime

## Choice of integration method
from scipyDAE.radauDAE import RadauDAE

method=RadauDAE


bPrint=False # if True, Radau prints additional information during the computation

#%% Define the ODE formulation
mass     = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]])
# mass     = diags([1.,1.,0.], 0, format='csc') #sparse format
modelfun_ODE = lambda t,y: np.array([-0.04*y[0] + 1e4*y[1]*y[2],
                                     0.04*y[0] - 1e4*y[1]*y[2] - 3e7*(y[1]**2),
                                     3e7*(y[1]**2)])

#%% Define the DAE system
modelfun_DAE = lambda t,y: np.array([-0.04*y[0] + 1e4*y[1]*y[2],
                                     0.04*y[0] - 1e4*y[1]*y[2] - 3e7*(y[1]**2),
                                     y[0] + y[1] + y[2] - 1])
var_index = np.array([0,0,1]) # algebraic index of the components

jac_dae = lambda t,y: np.array([[-0.04, 1e4*y[2], 1e4*y[1]],
                                [0.04, -1e4*y[2] - 2*3e7*y[1], -1e4*y[1]],
                                [1, 1, 1],])
# jac_dae = None # will be computed via finite-differences

# Integration parameters
y0 = np.array([1.,0.,0.]) # initial condition
rtol=1e-10; atol=rtol/10 #1e-6 # absolute and relative tolerances for time step adaptation
tf = 5e6 # final time


# solve the ODE formulation
t1 = pytime.time()
sol_ode = solve_ivp(fun=modelfun_ODE, t_span=(0., tf), y0=y0, max_step=np.inf,
                    rtol=rtol, atol=atol, jac=None, jac_sparsity=None,
                    method=method, vectorized=False, first_step=1e-8, dense_output=True,
                    max_newton_ite=6, max_bad_ite=0,
                    scale_residuals = True,
                    scale_newton_norm = False,
                    scale_error = True,
                    mass_matrix=None, bPrint=bPrint,
                    bReport=True)
t2 = pytime.time()
sol_ode.CPUtime = t2-t1

# solve the DAE formulation
t1 = pytime.time()
sol_dae = solve_ivp(fun=modelfun_DAE, t_span=(0., tf), y0=y0, max_step=np.inf,
                    rtol=rtol, atol=atol, jac=jac_dae, jac_sparsity=None,
                    method=method, vectorized=False, first_step=1e-8, dense_output=True,
                    max_newton_ite=6, max_bad_ite=0,
                    min_factor=0.2, max_factor=10,
                    var_index=var_index,
                    # newton_tol=1e-1,
                    scale_residuals = True,
                    scale_newton_norm = False,
                    scale_error = True,
                    mass_matrix=mass, bPrint=bPrint,
                    bReport=True)
t2 = pytime.time()
sol_dae.CPUtime = t2-t1
# print("DAE solved in {} time steps, {} fev, {} jev, {} LUdec, {} LU solves".format(
#   sol_dae.t.size, sol_dae.nfev, sol_dae.njev, sol_dae.nlu, sol_dae.solver.nlusove))
for sol,name in ((sol_dae, 'DAE'),(sol_ode,'ODE')):
  sol.reports = sol.solver.reports
  for key in sol.reports.keys():
     sol.reports[key] = np.array(sol.reports[key])
  if sol.success:
      state='solved'
  else:
      state='failed'

  print("\n{} {}".format(name, state))
  print("\t{} time steps in {}s\n\t({} = {} accepted + {} rejected + {} failed)".format(
    sol.t.size-1, sol.CPUtime, sol.solver.nstep, sol.solver.naccpt, sol.solver.nrejct, sol.solver.nfailed))
  print("\t{} fev, {} jev, {} LUdec, {} linsolves, {} linsolves for error estimation".format(
        sol.nfev, sol.njev, sol.nlu, sol.solver.nlusolve, sol.solver.nlusolve_errorest))


#%% Plot the solution and some useful statistics
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3,1,sharex=True, dpi=300, figsize=np.array([1.5,3])*5)
pltfun = ax[0].plot
# pltfun = ax[0].semilogx
pltfun(sol_ode.t, sol_ode.y[0,:],     color='tab:green', linestyle='-', linewidth=2, label=None)
pltfun(sol_ode.t, 1e4*sol_ode.y[1,:], color='tab:orange', linestyle='-', linewidth=2, label=None)
pltfun(sol_ode.t, sol_ode.y[2,:],     color='tab:blue', linestyle='-', linewidth=2, label=None)


pltfun(sol_dae.t, sol_dae.y[0,:],     color='tab:green', linestyle=':', linewidth=4, label=None)
pltfun(sol_dae.t, 1e4*sol_dae.y[1,:], color='tab:orange', linestyle=':', linewidth=4, label=None)
pltfun(sol_dae.t, sol_dae.y[2,:],     color='tab:blue', linestyle=':', linewidth=4, label=None)

# hack for legend
ax[0].semilogx(np.nan, np.nan, color='tab:green',  linestyle='-', label='y[0]')
ax[0].semilogx(np.nan, np.nan, color='tab:orange', linestyle='-', label='y[1]')
ax[0].semilogx(np.nan, np.nan, color='tab:blue',   linestyle='-', label='y[2]')
ax[0].plot(np.nan, np.nan, color=[0,0,0],   linestyle='-', label='ODE')
ax[0].scatter(np.nan, np.nan, color=[0,0,0],   marker='s', label='DAE')
ax[0].legend(ncol=2, frameon=False)
ax[0].grid()
ax[0].set_ylabel('solution\ncomponents')

ax[1].loglog(sol_ode.t[:-1], np.diff(sol_ode.t), color='tab:blue', linestyle='-', linewidth=1, label='ODE')
ax[1].loglog(sol_dae.t[:-1], np.diff(sol_dae.t), color='tab:orange', linestyle='-', linewidth=1, label='DAE')
ax[1].grid()
ax[1].legend(frameon=False)
ax[1].set_ylabel('dt (s)')

ax[2].loglog(sol_ode.t, range(len(sol_ode.t)), color='tab:blue', linestyle='-', linewidth=1, label='ODE')
ax[2].loglog(sol_dae.t, range(len(sol_dae.t)), color='tab:orange', linestyle='-', linewidth=1, label='DAE')
ax[2].legend(frameon=False)
ax[2].grid()
ax[2].set_ylabel('number\nof steps')

# ax[3].loglog(sol_ode.solver.info['cond']['t'], sol_ode.solver.info['cond']['LU_real'], color='tab:blue', linestyle='-', linewidth=2, label='ODE real')
# ax[3].loglog(sol_dae.solver.info['cond']['t'], sol_dae.solver.info['cond']['LU_real'], color='tab:orange', linestyle='-', linewidth=1, label='DAE real')
# ax[3].loglog(sol_ode.solver.info['cond']['t'], sol_ode.solver.info['cond']['LU_complex'], color='tab:blue', linestyle='--', linewidth=2, label='ODE complex')
# ax[3].loglog(sol_dae.solver.info['cond']['t'], sol_dae.solver.info['cond']['LU_complex'], color='tab:orange', linestyle='--', linewidth=1, label='DAE complex')
# ax[3].legend(frameon=False)
# plt.xlim(1e-10,tf*1.05)
# ax[3].grid()
# ax[3].set_ylabel('condition\nnumbers')

ax[-1].set_xlabel('t (s)')

# #%% Plot the evolution of all time steps tried by the method
# if bPrint:
#   plt.figure()
#   plt.plot(sol_dae.solver.info['cond']['t'], sol_dae.solver.info['cond']['h'])
#   plt.plot(sol_ode.solver.info['cond']['t'], sol_ode.solver.info['cond']['h'])
#   plt.grid()
#   plt.ylabel('dt (s)')
#   plt.xlabel('t (s)')

#%% A posteriori analysis of the condition number of the system's Jacobian
plt.figure()
cond_number = [np.linalg.cond(jac_dae(sol_dae.t[i],sol_dae.y[:,i])) for i in range(len(sol_dae.t))]
plt.loglog(sol_dae.t, cond_number)
plt.xlabel('t (s)')
plt.ylabel('cond(J)')
plt.grid()

#%% Tests (for Scipy integration)
assert_(sol_dae.success, msg=f'DAE solution failed with solver {method}')
assert_(sol_ode.success, msg=f'ODE solution failed with solver {method}')
assert_allclose(sol_ode.y[:,-1], sol_dae.y[:,-1], rtol=5*max(atol,rtol)**0.5, err_msg='the DAE and ODE solutions are not close enough')
# assert_(sol_ode.t.size==sol_dae.t.size, msg='the mass-matrix option should not affect the number of steps')
# assert_((sol_ode.nlu - sol_dae.nlu)/sol_ode.nlu < 0.3, msg='the mass-matrix option should not worsen the performance as much (number of LU-decomposition)')

# test that the dense output is correct
test_t = np.logspace(-3,np.log10(tf),10000) # avoid initial transient with values close to zero
denseout_dae = sol_dae.sol(test_t)
denseout_ode = sol_ode.sol(test_t)
# Relative error on the dense output (DAE vs ODE)
plt.figure()
for i in range(3):
  plt.semilogy(test_t, abs(denseout_ode[i,:] - denseout_dae[i,:])/(rtol+abs(denseout_ode[i,:])), linewidth=2/(i+1), label=f'var {i}')
plt.axhline(rtol, color=[0,0,0], linestyle='--', label='rtol')
plt.xlabel('t (s)')
plt.ylim(1e-20, 1e5)
plt.grid()
plt.legend()
plt.ylabel('relative error on\nthe dense output')

# assert_(np.max( np.abs(denseout_ode - denseout_dae)/(atol+denseout_ode) )<max(atol,rtol)**0.5, 'DAE dense output is not close to the ODE one.')
