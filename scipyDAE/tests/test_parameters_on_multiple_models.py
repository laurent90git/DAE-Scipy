# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 15:53:57 2025

Script to easily test many possible settings, so that decent default values may
be suggested for Scipy's ongoing integration

@author: lfrancoi
"""
import numpy as np
import time as pytime
from scipyDAE.radauDAE import RadauDAE
from scipyDAE.radauDAE import solve_ivp_custom as solve_ivp
# from scipy.optimize._numdiff import approx_derivative


def getCase(ncase):
  if ncase==0:
    case_name = "Jay with nonlinear multiplier"
    from jay_dae import generate_Jay1995
    y0, mass_matrix, var_index, fun, jac, t_span = generate_Jay1995(True)
    
  elif ncase==1:
    case_name = "Jay"
    from jay_dae import generate_Jay1995
    y0, mass_matrix, var_index, fun, jac, t_span = generate_Jay1995(False)
    
  elif ncase==2:
    case_name="index-3 pendulum"
    from pendulum import generateSystem
    t_span, fun, jac, mass_matrix, y0, var_index = generateSystem(chosen_index=3)
    
  elif ncase==3:
    case_name="index-2 pendulum"
    from pendulum import generateSystem
    t_span, fun, jac, mass_matrix, y0, var_index = generateSystem(chosen_index=2)
    
  elif ncase==4:
    case_name="index-1 pendulum"
    from pendulum import generateSystem
    t_span, fun, jac, mass_matrix, y0, var_index = generateSystem(chosen_index=1)
    
  elif ncase==5:
    case_name="index-0 pendulum"
    from pendulum import generateSystem
    t_span, fun, jac, mass_matrix, y0, var_index = generateSystem(chosen_index=3)

  elif ncase==6:
    case_name="index-3 recursive pendulum (n=20)"
    from recursive_pendulum import generateSystem
    t_span, fun, jac, sparsity, mass_matrix, y0, var_index, T_th, m = generateSystem(n=10)
    
  elif ncase==7:
    case_name="Roberston ODE"
    from robertson import generateSystem
    t_span, y0, fun, jac, mass_matrix, var_index = generateSystem(bDAE=False)
    

  elif ncase==8:
    case_name="Index-1 Roberston DAE"
    from robertson import generateSystem
    t_span, y0, fun, jac, mass_matrix, var_index = generateSystem(bDAE=True)
    
  elif ncase==9:
    case_name="Index-1 Transistor amplfiier"
    from transistor_amplifier import generateSystem
    t_span, y0, fun, jac, mass_matrix, var_index, Ue = generateSystem()
    
  else:
    raise Exception(f"case {ncase} not found")
    
  return case_name, t_span, y0, fun, jac, mass_matrix, var_index
    
#%%

{"scale_residuals": [False, True],
 "scale_newton_norm": [False, True],
 "scale_error": [False, True],
 "zero_algebraic_error": [False, True],
 "bUsePredictiveController": [False, True],
 "bUsePredictiveNewtonStoppingCriterion": [False, True],
 "bAlwaysApply2ndEstimate": [False, True],
 "max_bad_ite": [0,1,2,3],
 "e":[],
 }

results = [[] for i in range(10)]
for ncase in range(2):
  case_name, t_span, y0, fun, jac, mass_matrix, var_index = getCase(ncase)
  
  # jac = None
  rtol=1e-3; atol=rtol/10

  # solve the DAE formulation
  t1 = pytime.time()
  sol = solve_ivp(fun=fun, t_span=t_span, y0=y0, max_step=np.inf,
                  rtol=rtol, atol=atol,
                  jac=jac, jac_sparsity=None,
                  method=RadauDAE, vectorized=False,
                  first_step=1e-5, dense_output=True,
                  # min_factor=0.2, max_factor=10,
                  max_newton_ite=6, max_bad_ite=1,
                  var_index=var_index,
                  # newton_tol=1e-1,
                  scale_residuals = True,
                  scale_newton_norm = True,
                  # zero_algebraic_error = True,
                  scale_error = True,
                  mass_matrix=mass_matrix,
                  bAlwaysApply2ndEstimate=False,
                  bUsePredictiveController=True,
                  bUsePredictiveNewtonStoppingCriterion=False,
                  bPrint=False,
                  bReport=True)

  t2 = pytime.time()
  sol.CPUtime = t2-t1
  
  sol.reports = sol.solver.reports
  for key in sol.reports.keys():
     sol.reports[key] = np.array(sol.reports[key])
  if sol.success:
      state='solved'
  else:
      state='failed'

  print("\nDAE {} \"{}\"\n{}".format(ncase, case_name, state))
  print("\t{} time steps in {}s\n\t({} = {} accepted + {} rejected + {} failed)".format(
    sol.t.size-1, sol.CPUtime, sol.solver.nstep, sol.solver.naccpt, sol.solver.nrejct, sol.solver.nfailed))
  print("\t{} fev, {} jev, {} LUdec, {} linsolves, {} linsolves for error estimation".format(
        sol.nfev, sol.njev, sol.nlu, sol.solver.nlusolve, sol.solver.nlusolve_errorest))

  results[ncase].append(
    (sol.t.size-1, sol.CPUtime, sol.solver.nstep, sol.solver.naccpt, sol.solver.nrejct, sol.solver.nfailed,
     sol.nfev, sol.njev, sol.nlu, sol.solver.nlusolve, sol.solver.nlusolve_errorest))
