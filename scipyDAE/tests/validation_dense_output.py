#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:15 2022

Test de la dense output sur une DAE possédant une solution analytique qui est un
polynome du temps.

Ainsi, pour des polynomes de degré suffisamment faible, la dense output devrait
être exacte.

y(t) = sum(k=0 to k=n) a_k * t^k

y'(t) = sum(k=1 to k=n) k * a_k * t^(k-1)

@author: lfrancoi
"""
import numpy as np
from scipyDAE.radauDAE import RadauDAE
from scipyDAE.radauDAE import solve_ivp_custom as solve_ivp


def generate_problem(a=np.array([1.,0.]), p=1):
    """ The problem has the form y'=f(t,y,z), with 
        an additional algebraic and:
          0 = z - y (index 1)
          0 = y - polyval(a,t)
    
        z is therefore an index-1 algebraic variable
        f is computed such that y(t) = np.polyval(a,t)
        """
    poly_der = np.flip( np.array([k*ak for k,ak in enumerate(np.flip(a))])[1:])
    def fun(t,x):
        y,z=x
        ytarget = np.polyval(a,t)
        yder_target = np.polyval(poly_der,t)
        if p==1: # index 1
          return np.array([ yder_target,
                            z-y])
        elif p==2: # index 2
          return np.array([ yder_target + (z-y),
                            y - ytarget])
        else:
          raise Exception('not implemented')
    jac = None
    
    # construct singular mass matrix
    mass_matrix = np.eye(2)
    mass_matrix[-1, -1] = 0
    if p==1:
      var_index = np.array([0,1])
    elif p==2:
      var_index = np.array([0,2])

    # initial conditions
    y0 = np.array([a[-1], a[-1]])
    # y0 = np.array([a[-1], 0])
    t_span = (0, 2.)
    return y0, mass_matrix, var_index, fun, jac, t_span, poly_der


if __name__ == "__main__":
    
    # whished degree
    for deg in range(0,5):
        # deg = 3 # if <=3, dense output is the exact solution up to machine accuracy
        # if deg is higher than 3, the stage order is not sufficiently high
        # to exactly resolve the algebraic variables if the index is 2 or more
        a = np.array([0.01, 0.02, .05, -0.1, -0.4, 0.5, 1., 2.])
        assert deg<=(len(a)+1)
        a = np.flip(np.flip(a)[:deg+1])
        
        
        y0, mass_matrix, var_index, fun, jac, t_span, poly_der = generate_problem(
            a=a, p=2
            )
    
        dt = 1.0/4
        rtol=1e-4
        sol = solve_ivp(
                        fun=fun,
                        t_span=t_span,
                        y0=y0,
                        rtol=rtol,
                        atol=rtol/10,
                        jac=jac,
                        method=RadauDAE,
                        first_step=dt, max_step=dt,
                        min_factor=0.2, max_factor=10,
                        mass_matrix=mass_matrix,
                        bPrint=False,
                        dense_output=True,
                        max_newton_ite=6,
                        var_index=var_index,
                        # newton_tol=1e-10 / rtol,
                        zero_algebraic_error = True,
                        scale_residuals = True,
                        scale_newton_norm = False,
                        scale_error = True,
                        max_bad_ite=2,
                        jacobianRecomputeFactor=1e-3,
                        bAlwaysApply2ndEstimate=True,
                        bUsePredictiveController=True,
                        bUsePredictiveNewtonStoppingCriterion=False,
                        bUseExtrapolatedGuess=True,
                        bPerformAllNewtonIterations=True,
                        bReport=True,
                        return_substeps=True,
                        )
    
        assert sol.success
    
        sol.reports = sol.solver.reports
        for key in sol.reports.keys():
           sol.reports[key] = np.array(sol.reports[key])
        if sol.success:
            state='solved'
        else:
            state='failed'
      
        print("\nDAE {}".format(state))
        print("\t{} time steps in {}s\n\t({} = {} accepted + {} rejected + {} failed)".format(
          sol.t.size-1, sol.CPUtime, sol.solver.nstep, sol.solver.naccpt, sol.solver.nrejct, sol.solver.nfailed))
        print("\t{} fev, {} jev, {} LUdec, {} linsolves, {} linsolves for error estimation".format(
              sol.nfev, sol.njev, sol.nlu, sol.solver._nlusolve, sol.solver._nlusolve_errorest))
        
        #%%
        t,y=sol.t,sol.y
        import matplotlib.pyplot as plt
    
        fig, ax = plt.subplots()
        ax.plot(t, y[0], linestyle='-', color="tab:red", label="y")
        ax.plot(t, y[1], linestyle='', marker='.', linewidth=4, color="tab:green", label="z")
        t_analytical = np.linspace(t[0], t[-1], 1000)
        ax.plot(t_analytical, np.polyval(a,t_analytical), marker='', linestyle='--', color=[0,0,0], label='analytical')
        ax.grid()
        ax.legend()
        ax.set_title(f'Solution (without dense output) for deg={deg}')
    
        #%% dense output
        for ivar in range(y0.size):
          fig, ax = plt.subplots()
          # ax.plot(t, y[ivar], linestyle=':', color="tab:red", label="", marker='o',
                  # linewidth=4)
          ax.plot(t_analytical, sol.sol(t_analytical)[ivar], linestyle='-', marker=None,
                  color="tab:green", label="dense output")
          ax.plot(t_analytical, np.polyval(a,t_analytical), marker='', linestyle='--', color=[0,0,0], label='analytical')
          for tt in t: ax.axvline(tt, color=[0,0,0,0.5], label=None)
          # plot substeps
          ax.plot(sol.tsub, sol.ysub[ivar], linestyle='', marker='*', markersize=10, color='tab:red', label='substeps')
          ax.grid()
          ax.legend()
          ax.set_title(f'var {ivar}')
          fig.suptitle(f'Solution (without dense output) for deg={deg}')
        
        #%% Relative error between dense output of y (differential variable) and z (algebraic)
        sol_dense = sol.sol(t_analytical)
        # assert np.allclose(sol_dense[1], sol_dense[0], rtol=1e-12, atol=1e-12)
        rel_error = np.abs(sol_dense[1] - sol_dense[0]) / ( 1e-10 + abs(sol_dense[0]) )
        # plt.figure()
        # plt.plot(t_analytical, rel_error)
        # plt.xlabel('t (s)')
        # plt.ylabel('dense output error')
        
        if deg<=3:
            assert np.max(np.abs(rel_error))<1e-12, f"the dense output is not exact for degree {deg} whereas it should be !"
        
        #%% Compare y and z
        ivar=0
        plt.figure()
        plt.semilogy(sol.tsub, abs(sol.ysub[0]-sol.ysub[1]),
                     linestyle='', marker='*', markersize=10,
                     color='tab:red', label='substeps')
        plt.xlabel('t')
        plt.ylabel('|y-z|')
        plt.grid()
        
        #%% Residuals
        plt.figure()
        plt.semilogy(sol.tsub, abs(sol.fsub[1]))
        