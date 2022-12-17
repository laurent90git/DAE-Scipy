#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:54:33 2020

1) Modelling the pendulum

  The pendulum with constant rod-length may be represented in various manner:
  DAEs of index 3 to 0. The present script implement each possibility and compares
  the corresponding solution to the true solution obtained from the simple
  pendulum ODE y'' = -g*sin(y).

  The pendulum DAE of index 3 is obtained simply by using Newton's law on the
  pendulum of mass m in the (x,y) reference frame, x being the horizontal axis,
  and y the vertical axis (positive upwards). The rod force exerced on the mass
  is T. The system reads:

    dx/dt = vx                         (1)
    dydt = vy                          (2)
    d(vx)/dt = -(T/m)*sin(theta)       (3)
    d(vy)/dt =  (T/m)*cos(theta) - g   (4)
    x^2 + y^2 = r0^2                   (5)

  Equation (5) ensures that the rod-length remains constant.
  By expression sin(theta) as x/sqrt(x**2+y**2) and cos(theta)=-y/sqrt(x**2+y**2),
  we may introduce lbda=T/sqrt(x**2+y**2) and rewrite Equations (4) and (5) as:

    d(vx)/dt = -lbda*x/m               (3a)
    d(vy)/dt = -lbda*y/m - g           (4a)

  The new variable "lbda" plays the role of a Lagrange multiplier, which adjusts
  istelf in time so that equation (5) is not violated, i.e. the solution remains
  on the manifold x^2 +y^2 - r0^2 = 0.

  The corresponding system (1,2,3a,4a,5) is a DAE of index 3.
  We can obtain lower-index DAE by differentiating equation (5) with respect
  to time. If we do it once, we obtain:

    x*vx + y*vy = 0                    (6)

  Physically, this means that the velocity vector of the pendulum mass is
  always perpendicular to the rod. The system (1,2,3a,3b,6) is a DAE of index 1.
  If we differentiate it once more with respect to time, we obtain:

    lbda*(x^2 + y^2)  + m*(vx^2 + vy^2) - m*g*y = 0     (7)

  The system(1,2,3a,4a,7) is a DAE of index 1. Note that we could directly
  remove lambda from the problem as we now have its expression. Physically,
  equation (7) means that the rod force must equilibrate the centrifugal force
  and the gravity force, so that the mass remains attache at the rod.
  We may differentiate (7) once more to obtain a (long) ODE on lbda:

    d(lbda)/dt = f(x,y,vx,vy)          (8)

  The system (1,2,3a,4a,8) is a DAE Of index 0, i.e. an ODE ! This is actually
  how we define the original index of the first system: the number of times
  equation (5) must be derived so that only ODEs remain.

  2) Numerical solution

    The adapted version of Scipy's Radau, RadauDAE, is able to solve problems
    of the form:
      M y' = f(t,y)       (9)
    even when the mass matrix M is singular. This enables the computation of
    the solution of DAEs expressed in Hessenberg form.

    Radau5 is originally able to adapt the time step thanks to a 3rd-order
    error estimate. This estimate is however not suited for DAEs, as the error
    on DAEs is vastly overestimated (see [1]). Hence in this implementation, the
    error on the algebraic variables is forced to zero, when these are clearly
    identified (those that correspond to a row of zeros in M). Note that [1]
    recommends to multiply the algebraic error by the time step instead.

    Solution of (9) is also possible when M is singular, but without any rows
    full of zeros (see the transistor amplifier example in this git, based on
    [2]).

    Additional information can be printed during the Radau computation by setting
    bPrint to True:
      newton convergence, norm of the error estimate...

  References:
    [1] Hairer, Ernst & Lubich, Christian & Roche, Michel. (1989)
        The numerical solution of differential-algebraic systems
        by Runge-Kutta methods, Springer
    [2] Hairer, Ernst & Wanner, Gerhard. (1996)
        Solving Ordinary Differential Equations II:
        Stiff and Differential-Algebraic Problems

@author: laurent
"""
import numpy as np


#%% Setup the model based on the chosen formulation
def generateSystem(chosen_index, theta_0=np.pi/2, theta_dot0=0., r0=1., m=1, g=9.81):
    """ Generates the DAE function representing the pendulum with the desired
    index (0 to 3).
        Inputs:
          chosen_index: int
            index of the DAE formulation
          theta_0: float
            initial angle of the pendulum in radians
          theta_dot0: float
            initial angular velocity of the pendulum (rad/s)
          r0: float
            length of the rod
          m: float
            mass of the pendulum
          g: float
            gravitational acceleration (m/s^2)
        Outputs:
          dae_fun: callable
            function of (t,x) that represents the system. Will be given to the
            solver.
          jac_dae: callable
            jacobian of dae_fun with respect to x. May be given to the solver.
          mass: array_like
            mass matrix
          Xini: initial condition for the system

    """

    if chosen_index==3:
        def dae_fun(t,X):
          # X= (x,y,xdot=vx, ydot=vy, lbda)
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([vx,
                           vy,
                           -x*lbda/m,
                           -g-(y*lbda)/m,
                            x**2 + y**2 -r0**2])
        mass = np.eye(5) # mass matrix M
        mass[-1,-1]=0
        var_index = np.array([0,0,0,0,3])

        def jac_dae(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([[0.,0.,1.,0.,0.],
                           [0.,0.,0.,1.,0.],
                           [-lbda/m, 0., 0., 0., -x/m],
                           [0.,-lbda/m,  0., 0., -y/m],
                           [2*x, 2*y, 0., 0., 0.]])

    elif chosen_index==2:
        def dae_fun(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([vx,
                           vy,
                           -x*lbda/m,
                           -g-(y*lbda)/m,
                            x*vx+y*vy,])
        mass = np.eye(5)
        mass[-1,-1]=0
        var_index = np.array([0,0,0,0,2])

        def jac_dae(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([[0.,0.,1.,0.,0.],
                           [0.,0.,0.,1.,0.],
                           [-lbda/m, 0., 0., 0., -x/m],
                           [0.,-lbda/m,  0., 0., -y/m],
                           [vx, vy, x, y, 0.]])

    elif chosen_index==1:
        def dae_fun(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([vx,
                           vy,
                           -x*lbda/m,
                           -g-(y*lbda)/m,
                           lbda*(x**2+y**2)/m + g*y - (vx**2 + vy**2)])
        mass = np.eye(5)
        mass[-1,-1]=0
        var_index = np.array([0,0,0,0,1])
        def jac_dae(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([[0.,0.,1.,0.,0.],
                           [0.,0.,0.,1.,0.],
                           [-lbda/m, 0., 0., 0., -x/m],
                           [0.,-lbda/m,  0., 0., -y/m],
                           [2*x*lbda/m, 2*y*lbda/m + g, -2*vx, -2*vy, (x**2+y**2)/m]])

    elif chosen_index==0:
        def dae_fun(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          dvx = -lbda*x/m
          dvy = -lbda*y/m - g
          rsq = x**2 + y**2 # = r(t)^2
          dt_lbda = (1/m)*(  -g*vy/rsq + 2*(vx*dvx+vy*dvy)/rsq  + (vx**2+vy**2-g*y)*(2*x*vx+2*y*vy)/(rsq**2))
          return np.array([vx,
                           vy,
                           dvx,
                           dvy,
                           dt_lbda])
        mass=None # or np.eye(5) the identity matrix
        var_index = np.array([0,0,0,0,0])
        jac_dae = None # use the finite-difference routine, this expression would
                       # otherwise be quite heavy :)
    else:
      raise Exception('index must be in [0,3]')

    # alternatively, define the Jacobian via finite-differences, via complex-step
    # to ensure machine accuracy
    if jac_dae is None:
        import scipy.optimize._numdiff
        jac_dae = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                                    fun=lambda x: dae_fun(t,x),
                                    x0=x, method='cs',
                                    rel_step=1e-50, f0=None,
                                    bounds=(-np.inf, np.inf), sparsity=None,
                                    as_linear_operator=False, args=(),
                                    kwargs={})
    ## Otherwise, the Radau solver uses its own routine to estimate the Jacobian, however the
    # original Scip routine is only adapted to ODEs and may fail at correctly
    # determining the Jacobian because of it would chosse finite-difference step
    # sizes too small with respect to the problem variables (<1e-16 in relative).
    # jac_dae = None

    ## Initial condition (pendulum at an angle)
    x0 =  r0*np.sin(theta_0)
    y0 = -r0*np.cos(theta_0)
    vx0 = r0*theta_dot0*np.cos(theta_0)
    vy0 = r0*theta_dot0*np.sin(theta_0)
    lbda_0 = (m*r0*theta_dot0**2 +  m*g*np.cos(theta_0))/r0 # equilibrium along the rod's axis

     # the initial condition should be consistent with the algebraic equations
    Xini = np.array([x0,y0,vx0,vy0,lbda_0])

    return dae_fun, jac_dae, mass, Xini, var_index


if __name__=='__main__':
    # Test the pendulum
    from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
    import matplotlib.pyplot as plt
    from scipyDAE.radauDAE import RadauDAE
    from scipyDAE.radauDAE import solve_ivp_custom as solve_ivp
    # from scipy.integrate import solve_ivp
    
    
    ###### Parameters to play with
    chosen_index = 3 # The index of the DAE formulation
    tf = 10.0        # final time (one oscillation is ~2s long)
    rtol=1e-6; atol=rtol # relative and absolute tolerances for time adaptation
    # dt_max = np.inf
    # rtol=1e-3; atol=rtol # relative and absolute tolerances for time adaptation
    bPrint=False # if True, additional printouts from Radau during the computation
    bDebug=False # sutdy condition number of the iteration matrix
    method=RadauDAE

    ## Physical parameters for the pendulum
    theta_0=np.pi/6 # initial angle
    theta_dot0=0. # initial angular velocity
    r0=3.  # rod length
    m=1.   # mass
    g=9.81 # gravitational acceleration

    dae_fun, jac_dae, mass, Xini, var_index= generateSystem(chosen_index, theta_0, theta_dot0, r0, m, g)
    # jac_dae = None
    #%% Solve the DAE
    print(f'Solving the index {chosen_index} formulation')
    sol = solve_ivp(fun=dae_fun, t_span=(0., tf), y0=Xini, max_step=tf/10,
                    rtol=rtol, atol=atol, jac=jac_dae, jac_sparsity=None,
                    method=method, vectorized=False, first_step=1e-3, dense_output=True,
                    mass_matrix=mass, bPrint=bPrint, return_substeps=True,
                    max_newton_ite=6, min_factor=0.2, max_factor=10,
                    var_index=var_index,
                    # newton_tol=1e-4,
                    scale_residuals = True,
                    scale_newton_norm = True,
                    scale_error = True,
                    max_bad_ite=1,
                    bDebug=bDebug)
    print("DAE of index {} {} in {} time steps, {} fev, {} jev, {} LUdec".format(
          chosen_index, 'solved'*sol.success+(1-sol.success)*'failed',
          sol.t.size, sol.nfev, sol.njev, sol.nlu))

    # recover the time history of each variable
    x=sol.y[0,:]; y=sol.y[1,:]; vx=sol.y[2,:]; vy=sol.y[3,:]; lbda=sol.y[4,:]
    T = lbda * np.sqrt(x**2+y**2)
    theta= np.arctan(x/y)

    #%% Compute true solution (ODE on the angle in polar coordinates)
    def fun_ode(t,X):
      theta=X[0]; theta_dot = X[1]
      return np.array([theta_dot,
                       -g/r0*np.sin(theta)])

    Xini_ode= np.array([theta_0,theta_dot0])
    sol_ode = solve_ivp(fun=fun_ode, t_span=(0., tf), y0=Xini_ode,
                    rtol=rtol, atol=atol, max_step=tf/10, method=RadauDAE, bPrint=bPrint,
                    dense_output=True, return_substeps=True)
    theta_ode = sol_ode.y[0,:]
    theta_dot = sol_ode.y[1,:]
    x_ode =  r0*np.sin(theta_ode)
    y_ode = -r0*np.cos(theta_ode)
    vx_ode =  r0*theta_dot*np.cos(theta_ode)
    vy_ode =  r0*theta_dot*np.sin(theta_ode)
    T_ode = m*r0*theta_dot**2 + m*g*np.cos(theta_ode)

    #%% Compare the DAE solution and the true solution
    plt.figure()
    plt.plot(sol.t,x, color='tab:orange', linestyle='-', label=r'$x_{DAE}$')
    plt.plot(sol_ode.t,x_ode, color='tab:orange', linestyle='--', label=r'$x_{ODE}$')
    plt.plot(sol.t,y, color='tab:blue', linestyle='-', label=r'$y_{DAE}$')
    plt.plot(sol_ode.t,y_ode, color='tab:blue', linestyle='--', label=r'$y_{ODE}$')
    plt.plot(sol.t,y**2+x**2, color='tab:green', label=r'$x_{DAE}^2 + y_{DAE}^2$')
    plt.grid()
    plt.legend()
    plt.title('Comparison with the true solution')

    #%% Analyze constraint violations
    # we check how well equations (5), (6) and (7) are respected
    fig,ax = plt.subplots(3,1,sharex=True)
    constraint = [None for i in range(3)]
    constraint[2] = x**2 + y**2 - r0**2 #index 3
    constraint[1] = x*vx+y*vy #index 12
    constraint[0] = lbda*(x**2+y**2)/m + g*y - (vx**2 + vy**2) #index 1
    for i in range(len(constraint)):
      ax[i].plot(sol.t, constraint[i])
      # ax[i].semilogx(sol.t, np.abs(constraint[i]))
      ax[i].grid()
      ax[i].set_ylabel('index {}'.format(i+1))
    ax[-1].set_xlabel('t (s)')
    fig.suptitle('Constraints violation')


    #%% Plot the solution and some useful statistics
    fig, ax = plt.subplots(5,1,sharex=True, figsize=np.array([1.5,3])*5)
    i=0
    ax[i].plot(sol.t, x,     color='tab:orange', linestyle='-', linewidth=2, marker='.', label='x')
    ax[i].plot(sol.t, y,     color='tab:blue', linestyle='-', linewidth=2, marker='.', label='y')
    ax[i].plot(sol_ode.t, x_ode,     color='tab:orange', linestyle='--', linewidth=2, marker=None, label='x ODE')
    ax[i].plot(sol_ode.t, y_ode,     color='tab:blue', linestyle='--', linewidth=2, marker=None, label='y ODE')
    ax[i].set_ylim(-1.2*r0, 1.2*r0)
    ax[i].legend(frameon=False)
    ax[i].grid()
    ax[i].set_ylabel('positions')

    i+=1
    ax[i].plot(sol.t, vx,     color='tab:orange', linestyle='-', linewidth=2, marker='.', label='vx')
    ax[i].plot(sol.t, vy,     color='tab:blue', linestyle='-', linewidth=2, marker='.', label='vy')
    ax[i].plot(sol_ode.t, vx_ode,     color='tab:orange', linestyle='--', linewidth=2, marker=None, label='vx ODE')
    ax[i].plot(sol_ode.t, vy_ode,     color='tab:blue', linestyle='--', linewidth=2, marker=None, label='vy ODE')
    ax[i].grid()
    ax[i].legend(frameon=False)
    ax[i].set_ylabel('velocities')

    i+=1
    ax[i].plot(sol.t, T,     color='tab:orange', linestyle='-', linewidth=2, marker='.', label='lbda')
    ax[i].plot(sol_ode.t, T_ode,     color='tab:orange', linestyle='--', linewidth=2, marker=None, label='lbda ODE')
    ax[i].grid()
    ax[i].legend(frameon=False)
    ax[i].set_ylabel('Lagrange multiplier\n(rod force)')

    i+=1
    ax[i].semilogy(sol.t[:-1], np.diff(sol.t), color='tab:blue', linestyle='-', marker='.', linewidth=1, label=r'$\Delta t$ (DAE)')
    ax[i].semilogy(sol_ode.t[:-1], np.diff(sol_ode.t), color='tab:orange', linestyle='--', marker='.', linewidth=1, label=r'$\Delta t$ (ODE)')
    ax[i].grid()
    ax[i].legend(frameon=False)
    ax[i].set_ylabel(r'$\Delta t$ (s)')

    # condition numbers for the LU solves in the Newton loop
    i+=1
    try:
      ax[i].semilogy(sol.solver.info['cond']['t'],     sol.solver.info['cond']['LU_real'],     color='tab:blue', linestyle='-', linewidth=2, label='cond(real) (DAE)')
      ax[i].semilogy(sol_ode.solver.info['cond']['t'], sol_ode.solver.info['cond']['LU_real'], color='tab:blue', linestyle='--', linewidth=2, label='cond(real) (ODE)')
      ax[i].semilogy(sol.solver.info['cond']['t'],     sol.solver.info['cond']['LU_complex'],     color='tab:orange', linestyle='-', linewidth=1, label='cond(complexe) (DAE)')
      ax[i].semilogy(sol_ode.solver.info['cond']['t'], sol_ode.solver.info['cond']['LU_complex'], color='tab:orange', linestyle='--', linewidth=1, label='cond(complexe) (ODE)')
    except Exception as e:
      print(e)
    ax[i].legend(frameon=False)
    ax[i].grid()
    ax[i].set_ylabel('condition\nnumbers')

    ax[-1].set_xlabel('t (s)')

    #%% Plot dense output
    t_eval = []
    for i in range(sol.t.size-1):
        t_eval.extend(  np.linspace(sol.t[i], sol.t[i+1], 20).tolist() )
    t_eval.append(sol.t[-1])
    t_eval = np.array(t_eval)

    dense_sol = sol.sol(t_eval)
    lbda_dense = dense_sol[4,:]

    plt.figure()
    plt.plot(t_eval, lbda_dense, label='dense output')
    plt.plot(sol.t, lbda, label='solution points', linestyle='', marker='o')
    plt.plot(sol.tsub, sol.ysub[-1,:], label='sub solution points', linestyle='', marker='+', color='tab:red')
    plt.grid()
    plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel('lbda')
    plt.title('Dense output')

    #%% Plot dense output for differential variables
    ivar = 2
    plt.figure()
    plt.plot(t_eval, dense_sol[ivar,:], label='dense output')
    plt.plot(sol.t, sol.y[ivar,:], label='solution points', linestyle='', marker='o')
    plt.plot(sol.tsub, sol.ysub[ivar,:], label='sub solution points', linestyle='', marker='+', color='tab:red')
    plt.grid()
    plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel(f'var {ivar}')
    plt.title('Dense output')

    #%% Find period
    t_change = sol.t[:-1][np.diff(np.sign(vx))<0]
    print('Numerical period={:.2e} s'.format(np.mean(np.diff(t_change))))
    print('Theoretical period={:.2e} s'.format(2*np.pi*np.sqrt(r0/g)))
