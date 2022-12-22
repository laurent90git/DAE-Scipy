#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  This is a simple generalisation of the previously modelled pendulum.
  Multiple pendulums are attached in series.
  See the documentation of the single pendulum for more details.

  Note that, for the single pendulum, the Lagrange multiplier represented the
  rod tension divided by the rod length. Here, we have on Lagrange multiplier
  per pendulum, which represents the rod tension (without dividing by the rod
  length). The i-th Lagrange multiplier is thus the force exerted by the
  (i-1)-th rod on the i-th mass.

  Some treatment of the algebraic variables is inspired by [2].

      References
    ----------
    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [2] E. Hairer, C. Lubich, M. Roche, "The Numerical Solution of
           Differential-Algebraic Systems by Runge-Kutta Methods"

@author: laurent
"""
import numpy as np
import scipy.sparse
import scipy.optimize._numdiff
import time as pytime
from scipyDAE.tests.pendulum import computeAngle
g=9.81


#%% Setup the model based on the chosen formulation
def generateSystem(n=50,initial_angle=np.pi/4, chosen_index=3):
    """ Generates the DAE function representing the pendulum with the desired
    index (0 to 3).
        Inputs:
          n: int
            number of pendulums attached in series
          initial_angle:
            initial angle of the chord formed by the pendulum (0=vertical, pi/2=horizontal)
          chosen_index: int
            index of the DAE formulation
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

    ###### Initial condition
    ## Initial conditions for each pendulum
    theta_0     = np.array([initial_angle for i in range(n)])
    # theta_0[-1] = np.pi/2
    # theta_0     = np.array([np.pi/2 for i in range(n)])
    # # theta_0     = np.array([0. for i in range(n)])
    # theta_0[-1] = np.pi/3
    theta_dot0  = np.array([0. for i in range(n)]);

    if 1: # chord with a mass at the end
        m_chord = 1.
        m_final = 1.
        L_chord = 2.
        m_node = m_chord / (n-1)
        L_rod  = L_chord / n

        r0  = np.array([L_rod for i in range(n)])
        r0s = r0**2
        m   = np.array([m_node for i in range(n)])
        m[-1] = m_final
    else: # also at the middle
        m_chord  = 1.
        m_final  = 5.
        m_middle = 5.

        L_chord = 2.
        m_node = m_chord / n
        L_rod  = L_chord / n

        r0  = np.array([L_rod for i in range(n)])
        r0s = r0**2
        m   = np.array([m_node for i in range(n)])
        m[-1] = m_final
        m[len(m)//2]=m_middle

    ###### DAE function and Jacobian
    if chosen_index==3:
        def dae_fun(t,X):
          x=X[0::5]; y=X[1::5]; vx=X[2::5]; vy=X[3::5]; lbda=X[4::5]
          dXdt = np.empty((5,x.size), order='C', dtype=X.dtype)
          dt_x, dt_y, dt_vx, dt_vy, constraints = dXdt

          dt_x[:] = vx
          dt_y[:] = vy

          # dx = np.hstack((x[0], np.diff(x))) # distances between each node, accounting for the fixed zero-th node
          # dy = np.hstack((y[0], np.diff(y)))
          dx = np.empty_like(x)
          dy = np.empty_like(x)
          dx[0]  = x[0]; dx[1:] =  np.diff(x) # distances between each node, accounting for the fixed zero-th node
          dy[0]  = y[0]; dy[1:] =  np.diff(y) # distances between each node, accounting for the fixed zero-th node

          rs  = dx**2 + dy**2

          dt_vx[:-1] = (1/m[:-1]) * (+lbda[:-1]*dx[:-1]/rs[:-1] - lbda[1:]*dx[1:]/rs[1:])
          dt_vy[:-1] = (1/m[:-1]) * (+lbda[:-1]*dy[:-1]/rs[:-1] - lbda[1:]*dy[1:]/rs[1:] - m[:-1]*g)

          dt_vx[-1]  = (1/m[-1]) *  (+lbda[-1]*dx[-1]/rs[-1])
          dt_vy[-1]  = (1/m[-1]) *  (+lbda[-1]*dy[-1]/rs[-1] - m[-1]*g)

          # constraints = rs**0.5 - r0s**0.5
          constraints[:] = rs - r0s
          return dXdt.reshape((-1,), order='F')

        diag_mass = np.ones((n*5,))
        var_index = np.zeros((n*5,))
        for i in range(n):
          diag_mass[4 + i*5] = 0
          var_index[2 + i*5] = 2 # weird but so is it specified in [1]
          var_index[3 + i*5] = 2
          var_index[4 + i*5] = 3
        mass = scipy.sparse.diags(diag_mass, format=('csc'))
        jac_dae = None

    elif chosen_index==2:
        def dae_fun(t,X):
          x=X[0::5]; y=X[1::5]; vx=X[2::5]; vy=X[3::5]; lbda=X[4::5]
          dt_x = vx
          dt_y = vy
          dt_vx = np.zeros_like(x)
          dt_vy = np.zeros_like(x)
          constraints = np.zeros_like(x)

          dx = np.hstack((x[0], np.diff(x))) # distances between each node, accounting for the fixed zero-th node
          dy = np.hstack((y[0], np.diff(y)))
          rs  = dx**2 + dy**2

          dt_vx[:-1] = (1/m[:-1]) * (+lbda[:-1]*dx[:-1]/rs[:-1] - lbda[1:]*dx[1:]/rs[1:])
          dt_vy[:-1] = (1/m[:-1]) * (+lbda[:-1]*dy[:-1]/rs[:-1] - lbda[1:]*dy[1:]/rs[1:] - m[:-1]*g)

          dt_vx[-1]  = (1/m[-1]) *  (+lbda[-1]*dx[-1]/rs[-1])
          dt_vy[-1]  = (1/m[-1]) *  (+lbda[-1]*dy[-1]/rs[-1] - m[-1]*g)

          constraints[1:] = 2*x[1:]*vx[1:] + 2*x[:-1]*vx[:-1] - 2*x[1:]*vx[:-1] - 2*x[:-1]*vx[1:] + \
                            2*y[1:]*vy[1:] + 2*y[:-1]*vy[:-1] - 2*y[1:]*vy[:-1] - 2*y[:-1]*vy[1:]
          constraints[0] =  2*x[0]*vx[0] + 2*y[0]*vy[0] # first node is like a single pendulum
          return np.vstack((dt_x, dt_y, dt_vx, dt_vy, constraints)).reshape((-1,), order='F')

        diag_mass = np.ones((n*5,))
        var_index = np.zeros((n*5,))
        for i in range(n):
          diag_mass[4 + i*5] = 0.
          var_index[4 + i*5] = 2
        mass = np.diag(diag_mass)
        jac_dae = None

    else:
      raise Exception('Only index 3 is implemented')

    sparsity = scipy.sparse.diags(diagonals=[np.ones(n*5-abs(i)) for i in range(-10,10)], offsets=[i for i in range(-10,10)],
                                  format='csc')


    ## Initial condition (pendulum at an angle)
    x0 =  np.cumsum( r0*np.sin(theta_0))
    y0 =  np.cumsum(-r0*np.cos(theta_0))
    vx0 = np.cumsum( r0*theta_dot0*np.cos(theta_0))
    vy0 = np.cumsum( r0*theta_dot0*np.sin(theta_0))
    lbda0 = np.zeros(n)#(m*r0*theta_dot0**2 +  m*g*np.cos(theta_0))/r0 # equilibrium along the rod's axis
    # the initial value for lbda does not change the solution, since this variable will be readjusted by the solver

    # the initial condition should be consistent with the algebraic equations
    Xini =np.vstack((x0, y0, vx0, vy0, lbda0)).reshape((-1,), order='F')

    T_th = 2*np.pi*np.sqrt(L_chord/g) # theoretical period for a single pendulum of equal length

    return dae_fun, jac_dae, sparsity, mass, Xini, var_index, T_th, m

#%%
if __name__=='__main__':
    # Test the pendulum
    # from scipy.integrate import solve_ivp
    from scipyDAE.radauDAE import solve_ivp_custom as solve_ivp
    from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
    import matplotlib.pyplot as plt

    from scipyDAE.radauDAE import RadauDAE
    # from radauDAE_subjac import RadauDAE
    ###### Parameters to play with
    n = 1000
    initial_angle = np.pi/4
    chosen_index = 3 # The index of the DAE formulation
    rtol=1e-5; atol=rtol # relative and absolute tolerances for time adaptation
    bPrint=False # if True, additional printouts from Radau during the computation
    method=RadauDAE

    dae_fun, jac_dae, sparsity, mass, Xini, var_index, T_th, m = \
        generateSystem(n, initial_angle, chosen_index)

    tf = 2*T_th # simulate 2 periods
    # jac_dae = None

    #%%
    if 0:
        #%% Initial state, study Jacobian
        jacfull_fun = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                            fun=lambda x: dae_fun(t,x),
                            x0=x, method='2-point',
                            rel_step=1e-8, f0=None,
                            bounds=(-np.inf, np.inf), sparsity=None,
                            kwargs={})
        jacobian = jacfull_fun(0,Xini+(1e-3+ abs(Xini)*1e-3)*np.random.random(Xini.size))

        dae_fun(0,Xini+(1e-3+ abs(Xini)*1e-3)*np.random.random(Xini.size))

        plt.figure()
        plt.spy(jacobian)
        for i in range(n):
          plt.axhline(i*5-0.5, color='tab:gray', linewidth=0.5)
          plt.axvline(i*5-0.5, color='tab:gray', linewidth=0.5)
        # plt.grid()
        plt.title('Jacobian of the DAE function at perturbed initial state')

    #%% Solve the DAE
    print(f'Solving the index {chosen_index} formulation')
    t_start = pytime.time()
    sol = solve_ivp(fun=dae_fun, t_span=(0., tf), y0=Xini, max_step=tf/10,
                    rtol=rtol, atol=atol, jac=jac_dae, jac_sparsity=sparsity,
                    method=method, vectorized=False, first_step=1e-3, dense_output=True,
                    mass_matrix=mass, bPrint=bPrint, bPrintProgress=True,
                    max_newton_ite=7, min_factor=0.2, max_factor=10,
                    var_index=var_index,
                    # newton_tol=1e-4,
                    scale_residuals = True,
                    scale_newton_norm = True,
                    scale_error = True,
                    zero_algebraic_error = False,
                    max_bad_ite=2,
                    nmax_step=10000,
                    )
    t_end = pytime.time()
    print("DAE of index {} {} in {} s, {} time steps, {} fev, {} jev, {} LUdec".format(
          chosen_index, 'solved'*sol.success+(1-sol.success)*'failed', t_end-t_start,
          sol.t.size, sol.nfev, sol.njev, sol.nlu))
    # print("DAE of index {} {} in {} time steps, {} fev, {} jev, {} LUdec, {} LU solves".format(
    #       chosen_index, 'solved'*sol.success+(1-sol.success)*'failed',
    #       sol.t.size, sol.nfev, sol.njev, sol.nlu, sol.solver.nlusove))

    # recover the time history of each variable
    x=sol.y[0::5,:]; y=sol.y[1::5,:]; vx=sol.y[2::5,:]; vy=sol.y[3::5,:]; lbda=sol.y[4::5,:]
    theta= np.arctan(x/y)
    t = sol.t

    dx = np.vstack((x[0,:], np.diff(x,axis=0)))
    dy = np.vstack((y[0,:], np.diff(y,axis=0)))
    r = (dx**2 + dy**2)**0.5
    # theta = np.arctan(dx/dy)
    theta = computeAngle(dx,dy) #np.angle(dx + 1j * dy)

    np.diff(theta, axis=0)*180/np.pi
    length = np.sum(r,axis=0)

    #%% Spy Jacobian
    if jac_dae is not None:
      plt.figure()
      # jacobian = jac_dae(sol.t[0], sol.y[:,0])
      jacobian = jac_dae(sol.t[-1], sol.y[:,-1])
      plt.spy(jacobian)
      for i in range(n):
        plt.axhline(i*5-0.5, color='tab:gray', linewidth=0.5)
        plt.axvline(i*5-0.5, color='tab:gray', linewidth=0.5)
      # plt.grid()
      plt.title('Jacobian of the DAE function')

    #%% plot total energy
    Ec = 0.5*(vx**2+vy**2).T.dot(m)
    Ep = g*y.T.dot(m)

    plt.figure(dpi=200)
    plt.plot(t,Ec,color='r', label='Ec')
    plt.plot(t,Ep,color='b', label='Ep')
    plt.plot(t,Ec+Ep,color='g', label='Etot', linestyle='--')
    plt.legend()
    plt.grid()
    plt.xlabel('t (s)')
    plt.ylabel('Energy (J)')

    #%% Relative angles
    # It does not make much sense to use complex numbers, but anyway...
    plt.figure()
    # projvec = dx[:-1] + 1j*dy[:-1]
    # projvec_norm = (projvec.real**2 + projvec.imag**2)**0.5
    # projvec = projvec / projvec_norm

    # vectors = dx[1:] + 1j*dy[1:]
    # vectors_norm = (vectors.real**2 + vectors.imag**2)**0.5

    # colinear_part = (vectors.real * projvec.real +  vectors.imag * projvec.imag )* projvec
    # colinear_part_norm = (colinear_part.real**2 + colinear_part.imag**2)**0.5

    # perpendicular_part = vectors - colinear_part * projvec
    # perpendicular_part_norm = (perpendicular_part.real**2 + perpendicular_part.imag**2)**0.5

    # assert np.allclose(perpendicular_part_norm**2 + colinear_part_norm**2, vectors_norm**2)

    # angles = computeAngle(x=colinear_part, y=perpendicular_part)

    # plt.plot(t, angles.T)
    plt.plot(t, np.diff(theta.T,axis=1))
    plt.grid()
    plt.xlabel('t (s)')
    plt.ylabel('angle (rad)')
    plt.title('Relative angle between successive rods')

    #%% plot joint forces
    plt.figure()
    plt.semilogy(t, np.abs(lbda.T))
    plt.grid()
    plt.xlabel('t (s)')
    plt.ylabel('absolute rod tension (N)')

    #%%
    plt.figure(figsize=(5,5))
    for i in range(n):
      plt.plot(x[i,:], y[i,:], linestyle='--')
    plt.grid()

    # add first and last form
    plt.plot( np.hstack((0., x[:,-1])), np.hstack((0.,y[:,-1])), color='r')
    plt.plot( np.hstack((0., x[:,0])), np.hstack((0.,y[:,0])), color=[0,0,0])
    plt.scatter(0.,0.,marker='o', color=[0,0,0])
    for i in range(n):
      for j in [0,-1]:
        plt.scatter(x[i,j],y[i,j],marker='o', color=[0,0,0])

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    L0 = length[0]
    plt.ylim(-L0,L0)
    plt.xlim(-L0,L0)

    #%% Find period
    Ep_mean = (np.max(Ep)+np.min(Ep))/2
    t_change = t[:-1][np.diff(np.sign(Ep-Ep_mean))<0]
    print('Numerical period={:.2e} s'.format(2*np.mean(np.diff(t_change)))) # energy oscillates twice as fast
    print('Theoretical period={:.2e} s'.format(T_th))

#%%
if 0:
  #%% Create animation
  import numpy as np
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.animation  as animation

  fig = plt.figure(figsize=(5, 5), dpi=200)
  xmax = 1.2*np.max(np.abs(x))
  ymax = max(1.0, 1.2*np.max(y))
  ymin = 1.2*np.min(y)
  ax = fig.add_subplot(111, autoscale_on=False, xlim=(-xmax,xmax), ylim=(ymin,ymax))
  ax.grid()
  ax.set_aspect('equal')

  lines, = plt.plot([], [], linestyle='-', linewidth=1.5, marker=None, color='tab:blue') # rods
  marker='o'
  # if n<50:
  #     marker='o'
  # else:
  #     marker=None
  points, = plt.plot([],[], marker=marker, linestyle='', color=[0,0,0]) # mass joints
  # paths  = [plt.plot([],[], alpha=0.2, color='tab:orange')[0] for i in range(n)] # paths of the pendulum
  time_template = r'$t = {:.2f}$ s'
  time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

  def init():
      return lines,points

  def update(t):
      interped_sol = sol.sol(t)
      x,y = np.hstack((0.,interped_sol[::5])), np.hstack((0.,interped_sol[1::5]))
      lines.set_data(x,y)
      # points.set_data(x,y)
      # points.set_data(x[-1],y[-1]) # only plot last mass
      points.set_data(x[-1],y[-1]) # only plot last mass
      time_text.set_text(time_template.format(t))

      return lines, time_text, points

  if 0:# test layout
    init()
    update(sol.t[-1]/2)
  else:
    # compute how many frames we want for real time
    fps=30
    total_frames = np.ceil((sol.t[-1]-sol.t[0])*fps).astype(int)

    from tqdm import tqdm
    ani = animation.FuncAnimation(fig, update, frames=tqdm(np.linspace(sol.t[0], sol.t[-1],total_frames)),
                        init_func=init, interval=200, blit=True)
    ani.save('animation_new12.gif', writer='imagemagick', fps=30)
    # writer = animation.writers['ffmpeg'](fps=24, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('animation.mp4', writer=writer)
  plt.show()



