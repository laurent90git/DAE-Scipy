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

@author: laurent
"""
import numpy as np


n_pendulum = 100
n=n_pendulum

g=9.81

## Initial conditions for each pendulum
theta_0     = np.array([3*np.pi/4 for i in range(n)])
# theta_0[-1] = np.pi/2
# theta_0     = np.array([np.pi/2 for i in range(n)])
# # theta_0     = np.array([0. for i in range(n)])
# theta_0[-1] = np.pi/3
theta_dot0  = np.array([0. for i in range(n)]);


r0  = np.array([1. for i in range(n)])
r0s = r0**2
m   = np.array([1. for i in range(n)])
# m[-1] = 3e-1

#%% Setup the model based on the chosen formulation  
def generateSytem(n,chosen_index=3):
    """ Generates the DAE function representing the pendulum with the desired
    index (0 to 3).
        Inputs:
          n: int
            number of pendulums attached in series
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
    
    if chosen_index==3:
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
          
          constraints = rs - r0s
          return np.vstack((dt_x, dt_y, dt_vx, dt_vy, constraints)).reshape((-1,), order='F')
        
        mass = np.eye(n*5)
        for i in range(n):
          mass[4 + i*5] = 0.
        jac_dae = None
    else:
      raise Exception('Only index 3 is implemented')
      
    # alternatively, define the Jacobian via finite-differences, via complex-step
    # to ensure machine accuracy
    if jac_dae is None:
        import scipy.optimize._numdiff
        import scipy.sparse
        # sparsity = scipy.sparse.diags(diagonals=[np.ones(n*5-abs(i)) for i in range(-7,7)], offsets=[i for i in range(-7,7)])
        sparsity = None
        jac_dae = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                                    fun=lambda x: dae_fun(t,x),
                                    x0=x, method='cs',
                                    rel_step=1e-50, f0=None,
                                    bounds=(-np.inf, np.inf), sparsity=sparsity,
                                    as_linear_operator=False, args=(),
                                    kwargs={})      
    ## Otherwise, the Radau solver uses its own routine to estimate the Jacobian, however the
    # original Scip routine is only adapted to ODEs and may fail at correctly
    # determining the Jacobian because of it would chosse finite-difference step
    # sizes too small with respect to the problem variables (<1e-16 in relative).
    # jac_dae = None
    
    ## Initial condition (pendulum at an angle)    
    x0 =  np.cumsum( r0*np.sin(theta_0))
    y0 =  np.cumsum(-r0*np.cos(theta_0))
    vx0 = np.cumsum( r0*theta_dot0*np.cos(theta_0))
    vy0 = np.cumsum( r0*theta_dot0*np.sin(theta_0))
    lbda0 = np.zeros(n)#(m*r0*theta_dot0**2 +  m*g*np.cos(theta_0))/r0 # equilibrium along the rod's axis

    
    # the initial condition should be consistent with the algebraic equations
    Xini =np.vstack((x0, y0, vx0, vy0, lbda0)).reshape((-1,), order='F')
                                 
    return dae_fun, jac_dae, mass, Xini

#%%
if __name__=='__main__':
    # Test the pendulum
    from scipy.integrate import solve_ivp
    from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
    import matplotlib.pyplot as plt
    
    from radauDAE import RadauDAE
    ###### Parameters to play with
    chosen_index = 3 # The index of the DAE formulation
    tf = 50.0       # final time (one oscillation is ~2s long)
    rtol=1e-3; atol=rtol # relative and absolute tolerances for time adaptation
    bPrint=False # if True, additional printouts from Radau during the computation
    method=RadauDAE
    
    dae_fun, jac_dae, mass, Xini = generateSytem(n_pendulum, chosen_index)
    
    ##TODO: find the correct lbda by ensuring that constraints are satisfied at=0,
    ##    --> need to differentiate the cosntraint twice to let lbda appear
    # def objfun(x):
    #   X = np.zeros(Xini.shape, dtype=x.dtype)
    #   X[:] = Xini
    #   X[4::5] = x # replace lambda
    #   f = dae_fun(0.,X)
    #   return np.sum(f[4::5]**2)
    # 
    # jac_objfun = lambda x: scipy.optimize._numdiff.approx_derivative(
    #                             fun=objfun,
    #                             x0=x, method='cs',
    #                             rel_step=1e-50, f0=None,
    #                             bounds=(-np.inf, np.inf), sparsity=None,
    #                             as_linear_operator=False, args=(),
    #                             kwargs={})
    # 
    # import scipy.optimize
    # out = scipy.optimize.minimize(fun=objfun, jac=jac_objfun, x0=np.zeros(n_pendulum), tol=1e-9)
    # jac_dae = None
    
    #%% Solve the DAE
    print(f'Solving the index {chosen_index} formulation')
    sol = solve_ivp(fun=dae_fun, t_span=(0., tf), y0=Xini, max_step=tf/10,
                    rtol=rtol, atol=atol, jac=jac_dae, jac_sparsity=None,
                    method=method, vectorized=False, first_step=1e-3, dense_output=True,
                    mass=mass, bPrint=bPrint, bPrintProgress=True)
    print("DAE of index {} {} in {} time steps, {} fev, {} jev, {} LUdec".format(
          chosen_index, 'solved'*sol.success+(1-sol.success)*'failed',
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
    theta = np.angle(dx + 1j * dy)
    np.diff(theta, axis=0)*180/np.pi
    
    #%% Spy Jacobian
    if jac_dae is not None:
      plt.figure()
      # jacobian = jac_dae(sol.t[0], sol.y[:,0])
      jacobian = jac_dae(sol.t[-1], sol.y[:,-1])
      plt.spy(jacobian)
      for i in range(n_pendulum):
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
    
    #%% plot joint forces
    plt.figure()
    plt.semilogy(t, np.abs(lbda.T))
    plt.grid()
    plt.xlabel('t (s)')
    plt.ylabel('absolute rod tension (N)')
    
    #%%
    plt.figure()
    for i in range(n_pendulum):
      plt.plot(x[i,:], y[i,:], linestyle='--')
    plt.grid()
    
    # add first and last form
    plt.plot( np.hstack((0., x[:,-1])), np.hstack((0.,y[:,-1])), color='r')
    plt.plot( np.hstack((0., x[:,0])), np.hstack((0.,y[:,0])), color=[0,0,0])
    plt.scatter(0.,0.,marker='o', color=[0,0,0])
    for i in range(n_pendulum):
      for j in [0,-1]:
        plt.scatter(x[i,j],y[i,j],marker='o', color=[0,0,0])  
      
    plt.axis('equal')
    plt.xlabel('x') 
    plt.ylabel('y')
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
  points, = plt.plot([],[], marker='o', linestyle='', color=[0,0,0]) # mass joints
  # paths  = [plt.plot([],[], alpha=0.2, color='tab:orange')[0] for i in range(n_pendulum)] # paths of the pendulum
  time_template = r'$t = {:.2f}$ s'  
  time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

  def init():
      return lines,points #paths
  
  def update(t):
      print(t)
      interped_sol = sol.sol(t)
      x,y = np.hstack((0.,interped_sol[::5])), np.hstack((0.,interped_sol[1::5]))
      lines.set_data(x,y)
      points.set_data(x,y)
      time_text.set_text(time_template.format(t))

      # t_path = np.linspace(sol.t[0], t, )
      # interped_sol = sol.sol(t)
      # x,y = interped_sol[::5], interped_sol[1::5]
      # paths.set_data

      return lines, time_text, points#, paths
  if 0:# test layout
    init()
    update(sol.t[-1]/2)
  else:
    # compute how many frames we want for real time
    fps=8
    total_frames = np.ceil((sol.t[-1]-sol.t[0])*fps).astype(int)
    
    ani = animation.FuncAnimation(fig, update, frames=np.linspace(sol.t[0], sol.t[-1],total_frames),
                        init_func=init, interval=200, blit=True)
    # ani.save('/tmp/animation.gif', writer='imagemagick', fps=30)
    writer = animation.writers['ffmpeg'](fps=24, metadata=dict(artist='Me'), bitrate=1800)  
    ani.save('animation.mp4', writer=writer)
  plt.show()
  
  
    
