#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 11:23:12 2025


Problematic test case from
https://github.com/laurent90git/DAE-Scipy/issues/8

@author: lfrancoi
"""

# Load required libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
k   = 14.7 / 1e1               # reaction rate constant [1/(mol/L)/min]
Ca0 = 0.0209              # initial concentration of A [mol/L]
Cb0 = Ca0 / 3             # initial concentration of B [mol/L]
Cc0 = 0.0                 # initial concentration of C [mol/L]

# Time span
t_span = (0, 20)
t_eval = np.linspace(*t_span, 201)

# Solver tolerance settings for error control:
rtol=1e-6; atol=rtol/10 #1e-6 # absolute and relative tolerances for time step adaptation


#%% ODE Approach

# ODE system: returns [dCa/dt]
def batch_reactor_ode(t, y):
    Ca = y
    Cb = Cb0 - (Ca0 - Ca)
    dCadt = -k * Ca * Cb
    return dCadt

# Solve using BDF method with mass matrix
sol_ode = solve_ivp(fun=batch_reactor_ode,
                t_span=t_span, y0=[Ca0], rtol=rtol, atol=atol,
                method='Radau', t_eval=t_eval)

# Extract solution
t  = sol_ode.t
Ca = sol_ode.y[0]
Cb = Cb0 - (Ca0 - Ca)
Cc = Cc0 + (Ca0 - Ca)

# Plot results
plt.plot(t, Ca, 'b', label='Ca')
plt.plot(t, Cb, 'm', label='Cb')
plt.plot(t, Cc, 'g', label='Cc')
plt.xlabel('Time [min]')
plt.ylabel('Concentration [mol/L]')
plt.title('Batch Reactor DAE Solution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% DAE Approach

# Choice of integration method
from scipyDAE.radauDAE import RadauDAE

# Set the solution method
method=RadauDAE

# DAE system: returns [dCa/dt, algebraic1, algebraic2]
def batch_reactor_dae(t, y):
    Ca, Cb, Cc = y
    eq1 = -k * Ca * Cb
    eq2 = Cb - (Cb0 - (Ca0 - Ca))
    eq3 = Cc - (Ca0 - Ca)
    return np.array([eq1, eq2, eq3])

import scipy.optimize._numdiff
jac_dae = lambda t,x: scipy.optimize._numdiff.approx_derivative(fun=lambda x: batch_reactor_dae(t,x),
                                                                x0=x, method='2-point',
                                                                rel_step=1e-8, f0=None,
                                                                bounds=(-np.inf, np.inf), sparsity=None,
                                                                as_linear_operator=False, args=(), kwargs={})

# Initial conditions for the DAE system [X, T, rAV]:
y0 = np.array([Ca0, Cb0, Cc0]) # Initial values for the dependent variables


# Mass matrix for DAE system classification:
# Diagonal elements indicate equation type:
# Mass matrix: 1 for differential, 0 for algebraic
mass_matrix = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

# Variable index for DAE classification:
# 0 = differential variable (evolves according to differential equation)
# 1 = algebraic variable (constrained by algebraic equation)
var_index = np.array([0,1,1]) # index of the components

# Solve the Differential-Algebraic Equation system using RadauDAE method
sol_dae = solve_ivp(fun=batch_reactor_dae,     # DAE system function
                t_span=t_span,      # Time integration range [start, end]
                # jac=jac_dae, # jacobian
                y0=y0,                # Initial condition vector
                max_step=np.inf,       # No upper limit on time step size
                rtol=rtol,            # Relative tolerance for error control
                atol=atol,            # Absolute tolerance for error control
                method=method,         # Using RadauDAE solver for stiff DAEs
                vectorized=False,      # Function is not vectorized (single evaluation)
                first_step=1e-8,       # Initial time step size [s] (very small for startup)
                dense_output=True,     # Enable solution interpolation between points
                
                # RadauDAE-specific parameters for Newton iterations:
                max_newton_ite=20,    # Maximum Newton iterations per time step
                max_bad_ite=3,        # Maximum failed step attempts before reducing step size
                min_factor=0.2,        # Minimum step size reduction factor (80% reduction max)
                max_factor=10,         # Maximum step size increase factor (10x increase max)
                var_index=var_index,   # Variable type classification array
                # newton_tol=1e-3,       # Tolerance for Newton iteration convergence
                
                # Numerical scaling options for better conditioning:
                scale_residuals = False,   # Scale residuals to improve numerical stability
                scale_newton_norm = False, # Scale Newton update norms for better convergence
                scale_error = False,       # Scale error estimates for adaptive step control
                bPrint=False,
                mass_matrix=mass_matrix)      # Mass matrix defining DAE structure

    
for sol, name in ((sol_dae, 'DAE'), (sol_ode, 'ODE')):
    # Modern SciPy OdeResult already exposes status and message
    state = 'solved' if sol.success else 'failed'
    print(f"\n{name} {state}")
    #print(f"\t{sol.t.size-1} time steps in {sol.CPUtime:.3f}s")
    print(f"\tstatus={sol.status}, message={sol.message}")
    print(f"\t{sol.nfev} fev, {sol.njev} jev, {sol.nlu} LU dec")    