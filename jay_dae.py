#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:15 2022

Test Jay's problem

https://github.com/scipy/scipy/pull/13068
https://doi.org/10.1016/0168-9274(95)00013-K

@author: lfrancoi
"""
import numpy as np

from scipy.integrate import solve_ivp#, Radau
from radauDAE import RadauDAE as Radau
from scipy.optimize._numdiff import approx_derivative


def generate_Jay1995(nonlinear_multiplier):
    def fun(t, y):
        """Index3 DAE found in Jay1995 Example 7.1 and 7.2.

        References:
        -----------
        Jay1995: https://doi.org/10.1016/0168-9274(95)00013-K
        """
        y1, y2, z1, z2, la = y

        if nonlinear_multiplier:
            dy3 = -y1*(y2**2)*(z2**3)*(la**2)
        else:
            dy3 = -y1*(y2**2)*(z2**2)*la

        dy = np.array(
            [
                2*y1*y2*z1*z2,
                -y1*y2*(z2**2),
                (y1*y2 + z1*z2)*la,
                dy3,
                y1 * y2**2 - 1.0,
            ],
            dtype=y.dtype,
        )

        return dy

    # the jacobian is computed via finite-differences
    def jac(t, x):
        return approx_derivative(
            fun=lambda x: fun(t, x), x0=x, method="cs", rel_step=1e-50
        )

    # construct singular mass matrix
    mass_matrix = np.eye(5)
    mass_matrix[-1, -1] = 0

    # initial conditions
    y0 = np.ones(5, dtype=float)
    y0[-1] = 4.

    t_span = (0, 0.1)
    # var_index = None
    var_index = np.array([0,0,0,0,2])
    return y0, mass_matrix, var_index, fun, jac, t_span


if __name__ == "__main__":

    y0, mass_matrix, var_index, fun, jac, t_span = generate_Jay1995(
        nonlinear_multiplier=True
    )

    sol = solve_ivp(
                        fun=fun,
                        t_span=t_span,
                        y0=y0,
                        rtol=1e-5,
                        atol=1e-10,
                        jac=jac,
                        method=Radau,
                        first_step=1e-5,
                        mass_matrix=mass_matrix,
                        bPrint=True,
                        dense_output=True,
                        max_newton_ite=8, min_factor=0.2, max_factor=10,
                        var_index=var_index,
                        newton_tol=1e-4,
                        scale_residuals = True,
                        scale_newton_norm = True,
                        scale_error = True,
                        max_bad_ite=1,
                    )


    assert sol.success

    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    t,y=sol.t,sol.y
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(t, y[0], "--r", label="y1")
    ax.plot(t, y[1], "--g", label="y2")
    ax.plot(t, y[2], "-.r", label="z1")
    ax.plot(t, y[3], "-.g", label="z2")
    ax.plot(t, y[4], "--b", label="u", marker='+')

    ax.plot(t, np.exp(2 * t), "-r", label="y1/z1 true")
    ax.plot(t, np.exp(-t), "-g", label="y2/z2 true")
    ax.plot(t, np.exp(t), "-b", label="u true")

    ax.grid()
    ax.set_ylim(0,1.5)
    ax.legend()
