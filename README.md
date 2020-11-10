# DAE integrator for Scipy

This repository is based on Scipy's integrate module. The Radau method implemented in Scipy for solving ordinary differential equations (ODEs) is adapted to allow for the solution of differential-algebraic equations (DAEs) up to index 3.

This is done by introducing a mass matrix *M* and solving the following equation:
> M dX/dt = f(t,X)

Various examples are given:

- pendulum (index 3, 2, 1 or 0 DAE)

- transistor amplifier (index-1 DAE with singular mass-matrix)

- Roberston chemical system (stiff DAE of index 1, or stiff ODE formulation)

- "recursive" pendulum (index-3 DAE of size n, n being the number of serially connected pendulums)


![Hanging rope index-3 DAE animated](https://raw.githubusercontent.com/laurent90git/DAE-Scipy/main/docs/hanging_rope.gif "Hanging rope index-3 DAE")


This modification of Radau will hopefully be pushed to Scipy after some further testing :)

Further DAE integrators with time adaptation (ESDIRKs) will be implemented based on Scipy's solver class. A test case with a DAE from a discretised PDE will also be added.

A very good reference to read:
E. Hairer and G. Wanner, [Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Equations](https://www.springer.com/gp/book/9783540604525) 2nd Revised Edition 1996
