import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix, issparse, eye, diags
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from scipy.integrate._ivp.common import (validate_max_step, validate_tol, select_initial_step,
                     norm, num_jac, EPS, warn_extraneous,
                     validate_first_step)
from scipy.integrate._ivp.base import OdeSolver, DenseOutput

S6 = 6 ** 0.5

# Butcher tableau. A is not used directly, see below.
C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])
E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3

# Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
# and a complex conjugate pair. They are written below.
MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3) # inverse of the real eigenvalue of A
MU_COMPLEX = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
              - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
BPRINT = False

CODE_NON_CONV=5
CODE_JACOBIAN_UPDATE=-11
CODE_FACTORISATION=-10
CODE_ACCEPTED=0
CODE_REJECTED=2

# These are transformation matrices.
T = np.array([
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1, 1, 0]])
TI = np.array([
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])

# These linear combinations are used in the algorithm.
TI_REAL = TI[0]
TI_COMPLEX = TI[1] + 1j * TI[2]

# Interpolator coefficients.
P = np.array([
    [13/3 + 7*S6/3, -23/3 - 22*S6/3, 10/3 + 5 * S6],
    [13/3 - 7*S6/3, -23/3 + 22*S6/3, 10/3 - 5 * S6],
    [1/3, -8/3, 10/3]])



def predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, bUsePredictiveController):
    """Predict by which factor to increase/decrease the step size.

    The algorithm is described in [1]_.

    Parameters
    ----------
    h_abs, h_abs_old : float
        Current and previous values of the step size, `h_abs_old` can be None
        (see Notes).
    error_norm, error_norm_old : float
        Current and previous values of the error norm, `error_norm_old` can
        be None (see Notes).

    Returns
    -------
    factor : float
        Predicted factor.

    Notes
    -----
    If `h_abs_old` and `error_norm_old` are both not None then a two-step
    algorithm is used, otherwise a one-step algorithm is used.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.
    """
    if not bUsePredictiveController:
        multiplier=1
    elif error_norm_old is None or h_abs_old is None or error_norm == 0:
        multiplier = 1
    else:
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25
        if BPRINT:
          # print('error_norm_old={:.2e}, error_norm={:.2e}'.format(error_norm_old,error_norm))
          print('\terr_np1 / err_n = {:.2e}'.format(error_norm / error_norm_old))
          print('\th_np1   / h_n   = {:.2e}'.format(h_abs / h_abs_old))

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** -0.25
    if BPRINT:
        print('\t--> step controller multiplier = {:.2e}'.format(multiplier))
        print('\t    factor = {:.2e}'.format(factor))

    return factor

from tqdm import tqdm
class progress_wrapper():
    def __init__(self, npoints=1000):
        self.progress = 0
        self.npoints = npoints
        self.pbar = tqdm(range(npoints), bar_format='{l_bar}{bar}| {elapsed}>{remaining}{postfix}')
        # self.nupdate = 0
        # self.update(0)

    def update(self, nstep, t, dt, progress):
        """ Progress should be in [0,1] """
        gap = np.floor(self.npoints * progress) - np.floor(self.npoints * self.progress)
        if gap >= 1: # update every one percent
            self.pbar.update(gap)
            self.pbar.postfix = f'nt={nstep}, t={t:.2e}, dt={dt:.2e}'
            # self.pbar.n = np.floor(self.npoints * progress)
            self.pbar.refresh()
            self.progress = progress
        if progress==1:
          self.finalise()

    def finalise(self):
        if self.progress<1:
           self.pbar.update(self.npoints)
        self.pbar.close()

class RadauDAE(OdeSolver):
    """Implicit Runge-Kutta method of Radau IIA family of order 5.

    The implementation follows [1]_. The error is controlled with a
    third-order accurate embedded formula. A cubic polynomial which satisfies
    the collocation conditions is used for the dense output.
    Specific treatment of differential-algebraic systems follows [3]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below). The
        vectorized implementation allows a faster approximation of the Jacobian
        by finite differences (required for this solver).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : {None, array_like, sparse_matrix, callable}, optional
        Jacobian matrix of the right-hand side of the system with respect to
        y, required by this method. The Jacobian matrix has shape (n, n) and
        its element (i, j) is equal to ``d f_i / d y_j``.
        There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)`` as necessary.
              For the 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
        elements in *each* row, providing the sparsity structure will greatly
        speed up the computations [2]_. A zero entry means that a corresponding
        element in the Jacobian is always zero. If None (default), the Jacobian
        is assumed to be dense.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.
    mass : {None, array_like, sparse_matrix}, optional
        Defined the constant mass matrix of the system, with shape (n,n).
        It may be singular, thus defining a problem of the differential-
        algebraic type (DAE), see [1]. The default value is None.
    newton_tol: float, optional
        Overrides the Newton tolerance. This parameter should be left to None,
        as it might strongly affect the performance and/or convergence.
    constant_dt: boolean, optional
        If True, the algorithm does not adapt the time step and tries to
        maintain h = max_step. Must be used in conjunction to a properly set value
        of the `newton_tol` parameter (e.g. 1e-8 as a rule of thumb).

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.

    References
    ----------
    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
    .. [3] E. Hairer, C. Lubich, M. Roche, "The Numerical Solution of
           Differential-Algebraic Systems by Runge-Kutta Methods"
    """
    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
                 factor_on_non_convergence = 0.5,
                 safety_factor = 0.9,
                 vectorized=False, first_step=None, mass_matrix=None, bPrint=False,
                 newton_tol=None, constant_dt=False, bPrintProgress=False,
                 max_newton_ite=6, min_factor=0.2, max_factor=10, max_bad_ite=None,
                 var_index=None, scale_residuals=False, scale_newton_norm=False,
                 scale_error=True, zero_algebraic_error=False, bDebug=False,
                 jacobianRecomputeFactor=1e-3,
                 bAlwaysApply2ndEstimate=True, bUsePredictiveController=True,
                 bUsePredictiveNewtonStoppingCriterion=True,
                 bUseExtrapolatedGuess=True,
                 bReport=False,
                 bPerformAllNewtonIterations=False,
                 **extraneous):

        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized)
        self.t0 = t0

        self.NEWTON_MAXITER = max_newton_ite # Maximum number of Newton iterations
        self.MIN_FACTOR = min_factor # Minimum allowed decrease in a step size
        self.MAX_FACTOR = max_factor # Maximum allowed increase in a step size
        self.factor_on_non_convergence = factor_on_non_convergence # Time step decrease when Newton fails
        self.safety_factor = safety_factor # safety factor for the determination of a new time step
        self.bAlwaysApply2ndEstimate = bAlwaysApply2ndEstimate # If True, the L-stabilised error estimate is always used, otherwise it is only use after a rejected step
        self.bUsePredictiveNewtonStoppingCriterion = bUsePredictiveNewtonStoppingCriterion # If True, the Newton may stop early if the predicted error after all allowed iterations are performed is too large
        self.bUseExtrapolatedGuess=bUseExtrapolatedGuess # if True, the Newton is initialised with an extrapolated dense output
        self.bUsePredictiveController = bUsePredictiveController # If True, adaptive time step controller is used
        self.jacobianRecomputeFactor = jacobianRecomputeFactor # if convergence rate is lower than this value after a step, the Jacobian is updated
        self.bReport = bReport # if True, details (error, Newton iterations) of each step are stored
        self.bPerformAllNewtonIterations = bPerformAllNewtonIterations # if True, all Newton iterations are performed, unless convergence is reached
        if self.bReport:
          self.reports={"t":[],"dt":[],"code":[],'newton_iterations':[], 'bad_iterations':[],
                   "err1":[], "err2":[], "errors1":[], "errors2":[], "error_scale": []}
          

        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)
        # Select initial step assuming the same order which is used to control
        # the error.
        if first_step is None:
            if mass_matrix is None:
                self.h_abs = select_initial_step(
                    self.fun, self.t, self.y, self.f, self.direction,
                    3, self.rtol, self.atol)
            else: # as in [1], default to 1e-6
                self.h_abs = self.direction * min(1e-6, abs(self.t_bound-self.t0))
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.h_abs_old = None
        self.error_norm_old = None

        # newton tolerance (relative to the integration error tolerance)
        if newton_tol is None:
            self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
        else:
            self.newton_tol = newton_tol
        self.sol = None
        self.constant_dt = constant_dt

        self.jac_factor = None
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)


        self.bPrintProgress = bPrintProgress
        if self.bPrintProgress:
          self.progressBar = progress_wrapper()
        self.bDebug = bDebug
        global BPRINT
        BPRINT=bPrint

        if BPRINT:
          zzzjac = self.jac
          def debugJacprint(t,y,f):
            print('\tupdating jacobian')
            return zzzjac(t,y,f)
          self.jac = debugJacprint

        self.nlusolve=0
        self.nlusolve_errorest = 0
        self.nstep = 0
        self.naccpt = 0
        self.nfailed = 0
        self.nrejct = 0
        if issparse(self.J):
            def lu(A):
                self.nlu += 1
                return splu(A)

            def solve_lu(LU, b):
                self.nlusolve+=1
                return LU.solve(b)

            I = eye(self.n, format='csc')
        else:
            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            def solve_lu(LU, b):
                self.nlusolve+=1
                return lu_solve(LU, b, overwrite_b=True)

            I = np.identity(self.n)

        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I

        # DAE-specific treatment
        self.scale_residuals = scale_residuals
        self.scale_error = scale_error
        self.scale_newton_norm = scale_newton_norm
        self.zero_algebraic_error = zero_algebraic_error
        if self.scale_residuals:
            self.hscale = None # will be dynamically computed
        else:
            self.hscale = self.I

        self.mass_matrix = self._validate_mass_matrix(mass_matrix)
        if var_index is None: # vector of the algebraic index of each variable
            self.var_index = np.zeros((y0.size,)) # assume all differential
        else:
            assert isinstance(var_index, np.ndarray), '`var_index` must be an array'
            assert var_index.ndim == 1
            assert var_index.size == y0.size
            self.var_index = var_index

        self.var_exp = self.var_index - 1
        self.var_exp[ self.var_exp < 0 ] = 0 # for differential components

        if not ( max_bad_ite is None ):
            self.NMAX_BAD = max_bad_ite # Maximum number of bad iterations
        else:
            if np.any(self.var_index>1):
                self.NMAX_BAD = 1
            else:
                self.NMAX_BAD = 0

        self.index_algebraic_vars = np.where( self.var_index != 0 )[0] #self.var_index != 0
        self.nvars_algebraic = self.index_algebraic_vars.size


        self.current_jac = True
        self.LU_real = None
        self.LU_complex = None
        self.Z = None

        if self.bDebug:
            self.info = {'cond':{'LU_real':[], 'LU_complex':[], 't':[], 'h':[]}}

    def _report(self,t,dt,code=None,newt=-1,nbad=-1,err1=-1,err2=-1,
                errors1=None, errors2=None, err_scale=None):
      if not self.bReport:
        return
      self.reports['t'].append(t)
      self.reports['dt'].append(dt)
      self.reports['code'].append(code)
      self.reports['newton_iterations'].append(newt)
      self.reports['bad_iterations'].append(nbad)
      self.reports['err1'].append(err1)
      self.reports['err2'].append(err2)
      if errors1 is None:
        errors1 = np.nan * np.zeros_like(self.y)
      if errors2 is None:
        errors2 = np.nan * np.zeros_like(self.y)
      if err_scale is None:
        err_scale = np.nan * np.zeros_like(self.y)
      self.reports['errors1'].append(errors1)
      self.reports['errors2'].append(errors2)
      self.reports['error_scale'].append(err_scale)
        
          
          
    def _validate_mass_matrix(self, mass_matrix):
        if mass_matrix is None:
            M = self.I
        elif callable(mass_matrix):
            raise ValueError("`mass_matrix` should be a constant matrix, but is"
                             " callable")
        else:
            if issparse(mass_matrix):
                M = csc_matrix(mass_matrix)
            else:
                M = np.asarray(mass_matrix, dtype=float)
            if M.shape != (self.n, self.n):
                raise ValueError("`mass_matrix` is expected to have shape {}, "
                                 "but actually has {}."
                                 .format((self.n, self.n), M.shape))
        return M

    def _validate_jac(self, jac, sparsity):
        t0 = self.t
        y0 = self.y

        if jac is None:
            if sparsity is not None:
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                groups = group_columns(sparsity)
                sparsity = (sparsity, groups)

            def jac_wrapped(t, y, f):
                self.njev += 1 # does not contribute to nfev
                J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
                                             self.atol, self.jac_factor,
                                             sparsity)
                return J
            J = jac_wrapped(t0, y0, self.f)
        elif callable(jac):
            J = jac(t0, y0)
            self.njev = 1
            if issparse(J):
                J = csc_matrix(J)

                def jac_wrapped(t, y, _=None):
                    self.njev += 1
                    return csc_matrix(jac(t, y), dtype=float)

            else:
                J = np.asarray(J, dtype=float)

                def jac_wrapped(t, y, _=None):
                    self.njev += 1
                    return np.asarray(jac(t, y), dtype=float)

            if J.shape != (self.n, self.n):
                raise ValueError("`jac` is expected to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
        else:
            if issparse(jac):
                J = csc_matrix(jac)
            else:
                J = np.asarray(jac, dtype=float)

            if J.shape != (self.n, self.n):
                raise ValueError("`jac` is expected to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
            jac_wrapped = None

        return jac_wrapped, J

    def _step_impl(self):
        t = self.t
        y = self.y
        f = self.f
        n = y.size
        self.nstep +=1

        max_step = self.max_step
        atol = self.atol
        rtol = self.rtol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            h_abs_old = None
            error_norm_old = None
        elif self.h_abs < min_step:
            h_abs = min_step
            h_abs_old = None
            error_norm_old = None
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            error_norm_old = self.error_norm_old

        if abs(self.t_bound - (t + self.direction * h_abs)) < 1e-2*h_abs:
            # the next time step would be too small
            # # --> we cut the remaining time in half
            # if BPRINT:
            #     print('Reducing time step to avoid a last tiny step')
            # h_abs = abs(self.t_bound - t) / 2
            # we increase the time step to match the final time
            if BPRINT:
                print('Increasing time step to avoid a last tiny step')
            h_abs = abs(self.t_bound - t)
            # require refactorisation of the iteration matrices
            self.LU_real = None
            self.LU_complex = None

        J = self.J
        LU_real = self.LU_real
        LU_complex = self.LU_complex

        current_jac = self.current_jac
        jac = self.jac

        rejected = False
        step_accepted = False
        message = None
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t # may introduce numerical rounding errors
            h_abs = np.abs(h)

            # initial solution for the Newton solve
            if (self.sol is None) or not self.bUseExtrapolatedGuess:
                Z0 = np.zeros((3, y.shape[0]))
            else:
                Z0 = self.sol(t + h * C).T - y  # extrapolate using previous dense output

            newton_scale = atol + np.abs(y) * rtol
            if self.scale_newton_norm:
                newton_scale = newton_scale / (h**self.var_exp) # scale for algebraic variables in the Newton increment norm

            converged = False
            while not converged:
                if BPRINT:
                  print('solving system at t={} with dt={}'.format(t,h))
                if LU_real is None or LU_complex is None:
                  if self.scale_residuals:
                    # residuals associated with high-index components are scaled
                    # this may help with the stability of the matrix decomposition
                    # (see [3], p97)
                    # self.hscale = np.diag(h**(-self.var_index))
                    if issparse(self.I):
                        self.hscale = diags(h**(-np.minimum(1,self.var_index)), offsets=0, format='csc')
                    else:
                        self.hscale = np.diag(h**(-np.minimum(1,self.var_index))) # only by h or 1
                  try:
                      if BPRINT:
                          print('\tLU-decomposition of system Jacobian')
                      if self.bReport: self._report(t=t,dt=h,code=CODE_FACTORISATION)
                      LU_real    = self.lu( self.hscale @ (MU_REAL    / h * self.mass_matrix - J) )
                      LU_complex = self.lu( self.hscale @ (MU_COMPLEX / h * self.mass_matrix - J) )
                      self.nlu -= 1 # to match Fortran
                  except ValueError as e:
                    # import pdb; pdb.set_trace()
                    return False, 'LU decomposition failed ({})'.format(e)

                  if self.bDebug:
                      U_matrix = np.triu(LU_real[0],k=0)
                      L_matrix = np.tril(LU_real[0],k=-1)+self.I
                      self.info['cond']['LU_real'].append(np.linalg.cond(U_matrix*L_matrix))
                      U_matrix = np.triu(LU_complex[0],k=0)
                      L_matrix = np.tril(LU_complex[0],k=-1)+self.I
                      self.info['cond']['LU_complex'].append(np.linalg.cond(U_matrix*L_matrix))
                      self.info['cond']['t'].append(t)
                      self.info['cond']['h'].append(h)
                      print('\tcond(LU_real)    = {:.3e}'.format( self.info['cond']['LU_real'][-1] ))
                      print('\tcond(LU_complex) = {:.3e}'.format( self.info['cond']['LU_complex'][-1]  ))

                converged, n_iter, n_bad, Z, f_subs, rate = self.solve_collocation_system(
                    t, y, h, Z0, newton_scale, self.newton_tol,
                    LU_real, LU_complex, residual_scale=self.hscale)

                safety = self.safety_factor * (2 * self.NEWTON_MAXITER + 1) / (2 * self.NEWTON_MAXITER + n_iter)
                if BPRINT:
                  print('\tsafety={:.2e}'.format(safety))
                  
                if not converged:
                    if BPRINT:
                        print('no convergence at t={} with dt={}'.format(t,h))
                    if current_jac: # we only allow one Jacobian computation per time step
                        if BPRINT:
                            print('  Jacobian had already been updated')
                        break

                    J = self.jac(t, y, f)
                    if self.bReport: self._report(t=t,dt=h,code=CODE_JACOBIAN_UPDATE)
                    current_jac = True
                    LU_real = None
                    LU_complex = None

	    ## End of the convergence loop
            if not converged:
                if BPRINT:
                   print('   --> dt will be reduced')
                self.nfailed += 1
                if self.bReport: self._report(t=t,dt=h,code=CODE_NON_CONV,newt=n_iter,nbad=n_bad)
                h_abs *= self.factor_on_non_convergence
                LU_real = None
                LU_complex = None
                continue

            y_new = y + Z[-1]
            self.t_substeps = t + h * C
            self.y_substeps = [y + Z[i] for i in range(len(Z))]
            self.f_substeps = f_subs

            if self.constant_dt:
                step_accepted = True
                error_norm = 0.
            else:
                err1, err2 = np.nan, np.nan
                errors1, errors2 = None, None
                
                ZE = Z.T.dot(E) / h
                err_scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
                if self.scale_error:
                    err_scale = err_scale / (h**self.var_exp) # scale for algebraic variables

                # see [1], chapter IV.8, page 127
                error = self.solve_lu(LU_real, f + self.mass_matrix.dot(ZE))
                errors1 = error / err_scale
                self.nlusolve_errorest += 1
                self.nlusolve -= 1
                if self.zero_algebraic_error:
                    error[ self.index_algebraic_vars ] = 0.
                    error_norm = np.linalg.norm(error / err_scale) / (n - self.nvars_algebraic) ** 0.5
                else:
                    error_norm = norm(error / err_scale)
                err1=error_norm
                if BPRINT:
                  print('\t1st error estimate: {:.3e}'.format(error_norm))

                if (self.bAlwaysApply2ndEstimate and error_norm > 1) or (rejected and error_norm > 1): # try with stabilised error estimate
                    if BPRINT:
                      print('\t rejected')
                    error = self.solve_lu(LU_real, self.fun(t, y + error) + self.mass_matrix.dot(ZE)) # error is not corrected for algebraic variables
                    errors2 = error / err_scale
                    self.nlusolve_errorest += 1
                    self.nlusolve -= 1
                    if self.zero_algebraic_error:
                        error[ self.index_algebraic_vars ] = 0.
                        error_norm = np.linalg.norm(error / err_scale) / (n - self.nvars_algebraic) ** 0.5
                    else:
                        error_norm = norm(error / err_scale)
                    err2=error_norm
                    if BPRINT:
                      print('\t2nd error estimate: {:.3e}'.format(error_norm))

                if error_norm > 1:
                    if BPRINT:
                        print('\t step rejected')
                    if BPRINT and y_new.size<10:
                      print('\terror (scaled)=',error/err_scale)
                      
                    if self.bReport: self._report(t=t,dt=h,code=CODE_REJECTED,newt=n_iter,nbad=n_bad,
                                 err1=err1, err2=err2, errors1=errors1, errors2=errors2, err_scale=err_scale)

                    # TODO: Original Fortran code does something else ?
                    factor = predict_factor(h_abs, h_abs_old,
                                            error_norm, error_norm_old,
                                            self.bUsePredictiveController)
                    if BPRINT:
                        print('\tfactor={:.2e}'.format(safety))
                    
                    h_abs *= max(self.MIN_FACTOR, safety * factor)
                    LU_real = None
                    LU_complex = None
                    rejected = True
                    self.nrejct += 1
                else:
                    if BPRINT: print('\t step accepted2')
                    if self.bReport: self._report(t=t,dt=h,code=CODE_ACCEPTED,newt=n_iter,nbad=n_bad,
                                 err1=err1, err2=err2, errors1=errors1, errors2=errors2, err_scale=err_scale)
                    self.naccpt +=1
                    step_accepted = True

	## Step is converged and accepted
        if BPRINT: print('\t step accepted')
        recompute_jac = jac is not None and n_iter > 2 and rate > self.jacobianRecomputeFactor

        if self.constant_dt:
          factor = self.max_step/h_abs # return to the maximum value
        else:
          factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old,
                                  self.bUsePredictiveController)
          factor = min(self.MAX_FACTOR, safety * factor)

        if not recompute_jac and factor < 1.2:
            factor = 1
        else:
            LU_real = None
            LU_complex = None

        f_new = self.fun(t_new, y_new)
        if recompute_jac:
            J = jac(t_new, y_new, f_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_old = self.h_abs
        self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y

        self.t = t_new
        self.y = y_new
        self.f = f_new


        self.Z = Z

        self.LU_real = LU_real
        self.LU_complex = LU_complex
        self.current_jac = current_jac
        self.J = J

        self.t_old = t
        self.sol = self._compute_dense_output()

        if self.bPrintProgress:
          # print('t=',t)
          self.progressBar.update(self.nstep, t_new, self.h_abs_old, (t_new-self.t0)/(self.t_bound-self.t0) )
          # also display current time

        return step_accepted, message

    def solve_collocation_system(self, t, y, h, Z0, norm_scale, tol,
                                 LU_real, LU_complex, residual_scale):
        """Solve the collocation system.

        Parameters
        ----------
        t : float
            Current time.
        y : ndarray, shape (n,)
            Current state.
        h : float
            Step to try.
        Z0 : ndarray, shape (3, n)
            Initial guess for the solution. It determines new values of `y` at
            ``t + h * C`` as ``y + Z0``, where ``C`` is the Radau method constants.
        norm_scale : ndarray, shape (n)
            Problem tolerance scale, i.e. ``rtol * abs(y) + atol``.
        tol : float
            Tolerance to which solve the system. This value is compared with
            the normalized by `scale` error.
        LU_real, LU_complex
            LU decompositions of the system Jacobians.

        Returns
        -------
        converged : bool
            Whether iterations converged.
        n_iter : int
            Number of completed iterations.
        Z : ndarray, shape (3, n)
            Found solution.
        rate : float
            The rate of convergence.
        """
        n = y.shape[0]
        M_real = MU_REAL / h
        M_complex = MU_COMPLEX / h

        W = TI.dot(Z0) # state vector at each quadrature point (with complex transformation)
        Z = Z0 # state vector at each quadrature point

        F = np.empty((3, n))  # RHS evaluated at the quadrature points
        ch = h * C # quadrature time points

        dW_norm_old = None
        res_norm = None
        res_norm_old = None
        dW = np.empty_like(W)
        converged = False
        rate = None
        nbad_iter = 0
        if BPRINT: print('Newton increment scale =', norm_scale)

        for k in range(self.NEWTON_MAXITER):
            if BPRINT:
              print('\titer {}/{}'.format(k,self.NEWTON_MAXITER))
            for i in range(3):
                F[i] = self.fun(t + ch[i], y + Z[i])

            if not np.all(np.isfinite(F)):
                if BPRINT:
                  print('\t\tF contains non real numbers...')
                break

            # compute residuals
            f_real    = F.T.dot(TI_REAL)    - M_real    * self.mass_matrix.dot( W[0] )
            f_complex = F.T.dot(TI_COMPLEX) - M_complex * self.mass_matrix.dot( W[1] + 1j * W[2] )
            if BPRINT:
              print('\t\tresiduals (unscaled):  ||Re(f)||={:.2e}  ||Im(f)||={:.2e}'.format(norm(f_real), norm(f_complex)))

            # scale residuals
            f_real = residual_scale @ f_real
            f_complex = residual_scale @ f_complex
            if BPRINT:
                R1 = norm(f_real)
                R2 = norm(f_complex)
                res_norm = np.sqrt((R1**2 * n +  R2**2 * (2*n)) / (3*n))
                print('\t\tresiduals   (scaled):  ||Re(f)||={:.2e}  ||Im(f)||={:.2e}'.format(R1,R2))

            if res_norm_old is not None:
                rate_res = res_norm / res_norm_old
                print('\t\tresidual rate = {:.2e}'.format(rate_res))

            # compute Newton increment
            dW_real    = self.solve_lu(LU_real,    f_real)
            dW_complex = self.solve_lu(LU_complex, f_complex)
            self.nlusolve -= 1 # to match the original Fortran code

            dW[0] = dW_real
            dW[1] = dW_complex.real
            dW[2] = dW_complex.imag

            dW_norm = norm(dW / norm_scale)
            if BPRINT:
              print('\t\t||dW||   = {:.3e} (scaled), {:.3e} (unscaled)'.format(dW_norm, norm(dW)))

            W += dW
            Z = T.dot(W)

            if dW_norm_old is not None:
                rate = dW_norm / dW_norm_old
                dW_true = rate / (1 - rate) * dW_norm # estimated true error
                dW_true_max_ite = rate ** (self.NEWTON_MAXITER - k) / (1 - rate) * dW_norm # estimated true error after max number of iterations
                if BPRINT:
                  print('\t\trate={:.3e}'.format(rate))

            if dW_norm < tol:
                if BPRINT:
                    print('\t\t--> ||dW|| < tol={:.2e}'.format(tol))
                converged = True
                break

            if rate is not None:
                if BPRINT:
                    print('\t\t||dW*||  = {:.3e}'.format( dW_true ))
                    print('\t\t||dW**|| = {:.3e}'.format( dW_true_max_ite ))

                if rate >= 1: # Newton loop diverges
                    if BPRINT:
                        print('\t\tNewton loop diverges')
                    if rate<100:
                        if nbad_iter<self.NMAX_BAD:
                            # we accept a few number of bad iterations, which may be necessary
                            if BPRINT:
                                print('\t\tbad iteration (rate>1)')
                            nbad_iter+=1
                            continue
                    if not self.bPerformAllNewtonIterations:
                      break
                if dW_true < tol:
                    if BPRINT:
                        print('\t\t--> ||dW*|| < tol={:.2e}'.format(tol))
                    converged = True
                    break
                if (dW_true_max_ite > tol) and self.bUsePredictiveNewtonStoppingCriterion :
                    # Newton will not converge in the allowed number of iterations
                    if BPRINT:
                      print('\t\t--> ||dW**|| > tol={:.2e}'.format(tol))
                    if nbad_iter<self.NMAX_BAD:
                        nbad_iter+=1
                        if BPRINT:
                            print('\t\t/!\ bad iteration (||dW**|| too large)')
                        continue
                    if not self.bPerformAllNewtonIterations:
                      break

            dW_norm_old  = dW_norm
            res_norm_old = res_norm

        if BPRINT:
            if converged:
                print('\t\tstep converged')
            else:
                print('\t\tstep failed')

        return converged, k + 1, nbad_iter, Z, F, rate

    def _compute_dense_output(self):
        Q = np.dot(self.Z.T, P)
        return RadauDenseOutput(self.t_old, self.t, self.y_old, Q)

    def _dense_output_impl(self):
        return self.sol


class RadauDenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, Q):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q
        self.order = Q.shape[1] - 1
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        if t.ndim == 0:
            p = np.tile(x, self.order + 1)
            p = np.cumprod(p)
        else:
            p = np.tile(x, (self.order + 1, 1))
            p = np.cumprod(p, axis=0)
        # Here we don't multiply by h, not a mistake, because we use the dimensionless time x
        y = np.dot(self.Q, p)
        if y.ndim == 2:
            y += self.y_old[:, None]
        else:
            y += self.y_old

        return y



MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred.",
            2: "Too many time steps have been performed",
            3: "Integration has been interrupted by the user"}
def solve_ivp_custom(fun, t_span, y0, method=RadauDAE, t_eval=None, dense_output=False,
              events=None, vectorized=False, args=None, return_substeps=False,
              nmax_step=np.inf,
              **options):
    """Solve an initial value problem for a system of ODEs.
    Custom version, useful for substep export
    """
    from scipy.integrate._ivp.ivp import prepare_events, find_active_events, handle_events
    from scipy.integrate._ivp.ivp import inspect, METHODS#, MESSAGES
    from scipy.integrate._ivp.ivp import OdeSolver, OdeSolution, OdeResult
    if method not in METHODS and not (
            inspect.isclass(method) and issubclass(method, OdeSolver)):
        raise ValueError("`method` must be one of {} or OdeSolver class."
                         .format(METHODS))

    t0, tf = float(t_span[0]), float(t_span[1])

    if args is not None:
        # Wrap the user's fun (and jac, if given) in lambdas to hide the
        # additional parameters.  Pass in the original fun as a keyword
        # argument to keep it in the scope of the lambda.
        fun = lambda t, x, fun=fun: fun(t, x, *args)
        jac = options.get('jac')
        if callable(jac):
            options['jac'] = lambda t, x: jac(t, x, *args)

    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = np.diff(t_eval)
        if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        if tf > t0:
            t_eval_i = 0
        else:
            # Make order of t_eval decreasing to use np.searchsorted.
            t_eval = t_eval[::-1]
            # This will be an upper bound for slices.
            t_eval_i = t_eval.shape[0]

    if method in METHODS:
        method = METHODS[method]

    solver = method(fun, t0, y0, tf, vectorized=vectorized, **options)

    tsub=[]
    ysub=[]
    fsub=[]
    if t_eval is None:
        ts = [t0]
        ys = [y0]
    elif t_eval is not None and dense_output:
        ts = []
        ti = [t0]
        ys = []
    else:
        ts = []
        ys = []

    interpolants = []

    events, is_terminal, event_dir = prepare_events(events)

    if events is not None:
        if args is not None:
            # Wrap user functions in lambdas to hide the additional parameters.
            # The original event function is passed as a keyword argument to the
            # lambda to keep the original function in scope (i.e., avoid the
            # late binding closure "gotcha").
            events = [lambda t, x, event=event: event(t, x, *args)
                      for event in events]
        g = [event(t0, y0) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
    else:
        t_events = None
        y_events = None

    status = None
    nsteps=0
    try:
      while status is None:
          nsteps+=1
          if (nsteps>nmax_step):
            print('\tToo many steps')
            status=2
            break
          message = solver.step()
  
          if solver.status == 'finished':
              status = 0
          elif solver.status == 'failed':
              status = -1
              break
  
          t_old = solver.t_old
          t = solver.t
          y = solver.y
  
          if return_substeps:
              tres=solver.t_substeps
              yres=solver.y_substeps
              fres=solver.f_substeps
              assert tres[-1]==t
              assert np.all(yres[-1]==y)
  
          if dense_output:
              sol = solver.dense_output()
              interpolants.append(sol)
          else:
              sol = None
  
          if events is not None:
              g_new = [event(t, y) for event in events]
              active_events = find_active_events(g, g_new, event_dir)
              if active_events.size > 0:
                  if sol is None:
                      sol = solver.dense_output()
                  print('SCIPY:solve_ivp: solving for precise event occurence')
                  root_indices, roots, terminate = handle_events(
                      sol, events, active_events, is_terminal, t_old, t)
  
                  for e, te in zip(root_indices, roots):
                      t_events[e].append(te)
                      y_events[e].append(sol(te))
  
                  if terminate:
                      status = 1
                      t = roots[-1]
                      y = sol(t)
  
              g = g_new
  
          if return_substeps:
              tsub.extend(tres)
              ysub.extend(yres)
              fsub.extend(fres)
          if t_eval is None:
              ts.append(t)
              ys.append(y)
          else:
              # The value in t_eval equal to t will be included.
              if solver.direction > 0:
                  t_eval_i_new = np.searchsorted(t_eval, t, side='right')
                  t_eval_step = t_eval[t_eval_i:t_eval_i_new]
              else:
                  t_eval_i_new = np.searchsorted(t_eval, t, side='left')
                  # It has to be done with two slice operations, because
                  # you can't slice to 0th element inclusive using backward
                  # slicing.
                  t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]
  
              if t_eval_step.size > 0:
                  if sol is None:
                      sol = solver.dense_output()
                  ts.append(t_eval_step)
                  ys.append(sol(t_eval_step))
                  t_eval_i = t_eval_i_new
  
          if t_eval is not None and dense_output:
              ti.append(t)
    except KeyboardInterrupt:
      status=3
      
    message = MESSAGES.get(status, message)

    if t_events is not None:
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    else:
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    if dense_output:
        if t_eval is None:
            sol = OdeSolution(ts, interpolants)
        else:
            sol = OdeSolution(ti, interpolants)
    else:
        sol = None

    out = OdeResult(t=ts, y=ys, sol=sol, t_events=t_events, y_events=y_events,
                     nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu,
                     status=status, message=message, success=status >= 0)
    out.solver = solver
    if return_substeps:
        # import pdb; pdb.set_trace()
        out.ysub = np.vstack(ysub).T
        out.tsub = np.hstack(tsub)
        out.fsub = np.vstack(fsub).T
    return out

if __name__=='__main__':
  # some quick analysis
  import matplotlib.pyplot as plt
  rtol = np.logspace(-16,3,1000)
  newton_tol = np.maximum(10 * EPS / rtol, np.minimum(0.03, rtol ** 0.5))
  rtol_changed = 0.1*rtol**(2/3)
  plt.figure()
  plt.loglog(rtol, newton_tol, label='relative Newton', linestyle='--', linewidth=3)
  plt.loglog(rtol, rtol, label='rtol', linestyle='--', color=[0,0,0])
  plt.loglog(rtol, rtol_changed, label='modified rtol')
  plt.loglog(rtol, newton_tol*rtol, label='absolute rtol Newton')
  plt.loglog(rtol, newton_tol*rtol_changed, label='absolute rtol changed Newton')
  plt.loglog(rtol, 10 * EPS / rtol, label='10 * EPS / rtol')
  plt.loglog(rtol, rtol ** 0.5, label='rtol ** 0.5')
  plt.loglog(rtol, 0.03 + 0*rtol, label='0.03')
  plt.xlabel('rtol')
  plt.legend(ncol=2, framealpha=0.3, loc='upper center')
  plt.ylabel('tolerances')
  plt.grid()