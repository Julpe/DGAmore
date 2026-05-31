"""
This code was taken, condensed and modified from the ana_cont package (https://github.com/josefkaufmann/ana_cont) of
Josef Kaufmann. Reason is, that the ana_cont package (installed via pip), has a lot of unwanted package dependencies.
The code is published under the following MIT license:

MIT License

Copyright (c) 2018 Josef Kaufmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import collections
import sys

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt
from scipy.integrate import simpson


class Kernel:
    def __init__(self, kind=None, re_axis=None, im_axis=None):
        if kind is None or re_axis is None or im_axis is None:
            raise ValueError("Kernel not correctly initialized")
        self.kind = kind
        self.re_axis = re_axis
        self.im_axis = im_axis
        self.original_matrix = self.kernel_matrix()
        self.matrix = np.copy(self.original_matrix)
        self.nw = self.re_axis.shape[0]
        self.niw = self.im_axis.shape[0]

    def kernel_matrix(self):
        if self.kind == "freq_bosonic":
            with np.errstate(invalid="ignore"):
                kernel = (self.re_axis**2)[None, :] / ((self.re_axis**2)[None, :] + (self.im_axis**2)[:, None])
            WhereIsiwn0 = np.where(self.im_axis == 0.0)[0]
            WhereIsw0 = np.where(self.re_axis == 0.0)[0]
            if len(WhereIsiwn0 == 1) and len(WhereIsw0 == 1):
                kernel[WhereIsiwn0, WhereIsw0] = 1.0  # analytically with de l'Hospital
        elif self.kind == "time_bosonic":
            with np.errstate(invalid="ignore"):
                kernel = (
                    0.5
                    * self.re_axis[None, :]
                    * (
                        np.exp(-self.re_axis[None, :] * self.im_axis[:, None])
                        + np.exp(-self.re_axis[None, :] * (1.0 - self.im_axis[:, None]))
                    )
                    / (1.0 - np.exp(-self.re_axis[None, :]))
                )
            kernel[:, 0] = 1.0  # analytically with de l'Hospital
        elif self.kind == "freq_bosonic_xyz":
            kernel = -self.im_axis[:, None] / ((self.re_axis**2)[None, :] + (self.im_axis**2)[:, None])
            if self.im_axis[0] == 0.0:
                kernel[0] = 0.0
        elif self.kind == "freq_fermionic":
            kernel = 1.0 / (1j * self.im_axis[:, None] - self.re_axis[None, :])
        elif self.kind == "time_fermionic":

            def time_kernel(τ, ω):
                if np.exp(-ω) > 1.0e8:
                    return np.exp(-(τ - 1) * ω)
                else:
                    return np.exp(-τ * ω) / (1.0 + np.exp(-ω))

            time_kernel_vect = np.vectorize(time_kernel)
            kernel = time_kernel_vect(self.im_axis[:, None], self.re_axis[None, :])
        elif self.kind == "freq_fermionic_phsym":  # in this case, the data must be purely real (the imaginary part!)
            print("Warning: phsym kernels do not give good results in this implementation. ")
            kernel = -2.0 * self.im_axis[:, None] / ((self.im_axis**2)[:, None] + (self.re_axis**2)[None, :])
        elif self.kind == "time_fermionic_phsym":
            print("Warning: phsym kernels do not give good results in this implementation. ")
            kernel = (
                np.cosh(self.im_axis[:, None] * self.re_axis[None, :])
                + np.cosh((1.0 - self.im_axis[:, None]) * self.re_axis[None, :])
            ) / (1.0 + np.cosh(self.re_axis[None, :]))
        else:
            raise ValueError("Unknown kernel type")
        return kernel

    def preblur(self, blur_width):
        self.blur_width = blur_width
        if self.blur_width > 0.0 and (self.kind == "freq_fermionic" or self.kind == "freq_bosonic"):
            self.matrix = self.convolve_kernel()
        else:
            self.matrix = np.copy(self.original_matrix)

    def convolve_kernel(self):
        self.w_integration = np.linspace(-5.0 * self.blur_width, 5.0 * self.blur_width, num=201, endpoint=True)
        norm = 1.0 / (self.blur_width * np.sqrt(2.0 * np.pi))
        self.gaussian_numeric = norm * np.exp(-0.5 * (self.w_integration / self.blur_width) ** 2)

        if self.kind == "freq_fermionic":
            integrand = self.gaussian_numeric[None, None, :] / (
                1j * self.im_axis[:, None, None] - self.re_axis[None, :, None] - self.w_integration[None, None, :]
            )
        elif self.kind == "freq_bosonic":
            where_is_iwn0 = np.where(self.im_axis == 0.0)[0]
            with np.errstate(invalid="ignore"):
                integrand_1 = (
                    self.gaussian_numeric[None, None, :]
                    * (self.w_integration[None, None, :] + self.re_axis[None, :, None]) ** 2
                    / (
                        (self.w_integration[None, None, :] + self.re_axis[None, :, None]) ** 2
                        + (self.im_axis[:, None, None]) ** 2
                    )
                )
                integrand_2 = (
                    self.gaussian_numeric[None, None, :]
                    * (self.w_integration[None, None, :] - self.re_axis[None, :, None]) ** 2
                    / (
                        (self.w_integration[None, None, :] - self.re_axis[None, :, None]) ** 2
                        + (self.im_axis[:, None, None]) ** 2
                    )
                )
            if len(where_is_iwn0) > 0:  # analytically with de l'Hospital
                integrand_1[where_is_iwn0] = self.gaussian_numeric[None, None, :]
                integrand_2[where_is_iwn0] = self.gaussian_numeric[None, None, :]
            integrand = 0.5 * (integrand_1 + integrand_2)
        else:
            raise NotImplementedError("No preblur implemented for this kernel.")
        return simpson(integrand, x=self.w_integration, axis=-1)

    def blur(self, hidden_spectrum):
        if self.kind == "freq_bosonic":
            h_interp = interp.InterpolatedUnivariateSpline(
                np.concatenate((-self.re_axis[:0:-1], self.re_axis)),
                np.concatenate((hidden_spectrum[:0:-1], hidden_spectrum)),
                ext="zeros",
            )
        else:
            h_interp = interp.InterpolatedUnivariateSpline(self.re_axis, hidden_spectrum, ext="zeros")

        integrand = self.gaussian_numeric[None, :] * h_interp(self.re_axis[:, None] + self.w_integration[None, :])
        return simpson(integrand, x=self.w_integration, axis=-1)

    def real_matrix(self):
        if self.kind == "freq_fermionic":
            return np.concatenate((self.matrix.real, self.matrix.imag))
        else:
            return self.matrix

    def rotate_to_cov_eb(self, ucov):
        self.ucov = ucov
        self.matrix = np.dot(self.ucov.T.conj(), self.matrix)


class AnalyticContinuationProblem:
    def __init__(
        self,
        im_axis: np.ndarray,
        re_axis: np.ndarray,
        im_data: np.ndarray,
        beta: float,
        kernel_mode: str = "freq_fermionic",
    ):
        self.kernel_mode = kernel_mode
        if np.allclose(im_axis.imag, np.zeros_like(im_axis, dtype=float)):
            self.im_axis = im_axis
        else:
            raise ValueError(
                "Parameter im_axis takes only the imaginary part of the imaginary axis (i.e. only real values)"
            )
        self.re_axis = re_axis
        self.im_data = im_data
        if self.kernel_mode == "freq_bosonic":
            pass  # not necessary to do anything additionally here
        elif self.kernel_mode == "time_bosonic":
            self.im_axis = im_axis / beta
            self.re_axis = re_axis * beta
            self.im_data = im_data * beta
            self.beta = beta
        elif self.kernel_mode == "freq_fermionic":
            pass
        elif self.kernel_mode == "freq_fermionic_phsym":
            # check if data are purely imaginary
            if np.allclose(im_axis.real, 0.0):
                self.im_data = im_data.imag
            elif np.allclose(im_axis.imag, 0.0):  # if only the imaginary part is passed
                self.im_data = im_data.real
            else:
                print("The data are neither purely real nor purely imaginary,")
                print("you cannot use a ph-symmetric kernel in this case.")
                sys.exit()
        elif self.kernel_mode == "time_fermionic" or self.kernel_mode == "time_fermionic_phsym":
            self.im_axis = im_axis / beta
            self.re_axis = re_axis * beta
            self.im_data = im_data
            self.beta = beta
        else:
            raise ValueError("Unsupported kernel_mode.")

    def solve(self, **kwargs):
        self.solver = MaxentSolverSVD(self.im_axis, self.re_axis, self.im_data, kernel_mode=self.kernel_mode, **kwargs)
        sol = self.solver.solve(**kwargs)
        if self.kernel_mode == "time_fermionic":
            sol[0].A_opt *= self.beta
        elif self.kernel_mode == "time_bosonic":
            sol[0].A_opt *= self.beta
            sol[0].backtransform /= self.beta
        return sol


class MaxentSolverSVD:
    def __init__(
        self,
        im_axis: np.ndarray,
        re_axis: np.ndarray,
        im_data: np.ndarray,
        kernel_mode: str = "freq_fermionic",
        model: np.ndarray = None,
        stdev: np.ndarray = None,
        cov=None,
        offdiag=False,
        preblur=True,
        blur_width=0.16,
        **kwargs,
    ):
        self.kernel_mode = kernel_mode
        if np.allclose(im_axis.imag, np.zeros_like(im_axis, dtype=float)):
            self.im_axis = im_axis
        else:
            raise ValueError(
                "Parameter im_axis takes only the imaginary part of the imaginary axis (i.e. only real values)"
            )
        self.re_axis = re_axis
        self.im_data = im_data
        self.offdiag = offdiag

        self.nw = self.re_axis.shape[0]
        self.wmin = self.re_axis[0]
        self.wmax = self.re_axis[-1]
        self.dw = np.diff(np.concatenate(([self.wmin], (self.re_axis[1:] + self.re_axis[:-1]) / 2.0, [self.wmax])))
        if not self.offdiag:
            self.model = model  # the model should be normalized by the user himself
        else:
            self.model_plus = model  # the model should be normalized by the user himself
            self.model_minus = model  # the model should be normalized by the user himself

        if cov is None and stdev is not None:
            self.var = stdev**2
            self.cov = np.diag(self.var)
            self.ucov = np.eye(self.im_axis.shape[0])
        elif stdev is None and cov is not None:
            self.cov = cov
            self.var, self.ucov = np.linalg.eigh(self.cov)  # go to eigenbasis of covariance matrix
            self.var = np.abs(self.var)  # numerically, var can be zero below machine precision

        self.im_data = np.dot(self.ucov.T.conj(), self.im_data)
        self.E = 1.0 / self.var
        self.niw = self.im_axis.shape[0]

        # set the kernel
        self.kernel = Kernel(kind=kernel_mode, im_axis=self.im_axis, re_axis=self.re_axis)

        # PREBLUR
        self.preblur = preblur
        if self.preblur:
            self.kernel.preblur(blur_width=blur_width)

        # rotate kernel to eigenbasis of covariance matrix
        self.kernel.rotate_to_cov_eb(ucov=self.ucov)

        # special treatment for complex data of fermionic frequency kernel
        if kernel_mode == "freq_fermionic":
            self.niw *= 2
            self.im_data = np.concatenate((self.im_data.real, self.im_data.imag))
            self.var = np.concatenate((self.var, self.var))
            self.E = np.concatenate((self.E, self.E))

        # singular value decomposition of the kernel
        U, S, Vt = np.linalg.svd(self.kernel.real_matrix(), full_matrices=False)

        self.n_sv = np.arange(min(self.nw, self.niw))[S > 1e-10][-1]  # number of singular values larger than 1e-10

        self.U_svd = np.array(U[:, : self.n_sv], dtype=np.float64, order="C")
        self.V_svd = np.array(Vt[: self.n_sv, :].T, dtype=np.float64, order="C")  # numpy.svd returns V.T
        self.Xi_svd = S[: self.n_sv]

        if not self.offdiag:  # precompute matrices W_ml (W2), W_mil (W3)
            self.W2 = np.einsum(
                "k,km,m,kn,n,ln,l,l->ml",
                self.E,
                self.U_svd,
                self.Xi_svd,
                self.U_svd,
                self.Xi_svd,
                self.V_svd,
                self.dw,
                self.model,
                optimize=True,
            )
            self.W3 = np.array(self.W2[:, None, :] * (self.V_svd[None, :, :]).transpose((0, 2, 1)), order="C")

        else:  # precompute matrices M_ml (M2), M_mil (M3)
            self.M2 = np.einsum(
                "k,km,m,kn,n,ln,l->ml",
                self.E,
                self.U_svd,
                self.Xi_svd,
                self.U_svd,
                self.Xi_svd,
                self.V_svd,
                self.dw,
                optimize=True,
            )
            self.M3 = np.array(self.M2[:, None, :] * (self.V_svd[None, :, :]).transpose((0, 2, 1)), order="C")

        # precompute the evidence vector Evi_m
        self.Evi = np.einsum("m,km,k,k->m", self.Xi_svd, self.U_svd, self.E, self.im_data, optimize=True)

        # precompute curvature of likelihood function
        self.d2chi2 = np.einsum(
            "i,j,ki,kj,k->ij",
            self.dw,
            self.dw,
            self.kernel.real_matrix(),
            self.kernel.real_matrix(),
            self.E,
            optimize=True,
        )

        self.chi2arr = []
        self.specarr = []
        self.backarr = []
        self.entrarr = []
        self.alpharr = []
        self.uarr = []
        self.bayesConv = []

    def compute_f_J_diag(self, u, alpha):
        v = np.dot(self.V_svd, u)
        w = np.exp(v)
        term_1 = np.dot(self.W2, w)
        term_2 = np.dot(self.W3, w)
        f = alpha * u + term_1 - self.Evi
        J = alpha * np.eye(self.n_sv) + term_2
        return f, J

    def compute_f_J_offdiag(self, u, alpha):
        v = np.dot(self.V_svd, u)
        w = np.exp(v)
        a_plus = self.model_plus * w
        a_minus = self.model_minus / w
        a1 = a_plus - a_minus
        a2 = a_plus + a_minus
        f = alpha * u + np.dot(self.M2, a1) - self.Evi
        J = alpha * np.eye(self.n_sv) + np.dot(self.M3, a2)
        return f, J

    def singular_to_realspace_diag(self, u):
        return self.model * np.exp(np.dot(self.V_svd, u))

    def singular_to_realspace_offdiag(self, u):
        v = np.dot(self.V_svd, u)
        w = np.exp(v)
        return self.model_plus * w - self.model_minus / w

    def backtransform(self, A):
        return np.trapezoid(np.dot(self.ucov, self.kernel.matrix) * A[None, :], self.re_axis, axis=-1)

    def chi2(self, A):
        return np.sum(
            self.E * (self.im_data - np.trapezoid(self.kernel.real_matrix() * A[None, :], self.re_axis, axis=-1)) ** 2
        )

    def entropy_pos(self, A, u):
        return np.trapezoid(A - self.model - A * np.dot(self.V_svd, u), self.re_axis)

    def entropy_posneg(self, A, u):
        root = np.sqrt(A**2 + 4.0 * self.model_plus * self.model_minus)
        return np.trapezoid(
            root - self.model_plus - self.model_minus - A * np.log((root + A) / (2.0 * self.model_plus)), self.re_axis
        )

    def bayes_conv(self, A, entr, alpha):
        LambdaMatrix = np.sqrt(A / self.dw)[:, None] * self.d2chi2 * np.sqrt(A / self.dw)[None, :]
        try:
            lam = np.linalg.eigvalsh(LambdaMatrix)
        except np.linalg.LinAlgError:
            lam = np.diag(LambdaMatrix)
        ng = -2.0 * alpha * entr
        tr = np.sum(lam / (alpha + lam))
        conv = tr / ng
        return ng, tr, conv

    def bayes_conv_offdiag(self, A, entr, alpha):
        A_sq = np.power((A**2 + 4.0 * self.model_plus * self.model_minus) / self.dw**2, 0.25)
        LambdaMatrix = A_sq[:, None] * self.d2chi2 * A_sq[None, :]
        lam = np.linalg.eigvalsh(LambdaMatrix)
        ng = -2.0 * alpha * entr
        tr = np.sum(lam / (alpha + lam))
        conv = tr / ng
        return ng, tr, conv

    def posterior_probability(self, A, alpha, entr, chisq):
        lambda_matrix = np.sqrt(A / self.dw)[:, None] * self.d2chi2 * np.sqrt(A / self.dw)[None, :]
        try:
            lam = np.linalg.eigvalsh(lambda_matrix)
        except np.linalg.LinAlgError:
            lam = np.diag(lambda_matrix)
        try:
            eig_sum = np.sum(np.log(alpha / (alpha + lam)))
        except RuntimeWarning:
            print(lam)
        log_prob = alpha * entr - 0.5 * chisq + np.log(alpha) + 0.5 * eig_sum
        return np.exp(log_prob)

    def maxent_optimization(self, alpha, ustart, use_bayes=False, **kwargs):
        if not self.offdiag:
            self.compute_f_J = self.compute_f_J_diag
            self.singular_to_realspace = self.singular_to_realspace_diag
            self.entropy = self.entropy_pos
        else:
            self.compute_f_J = self.compute_f_J_offdiag
            self.singular_to_realspace = self.singular_to_realspace_offdiag
            self.entropy = self.entropy_posneg

        newton_solver = NewtonOptimizer(self.n_sv, initial_guess=ustart, max_hist=1)
        sol = newton_solver(self.compute_f_J, alpha)

        u_opt = sol.x
        A_opt = self.singular_to_realspace(sol.x)
        entr = self.entropy(A_opt, u_opt)
        chisq = self.chi2(A_opt)  # has to be applied before blurring
        norm = np.trapezoid(A_opt, self.re_axis)  # is not changed by blurring

        # result = OptimizationResult()
        result_dict = {}
        result_dict.update({"u_opt": u_opt})
        if self.preblur:
            result_dict.update({"A_opt": self.kernel.blur(A_opt), "blur_width": self.kernel.blur_width})
        else:
            result_dict.update({"A_opt": A_opt, "blur_width": 0.0})
        result_dict.update(
            {
                "alpha": alpha,
                "entropy": entr,
                "chi2": chisq,
                "backtransform": self.backtransform(A_opt),
                "norm": norm,
                "Q": alpha * entr - 0.5 * chisq,
            }
        )

        if use_bayes:
            if not self.offdiag:
                ng, tr, conv = self.bayes_conv(A_opt, entr, alpha)
            else:
                ng, tr, conv = self.bayes_conv_offdiag(A_opt, entr, alpha)
            result_dict.update({"n_good": ng, "trace": tr, "convergence": conv})
            if not self.offdiag:
                prob = self.posterior_probability(A_opt, alpha, entr, chisq)
                result_dict.update({"probability": prob})
        return OptimizationResult(**result_dict)

    def solve_chi2kink(self, alpha_start=1e9, alpha_end=1e-3, alpha_div=10.0, fit_position=2.5, **kwargs):
        alpha = alpha_start
        chi = []
        alphas = []
        optarr = []
        ustart = np.zeros((self.n_sv))
        while True:
            try:
                o = self.maxent_optimization(alpha=alpha, ustart=ustart)
                optarr.append(o)
                ustart = o.u_opt
                chi.append(o.chi2)
                alphas.append(alpha)
            except:
                # For small alphas sometimes the optimization fails
                # Usually this happens at values of alpha that
                # are too small anyway, so don't worry.
                pass
            alpha = alpha / alpha_div
            if alpha < alpha_end:
                break

        alphas = np.asarray(alphas)
        chis = np.asarray(chi)

        def fitfun(x, a, b, c, d):
            return a + b / (1.0 + np.exp(-d * (x - c)))

        try:
            good_numbers = np.isfinite(chis)
            popt, pcov = opt.curve_fit(
                fitfun, np.log10(alphas[good_numbers]), np.log10(chis[good_numbers]), p0=(0.0, 5.0, 2.0, 0.0)
            )
        except ValueError:
            print("Fermi fit failed.")
            return optarr[-1], optarr

        a, b, c, d = popt

        if d < 0.0:
            raise RuntimeError("Fermi fit temperature negative.")

        a_opt = c - fit_position / d
        alpha_opt = 10.0**a_opt

        closest_idx = np.argmin(np.abs(np.log10(alphas) - a_opt))
        ustart = optarr[closest_idx].u_opt
        sol = self.maxent_optimization(alpha_opt, ustart)

        return sol, optarr

    def solve(self, **kwargs):
        return self.solve_chi2kink(**kwargs)


class OptimizationResult(object):
    def __init__(
        self,
        u_opt=None,
        A_opt=None,
        chi2=None,
        backtransform=None,
        entropy=None,
        n_good=None,
        probability=None,
        alpha=None,
        convergence=None,
        trace=None,
        Q=None,
        norm=None,
        blur_width=None,
        numerator=None,
        denominator=None,
        numerator_function=None,
        denominator_function=None,
        check=None,
        ivcheck=None,
        g_ret=None,
    ):
        self.u_opt = u_opt
        self.A_opt = A_opt
        self.chi2 = chi2
        self.backtransform = backtransform
        self.entropy = entropy
        self.n_good = n_good
        self.probability = probability
        self.alpha = alpha
        self.convergence = convergence
        self.trace = trace
        self.Q = Q
        self.norm = norm
        self.blur_width = blur_width
        self.numerator = numerator
        self.denominator = denominator
        self.numerator_function = numerator_function
        self.denominator_function = denominator_function
        self.check = check
        self.ivcheck = ivcheck
        self.g_ret = g_ret


class NewtonOptimizer(object):
    def __init__(self, opt_size, max_hist=1, max_iter=20000, initial_guess=None):
        if initial_guess is None:
            initial_guess = np.zeros((opt_size))

        self.props = [initial_guess]
        self.res = []
        self.max_hist = max_hist
        self.max_iter = max_iter
        self.opt_size = opt_size
        self.return_object = collections.namedtuple("NewtonResult", ["x", "nfev"])

    def iteration_function(self, proposal, function_vector, jacobian_matrix):
        try:
            increment = -np.dot(np.linalg.pinv(jacobian_matrix), function_vector)
        except np.linalg.LinAlgError:
            print("LinAlgError in Newton Solver, setting increment to zero")
            increment = np.zeros_like(proposal)
        step_reduction = 1.0
        significance_limit = 1e-4
        if np.any(np.abs(proposal) > significance_limit):
            ratio = np.abs(increment / proposal)
            max_ratio = np.amax(ratio[np.abs(proposal) > significance_limit])
            if max_ratio > 1.0:
                step_reduction = 1.0 / max_ratio
        result = proposal + step_reduction * increment
        return result

    def __call__(self, function_and_jacobian, alpha):
        f, J = function_and_jacobian(self.props[0], alpha)
        initial_result = self.iteration_function(self.props[0], f, J)
        self.res.append(initial_result)

        counter = 0
        converged = False
        while not converged:
            prop = self.get_proposal()
            f, J = function_and_jacobian(prop, alpha)
            result = self.iteration_function(prop, f, J)
            self.props.append(prop)
            self.res.append(result)
            converged = counter > self.max_iter or np.max(np.abs(result - prop)) < 1e-4
            counter += 1
            if np.any(np.isnan(result)):
                raise RuntimeWarning("Function returned NaN.")
        if counter > self.max_iter:
            raise RuntimeWarning("Failed to get optimization result in {} iterations".format(self.max_iter))

        self.return_object.x = result
        self.return_object.nfev = counter
        return self.return_object

    def get_proposal(self, mixing=0.5):
        n_iter = len(self.props)
        new_proposal = self.props[n_iter - 1]
        f_i = self.res[n_iter - 1] - self.props[n_iter - 1]
        update = mixing * f_i
        return new_proposal + update


class RealFreqTwoPoint:
    def __init__(self, spectrum=None, wgrid=None, kind=""):
        self.spectrum = spectrum
        self.wgrid = wgrid
        self.wmin = self.wgrid[0]
        self.wmax = self.wgrid[-1]
        self.dw = np.diff(np.concatenate(([self.wmin], (self.wgrid[1:] + self.wgrid[:-1]) / 2.0, [self.wmax])))
        self.kind = kind  # fermionic_phsym, bosonic,       fermionic
        #            or: symmetric,       antisymmetric, general

    def kkt(self):
        if self.kind == "fermionic_phsym" or self.kind == "symmetric":
            if self.wmin < 0.0:
                print("warning: wmin<0 not permitted for fermionic_phsym greens functions.")
            with np.errstate(divide="ignore"):
                m = (
                    2.0
                    * self.dw[:, None]
                    * self.wgrid[:, None]
                    * self.spectrum[:, None]
                    / (self.wgrid[None, :] ** 2 - self.wgrid[:, None] ** 2)
                )

        elif self.kind == "bosonic" or self.kind == "antisymmetric":
            if self.wmin < 0.0:
                print("warning: wmin<0 not permitted for bosonic (antisymmetric) spectrum.")
            with np.errstate(divide="ignore"):
                m = (
                    2.0
                    * self.dw[:, None]
                    * self.wgrid[None, :]
                    * self.spectrum[:, None]
                    / (self.wgrid[None, :] ** 2 - self.wgrid[:, None] ** 2)
                )

        elif self.kind == "fermionic" or self.kind == "general":
            with np.errstate(divide="ignore"):
                m = self.dw[:, None] * self.spectrum[:, None] / (self.wgrid[None, :] - self.wgrid[:, None])

        else:
            raise ValueError("Unknown kind of Greens function.")

        np.fill_diagonal(m, 0.0)  # set manually where w==w'
        self.g_real = np.sum(m, axis=0)
        self.g_imag = -self.spectrum * np.pi
        return self.g_real + 1j * self.g_imag
