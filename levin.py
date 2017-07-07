#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:46:18 2017

@author: jolyon
"""

import numpy as np
import scipy
from scipy import special
from scipy import integrate as sciint
from math import pi

class LevinIntegrals(object) :

    def __init__(self, numpoints) :
        self.numpoints = numpoints
        theta = pi * np.linspace(0.0, 1.0, numpoints)
        # Collocation points in x space
        self.colloc = -np.cos(theta)

        # These are (n, collocation) indexed
        self.chebyTs = np.array([self.chebyT(theta, n) for n in range(numpoints)])
        self.dchebyTs = np.array([n * self.chebyU(theta, n-1) for n in range(0, numpoints)])

    def kvals(self, a, b) :
        """Return k values for a given set of collocation points, given a and b"""
        return (b-a)/2*self.colloc + (b+a)/2

    @staticmethod
    def chebyT(theta, n) :
        """Returns Chebyshev polynomials of the first kind of order n at value -theta"""
        return (-1) ** (n % 2) * np.cos(n * theta)

    @staticmethod
    def chebyU(theta, n) :
        """Returns Chebyshev polynomials of the second kind of order n at value -theta"""
        if n == -1 :
            return np.zeros_like(theta)
        subtheta = theta[1:-1]
        n1 = n + 1
        return (-1) ** (n % 2) * np.concatenate(([n1], np.sin(n1 * subtheta) / np.sin(subtheta), [(-1) ** (n % 2) * n1]))


    def compute_K(self, a, b, alpha, beta, l, func) :
        """Computes int(func(k) j_l(alpha k) j_l(beta k), {k, a, b})"""

        Abase = np.zeros((4,4))
        Abase[0,1] = -alpha
        Abase[0,2] = -beta
        Abase[1,0] = alpha
        Abase[2,0] = beta
        Abase[1,3] = -beta
        Abase[3,1] = beta
        Abase[2,3] = -alpha
        Abase[3,2] = alpha

        diagbase = np.diag([2*l, -2, -2, -2*(2+l)])

        ks = self.kvals(a, b)
        derivs = 2 / (b - a) * self.dchebyTs

        sphlaa = scipy.special.spherical_jn(l, alpha * a)
        sphlba = scipy.special.spherical_jn(l, beta * a)
        sphl1aa = scipy.special.spherical_jn(l+1, alpha * a)
        sphl1ba = scipy.special.spherical_jn(l+1, beta * a)
        sphlab = scipy.special.spherical_jn(l, alpha * b)
        sphlbb = scipy.special.spherical_jn(l, beta * b)
        sphl1ab = scipy.special.spherical_jn(l+1, alpha * b)
        sphl1bb = scipy.special.spherical_jn(l+1, beta * b)

        weightsa = np.array([
                    sphlaa * sphlba,
                    sphl1aa * sphlba,
                    sphlaa * sphl1ba,
                    sphl1aa * sphl1ba
                   ])
        weightsb = np.array([
                    sphlab * sphlbb,
                    sphl1ab * sphlbb,
                    sphlab * sphl1bb,
                    sphl1ab * sphl1bb
                   ])

        # Indexed by weight, then collocation point
        flist = np.transpose(np.array([[func(k), 0.0, 0.0, 0.0] for k in ks]))

        # Construct the matrix equations a_{ijkl} c_{ik} = f_{jl}
        # a has indices of weight, weight, Chebyshev order, collocation index
        # F_i(x) = sum_k c_{ik} T_k(x)
        a = np.zeros([4, 4, self.numpoints, self.numpoints])
        # Loop over collocation points
        for l, kval in enumerate(ks) :
            Amat = Abase + diagbase / kval
            a[:,:,:,l] = np.tensordot(Amat, self.chebyTs[:,l], 0) + np.tensordot(np.eye(4), derivs[:,l], 0)

        # Reorganize the tensor so that the contracted indices are rightmost
        atensor = np.transpose(a, (1, 3, 0, 2))

        # Solve the system
        cij = np.linalg.tensorsolve(atensor, flist)

        # Construct the integral
        resulta = np.dot(np.dot(cij, self.chebyTs[:, 0]), weightsa)
        resultb = np.dot(np.dot(cij, self.chebyTs[:, -1]), weightsb)

        # Return the result
        return resultb - resulta

    def compute_H(self, a, b, alpha, l, func) :
        """Computes int(func(k) j_l(alpha k)^2, {k, a, b})"""

        Abase = np.zeros((3,3))
        Abase[0,1] = -2.0*alpha
        Abase[1,0] = alpha
        Abase[1,2] = -alpha
        Abase[2,1] = 2.0*alpha

        diagbase = np.diag([2*l, -2, -2*(2+l)])

        ks = self.kvals(a, b)
        derivs = 2 / (b - a) * self.dchebyTs

        sphla = scipy.special.spherical_jn(l, alpha * a)
        sphl1a = scipy.special.spherical_jn(l+1, alpha * a)
        sphlb = scipy.special.spherical_jn(l, alpha * b)
        sphl1b = scipy.special.spherical_jn(l+1, alpha * b)

        weightsa = np.array([
                    sphla**2,
                    sphla * sphl1a,
                    sphl1a**2
                   ])
        weightsb = np.array([
                    sphlb**2,
                    sphlb * sphl1b,
                    sphl1b**2
                   ])

        # Indexed by weight, then collocation point
        flist = np.transpose(np.array([[func(k), 0.0, 0.0] for k in ks]))

        # Construct the matrix equations a_{ijkl} c_{ik} = f_{jl}
        # a has indices of weight, weight, Chebyshev order, collocation index
        # F_i(x) = sum_k c_{ik} T_k(x)
        a = np.zeros([3, 3, self.numpoints, self.numpoints])
        # Loop over collocation points
        for l, kval in enumerate(ks) :
            Amat = Abase + diagbase / kval
            a[:,:,:,l] = np.tensordot(Amat, self.chebyTs[:,l], 0) + np.tensordot(np.eye(3), derivs[:,l], 0)

        # Reorganize the tensor so that the contracted indices are rightmost
        atensor = np.transpose(a, (1, 3, 0, 2))

        # Solve the system
        cij = np.linalg.tensorsolve(atensor, flist)

        # Construct the integral
        resulta = np.dot(np.dot(cij, self.chebyTs[:, 0]), weightsa)
        resultb = np.dot(np.dot(cij, self.chebyTs[:, -1]), weightsb)

        # Return the result
        return resultb - resulta

    def compute_I(self, a, b, alpha, l, func) :
        """Computes int(func(k) j_l(alpha k), {k, a, b})"""

        Abase = np.zeros((2,2))
        Abase[0,1] = -alpha
        Abase[1,0] = alpha

        diagbase = np.diag([l, -(2+l)])

        ks = self.kvals(a, b)
        derivs = 2 / (b - a) * self.dchebyTs

        sphla = scipy.special.spherical_jn(l, alpha * a)
        sphl1a = scipy.special.spherical_jn(l+1, alpha * a)
        sphlb = scipy.special.spherical_jn(l, alpha * b)
        sphl1b = scipy.special.spherical_jn(l+1, alpha * b)

        weightsa = np.array([
                    sphla,
                    sphl1a
                   ])
        weightsb = np.array([
                    sphlb,
                    sphl1b
                   ])

        # Indexed by weight, then collocation point
        flist = np.transpose(np.array([[func(k), 0.0] for k in ks]))

        # Construct the matrix equations a_{ijkl} c_{ik} = f_{jl}
        # a has indices of weight, weight, Chebyshev order, collocation index
        # F_i(x) = sum_k c_{ik} T_k(x)
        a = np.zeros([2, 2, self.numpoints, self.numpoints])
        # Loop over collocation points
        for l, kval in enumerate(ks) :
            Amat = Abase + diagbase / kval
            a[:,:,:,l] = np.tensordot(Amat, self.chebyTs[:,l], 0) + np.tensordot(np.eye(2), derivs[:,l], 0)

        # Reorganize the tensor so that the contracted indices are rightmost
        atensor = np.transpose(a, (1, 3, 0, 2))

        # Solve the system
        cij = np.linalg.tensorsolve(atensor, flist)

        # Construct the integral
        resulta = np.dot(np.dot(cij, self.chebyTs[:, 0]), weightsa)
        resultb = np.dot(np.dot(cij, self.chebyTs[:, -1]), weightsb)

        # Return the result
        return resultb - resulta

class SphericalBesselIntegrator(object) :
	"""
	A wrapper class for LevinIntegrals. Instantiates two LevinIntegrals
	objects at 21 and 10 collocation points to estimate error (persistence
	of these objects requires a wrapper class, instead of a function).
	Also implements case-checking and -handling for alpha=0, beta=0 (for K
	integrals), or for alpha=0 (for H/I integrals).
	"""

	def __init__(self) :
		self.integrator_hiprec = LevinIntegrals(21)
		self.integrator_loprec = LevinIntegrals(10)
		self.rel_tol = 1e-8

	def KCalc(self, a, b, alpha_tup, beta_tup, l, func):
		"""
		Assuming the first entry in the radial coordinate array 
		is zero. Check to see if both alpha and beta have index 
		zero and l==0. If so, the collocation method fails because the 
		differential matrix is singular. But in this case the 
		covariance is simply the integral of the power spectrum,
		which is easily done with scipy's built-in quadrature routine. 
		"""
	
		if (alpha_tup[1]==0 and beta_tup[1]==0 and l==0):
			#Absolute error tolerance
			intepsrel=1e-12
			#Compute integral using quadrature.
			(result,err) = sciint.quad(func,a,b,epsrel=intepsrel)		
			#Estimate and check relative error.
			rel_err = abs(err/result)
			if rel_err > self.rel_tol:
				#FIXME: handle this error properly!
				print(str(self.rel_tol)+" Relative Error Bound Exceeded (Quadrature)")
		else:
			kresult_hiprec = self.integrator_hiprec.compute_K(a, b, alpha_tup[0], 
				beta_tup[0], l, func)
			kresult_loprec = self.integrator_loprec.compute_K(a, b, alpha_tup[0], 
				beta_tup[0], l, func)
			result = kresult_hiprec
			#Estimate and check relative error.
			err = abs(kresult_hiprec-kresult_loprec)
			rel_err = abs(err/result)
			if rel_err > self.rel_tol:
				#FIXME: handle this error properly!
				print(str(self.rel_tol)+" Relative Error Bound Exceeded (LevinCollocation)")
		return result

	def HCalc(self, a, b, alpha_tup, l, func):
		"""
		Assuming the first entry in the radial coordinate array 
		is zero. Check to see if both alpha has index 
		zero and l==0. See KCalc().
		"""
	
		if (alpha_tup[1]==0 and l==0):
			#Absolute error tolerance
			intepsrel=1e-12
			#Compute integral using quadrature.
			(result,err) = sciint.quad(func,a,b,epsrel=intepsrel)		
			#Estimate and check relative error.
			rel_err = abs(err/result)
			if rel_err > self.rel_tol:
				#FIXME: handle this error properly!
				print(str(self.rel_tol)+" Relative Error Bound Exceeded (Quadrature)")
		else:
			hresult_hiprec = self.integrator_hiprec.compute_H(a, b, alpha_tup[0], l, func)
			hresult_loprec = self.integrator_loprec.compute_H(a, b, alpha_tup[0], l, func)
			result = hresult_hiprec
			#Estimate and check relative error.
			err = abs(hresult_hiprec-hresult_loprec)
			rel_err = abs(err/result)
			if rel_err > self.rel_tol:
				#FIXME: handle this error properly!
				print(str(self.rel_tol)+" Relative Error Bound Exceeded (LevinCollocation)")
		return result

	def ICalc(self, a, b, alpha_tup, l, func):
		"""
		Assuming the first entry in the radial coordinate array 
		is zero. Check to see if both alpha has index 
		zero and l==0. See KCalc().
		"""
		if (alpha_tup[1]==0 and l==0):
			#Absolute error tolerance
			intepsrel=1e-12
			#Compute integral using quadrature.
			(result,err) = sciint.quad(func,a,b,epsrel=intepsrel)		
			#Estimate and check relative error.
			rel_err = abs(err/result)
			if rel_err > self.rel_tol:
				#FIXME: handle this error properly!
				print(str(self.rel_tol)+" Relative Error Bound Exceeded (Quadrature)")
		else:
			iresult_hiprec = self.integrator_hiprec.compute_I(a, b, alpha_tup[0], l, func)
			iresult_loprec = self.integrator_loprec.compute_I(a, b, alpha_tup[0], l, func)
			result = iresult_hiprec
			#Estimate and check relative error.
			err = abs(iresult_hiprec-iresult_loprec)
			rel_err = abs(err/result)
			if rel_err > self.rel_tol:
				#FIXME: handle this error properly!
				print(str(self.rel_tol)+" Relative Error Bound Exceeded (LevinCollocation)")
		return result
