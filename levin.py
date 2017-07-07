#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:46:18 2017

@author: jolyon
"""

import numpy as np
import scipy
from scipy import special
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

def testfunc(x) :
    return x**6

integrator = LevinIntegrals(21)
integrator2 = LevinIntegrals(10)

kresult = integrator.compute_K(1.0, 10.0, 0.01, 10.0, 10, testfunc)
kresult2 = integrator2.compute_K(1.0, 10.0, 0.01, 10.0, 10, testfunc)

hresult = integrator.compute_H(1.0, 10.0, 0.01, 10, testfunc)
hresult2 = integrator2.compute_H(1.0, 10.0, 0.01, 10, testfunc)

iresult = integrator.compute_I(1.0, 10.0, 0.01, 10, testfunc)
iresult2 = integrator2.compute_I(1.0, 10.0, 0.01, 10, testfunc)

print("K Integrals")
print('{:.15e}'.format(kresult))
print('{:.15e}'.format(kresult2))
print("H Integrals")
print('{:.15e}'.format(hresult))
print('{:.15e}'.format(hresult2))
print("I Integrals")
print('{:.15e}'.format(iresult))
print('{:.15e}'.format(iresult2))
