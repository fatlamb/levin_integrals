#!/usr/bin/python

import numpy as np
import scipy
from scipy import special
from scipy import integrate as sciint
from math import pi
import time

import levin

def testfunc(x) :
    return x**6
    #return 1.0

integrator = levin.SphericalBesselIntegrator()

a=1.0
b=10.0
#alpha=[0.01,1]
alpha=[0.0,0]
#beta=[10.0,1]
beta=[0.0,0]
l=0

start = time.clock()
kresult = integrator.KCalc(a, b, alpha, beta, l, testfunc)
end = time.clock()
ktime = end-start

start = time.clock()
hresult = integrator.HCalc(a, b, alpha, l, testfunc)
end = time.clock()
htime = end-start

start = time.clock()
iresult = integrator.ICalc(a, b, alpha, l, testfunc)
end = time.clock()
itime = end-start

print
print("K Integrals")
print("Result: "+'{:.15e}'.format(kresult))
print("Time: "+'{:.2e}'.format(ktime)+" Seconds")
print
print("H Integrals")
print("Result: "+'{:.15e}'.format(hresult))
print("Time: "+'{:.2e}'.format(htime)+" Seconds")
print
print("I Integrals")
print("Result: "+'{:.15e}'.format(iresult))
print("Time: "+'{:.2e}'.format(itime)+" Seconds")
print
