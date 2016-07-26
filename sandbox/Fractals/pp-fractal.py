
import pp
import math
import numpy as np
import pylab as plb
import matplotlib.pyplot as plt
from numpy import pi
from colorsys import hls_to_rgb
from textwrap import wrap

'''
    Parallel version of newton.py to generate fractals based on the mapping of an initial guess to a Newton-Raphson
    root finding algorithm to the root it finds.

    Uses the parallel python library PP (available from: http://www.parallelpython.com)

'''


#The polynomial
def f(z, coeff):
    value = 0.0
    for i in range(len(coeff)):
        value += coeff[i]*z**(len(coeff)-i)
    return value

#The derivative f'(z)
def f_prime(z, coeff):
    value = 0.0
    for i in range(len(coeff)-1):
        value += coeff[i]*(len(coeff)-i)*z**(len(coeff)-i-1)
    return value

#The second derivative f''(z)
def f_doubleprime(z, coeff):
    value = 0.0
    for i in range(len(coeff)-2):
        value += coeff[i]*(len(coeff)-i)*(len(coeff)-i-1)*z**(len(coeff)-i-2)
    return value


#Newton-Raphson Solver
def newton(x,i,j,coeff):

    while abs(f(x,coeff)) >= 1e-12: #Tolerance value hardcoded
        
        if f_prime(x,coeff) == 0:
            x = x - f(x)/(f_prime(x)**2-f(x)*f_doubleprime(x))

        else:
            x = x - f(x,coeff)/f_prime(x,coeff)           

    return x,i,j

#Coloring complex valued things
#Lifted from the internets
def colorize(z):
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    return c

       
#Main program
print ' Enter the coefficients of your polynomial (up to degree 10) seperated by spaces. \n For example the polynomial \n\n\t x^2 + 2x + 3\n\n would be entered as \"1 2 3\":'
coeff = raw_input().split()
left_bc = -2000
right_bc = 2000
size = 500

#Cast the entries as floating point numbers
coeff = [float(i) for i in coeff]

#2D array to hold data (initialized to zero)
data = np.zeros((size,size), dtype = complex)

#x and y values for image (each represents some number of pixels?)
x = np.linspace(left_bc, right_bc, size)

#Paralell type things here
job_server=pp.Server()

print "Starting ", job_server.get_ncpus(), " workers"

jobs = [] #Container for all jobs to be run in parallel

#Fill job list 
for i in range(len(x)):
    for j in range(len(x)):
        jobs.append(job_server.submit(newton, (complex(x[i]/float(size),x[j]/float(size)),i,j,coeff), (f,f_prime,f_doubleprime)))

#Execute in parallel and save results
x = [job() for job in jobs]

#End parallel block
job_server.print_stats()

#Put data where it needs to be for image
for a in x:
    data[a[1]][a[2]] = a[0]

#Generating a string of the entered polynomial
polynomial = ""
for i in range(len(coeff)):
    if i == 0:
        polynomial += "%d * x^%d " % (coeff[i], len(coeff) - i)
    else:
        polynomial += "+ %d * x^%d " % (coeff[i], len(coeff) - i)


#Plotting
img = colorize(data)
im = plb.imshow(img)
im.axes.get_xaxis().set_visible(False)
im.axes.get_yaxis().set_visible(False)
#plb.title("\n".join(wrap(polynomial)))
plb.savefig('fractal.pdf', bbox_inches='tight')
plt.show()


