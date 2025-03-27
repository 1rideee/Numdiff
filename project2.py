#Required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def partition(M):
    X = np.linspace(0,1, M)

    Tau_h = np.array([[a,b] for a, b in zip(X[0:-1:2],X[2::2])])
    return Tau_h
test_partition = np.array([[0,0.1],[0.1,0.45],[0.45,0.9],[0.9,1]])

def phi0(x):
    return 2*x*x-3*x+1
def phi1(x):
    return -4*x*x+4*x
def phi2(x):
    return 2*x*x-x

def mapping(k, xi):
    h = k[1] -k[0]
    return k[0] + h*xi

#simpson algorithm from wikipedia
def simpson(f, k):
    h = k[1] - k[0]
    c = (k[1] + k[0])/2
    return h/6 * (f(k[0]) + 4*f(c) + f(k[1]))

#Matrix calculated from integrals in the overleaf document
def Ak(h):
    return 1/(3*h)* np.array([[7, -8, 1],[-8, 16 ,-8], [1, -8, 7]])

def Fk(f, k):
    fk = np.zeros(3)
    h= k[1] - k[0]
    fk[0] = h * simpson(lambda x : f(x)*phi0(x), k)
    fk[1] = h * simpson(lambda x : f(x)*phi1(x), k)
    fk[2] = h * simpson(lambda x : f(x)*phi2(x), k)
    return fk


def local_to_global(k,a):
    return 2*k+a

#Makes the big Matrix
def assembly(partition,func):
    p = len(partition)
    m = 3*p -(p-1)
    a = np.zeros((m,m))
    f = np.zeros((m))
    for k in range(p):
        hk = partition[k][1] - partition[k][0]
        ak = Ak(hk)
        fk = Fk(func, partition[k])
        for alpha in range(3):
            for beta in range(3):
                i = local_to_global(k,alpha)
                j = local_to_global(k,beta)
                a[i][j] = a[i][j] + ak[alpha][beta]
            f[i] += fk[alpha]
    return a , f

def f(x):
    return 1

#modifies the boundary of the big matrix
def modify_boundary(A):
    A[0]=np.zeros(len(A[0]))
    A[len(A)-1]=np.zeros(len(A[0]))
    A[0][0]=1
    A[len(A)-1][len(A)-1]=1
    return A
#modified the boundary of the vector for the np.linalg.solve
def modify_vector_boundary(v,a,b):
    v[0]=a
    v[len(v)-1]=b
    return v

#gives the coefficient vector for u. Missing the varphi function added thing for the solution to work
def solve(a,f):
    u = (np.linalg.solve(a,f))
    return u



#exact solution for comparison
def exact_solution(x):
    return -(1/2)*x*(x-1)



a, f = assembly(partition(20),f)
a = modify_boundary(a)
f = modify_vector_boundary(f,0,0)
u = solve(a,f)

x = np.linspace(0,1,len(u))
plt.plot(x, u, label='u')
plt.plot(x,exact_solution(x), label='exact')
plt.show()









x = np.linspace(0,1,100)
plt.plot(x, phi0(x), label='phi0')
plt.plot(x, phi1(x), label='phi1')
plt.plot(x, phi2(x), label='phi2')
#plt.show()

a,b = partition(10)[0][0], partition(10)[0][1]
xhi = np.linspace(a, b, 100) 

plt.plot(xhi, phi0(x), label='phi0')
plt.plot(xhi, phi1(x), label='phi1')
plt.plot(xhi, phi2(x), label='phi2')
#plt.show()


