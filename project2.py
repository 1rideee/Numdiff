#Required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def partition(X):  

    Tau_h = np.array([[a,(a+b)/2, b] for a, b in zip(X[0:-1], X[1:])])
    return Tau_h


def phi0(x):
    return 2*x*x-3*x+1

def phi1(x):
    return -4*x*x+4*x

def phi2(x):
    return 2*x*x-x

def mapping(K, xi):
    h = K[2] - K[0]
    return K[0] + h*xi

def local_to_global(k,a):
    return 2*k+a

#simpson algorithm from wikipedia
def simpson(f):

    return (1/6) * (f(0) + 4*f(0.5) + f(1))

#Matrix calculated from integrals in the overleaf document
def Ak(h):
    return 1/(6*h)* np.array([[7, -8, 1], [-8, 16 ,-8], [1, -8, 7]])

def Fk(f, K):
    fk = np.zeros(3)
    h = K[2] - K[0]
    fk[0] = h * simpson(lambda x : f(mapping(K,x))*phi0(x))
    fk[1] = h * simpson(lambda x : f(mapping(K,x))*phi1(x))
    fk[2] = h * simpson(lambda x : f(mapping(K,x))*phi2(x))
    return fk


#Makes the big Matrix
def assembly(partition, func):
    
    p = np.shape(partition)[0]
    m = 3*p - (p-1)
    A = np.zeros((m,m))
    F = np.zeros((m))

    for k in range(p):
        hk = partition[k][1] - partition[k][0]
        ak = Ak(hk)
        fk = Fk(func, partition[k])
        for alpha in range(3):
            i = local_to_global(k,alpha)
            for beta in range(3):
                j = local_to_global(k,beta)
                A[i][j] = A[i][j] + ak[alpha][beta]
            F[i] += fk[alpha]
    return A , F

def g(x):
    return 1



#modifies the boundary of the big matrix
def modify_boundary(A):
    A[0] = np.zeros(len(A[0]))
    A[-1] = np.zeros(len(A[0]))
    A[0][0] = 1
    A[-1][-1] = 1
    return A

#modified the boundary of the vector for the np.linalg.solve
def modify_vector_boundary(v, a, b):
    v[0]=a
    v[len(v)-1]=b
    return v



#exact solution for comparison
def exact_solution(x):
    return -(1/2)*x*(x-1)


def problem1(exact_solution, g): 
    
    X = [0,0.1,0.2,0.3,0.35,0.4,0.45,0.6,0.9, 0.96,1]  


    tau = partition(X)
    

    combined = []

    for i in range(len(X) - 1):
        combined.append(X[i])  # Add original point
        combined.append((X[i] + X[i+1]) / 2)  # Add midpoint

    combined.append(X[-1])  # Add the last original point


    a, F = assembly(tau,g)
    a = modify_boundary(a)
    F = modify_vector_boundary(F,0,0)
    print("F:",F)
    u = np.linalg.solve(a,F)

    x = np.linspace(0,1,len(u))
    plt.plot(combined, u, "-o", label='u')
    plt.plot(x, exact_solution(x), label='exact')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.show()


problem1(exact_solution, g)

def u_1(x):
    return np.sin(2*np.pi*x)

def f_1(x):
    return 4*np.pi**2 * np.sin(2*np.pi*x)

problem1(u_1, f_1)


def Ak2(h):
    return h/(15)* np.array([[2, 1, -1/2], [1, 8 , 1], [-1/2, 1, 2]])

def assembly2(partition):
    
    p = np.shape(partition)[0]
    m = 3*p - (p-1)
    A = np.zeros((m,m))

    for k in range(p):
        hk = partition[k][1] - partition[k][0]
        ak = Ak2(hk)
        
        for alpha in range(3):
            i = local_to_global(k,alpha)
            for beta in range(3):
                j = local_to_global(k,beta)
                A[i][j] = A[i][j] + ak[alpha][beta]
            
    return A 

def y_d1(x):
    return (1/2)*x*(1-x)

def y_d2(x):
    return 1

def y_d3(x):
    if x >0.25 and x < 0.75:
        return 1
    else:
        return 0


def optim(alpha, y_d):
    X = [0,0.1,0.2,0.3,0.35,0.4,0.45,0.6,0.9, 0.96,1]  
    midpoints = [(X[i] + X[i+1]) / 2 for i in range(len(X)-1)]

    tau = partition(X)
    
    combined = []

    for i in range(len(X) - 1):
        combined.append(X[i])  # Add original point
        combined.append((X[i] + X[i+1]) / 2)  # Add midpoint

    combined.append(X[-1])  # Add the last original point
    
    A = assembly2(tau)
    A = A[1:-1, 1:-1]

    Abar = assembly2(tau)
    Abar = Abar[:,1:-1]
    
    Aprime, F = assembly(tau,g)
    Aprime = Aprime[1:-1, 1:-1]

    big_A = np.block([[A, np.zeros_like(A), -Aprime.T], 
                      [np.zeros_like(A), (alpha)*A, A.T],
                      [-Aprime, A.T, np.zeros_like(A)]])
    
    
    big_b = np.array([np.dot(Abar.T, [y_d(x) for x in combined])])
    big_b = np.append(big_b, np.zeros(len(A)*2))
    print(np.shape(big_b), np.shape(big_A))

    sol = np.linalg.solve(big_A, big_b)

    y = sol[0:len(A)]
    u = sol[len(A):2*len(A)]
    lambda_ = sol[2*len(A):]
    print("y:", y)
    print("u", u)
    print("lambda:", lambda_)

    # plt.plot(combined[1:-1], y, "-", label='y')
    plt.plot(combined[1:-1],u, "-o", label='u')
    # plt.plot(combined[1:-1],lambda_, "-o", label='lambda')
    
    plt.plot(combined, [y_d(x) for x in combined], "o", label='y_d')
    plt.legend()
    plt.title(f"Numerical solution with alpha = {alpha}")
    plt.grid()


plot= True
if plot:
    alpha = 1
    optim(alpha, y_d1)
    optim(alpha, y_d2)
    optim(alpha, y_d3)
    plt.show()

    optim(1/100, y_d1)
    optim(1/100, y_d2)
    optim(1/100, y_d3)
    plt.show()

    optim(1e-6, y_d1)
    optim(1e-6, y_d2)
    optim(1e-6, y_d3)
    plt.show()

    optim(1e-8, y_d1)
    optim(1e-8, y_d2)
    optim(1e-8, y_d3)
    plt.show()