import numpy as np
from typing import Tuple

def Cholesky_decomposition(A: np.ndarray
    ) -> np.ndarray:
    assert A.shape[0]==A.shape[1]
    n = A.shape[0]
    L = np.zeros((n, n))
    for r in range(n): #rows
        for c in range(r+1): #columns
            sum = 0
            for k in range(c):
                sum += L[r,k] * L[c,k]
            if r == c:
                L[r,c] = np.sqrt(A[r,c] - sum)
            else:
                L[r,c] = (A[r,c] - sum)/L[c,c]
    return L

def forward_substitution(
    LHS: np.ndarray,
    RHS: np.ndarray,
    ) -> np.ndarray:
    assert np.allclose(LHS, np.tril(LHS))
    assert LHS.shape[0]==LHS.shape[1]
    assert (RHS.shape==(LHS.shape[1],1)) or (RHS.shape==(LHS.shape[1],))
    P = LHS.shape[0]
    x = np.zeros(P)
    for m in range(P):
        sum = 0
        for i in range(m):
            sum += LHS[m,i] * x[i]
        x[m] = (RHS[m] - sum)/LHS[m,m]
    return x

def backward_substitution(
    LHS: np.ndarray,
    RHS: np.ndarray,
    ) -> np.ndarray:
    assert np.allclose(LHS, np.triu(LHS))
    assert LHS.shape[0]==LHS.shape[1]
    assert (RHS.shape==(LHS.shape[1],1)) or (RHS.shape==(LHS.shape[1],))
    P = LHS.shape[0]
    x = np.zeros(P)
    for m in reversed(range(P)):
        sum = 0
        for i in reversed(range(m)):
            sum += LHS[m,i] * x[i]
        x[m] = (RHS[m] - sum)/LHS[m,m]
    return x

def solve_linear_Cholesky(
    LHS: np.ndarray,
    RHS: np.ndarray
    ) -> np.ndarray:

    if np.all(np.linalg.eigvals(LHS) >= 0):
        L = Cholesky_decomposition(LHS)
        z = forward_substitution(LHS=L, RHS=RHS)
        a = backward_substitution(LHS=L.T, RHS=z)
        return a
    else:
        print("LHS matrix is not positive semi-definite, redefine.")
        return -1

def Gauss_Seidel_update(
    LHS: np.ndarray,
    RHS: np.ndarray,
    prev_x: np.ndarray,
    ) -> np.ndarray:

    x = np.copy(prev_x)
    P = LHS.shape[0]
    for i in range(P):
        sum_new = 0.
        sum_old = 0.
        for j in range(i):
            sum_new += LHS[i,j] * x[j]
        for j in range(i+1, P):
            sum_old = LHS[i,j] * x[j]
        x[i] = (RHS[i] - sum_new - sum_old)/LHS[i,i]
    return x

def solve_linear_Gauss_Seidel(
    LHS: np.ndarray,
    RHS: np.ndarray,
    a: np.ndarray,  # SHOULDN"T BE HERE
    max_Niter: int=1000,
    tol: float=5*10**(-3),
    ) -> np.ndarray:
        
    assert LHS.shape[0]==LHS.shape[1]
    assert (RHS.shape==(LHS.shape[1],1)) or (RHS.shape==(LHS.shape[1],))
    P = LHS.shape[0]
    if np.all(np.linalg.eigvals(LHS) >= 0):
        prev_x = np.zeros(P)
        mse = 10
        counter = 0
        while mse > tol:
            x = Gauss_Seidel_update(LHS=LHS, RHS=RHS, prev_x=prev_x)
            mse = np.sum((x-a)**2)/P
            prev_x = x
            counter += 1
            if counter > max_Niter:
                break
        return x
    else:
        print("LHS matrix is not positive semi-definite, redefine.")
        return -1

def PCG(
    LHS: np.ndarray,
    RHS: np.ndarray,
    x0: np.ndarray,
    tol: float=10**(-3),
) -> np.ndarray:

    assert LHS.shape[0]==LHS.shape[1]
    assert (RHS.shape==(LHS.shape[1],1)) or (RHS.shape==(LHS.shape[1],))
    P = LHS.shape[0]
    if np.all(np.linalg.eigvals(LHS) >= 0):
        x = np.copy(x0)
        r = RHS - LHS @ x
        p = np.copy(r)
        rsold = r.T @ r 
        for i in range(P):
            Ap = LHS @ p 
            alpha = rsold / (p.T @ Ap)
            x += alpha * p 
            r -= alpha * Ap
            rsnew = r.T @ r 
            if np.sqrt(rsnew) < tol:
                break
            p = r + (rsnew/ rsold) * p 
            rsold= np.copy(rsnew)
    return x

def standartize_X(X: np.ndarray) -> np.ndarray:
    """Standartizes the genotype matrix X: Xj = (Gj-muj)/sigma_j for each snp j. 
        If var(Gj)==0 then Xj=0. """
    X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)
    return X

def generate_single_population_X(
    n: int, 
    p: int, 
    rng: np.random._generator.Generator
    ) -> np.ndarray:
    """Generates a column-wise standartized [nxp] np.array where each column is 
    sampled from a binomial distribution with minor allele frequency MAF, which 
    is sampled from a uniform distribution."""

    X = np.empty((n, p))
    for j, maf in enumerate(rng.uniform(low=0, high=0.5, size=p)):
        col = np.zeros(p)
        while (np.all(col == 0)):
            col = rng.binomial(n=2, p=maf, size=n)
        X[:, j] = col
    assert X.shape == (n, p)
    return standartize_X(X)


def generate_genetic_cov_matrix(
    p: int,
    incl_pr: float,
    h2: np.ndarray,
    ) -> np.ndarray:
    """Generates a kxk genetic covariance matrix from Inverse-Wishart distribution"""
    G = h2/(p*incl_pr) 
    return G

def generate_effects(
    p: int, 
    G:np.ndarray,
    incl_pr: float,
    rng: np.random._generator.Generator
) -> np.ndarray:

    a_inclusion = rng.choice(np.array([0,1]), size=p, p=np.array([1-incl_pr,incl_pr]), replace=True)
    a = np.array([rng.normal(loc=0, scale=np.sqrt(a_inclusion[j] * G)) 
        for j in range(p)])

    return a, a_inclusion

def generate_R(y_hat: np.ndarray) -> np.ndarray:
    R = 1 - np.var(y_hat)
    return R

def linear(
    h2: float, 
    n: int, # for train data
    p: int, 
    incl_pr: float, 
    rng: np.random._generator.Generator,
    mu: float=0
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate genotypes and phenotypes following the multi-population 
        standard linear model."""

    X = generate_single_population_X(n=n, p=p, rng=rng)
    G = generate_genetic_cov_matrix(p=p, incl_pr=incl_pr, h2=h2)
    a, a_inclusion = generate_effects(p=p, G=G, incl_pr=incl_pr, rng=rng)

    y_hat = mu + X @ a
    R = generate_R(y_hat)
    Y = y_hat + rng.normal(loc=np.zeros(1), scale=np.sqrt(R), size=n)

    return X, Y, mu, a, R, G, a_inclusion 