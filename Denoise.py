import numpy as np
from matplotlib import pyplot as plt
import pdb


def descend(data, thresh, iota, k_max):
    print("Initialize starting parameters")
    # Step
    k = 0
    # Sensing Matrix
    print("initializing A Matrix!")

    f0 = 0.9 * data + np.random.normal(size = data.size).reshape(data.shape)

    A = np.ones_like(data)

    # Set the initial starting point to be f at current step k
    print("initializing f_k")


    # Euclidean distance between current guess and previous guess
    # This will be loop termination criterion
    dx = 2 * thresh  # can't equal zero
    f_kmin1 = 0.9 * f0 - 0.1  # can't equal f0

    # Approximate gradient

    dx_k = f0 - f_kmin1

    # Choose intial alpha_k using modified Barzilai-Borwein gradient descent
    print("calculating new alpha_k")
    alpha_k = np.linalg.norm(np.sqrt(data) * np.matmul(A, dx_k) / np.matmul(A, f0)) ** 2 / np.linalg.norm(dx_k) ** 2
    lst = []
    N = 0
    while (dx > thresh and k < k_max) or k < 1:
        # calculate gradent at current step
        g_k = approx_grad(data, A, f0)
        # Determine how much to move towards the local minimum
        s_k = f0 - g_k / alpha_k
        # Calculate next guess f_kplus1
        tv = TV(f0)
        f_kplus1 = s_k + (1 / alpha_k) * tv
        # Update dx

        dx = np.abs(np.linalg.norm(f_kplus1 - f0) / np.linalg.norm(f0))

        # update current step
        k += 1
        f_kmin1 = f0
        f0 = f_kplus1
        # increment alpha_k by iota
        alpha_k += iota
        print("dx = {} iter = {}".format(dx, N))
        lst.append(dx)
        N += 1
    # This is your true signal
    print("Finished iterations!")
    print("Number of iterations: {}".format(N))
    print("final dx: {}".format(dx))

    lst.pop(0)
    fig = plt.figure()
    xs = np.arange(0, len(lst))
    plt.plot(xs, lst)
    plt.xlabel("Iterations")
    plt.ylabel("dx")
    plt.savefig("descend.png")
    plt.close(fig)

    return f0


def approx_grad(data, A, f_k):
    e_i = np.eye(data.shape[0])
    denom = np.matmul(e_i.T, np.matmul(A, f_k)) + 1e-20
    g = np.matmul(data, np.matmul(A.T, e_i))
    return g / denom


def TV(img):
    cp1 = np.zeros_like(img)
    cp2 = np.zeros_like(img)

    N = img.shape[0]
    K = img.shape[1]

    for n in range(0, N-1):
        for k in range(0, K):
            cp1[n][k] = np.abs(img[n][k] - img[n+1][k])

    for n in range(0, N):
        for k in range(0, K-1):
            cp1[n][k] = np.abs(img[n][k] - img[n][k+1])

    return cp1 + cp2