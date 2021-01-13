import numpy as np
import pdb
from matplotlib import pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from numba import jit, cuda


def descend(data, f0, thresh, iota, k_max):
    # Initialize starting parameters
    # Step
    k = 0
    n = data.size
    m = n
    # Sensing Matrix
    print("initializing A Matrix!")
    # RUN THIS ON GPU
    A = np.random.randint(1,100,(m, n))

    # Set the initial starting point to be f at current step k
    print("initializing f_k")


    # Euclidean distance between current guess and previous guess
    # This will be loop termination criterion
    dx = 2 * thresh # can't equal zero
    #Run tu
    f_kmin1 = 0.9*f0-0.1 # can't equal f0

    # Approximate gradient
    g = aprox_grad(data, A, f_kmin1, m)
    g_k = None

    dx_k = f0 - f_kmin1

    # Choose intial alpha_k using modified Barzilai-Borwein gradient descent
    print("calculating new alpha_k")
    alpha_k = np.linalg.norm(np.sqrt(data) * A * dx_k / A * f0) ** 2 / np.linalg.norm(dx_k) ** 2
    lst = []
    N = 0
    while(dx > thresh and k < k_max) or k < 1:
        print("Calculating TV Seminorm for signal at step k")
        tv = denoise_tv_chambolle(f0)

        print("Calculating f_k+1")
        # TODO start CUDA here
        dx, f_kplus1 = helper(data, A, f0, alpha_k, tv, f_kmin1, m)

        # TODO end CUDA here
        # update current step
        k += 1
        f_kmin1 = f0
        f0 = f_kplus1
        # increment alpha_k by iota
        print("Incrementing alpha_k y iota")
        alpha_k += iota
        print("iter = {}".format(N))
        lst.append(dx)
        N += 1
    # This is your true signal
    print("Finished iterations!")
    print("Number of iterations: {}".format(N))
    print("final dx: {}".format(dx))

    lst.pop(0)
    fig = plt.figure(figsize=(15,15))
    xs = np.arange(0,len(lst))
    plt.plot(xs,lst)
    plt.xlabel("Iterations")
    plt.ylabel("dx")
    plt.savefig("descend.png")
    plt.close(fig)

    return f0

#@jit(target ="cuda")
def helper(data, A, f0, alpha_k, tv, f_kmin1, m):
    # calculate gradent at current step
    g_k = aprox_grad(data, A, f_kmin1, m)

    # Determine how much to move towards the local minimum
    s_k = f0 - g_k / alpha_k
    # Calculate next guess f_kplus1
    f_kplus1 = s_k + (1 / alpha_k) * tv
    # Update dx
    dx = np.abs(np.linalg.norm(f_kplus1 - f0) / np.linalg.norm(f0))

    return dx, f_kplus1

def aprox_grad(data, A, f_k, m):
    holder = data / (A * f_k)
    return A - sum(holder)




#data = np.array([200])
#f0 = 0.9*data-0.1
#d = Denoiser(data,f0)
#thresh = 1e-9
#d.descend(thresh, 1)




