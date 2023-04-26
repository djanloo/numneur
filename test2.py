from numneur.stochastic import thresholded_OU as throu, poisson_process as pp
import matplotlib.pyplot as plt
import numpy as np

# N = 100_000
# M = 60

# R = np.zeros((M,M))

# Is = np.linspace(0.0, 2.0, M)
# sigmas = np.linspace(0.0, 2.0, M)

# for i in range(M):
#     print(i)
#     for j in range(M):
#         I = Is[j]*np.ones(N)
#         v, ft = throu(I, sigma=sigmas[i], tau=1)
#         R[i,j] = len(ft)

# X, Y = np.meshgrid(Is, sigmas)

# plt.contourf(X, Y, R)
# plt.colorbar()

# plt.xlabel("I")
# plt.ylabel("sigma")
# plt.title("Firing rate")

# plt.show()

poiss, interspike = pp(0.5, 100_000, 0.01)
plt.figure(1)
plt.hist(interspike)
plt.figure(2)
plt.plot(poiss)
print( np.sum(poiss > 0.5)/len(poiss)/0.01)
plt.show()