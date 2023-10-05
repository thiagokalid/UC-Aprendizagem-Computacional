import numpy as np
import scipy

data = np.loadtxt("heightWeightData.csv", delimiter=",", dtype=float)
G = data[:, 0]
H = data[:, 1]
W = data[:, 2]

# Compute the samples average:
mu_G = np.mean(G)
mu_H = np.mean(H)
mu_W = np.mean(W)

GHW = np.stack((G, H, W), axis=0)
cov_ghw = np.cov(GHW)

HW = np.stack((H, W), axis=0)
cov_hw = np.cov(HW)

#
pdf_GHW = scipy.stats.multivariate_normal(mean=[mu_G, mu_H, mu_W], cov=cov_ghw)

pdf_HW = scipy.stats.multivariate_normal(mean=[mu_H, mu_W], cov=cov_hw)

pdf_G_given_HW = lambda gender, height, width: pdf_GHW.pdf([gender, height, width]) / pdf_HW.pdf([height, width])

# What is the probability of the individual be a male given a certain height "h" and a weight "w"?
for i in range(len(G)):
    g = G[i]
    h = H[i]
    w = W[i]
    probability = pdf_G_given_HW(1, h, w)*100
    print(f"Probability of beeing a male is {probability:.2f} %. The answer is {g}")

g_probability = [pdf_G_given_HW(1, H[i], W[i]) for i in range(len(G))]