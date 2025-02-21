# coding: utf-8
# Script for performing change point detection on Grassmann manifolds
#
# Reference: 
# Riemannian change point Detection on Manifolds with Robust Centroid Estimation
# Xiuheng Wang, Ricardo Borsoi, Arnuad Breloy, Cédric Richard
#
# 2024/07
# Implemented by
# Xiuheng Wang
# dr.xiuheng.wang@gmail.com

import pymanopt
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from matplotlib.pyplot import MultipleLocator

from utils.draw_figure import comp_arl_mdd, makedir
from utils.riemannian_cpd import riemannian_cpd_grassmann, cpd_rkm_grassmann
from utils.functions import generate_random_SPD_mtx, generate_random_mtx_normal

figure_path = './figures/'
# parameter settings
lambda_0 = 1e-2
lambda_1 = 2e-2
lambda_rkm = 1e-1
Huber_parameter = 5e-2

# experiment setups
T = 2000
Tc = 1500
N = 20 # Dimension of the ambient space.
P = 5 # Dimension of the subspaces.
Iter = 1e3

# generate parameters for two matrix normal distributions
np.random.seed(1)
temp = np.random.randn(N,N)
eigsv = np.random.rand(N) + 1e-6 # positive
U = generate_random_SPD_mtx(temp, eigsv)
temp = np.random.randn(N,N)
eigsv = np.random.rand(N) + 1e-6 # positive
V = generate_random_SPD_mtx(temp, eigsv)
M0 = np.random.randn(N,N) + 1
M1 = M0 + 0.03 * np.random.randn(N,N)

# define manifold
manifold = pymanopt.manifolds.grassmann.Grassmann(N, P)

stat_all = []
stat_all_rkm = []
for _ in tqdm(range(int(Iter))):
    X = []
    for t in range(T):
        if t < Tc:
            temp = generate_random_mtx_normal(M0, U, V)
            X.append(np.linalg.svd(temp)[0][:, :P]) # Truncated SVD 
        else:
            temp = generate_random_mtx_normal(M1, U, V)
            X.append(np.linalg.svd(temp)[0][:, :P]) # Truncated SVD
    stat_all.append(riemannian_cpd_grassmann(manifold, X, lambda_0, lambda_1))
    stat_all_rkm.append(cpd_rkm_grassmann(manifold, X, lambda_rkm, Huber_parameter))

# gather all test statistics
stats = []
stats.append(stat_all)
stats.append(stat_all_rkm)

# set names and colors
names = ["Baseline", "Proposed"]
colors = ["#82B0D2", "#FA7F6F"] # ["#BEB8DC", "#82B0D2", "#8ECFC9", "#FFBE7A", "#FA7F6F"]

# draw figures
start_point = 400
if not os.path.exists(figure_path):
    makedir(figure_path)
fig = plt.figure(figsize = (6, 4), dpi = 120)
for index in range(len(names)):
    ax = fig.add_subplot(len(names), 1, index+1)
    avg = np.mean(stats[index], axis = 0)
    std = np.std(stats[index], axis = 0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
    ax.plot(range(0, T), avg, color = "#2F7FC1")
    ax.fill_between(range(0, T), r1, r2, alpha=0.2)
    plt.axvline(Tc, color = "#FA7F6F")
    plt.legend([names[index]], loc = 1)
    plt.xlim(start_point, T)
    plt.ylim(0.9*np.min(r1[start_point:]), 1.1*np.max(r2[start_point:]))
plt.tight_layout()
plt.subplots_adjust(hspace = 0.28)
plt.savefig(figure_path + "simulation_grassmann.pdf", bbox_inches='tight')

N_th = 1000
fig = plt.figure(figsize = (3.2, 3.0), dpi = 150)
for index in range(len(names)):
    arl, mdd = comp_arl_mdd(stats[index], Tc, N_th, start_point)
    plt.plot(arl, mdd, color=colors[index], label=names[index])
plt.xlim(0, 250)
plt.ylim(0, 8)
y_major_locator = MultipleLocator(2)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.xlabel("Average run length")
plt.ylabel("Mean detection delay")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(figure_path + "arl_mdd_grassmann.pdf", bbox_inches='tight')

plt.show()