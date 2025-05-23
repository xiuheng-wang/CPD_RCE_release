# coding: utf-8
# Script for performing change point detection in voice activity detection
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
from utils.riemannian_cpd import riemannian_cpd_spd, riemannian_cpd_grassmann, cpd_rkm_spd, cpd_rkm_grassmann
from utils.functions import import_vad_data

# parameter settings
lambda_0_spd = 1e-2
lambda_1_spd = 2e-2
lambda_0_sub = 1e-2
lambda_1_sub = 2e-2
lambda_rkm_spd = 1e-1
lambda_rkm_sub = 1e-1
Huber_parameter_spd = 1e0
Huber_parameter_sub = 5e-2

# paths of data and figures
root_path = "../data/"
figure_path = './figures/'
if not os.path.exists(figure_path):
    makedir(figure_path)

# experiment setups
nb_change = 1e3
length_noise = 15
length_speech = 4
SNR = 0.5  # 0: only noise, 1: only speech
nperseg = 128*2
sample_factor = 8
no_show = 1
print("SNR:", 10*np.log10(SNR))
X, X_full = import_vad_data(root_path, nb_change, length_noise, length_speech, SNR, nperseg, sample_factor, no_show)
window_length = 32

# define manifold
N = np.shape(X)[-1] # dimensionality of SPD
manifold_cov = pymanopt.manifolds.positive_definite.SymmetricPositiveDefinite(N)
P = 1
manifold_sub = pymanopt.manifolds.grassmann.Grassmann(N, P)

# compute covariance matrices and subspaces
print("Compute features:")
X_cov = []
X_sub = []
for x in tqdm(X):
    i = window_length
    x_cov = []
    x_sub = []
    while i <= np.shape(x)[0]:
        samples = x[i-window_length: i]
        covariance = np.cov(samples.T)
        x_cov.append(covariance)
        samples -= samples.mean(axis=0)
        subspace = np.linalg.svd(samples / np.sqrt(N*window_length))[2][:P, :].T
        x_sub.append(subspace)
        i += 1
    X_cov.append(x_cov)
    X_sub.append(x_sub)

print("Detect change points:")
stat_spd_all = []
stat_sub_all = []
stat_spd_all_rkm = []
stat_sub_all_rkm = []
d = np.size(X_full[0][0])
for index in tqdm(range(int(nb_change))):
    # baselines
    stat_spd_all.append(riemannian_cpd_spd(manifold_cov, X_cov[index], lambda_0_spd, lambda_1_spd))
    stat_sub_all.append(riemannian_cpd_grassmann(manifold_sub, X_sub[index], lambda_0_sub, lambda_1_sub))
    stat_spd_all_rkm.append(cpd_rkm_spd(manifold_cov, X_cov[index], lambda_rkm_spd, Huber_parameter_spd))
    stat_sub_all_rkm.append(cpd_rkm_grassmann(manifold_sub, X_sub[index], lambda_rkm_sub, Huber_parameter_sub))

# gather all test statistics
stats_spd = []
stats_spd.append(stat_spd_all)
stats_spd.append(stat_spd_all_rkm)
stats_sub = []
stats_sub.append(stat_sub_all)
stats_sub.append(stat_sub_all_rkm)

# set names and colors
names = ["Baseline", "Proposed"]
colors = ["#82B0D2", "#FA7F6F"] # ["#BEB8DC", "#82B0D2", "#8ECFC9", "#FFBE7A", "#FA7F6F"]

# draw figures
T = np.shape(X)[1]
Tc = int(T * (length_noise - length_speech) / length_noise) - window_length + 1
T -=  window_length - 1
start_point = 300
N_th = 1000

fig = plt.figure(figsize = (6, 4), dpi = 120)
for index in range(len(names)):
    ax = fig.add_subplot(len(names), 1, index+1)
    avg = np.mean(stats_spd[index], axis = 0)
    std = np.std(stats_spd[index], axis = 0)
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
plt.savefig(figure_path + "vad_spd.pdf", bbox_inches='tight')

fig = plt.figure(figsize = (3.2, 3.0), dpi = 150)
for index in range(len(names)):
    arl, mdd = comp_arl_mdd(stats_spd[index], Tc, N_th, start_point)
    if index == 0:
        plt.plot(arl, mdd, color=colors[index], label=names[index], linestyle='dashed')
    else: 
        plt.plot(arl, mdd, color=colors[index], label=names[index])
plt.xlim(0, 1000)
plt.ylim(0, 40)
y_major_locator = MultipleLocator(10)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.xlabel("Average run length")
plt.ylabel("Mean detection delay")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(figure_path + "arl_mdd_spd_vad.pdf", bbox_inches='tight')

fig = plt.figure(figsize = (6, 4), dpi = 120)
for index in range(len(names)):
    ax = fig.add_subplot(len(names), 1, index+1)
    avg = np.mean(stats_sub[index], axis = 0)
    std = np.std(stats_sub[index], axis = 0)
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
plt.savefig(figure_path + "vad_grassmann.pdf", bbox_inches='tight')

fig = plt.figure(figsize = (3.2, 3.0), dpi = 150)
for index in range(len(names)):
    arl, mdd = comp_arl_mdd(stats_sub[index], Tc, N_th, start_point)
    if index == 0:
        plt.plot(arl, mdd, color=colors[index], label=names[index], linestyle='dashed')
    else: 
        plt.plot(arl, mdd, color=colors[index], label=names[index])
plt.xlim(0, 1000)
plt.ylim(0, 60)
y_major_locator = MultipleLocator(15)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.xlabel("Average run length")
plt.ylabel("Mean detection delay")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(figure_path + "arl_mdd_grassmann_vad.pdf", bbox_inches='tight')

plt.show()
