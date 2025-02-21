# Riemannian change point Detection on Manifolds with Robust Centroid Estimation

In this repository you can find the code to reproduce the results of the paper "Riemannian change point Detection on Manifolds with Robust Centroid Estimation".

Steps:

1. Run main_spd.py to plot Fig. 1 (left);

2. Run main_grassmann.py to plot Fig. 1 (right);

3. Download the <a href="https://figshare.com/articles/dataset/TIMIT_zip/5802597">TIMIT</a> and <a href="https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/">QUT_NOISE_STREET</a> database, convert WAV files in QUT-NOISE <a href="https://stackoverflow.com/questions/5120555/how-can-i-convert-a-wav-from-stereo-to-mono-in-python">from stereo to mono</a>, and run main_vad.py to plot Fig. 2;

For any questions, feel free to email us at dr.xiuheng.wang@gmail.com.

If this code is helpful for you, please cite our paper as follows:

    @inproceedings{wang2025riemannian,
      title={Riemannian change point Detection on Manifolds with Robust Centroid Estimation},
      author={Wang, Xiuheng and Borsoi, Ricardo Augusto, Arnuad Breloy and Richard, C{\'e}dric},
      booktitle={Submitted},
      year={2025}
    }

This experimental code is mainly built on the following repository: <a href="https://github.com/xiuheng-wang/CPD_manifold_release">CPD_manifold_release</a>.

    @inproceedings{wang2024nonparametric,
      title={Non-parametric Online Change Point Detection on Riemannian Manifolds},
      author={Wang, Xiuheng and Borsoi, Ricardo Augusto and Richard, C{\'e}dric},
      booktitle={International Conference on Machine Learning (ICML)},
      year={2024},
      organization={PMLR}
    }

Note that the copyright of the pymanopt toolbox is reserved by https://pymanopt.org/.

**Requirements**
```
pymanopt==2.0.1
numpy==1.22.4
matplotlib==3.4.3
tqdm==4.62.3
scipy==1.7.1
sphfile==1.0.3
torch==1.10.0
```
