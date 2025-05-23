# Riemannian Change Point Detection on Manifolds with Robust Centroid Estimation

In this repository you can find the code to reproduce the results of the paper "Riemannian Change Point Detection on Manifolds with Robust Centroid Estimation".

Steps:

1. Run main_spd.py to plot Fig. 1 (left);

2. Run main_grassmann.py to plot Fig. 1 (right);

3. Download the <a href="https://figshare.com/articles/dataset/TIMIT_zip/5802597">TIMIT</a> and <a href="https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/">QUT_NOISE_STREET</a> database, convert WAV files in QUT-NOISE <a href="https://stackoverflow.com/questions/5120555/how-can-i-convert-a-wav-from-stereo-to-mono-in-python">from stereo to mono</a>.

4. Run main_vad.py to plot Fig. 2;

For any questions, feel free to email us at dr.xiuheng.wang@gmail.com.

If this code is helpful for you, please cite our paper as follows:

    @inproceedings{wang2025riemannian,
      title={Riemannian Change Point Detection on Manifolds with Robust Centroid Estimation},
      author={Wang, Xiuheng and Borsoi, Ricardo Augusto, Arnuad Breloy and Richard, C{\'e}dric},
      booktitle={2025 33st European Signal Processing Conference (EUSIPCO)},
      year={2025},
      organization={IEEE}
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