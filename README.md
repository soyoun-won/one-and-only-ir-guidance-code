<div align="center">
  <h1> Intermediate Text Representation Guided Text-to-Image Generation for Enhancing One-and-Only Alignment </h1>
</div>

<div align="center">

[![ECCV](https://img.shields.io/badge/ECCV-2026-yellow.svg)](https://eccv.ecva.net/virtual/2026/poster/5537)
[![ArXiv](https://img.shields.io/badge/Paper-arXiv.2606.30262-B31B1B.svg)](https://arxiv.org/abs/2606.30262)
[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-blue.svg)](https://soyoun-won.github.io/one-and-only-ir-guidance/)
</div>


## IR-Guidance

```
python run.py \
    --dataroot dataroot \
    --dataset OAO_attackbench \
    --n_images 1 \
    --target_hidden_layer 1 \
    --where_to_intervene embed \
    --process injection_0.2
```

## Citation
```bibtex
@article{won2026intermediate,
  title={Intermediate Text Representation Guided Text-to-Image Generation for Enhancing One-and-Only Alignment},
  author={Won, Soyoun and Parast, Aryan Yazdan and Azam, Basim and Honorio, Jean and Akhtar, Naveed},
  journal={arXiv preprint arXiv:2606.30262},
  year={2026}
}
```

