# Nov 4 checkup: bibliography

The links Jeong asked for, in the order the deck uses them, with what each one backs. The same list is the `References` slide at the end of the deck copy, every entry clickable. Lecture links are on Canvas and are left for Scott to add.

| # | Entry | Link | Backs on the deck |
|---|---|---|---|
| 1 | Casao, Peña, Sabater, Castillón, Suárez, Montijano, Murillo. *SpectralWaste Dataset: Multimodal Data for Waste Sorting Automation*. IROS 2024. | <https://arxiv.org/abs/2403.18033> | every baseline number (Table IV), instance counts (Table I), the 79.6 label-transfer agreement (Table II), the camera and the 224 bands (Sec III-A), the 256 by 256 resize (Sec IV-A), label transfer (Sec IV-C), the training protocol (Sec V-A) |
| 2 | SpectralWaste segmentation release, Zenodo record 10880544 (CC BY 4.0). | <https://doi.org/10.5281/zenodo.10880544> | the cubes, the masks, the 514 / 167 / 171 split counted on disk |
| 3 | SpectralWaste segmentation mirror on Hugging Face, `ferpb/spectralwaste-segmentation` (same release, plus the RGB frames and RGB-frame labels as WebDataset shards). | <https://huggingface.co/datasets/ferpb/spectralwaste-segmentation> | the RGB panel in the data example; the label-set IoU figure |
| 4 | Official code, `ferpb/spectralwaste-segmentation`. | <https://github.com/ferpb/spectralwaste-segmentation> | batch 12, class-weight exponent 0.12 from RGB-label counts, no pretrained weights loaded, best-val checkpoint |
| 5 | Alonso, Riazuelo, Murillo. *MiniNet: An Efficient Semantic Segmentation ConvNet for Real-Time Robotic Applications*. IEEE Transactions on Robotics, 2020. | <https://doi.org/10.1109/TRO.2020.2974099> | the MiniNet-v2 card (52.8, 0.585M) and the family our CNN control is drawn from |
| 6 | Xie, Wang, Yu, Anandkumar, Alvarez, Luo. *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS 2021. | <https://arxiv.org/abs/2105.15203> | the SegFormer-B0 card (54.3, 4.06M), the paper's transformer baseline |
| 7 | Zhang, Liu, Yang, Peng, Stiefelhagen. *CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers*. 2022. | <https://arxiv.org/abs/2203.04838> | the CMX-B0 card (58.2, RGB plus HSI), context only |
| 8 | Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. | <https://arxiv.org/abs/2010.11929> | patch and position embedding, the Lecture 1 tie-in for the band tokenizer |
| 9 | Hong et al. *SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers*. IEEE TGRS 2022. | <https://arxiv.org/abs/2107.02988> | the precedent for grouping neighbouring bands into one token |
| 10 | Musiat, Ebert, Wasenmüller. *SSFT: A Lightweight Spectral-Spatial Fusion Transformer for Generic Hyperspectral Classification*. 2026. | <https://arxiv.org/abs/2604.15828> | the precedent for a 0.5M spectral-spatial transformer (D = 64, 4 heads); classification, not segmentation |
| 11 | Touvron et al. *Training data-efficient image transformers and distillation through attention* (DeiT). ICML 2021. Steiner et al. *How to train your ViT?* TMLR 2022. | <https://arxiv.org/abs/2012.12877> · <https://arxiv.org/abs/2106.10270> | the from-scratch training recipe: AdamW, warmup and cosine, stochastic depth, augmentation over regularization on small data |
| 12 | Our repository, DATA 255 Group 1. | <https://github.com/scottn66/255-final-project> | loader, metric, attention, tokenizer, CNN control, the training harness, these figures |
| 13 | Lecture 1 (ViT components: patch embedding, position embedding, attention) and Lecture 9 (segmentation, IoU and mIoU), DATA 255, Canvas. | Canvas, Scott adds the links | the lecture tie-ins on the approach figure |

Every arXiv and DOI link was checked in this session or in `docs/RECIPE.md` (Oct 6). The Hugging Face mirror was read directly on Nov 4 (its README, shard listing, and sample layout).
