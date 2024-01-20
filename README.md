# ATE
## Domain Generalization with Anti-background Perturbation Consistency and Texture Reduction Ensemble Models for Hepatocyte Nucleus Segmentation

This paper has been accepted for presentation at the ISCAS conference.

Abstract—Hepatocyte nucleus segmentation in histopathology images is vital for diagnostics. However, varying slide background due to cutting and staining poses a great challenge for domain-agnostic segmentation, which limits model generalization. This paper first proposes anti-background perturbation consistency (APC) loss to mitigate the influence of background influence model decisions and controlled background perturbations to enhance generalizability, while still maintaining feature consistency. Since convolutional neural networks (CNNs) often prioritize local texture over global shape which also limit generalization, it then introduces the concept of local self-information into the texture probability (TP) loss to reduce over-focus of CNN on local textures. To avoid model convergence to saddle points during training which yields varying outcomes and unstable performance, it finally concludes a meta-learner which combines results from multiple models to improve stability and better decision-making. At the end, our developed APC coupled with the texture reduction ensemble model (TREM) effectively increases model generalizability across diverse data without fine-tuning model parameters. 
Keywords—segmentation, hepatocyte nuclei segmentation, robustness, domain generalization, Histopathology image

```python
sh KDE_ensemble_contrative_DG_shape_aware_1DS_4gpu_min.sh
```
