# macsima-qc

**Quality control pipeline for MACSima / MACSiQView cell segmentation**

Python package for evaluating and comparing cell segmentation results from the MACSima cyclic immunofluorescence platform, using manual ROI references from Fiji/ImageJ.

---

## Installation

```bash
pip install macsima-qc
```

Or from source (development mode) :

```bash
git clone https://github.com/mathisbouvet/MACSima_Advanced-Spatial-Omics-Pipeline
cd MACSima_Advanced-Spatial-Omics-Pipeline
pip install -e .
```

---

## Workflow

```
Fiji ROIs (.zip) ──► generate_masks() ──► MACSiQView segmentations (.csv)
                                                    │
                                          run_comparison()  ← KS distances + KDE
                                                    │
                                          run_qc_pipeline() ← Isolation Forest + Mann-Whitney
                                                    │
                                          macsiq_param_suggestions.csv
```

---

## Quick start

### 1. Generate masks from Fiji ROIs

```python
from macsima_qc import load_image, load_rois_from_zip, generate_masks

image   = load_image("C2.tif")
rois    = load_rois_from_zip("RoiSetC2.zip")
masks   = generate_masks(image, rois, output_dir="masks/")
# → mask_1.tif, mask_2.tif, mask_3.tif, mask_4.tif
```

### 2. Compare segmentations (KS test)

```python
from macsima_qc import run_comparison

distances = run_comparison(
    manual_path  = "Segmentation_manuelle.csv",
    auto_paths   = ["Mask_3_Single_Cell.csv", "Mask_4_Import_mask.csv"],
    auto_names   = ["Mask 3 - Single Cell", "Mask 4 - Import Mask"],
    output_dir   = "figures/",
)
```

### 3. QC with Isolation Forest

```python
from macsima_qc import run_qc_pipeline

df_annotated, df_suggestions = run_qc_pipeline(
    ref_path    = "Segmentation_ref.csv",
    test_path   = "Segmentation_EH3524.csv",
    contamination = 0.10,
    output_dir  = "figures/",
)
# → segmentation_test_annotated.csv
# → macsiq_param_suggestions.csv
```

---

## Outputs

| File | Description |
|------|-------------|
| `mask_1–4.tif` | ROI masks for MACSiQView import |
| `figures/ks_average_distances.png` | KS distance barplot per segmentation |
| `figures/kde_comparative_distributions.png` | KDE distribution comparison |
| `figures/contamination_sensitivity.png` | Isolation Forest sensitivity curve |
| `figures/anomaly_scores_distribution.png` | Anomaly score histogram |
| `segmentation_test_annotated.csv` | Cells annotated OK/KO + anomaly score |
| `macsiq_param_suggestions.csv` | MACSiQView parameter adjustment suggestions |

---

## Requirements

- Python ≥ 3.9
- numpy, opencv-python, matplotlib, scikit-image, read-roi
- pandas, scikit-learn, scipy, seaborn

---

## Citation

If you use this package in a publication, please cite:

> Bouvet M. (2024). *Cartographie intégrée de la région lombo-sacrée embryonnaire humaine par imagerie 3D et immunofluorescence multiplexée*. Institut de la Vision, Paris.

---

## License

MIT
