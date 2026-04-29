# Project 3 Requirement Notes

## Goal
- Build a 15-class scene image classification system.
- Use BoVW + SPM + SVM pipeline.
- Report test performance with classification report and confusion matrix.

## Dataset Rule
- Data path: `project_3/data`
- Classes: `00` to `14` (15 classes)
- Split per class after numeric filename sort:
  - First 150 images -> training set
  - Remaining images -> test set
- Expected totals:
  - Training: 2250 images
  - Test: 2235 images

## Required Pipeline
1. Extract SIFT features on training images.
2. Build visual vocabulary by KMeans clustering on SIFT descriptors.
3. Build SPM feature with grid levels `1x1`, `2x2`, `4x4`.
4. Concatenate histograms into `21 * num_clusters` feature vector.
5. Train SVM classifier on training features.
6. Apply same feature pipeline on test images and predict.
7. Output:
   - `classification_report.txt`
   - `confusion_matrix.csv`
   - `confusion_matrix.png`

## Submission Constraint
- Code part should be a single `.py` file (`project_3/main.py`).
- Submit design PDF separately.
- Do not include dataset in submission package.

## Environment
- Conda env name: `mv-course`
- Python: `3.10`
- Core packages:
  - `opencv`
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `matplotlib`
  - `pillow`
  - `tqdm`

## Run
```powershell
conda activate mv-course
python .\project_3\main.py --data-dir .\project_3\data --output-dir .\project_3\result
```
