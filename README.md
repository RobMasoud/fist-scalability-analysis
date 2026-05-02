# fist-scalability-analysis
Evaluating the scalability and performance of the FIST algorithm for imputing spatial transcriptomics data.

## Current Progress (Rabeh)

### What I’ve done so far
- Set up the overall project structure (data, preprocessing, models, evaluation, experiments)
- Built a simple baseline model (fills missing values with the mean)
- Created a masking function to simulate missing data
  - Updated it to work efficiently on large tensors (avoids memory issues)
- Implemented evaluation metrics: MAE, MAPE, and R^2
- Downloaded and added the FIST dataset (10x_data)
- Loaded the tensor from the MATLAB file and converted it to a NumPy array
- Integrated ZIFA (Zero-Inflated Factor Analysis) model
- Integrated REMAP (Dual-regularized collaborative filtering) model
- Ran the full pipeline on real data:
  Tensor -> Mask -> Baseline/ZIFA/REMAP -> Evaluation

- Baseline results on HBA1 dataset:
  - MAE: 3.25
  - MAPE: 71.76%
  - R^2: -0.069

- REMAP results on HBA1 dataset (full 5k dataset):
  - MAE: 1.47
  - MAPE: 49.75%  
  - R^2: 0.715

---

### Notes
- The baseline performs poorly, which is expected. (gives us a reference point)
- Masking was updated to avoid memory issues with large datasets
- Metrics are computed only on non-zero values to make results more meaningful

---

### What’s next
- Run the pipeline on more datasets (HBA2, etc.)
- Turn the test script into a reusable experiment script
- Start adding comparison methods (ZIFA, REMAP, GWNMF)
- Compare all methods using the same pipeline
- Use results for analysis and report

---

### Important files
- models/baseline.py -> baseline model
- preprocessing/mask_data.py -> masking function
- evaluation/metrics.py -> metrics (MAE, MAPE, R^2)
- experiments/test_real_data.py -> current test script

---

## How to run

Install dependencies:
pip install -r requirements.txt

Run the experiment:
py -m experiments.test_real_data
