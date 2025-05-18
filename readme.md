## ECG Transformer-based Classification and Anomaly Detection

## Overview

This project implements deep learning models based on Transformer architectures for ECG (Electrocardiogram) signal classification and anomaly detection. The pipeline includes data preprocessing, model training with class-imbalance handling, and evaluation. The MIT-BIH Arrhythmia Database (mitdb) is used as the primary dataset.

## Dataset: MIT-BIH Arrhythmia Database (mitdb)

- **Source:** [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- **Description:**  
  The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects. Each record includes ECG signals and expert-annotated beat labels.
- **Usage in this project:**  
  - Only the first lead is used for each record.
  - Each ECG record is segmented into 1-second windows (typically 360 samples per segment).
  - Each segment is labeled for both anomaly detection (binary: normal vs. abnormal) and multi-class classification (10 beat types).


## Project Structure
```
ECG Detection/
│
├── src/
│   ├── anomaly_detection.py      # Transformer model for binary anomaly detection
│   ├── classification.py         # Transformer model for 10-class beat classification
│   ├── preprocessing.py          # Data loading, segmentation, labeling, and augmentation
│   ├── train.py                  # Training and validation script with weighted loss
│   ├── test.py                   # Model evaluation and metrics
│
├── data/
│   └── mitdb/                    # MIT-BIH Arrhythmia Database files (.dat, .hea, .atr)
│
├── models/                       # Saved model weights
│
├── readme.txt                    # Project documentation
└── requirements.txt              # Python dependencies
```
## Data Preprocessing

- **Segmentation:**  
  Each ECG record is split into 1-second segments.
- **Labeling:**  
  - **Anomaly label:** 0 (normal) or 1 (anomaly), based on beat annotation.
  - **Class label:** Integer from 0 to 9, representing different beat types:
    - 0: Normal beat (N)
    - 1: Left bundle branch block beat (L)
    - 2: Right bundle branch block beat (R)
    - 3: Atrial premature contraction (A)
    - 4: Aberrated atrial premature contraction (a)
    - 5: Premature ventricular contraction (V)
    - 6: Fusion of ventricular and normal beat (F)
    - 7: Nodal (junctional) premature beat (J)
    - 8: Ventricular escape beat (E)
    - 9: Paced beat (/)
- **Augmentation:**  
  Optional Gaussian noise and scaling applied during training.
- **Class Imbalance:**  
  Weighted loss is computed based on class frequencies in the training set.


## Model Architectures

### 1. **ECGClassifier (classification.py)**
- **Purpose:** 10-class ECG beat classification.
- **Architecture:**
  - Input: [batch_size, sequence_length]
  - Linear embedding layer
  - 2-layer Transformer Encoder (d_model=128, nhead=4)
  - Global average pooling
  - Fully connected output layer (10 classes)

### 2. **TransformerModel (anomaly_detection.py)**
- **Purpose:** Binary anomaly detection (normal vs. abnormal).
- **Architecture:**
  - Input: [batch_size, sequence_length]
  - Linear embedding layer
  - 2-layer Transformer Encoder (d_model=128, nhead=4)
  - Global average pooling
  - Fully connected output layer (2 classes)


## Training (train.py)

- **Weighted Loss:**  
  Class weights are computed for both anomaly and classification tasks to address class imbalance.
- **Optimizer:**  
  Adam optimizer is used for both models.
- **Validation:**  
  After each epoch, models are evaluated on the test set. The best models are saved based on validation accuracy.


## Evaluation (test.py)

- **Metrics:**  
  - Accuracy, Recall, F1-Score for both anomaly detection and classification.
  - Detailed classification report for all classes.
- **Handling Class Imbalance:**  
  Weighted loss and metrics ensure fair evaluation even for rare classes.


## How to Run

1. **Prepare the Dataset:**  
   Download the MIT-BIH Arrhythmia Database and place the `.dat` and annotation files in `data/mitdb/`.

2. **Train the Models:**  
   Run `train.py` for both classification and anomaly detection models. Adjust hyperparameters as needed.

3. **Evaluate the Models:**  
   Run `test.py` to evaluate the trained models on the test set. Check the metrics and classification reports.


## References

1. PhysioNet: MIT-BIH Arrhythmia Database - [Link](https://physionet.org/content/mitdb/1.0.0/)
2. Original paper describing the MIT-BIH Arrhythmia Database:
   - Goldberger AL, et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals." Circulation 2000; 101(23): e215-e220.
3. Code for ECG classification using Transformer:
   - [Link to ECG Transformer Code](https://github.com/yourusername/ecg-transformer)
4. Code for anomaly detection using Transformer:
   - [Link to Anomaly Detection Code](https://github.com/yourusername/ecg-anomaly-detection)


## Acknowledgments

- The authors of the MIT-BIH Arrhythmia Database for providing the dataset.
- The developers of PyTorch and other libraries used in this project.

