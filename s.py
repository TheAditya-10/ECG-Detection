import wfdb
from wfdb import processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the record name
record_name = "100"  # Example record (can be changed)

# Load the ECG signal
record = wfdb.rdrecord(f"mitdb/{record_name}", sampfrom=0, sampto=10000)  # Load 10,000 samples
annotation = wfdb.rdann(f"mitdb/{record_name}", "atr")

# Extract signal data
ecg_signal = record.p_signal[:, 0]  # First lead (MLII)
fs = record.fs  # Sampling frequency (360 Hz)

# Print basic info
print(f"Record Name: {record_name}")
print(f"Sampling Frequency: {fs} Hz")
print(f"ECG Shape: {ecg_signal.shape}")

# Display metadata
record.__dict__

