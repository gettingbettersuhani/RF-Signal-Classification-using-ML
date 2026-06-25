import pickle
from pathlib import Path

DATASET_PATH = r"C:\Users\HP\Desktop\RF Signal Processing\RF-Signal-Classification-using-ML\data\raw\RML2016.10a_dict.pkl"

with open(DATASET_PATH, "rb") as file:
    dataset = pickle.load(file, encoding="latin1")

print("Dataset loaded successfully!")