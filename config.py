# config.py
import pandas as pd

AVAILABLE_MODELS = pd.read_csv("conversational_models_sorted.csv")["model"].dropna().unique().tolist()