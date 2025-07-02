import pandas as pd

try:
    df = pd.read_csv('data.csv')
    print(" File loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print(" ERROR: data.csv file not found in this folder.")
