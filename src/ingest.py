import os
import pandas as pd

def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(file_path)
    
    print("Shape (rows, columns):", df.shape)


    print("\nFirst 5 rows:")
    print(df.head())


    print("\nColumn Data Types:")
    print(df.dtypes)


    missing_totalcharges = df['TotalCharges'].isna().sum()
    blank_totalcharges = (df['TotalCharges'].str.strip() == '').sum()

    print("\nMissing values in TotalCharges:", missing_totalcharges)
    print("Blank values in TotalCharges:", blank_totalcharges)
    print("Total missing + blank:", missing_totalcharges + blank_totalcharges)
    
    return df

if __name__ == "__main__":
    df = load_data()