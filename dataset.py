import pandas as pd

def load_dataset(filepath='fraudTest.csv'):
    data = pd.read_csv(filepath)
    return data

if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset.head())
