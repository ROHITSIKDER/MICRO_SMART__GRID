import pandas as pd
import numpy as np

def create_sequences(data, sequence_length=7):
    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length][2])

    return np.array(X), np.array(y)


def main():
    df = pd.read_csv("Data/cleaned_data.csv")
    df = df.drop(columns=["DATE"])

    data = df.values

    X, y = create_sequences(data, sequence_length=7)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    np.save("model/X.npy", X)
    np.save("model/y.npy", y)


if __name__ == "__main__":
    main()