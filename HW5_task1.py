from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import numpy as np
from sklearn.linear_model import LinearRegression


class CreationalPatternName:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

    def get_subsample(self, df_share):
        X = X_train.copy()
        y = y_train.copy()
        X, y = shuffle(X, y, random_state=42)
        take = int(len(y) * df_share / 100.0)
        return (X_train[:take, :], y_train[:take])

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X, y = shuffle(X,y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    pattern_item = CreationalPatternName(X_train, y_train)
    for df_share in range(10, 101, 10):
        curr_X_train, curr_y_train = pattern_item.get_subsample(df_share)

        lr = LinearRegression().fit(curr_X_train, curr_y_train)
        print(f"score: {lr.score(X_test, y_test):0.2f} items: {len(curr_X_train)}")
