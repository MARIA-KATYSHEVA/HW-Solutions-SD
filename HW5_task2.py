from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class FacadeClass:
    def __init__(self, x_train, y_train) -> None:
        self.X_train = x_train
        self.Y_train = y_train

        assert (len(self.X_train) == len(self.Y_train))

        self.predictors = [
            LinearRegression(),
            SVC(gamma='auto', random_state=42),
            RandomForestClassifier(max_depth=5, random_state=42),
            KNeighborsClassifier(7)
        ]
    def fit(self):
         for predictor in self.predictors:
             predictor.fit(self.X_train, self.Y_train)

    def predict(self, x):
        y = []
        for predictor in self.predictors:
            y.append(np.round(np.array(predictor.predict(x))))

        y_best, _ = np.unique(y, axis=0)
        return y_best

if __name__ == "__main__":
     data = load_iris()
     X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42, shuffle=True)
     Estim = FacadeClass(X_train, Y_train)
     Estim.fit()
     preds = Estim.predict(X_test)
     print(f"Accuracy {100*sum(preds == Y_test)/len(Y_test):0.1f}%")