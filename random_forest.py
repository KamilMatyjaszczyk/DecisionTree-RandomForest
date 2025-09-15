import numpy as np
from decision_tree import DecisionTree, most_common


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.trees = []

    def _bootstrap(self, X: np.ndarray, y: np.ndarray):
        """Take a bootstrap sample and put it back, from dataset"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train n_estimators decision trees on bootstrap samples"""
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap(X, y)
            tree = DecisionTree(
                max_depth=self.max_depth,
                criterion=self.criterion,
                max_features=self.max_features,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make prediction by aggregating choices from all trees"""
        tree_prediction = np.array([tree.predict(X) for tree in self.trees])
        final_prediction = [most_common(col) for col in tree_prediction.T]
        return np.array(final_prediction)

if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 42

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    rf = RandomForest(
        n_estimators=20, max_depth=5, criterion="entropy", max_features="sqrt"
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
