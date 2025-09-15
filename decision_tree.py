import numpy as np
from typing import Self, Tuple, Any
from numpy import ndarray, dtype, float64

def count(y: np.ndarray) -> np.ndarray:
    """
    Calculate relative frequency of each unique label in y

    Parameters:
    y : Array of labels.

    Returns:
    Relative frequency of each label
    """
    labels, counted = np.unique(y, return_counts=True)
    relative_counted = counted / len(y)
    return relative_counted


def gini_index(y: np.ndarray) -> float:
    """
    Compute the Gini impurity for a label array

    Parameters:
    y : Array of labels

    Returns:
    float: Gini impurity score
    """
    if y.size == 0:
        return 0.0

    probs = count(y)
    return float(1.0 - np.sum(probs ** 2))


def entropy(y: np.ndarray) -> float:
    """
    Compute the entropy for a label array

    Parameters:
    y : Array of labels

    Returns:
    float: Entropy score
    """
    if y.size == 0:
        return 0.0

    probs = count(y)
    probs = probs[probs > 0]

    return float(-np.sum(probs * np.log2(probs)))


def split(x: np.ndarray, value: float) -> np.ndarray:
    """
    Create a boolean mask for splitting a feature array at a threshold

    Parameters:
    x (np.ndarray): Feature values
    value (float): Threshold for split

    Returns:
    np.ndarray : Boolean mask
    """
    return x <= value


def most_common(y: np.ndarray) -> int:
    """
    Return the most common class label in y

    Parameters:
    y : Array of labels

    Returns:
    int: Most common label
    """
    labels, counted = np.unique(y, return_counts=True)
    return int(labels[np.argmax(counted)])


class Node:
    """
    A node in the decision tree
    """

    def __init__(
            self,
            feature: int = 0,
            threshold: float = 0.0,
            left: int | Self | None = None,
            right: int | Self | None = None,
            value: int | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf node

        Returns:
        bool: True if the node is a leaf, False otherwise
        """
        return self.value is not None


class DecisionTree:
    """
    Decision tree classifier
    """

    def __init__(
            self,
            max_depth: int | None = None,
            criterion: str = "entropy",
            max_features: str | int | None = None,
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features


    def _impurity(self, y: ndarray) -> float:
        """
        Compute impurity of a label array using the chosen criterion

        Parameters:
        y : Array of labels

        Returns:
        float: Impurity score
        """
        if self.criterion == "gini":
            return gini_index(y)
        else:
            return entropy(y)

    def _ID3(self, X: ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """
        Find the best split for the current dataset

        Parameters:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels

        Returns:
        Tuple: best feature index, threshold, information gain
        """
        samples, features = X.shape
        parent_impurity = self._impurity(y)
        best_feature, best_threshold, best_gain = -1, None, -1.0

        if self.max_features is None:
            n_consider = features
        elif self.max_features == "sqrt":
            n_consider = max(1, int(np.sqrt(features)))
        elif self.max_features == "log2":
            n_consider = max(1, int(np.log2(features)))
        else:
            n_consider = min(features, int(self.max_features))

        feature_subset = np.random.choice(features, size=n_consider, replace=False)

        for x in feature_subset:
            col = X[:, x]
            threshold = np.median(col)
            left_mask = split(col, threshold)
            right_mask = np.logical_not(left_mask)
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            left_impurity = self._impurity(y[left_mask])
            right_impurity = self._impurity(y[right_mask])

            n, n_left, n_right = len(y), left_mask.sum(), right_mask.sum()
            total_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
            reduced_gain = parent_impurity - total_impurity

            if reduced_gain > best_gain:
                best_gain, best_feature, best_threshold = reduced_gain, x, threshold

        return best_feature, best_threshold, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree

        Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        depth (int): Current depth of the tree

        Returns:
        Node
        """
        if len(np.unique(y)) == 1:
            return Node(value=int(y[0]))
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=most_common(y))

        feature, threshold, gain = self._ID3(X, y)
        if feature == -1 or gain <= 0:
            return Node(value=most_common(y))
        col = X[:, feature]
        left_mask = split(col, threshold)
        right_mask = np.logical_not(left_mask)

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the decision tree classifier on the training data

        Parameters:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels

        Returns:
        DecisionTree
        """
        self.root = self._build(X, y, 0)
        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for a set of samples

        Parameters:
        X (np.ndarray): Feature matrix

        Returns:
        np.ndarray: Predicted labels
        """
        predictions = []

        for sample in X:
            node = self.root

            while not node.is_leaf():
                if sample[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            predictions.append(node.value)
        return np.array(predictions)

    def print_tree(self, node : Node = None, depth:int = 0):
        """
        Prints the decision tree in a readable if/else format for debugging and visualization

        Parameters:
        node : Current node
        depth (int): Current depth

        """
        if node is None:
            node = self.root

        indent = " " * depth

        if node.is_leaf():
            print(f"{indent}Predict: {node.value}")
        else:
            print(f"{indent}if feature[{node.feature}] <= {node.threshold:.4f}:")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}else:  # feature[{node.feature}] > {node.threshold:.4f}")
            self.print_tree(node.right, depth + 1)


if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
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

    # Expect the training accuracy to be 1.0 when max_depth=None
    rf = DecisionTree(max_depth=3, criterion="entropy")
    rf.fit(X_train, y_train)
    rf.print_tree()

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
