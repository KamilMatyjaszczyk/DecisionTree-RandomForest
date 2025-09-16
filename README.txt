README
This project contains an implementation of a Decision Tree and a Random Forest
classifier, as well as experiments for evaluating them.

FILES:
 decision_tree.py      : Implementation of the DecisionTree class
 random_forest.py      : Implementation of the RandomForest class (using DecisionTree)
 run_experiments.ipynb : Jupyter notebook to run experiments, cross-validation,
                          model selection, comparison with scikit-learn, and feature importance.

HOW TO RUN:
1. Open and run run_experiments.ipynb in Jupyter Notebook.
    This notebook loads the dataset (letters.csv), performs train/test split,
     runs k-fold cross-validation, selects best hyperparameters, trains the final models,
     compares them with scikit-learn implementations, and plots feature importance.

2. If you just want to test the models independently:
    Run `python decision_tree.py` from the command line to test the Decision Tree on a synthetic dataset.
    Run `python random_forest.py` from the command line to test the Random Forest on a synthetic dataset.

REPRODUCIBILITY:
- Random seeds are fixed (42) in all code files.
- All reported numbers can be reproduced by re-running run_experiments.ipynb.

DEPENDENCIES:
 Python
 NumPy
 scikit-learn
 Matplotlib

NOTE:
 The dataset file `letters.csv` must be placed in the same directory as the notebook
 for the experiments to run correctly.