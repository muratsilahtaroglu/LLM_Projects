import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

class ModelSelector:
    """
    A class to perform K-Fold Cross-Validation on a set of models and select the best model.
    """

    def __init__(self, models, k_folds=5, scoring=None):
        """
        Initializes the ModelSelector class.

        Args:
            models (dict): A dictionary of models to evaluate.
            k_folds (int): The number of folds for K-Fold Cross-Validation.
            scoring (function): The scoring function for model evaluation.
        """
        self.models = models
        self.k_folds = k_folds
        self.scoring = scoring if scoring else make_scorer(mean_squared_error, greater_is_better=False)

        self.best_model = None
        self.best_model_name = None
        self.best_mean_mse = float('inf')
        self.best_std_mse = float('inf')

    def evaluate_models(self, X, y):
        """
        Evaluates all models using K-Fold Cross-Validation and selects the best model.

        Args:
            X (DataFrame): Feature matrix.
            y (Series): Target vector.

        Returns:
            dict: A dictionary containing the best model and its performance.
        """
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        for model_name, model in self.models.items():
            print(f"Evaluating {model_name} with {self.k_folds}-Fold Cross-Validation...")

            # Perform Cross-Validation
            neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring=self.scoring)
            mse_scores = -neg_mse_scores  # Convert negative MSE to positive
            mean_mse = np.mean(mse_scores)
            std_mse = np.std(mse_scores)

            print(f"{model_name} - Mean MSE: {mean_mse:.4f} (±{std_mse:.4f})")

            # Update the best model
            if (mean_mse < self.best_mean_mse) or (mean_mse == self.best_mean_mse and std_mse < self.best_std_mse):
                self.best_model = model
                self.best_model_name = model_name
                self.best_mean_mse = mean_mse
                self.best_std_mse = std_mse

        return {
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "mean_mse": self.best_mean_mse,
            "std_mse": self.best_std_mse
        }

    def get_best_model(self):
        """
        Returns the best model after evaluation.

        Returns:
            model: The best performing model.
        """
        if self.best_model is None:
            raise ValueError("No model has been evaluated yet. Call evaluate_models() first.")
        return self.best_model
