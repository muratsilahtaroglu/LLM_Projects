import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from itertools import product, combinations
from rag_benchmark_analysis.ablation_experiment.__select_best_ML_model import ModelSelector  # Import ModelSelector class

# Generate DataFrame with random scores and helper columns
def generate_rag_data(all_combinations):
    # Parameters for generating combinations
    """
    Generate a DataFrame with random scores and helper columns

    Parameters
    ----------
    banchmark_scores: array_like (default=None)
        The benchmark scores to be used for generating the DataFrame. If not
        provided, random scores will be generated.

    Returns
    -------
    df: pd.DataFrame
        A DataFrame with columns k, coefficient, threshold, helpers, merge_type,
        and score. The score is the benchmark score if provided, otherwise
        a random score.
    helper_types: list
        The list of possible helper types
    """
    # k = [5, 10, 20, 50, 100, 200, 500, 1000]
    # coefficient_values = np.arange(0.6, 1.6, 0.15)
    # coefficient_values = np.insert(coefficient_values, 0, 0)
    # threshold_score_values = np.arange(0.75, 1.05, 0.05)
    helper_types = ["sub_queries", "keywords", "helper_keywords", "keywords_join_list"]
    # coefficient_helpers_values = [
    #     list(combo) for r in range(1, len(helper_types) + 1) for combo in combinations(helper_types, r)
    # ]
    # merge_type_values = ["sum", "proud", "square", "square_sum", "square_sum2", "square_proud", "square_proud2"]

    # # Generate all combinations
    # all_combinations = list(product(k, coefficient_values, threshold_score_values, coefficient_helpers_values, merge_type_values))
    df = pd.DataFrame(all_combinations, columns=["k", "coefficient", "threshold", "helpers", "merge_type"])

    # Simulate random scores
    np.random.seed(42)
    # Step 1: Process helpers and coefficients
    df = process_helpers_and_coefficients(df, helper_types)

    # Step 2: Clean feature names
    df = clean_feature_names(df)
    df = pd.get_dummies(df, columns=["merge_type"])

    return df

# Process helpers and coefficients to create new columns
def process_helpers_and_coefficients(df, helper_types):

    """
    Processes the DataFrame to create new columns for each helper type with their respective coefficients.

    This function applies a transformation to each row of the DataFrame, generating new columns for 
    each specified helper type. If a helper type is present in the 'helpers' column of the row, 
    the coefficient from the 'coefficient' column is assigned to the new column. If the helper type 
    is not present, a default value of 1 is assigned. The original 'coefficient' and 'helpers' 
    columns are dropped after processing.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing combinations of parameters and scores.
    helper_types : list of str
        A list of helper types to create columns for in the DataFrame.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with new columns for each helper type coefficient.
    """

    def helper_processing(row):
        result = {}
        for helper in helper_types:
            if helper in row["helpers"]:
                result[f"{helper}_coefficient"] = row["coefficient"]
            else:
                result[f"{helper}_coefficient"] = 1  # Default value
        return result

    helper_coefficients = df.apply(helper_processing, axis=1)
    helper_coefficients_df = pd.DataFrame(helper_coefficients.tolist())

    # Add new columns to the original DataFrame
    df = pd.concat([df, helper_coefficients_df], axis=1)
    df = df.drop(columns=["coefficient", "helpers"])

    return df

# Clean column names for compatibility
def clean_feature_names(df: pd.DataFrame):
    """Cleans column names by replacing or removing invalid characters."""
    df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)
    return df

# Train and evaluate models
def evaluate_models(X_train, y_train):
    models = {
        #"RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, verbose=0),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,verbose=0),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42,verbose=0),
        "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=0),
     "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,min_samples_split=10,min_samples_leaf=5, random_state=42,verbose=0),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,subsample=0.8,colsample_bytree=0.8, random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=200,num_leaves=25, learning_rate=0.05, max_depth=-1,min_data_in_leaf=10, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=200, learning_rate=0.05, depth=4, l2_leaf_reg=3,random_state=42, verbose=0),
    }

    model_selector = ModelSelector(models=models, k_folds=5)
    results = model_selector.evaluate_models(X_train, y_train)

    return results
def create_results_figure(model_name, importances,columns, version="4"):
    # Plot Feature Importance

    """
    Creates a bar plot of feature importances for a given model.

    Parameters
    ----------
    model_name : str
        The name of the model.
    importances : array-like
        The feature importances as computed by the model.
    columns : list of str
        The column names of the features.
    version : str, default="4"
        The version of the results to save.

    Returns
    -------
    None
    """

    plt.figure(figsize=(10, 6)) 
    plt.bar(range(len(importances)), importances)
    plt.xticks(range(len(importances)), columns, rotation=90)
    plt.ylabel("Feature Importance")
    plt.title(f"{model_name}: Parametrelerin Göreceli Önemi")
    
    # Save the figure
    filename = f"ablation_results/feature_importance_{model_name.lower()}_{version}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved as {filename}.")
    plt.close()

# Main function to execute all steps
def create_ablation_results(ablation_type:str="rag", scores:list=None,all_combinations:list=None, metric_type = None,verion:str="V1_deneme"):
    
    """
    Creates the ablation results for the given benchmark scores.

    Parameters
    ----------
    scores : list, default=None
        The benchmark scores to use for generating the data.

    Returns
    -------
    None
    """
    assert isinstance(all_combinations, list)
    assert len(scores) == len(all_combinations), "len scores and all_combinations must be equal"
    print(f"ablation_type: {ablation_type}\nAblation calculating for Metric: ", metric_type)
    # Step 1: Generate data
    if scores is None:
        scores = np.random.rand(len(df))
    else:
        assert isinstance(scores, list)
        scores = np.array(scores)
    if ablation_type == "rag":
        df = generate_rag_data(all_combinations)
    else:
        raise f"{ablation_type} is not found"
    
    # Step 2: Prepare data for model training
    
    X, y = df, scores 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Evaluate models
    results = evaluate_models(X_train, y_train)

    # Step 4: Print best model results
    print("\nBest Model:")
    print(f"Model Name: {results['best_model_name']}")
    print(f"Mean MSE: {results['mean_mse']:.4f} (±{results['std_mse']:.4f})")

    # Step 5: Train the best model
    best_model = results['best_model']
    best_model.fit(X_train, y_train)
    print("Best model training completed!")
    print("Best Model name: ",results["best_model_name"],"\tMean MSE: ",results["mean_mse"],"\tStd MSE: ",results["std_mse"])
    # Step 6: Feature Importance
    importances = best_model.feature_importances_
    create_results_figure(results['best_model_name'], importances, X.columns,version=f"{metric_type}_{verion}")
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{best_model} - Mean Squared Error: {mse:.4f}\n")


if __name__ == "__main__":
    create_ablation_results(kwargs=None,args=None)
    
