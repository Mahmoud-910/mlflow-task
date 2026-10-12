import csv
import math
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_DIR / "mlflow.db"
RESULTS_PATH = PROJECT_DIR / "results.csv"

CONFIGURATIONS = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]

def main():
    mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH}")
    mlflow.set_experiment("California Housing - Gradient Boosting")

    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    results = []

    for config in CONFIGURATIONS:
        model = GradientBoostingRegressor(
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=42,
        )

        with mlflow.start_run(run_name=config["run_name"]):
            model.fit(X_train, y_train)
            predictions = model.predict(X_validation)

            metrics = {
                "rmse": math.sqrt(mean_squared_error(y_validation, predictions)),
                "mae": mean_absolute_error(y_validation, predictions),
                "r2": r2_score(y_validation, predictions),
            }

            mlflow.log_params({
                "max_depth": config["max_depth"],
                "learning_rate": config["learning_rate"],
                "random_state": 42,
            })
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=X_train.head(5),
                serialization_format="cloudpickle",
            )

            results.append({
                "run": config["run_name"],
                "max_depth": config["max_depth"],
                "learning_rate": config["learning_rate"],
                **metrics,
            })

            print(
                f"{config['run_name']}: "
                f"RMSE={metrics['rmse']:.6f}, "
                f"MAE={metrics['mae']:.6f}, "
                f"R²={metrics['r2']:.6f}"
            )

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    best_run = min(results, key=lambda result: result["rmse"])

    print("\\nComparison:")
    print(f"{'Run':<8} {'Depth':>5} {'LR':>8} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    for result in results:
        print(
            f"{result['run']:<8} "
            f"{result['max_depth']:>5} "
            f"{result['learning_rate']:>8.2f} "
            f"{result['rmse']:>10.6f} "
            f"{result['mae']:>10.6f} "
            f"{result['r2']:>10.6f}"
        )

    print(f"\\nBest model by validation RMSE: {best_run['run']}")
    print(f"Results saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
