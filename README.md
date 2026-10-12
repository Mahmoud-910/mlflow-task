# MLflow House Price Prediction

I used MLflow to track and compare three Gradient Boosting experiments using the California Housing dataset.

## Run the project

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py


```

## Results

| Run | Max depth | Learning rate | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| Run 1 | 3 | 0.10 | 0.542215 | 0.371643 | 0.775645 |
| Run 2 | 5 | 0.05 | 0.519779 | 0.353132 | 0.793828 |
| Run 3 | 7 | 0.01 | 0.697287 | 0.531098 | 0.628964 |

## Best model

I selected Run 2 (`max_depth=5`, `learning_rate=0.05`) because it had the lowest validation RMSE: 0.519779.
