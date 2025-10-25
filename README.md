# Employee Churn Prediction

This repository contains a simple churn prediction project built for demonstration.

## Structure
- `data/` - raw and cleaned datasets
- `notebooks/` - analysis notebook (code + links to outputs)
- `models/` - saved trained pipelines (.joblib)
- `outputs/` - evaluation CSVs and charts
- `scripts/` - scripts to clean data and train models

## How to run
1. Create a virtual environment and install requirements:
```
pip install -r requirements.txt
```
2. Clean the raw data:
```
python scripts/data_cleaning_employee_churn.py
```
3. Train models:
```
python scripts/churn_training_pipeline.py
```

Charts from EDA and model evaluation are in `outputs/charts/`.

