import pandas as pd, numpy as np
df = pd.read_csv('data/employee_churn_dataset.csv')
df.drop_duplicates(inplace=True)
id_like = [c for c in df.columns if ('id' in c.lower()) or c.lower().startswith('emp')]
df.drop(columns=id_like, errors='ignore', inplace=True)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].replace('', 'missing').fillna('missing')
df.to_csv('data/employee_churn_clean.csv', index=False)
print('Saved cleaned dataset to data/employee_churn_clean.csv')
