# churn_training_pipeline.py
# This script trains models from cleaned data and saves pipelines.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
df = pd.read_csv('data/employee_churn_clean.csv')
target_candidates = [c for c in df.columns if c.strip().lower()=='churn' or 'churn' in c.lower()]
if not target_candidates:
    raise SystemExit('No churn column found')
target = target_candidates[0]
df[target] = df[target].astype(str).str.strip().str.lower().map({'yes':1,'y':1,'true':1,'t':1,'1':1,'no':0,'n':0,'false':0,'f':0,'0':0}).fillna(0).astype(int)
X = df.drop(columns=[target])
y = df[target]
id_like = [c for c in X.columns if 'id' in c.lower() or c.lower().startswith('emp')]
X = X.drop(columns=id_like, errors='ignore')
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
cat_cols = [c for c in cat_cols if X[c].nunique(dropna=True)<=50]
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))])
preprocessor = ColumnTransformer([('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)], remainder='drop')
models = {'logistic': LogisticRegression(max_iter=1000), 'rf': RandomForestClassifier(n_estimators=100, random_state=42), 'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)}
for name, clf in models.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    pipe.fit(X, y)
    joblib.dump(pipe, f'models/{name}_pipeline.joblib')
print('Models trained and saved to models/')