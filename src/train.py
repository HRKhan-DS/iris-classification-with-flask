import pandas as pd 
import numpy as np 

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from joblib import dump, load

path = r"G:\3. Machine Learning-24\Iris Classification-24\data\cleaned_data\ready.csv"
df = pd.read_csv(path)

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

cat_features = ['soil_type']
num_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

preprocessor = ColumnTransformer(transformers=[
      ('cat', OrdinalEncoder(), cat_features),
      ('num', StandardScaler(), num_features)
])

lr_pipeline = Pipeline([
      ('preprocessor', preprocessor),
      ('classifier', LogisticRegression())
])

lr_pipeline.fit(X_train, y_train)

y_pred_test = lr_pipeline.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
print(accuracy_test)

cm = confusion_matrix(y_test, y_pred_test)
print(cm)

classification_report_lr = classification_report(y_test, y_pred_test)
print(classification_report_lr)

# Save the trained model to a file
model_path = r"G:\3. Machine Learning-24\Iris Classification-24\data\model\best_model.pkl"
dump(lr_pipeline, model_path)
print(f"Model saved to {model_path}")