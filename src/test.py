import pandas as pd 
from joblib import load

model_path = r"G:\3. Machine Learning-24\Iris Classification-24\data\model\best_model.pkl"
pipeline = load(model_path)

csv_path = r"G:\3. Machine Learning-24\Iris Classification-24\data\cleaned_data\ready.csv"
df_train = pd.read_csv(csv_path)

print(df_train.head())

def predict_iris():
    # Define input data
    input_data = [[8, 7, 6, 4, 'clay']]  # Example input data point
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'soil_type'])
    # Make predictions
    prediction = pipeline.predict(input_df)  # Fix here: call predict on the pipeline, not on model
    print("Iris flower is:", prediction)
      
    return prediction

predict_iris()