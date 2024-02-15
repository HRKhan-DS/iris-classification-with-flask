
from src.load import load_data

import pandas as pd 

def preprocess_data(data):
      
     data.dropna(inplace=True)
     
     # Drop specified columns
     data.drop(columns=['sepal_area','petal_area','sepal_aspect_ratio','petal_aspect_ratio','sepal_to_petal_length_ratio',
                    'sepal_to_petal_width_ratio','sepal_petal_length_diff','sepal_petal_width_diff','petal_curvature_mm',
                    'petal_texture_trichomes_per_mm2', 'leaf_area_cm2',	'sepal_area_sqrt','petal_area_sqrt','area_ratios'], inplace=True)
     
     
     # Saving the new DataFrame to a CSV file
     data.to_csv(r"G:\3. Machine Learning-24\Iris Classification-24\data\cleaned_data\iris_cleaned.csv", index=False)
     
     return data

if __name__=="__main__":
      preprocessed_data = preprocess_data(load_data())
      
      print(preprocessed_data.head())