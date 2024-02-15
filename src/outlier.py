from src.load import load_data
from src.preprocess import preprocess_data

import numpy as np 

def outlier_handling(data):
      
      columns_to_plot = ['sepal_length','sepal_width','petal_length','petal_width']
      
      for column in columns_to_plot:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3-Q1
      
            lower_bound = Q1-IQR*1.5
            upper_bound = Q3+IQR*1.5
            data[column] = data[column].clip(lower = lower_bound, upper = upper_bound)

      return data

if __name__=="__main__":
      ready_data = outlier_handling(preprocess_data(load_data()))
      
      print(ready_data.head())
      
path = r"G:\3. Machine Learning-24\Iris Classification-24\data\cleaned_data\ready.csv"
ready_data.to_csv(path, index=False)