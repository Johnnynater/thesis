# Automated string cleaning and encoding
## Introduction
This framework was developed during my master thesis on automated string handling in tabular data.
The thesis was conducted under supervision of Joaquin Vanschoren at Eindhoven University of Technology.

## Installing the framework
Users can install the framework like so:

```
# pip
pip install git+https://github.com/ml-tue/automated-string-cleaning

# GitHub clone
git clone https://github.com/ml-tue/automated-string-cleaning
```

When cloning from GitHub, it might be necessary to install relevant packages like so:

```
pip install -r requirements.txt
```

## Usage examples
### Example 1: clean and encode the ... dataset

### Example 2: clean the Wine Reviews dataset
1. Download the Wine Reviews dataset from https://www.kaggle.com/zynicide/wine-reviews.
2. In a Notebook, open the dataset as a pandas DataFrame:
    ```
   import pandas as pd
   X = pd.read_csv(r'<path-to-csv>/<filename>.csv')
   
   # Optional: consider a subsample
   X = X.iloc[:200, :]
    ```
   
3. Run the string cleaner:
    ```
   from auto_string_cleaner import main
   X = main.run(X, encode=False)
    ```
   
4. Display the cleaned DataFrame:
    ```
   display(X)
    ```
