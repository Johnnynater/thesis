from src import infer_ptype, infer_stattype, encode
from src.gbc import heuristics
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


def load_dataset():
    pass


def inference_ptype(data):
    return infer_ptype.infer(data)


def inference_statistical_type(data):
    # result = []
    # for col in data:
    #     result.append(infer_stattype.run_inference(data[col].to_frame()))
    # return result
    return infer_stattype.infer(data)


def inference_heuristics(data):
    return heuristics.run(data)


def handle_outliers():
    pass


def handle_missing(column, type, missing, names):
    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    return imp_freq.fit_transform(column)


def apply_encoding():
    return


def output_dataset():
    pass


if __name__ == "__main__":
    # Selftest: create pfsm
    # pfsm = src.selfmade_pfsm.PFSM(r'([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)')
    # pfsm.create_pfsm()

    # Load in the dataset
    # TODO: when we create a callable method we will probably require it to have a data param, so this won't be needed
    load_dataset()
    data = pd.read_csv('datasets/fifa.csv')

    # Infer data type
    schema, names = inference_ptype(data)
    names.append('string')
    datatypes = [col.type for _, col in schema.cols.items()]
    missing_vals = [col.get_na_values() for _, col in schema.cols.items()]
    print(schema.show().to_string())
    print(missing_vals)
    print(names)

    for column, type, missing in zip(data, datatypes, missing_vals):
        # Fill in any missing values
        if missing:
            data[column] = handle_missing(data[column].to_frame(), type, missing, names)
    # Take the columns that were inferred as a string (feature)
    string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in names]]
    print(string_cols)



    # If cannot infer using given PFSMs, infer nominal / ordinal
    # print(inference_statistical_type(string_cols))

    # If cannot infer using given PFSMs, gather features and infer nominal / ordinal using GradientBoostingClassifier
    results_heur = inference_heuristics(string_cols)
    results_gbc = inference_statistical_type(results_heur)
    # 0 = ordinal, 1 = nominal
    print(results_gbc)


    # Encode the string columns based on the results from the GBC
    for column, pred in zip(string_cols, results_gbc):
        # encoded_column = encode.run(string_cols[column], pred)
        # encoded_column = [encoded_column[i] for i in range(len(encoded_column))]
        string_cols.loc[:, column] = encode.run(string_cols[column], pred) # encoded_column
    print(string_cols.to_string())




    # Remove or repair any detected outliers
    # handle_outliers()

    # Encode (if desired)
    # apply_encoding()

    # Output the cleaned dataset
    output_dataset()
