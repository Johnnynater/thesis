from src import infer_ptype, infer_stattype, process_stringtype, encode
from src.gbc import heuristics
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


def load_dataset():
    pass


def inference_ptype(df):
    return infer_ptype.infer(df)


def inference_statistical_type(df):
    # result = []
    # for col in data:
    #     result.append(infer_stattype.run_inference(data[col].to_frame()))
    # return result
    return infer_stattype.infer(df)


def inference_heuristics(df):
    return heuristics.run(df)


def handle_outliers():
    pass


def handle_missing(column, type, missing, names):
    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    return imp_freq.fit_transform(column)


def apply_process_unique(df, stringtypes):
    processed_df = pd.DataFrame()
    for col, type in zip(df, stringtypes):
        processed_df = pd.concat([processed_df, process_stringtype.run(df[col].to_frame(), type)], axis=1)
    return processed_df


def apply_encoding(cols, results):
    # Encode the string columns based on the results from the GBC
    for col, val in zip(cols, results):
        cols.loc[:, col] = encode.run(cols[col], val)  # encoded_column
    print(cols.to_string())
    return cols


def output_dataset():
    pass


if __name__ == "__main__":
    # Selftest: create pfsm
    # pfsm = src.selfmade_pfsm.PFSM(r'([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)')
    # pfsm.create_pfsm()

    # Load in the dataset
    # TODO: when we create a callable method we will probably require it to have a data param, so this won't be needed
    load_dataset()
    data = pd.read_csv('datasets/winemag-data-130k-v2.csv')

    # Infer data / string type using ptype
    schema, names = inference_ptype(data)
    # names.append('string')
    datatypes = [col.type for _, col in schema.cols.items()]
    missing_vals = [col.get_na_values() for _, col in schema.cols.items()]
    print(schema.show().to_string())
    # print(missing_vals)
    print(names)

    # TODO: put the for loop inside of the handle_missing method
    for column, type, missing in zip(data, datatypes, missing_vals):
        # Fill in any missing values
        if missing:
            data[column] = handle_missing(data[column].to_frame(), type, missing, names)


    # Take the columns that were inferred as a string (feature)
    # string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in names]]
    unique_string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in names]]
    stringtypes = [datatypes[i] for i in range(len(datatypes)) if datatypes[i] in names]
    string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] == 'string']]

    # Process unique strings etc
    unique_string_cols = apply_process_unique(unique_string_cols, stringtypes)
    print('wassup', unique_string_cols)

    # If cannot infer strings using given PFSMs, infer nominal / ordinal
    # print(inference_statistical_type(string_cols))

    # If cannot infer using given PFSMs, gather features and infer nominal / ordinal using GradientBoostingClassifier
    results_heur = inference_heuristics(string_cols)
    results_gbc = inference_statistical_type(results_heur)
    # 0 = ordinal, 1 = nominal
    print(results_gbc)

    # Encode the string columns based on the results from the GBC
    string_cols = apply_encoding(string_cols, results_gbc)

    # Remove or repair any detected outliers
    # handle_outliers()

    # Output the cleaned dataset
    output_dataset()
