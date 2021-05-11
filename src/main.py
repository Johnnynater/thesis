from src import infer_ptype, infer_stattype, process_stringtype, encode
from src.gbc import heuristics
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


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
    require_encoding = []
    for col, stringtype in zip(df, stringtypes):
        result, encode = process_stringtype.run(df[col].to_frame(), stringtype)
        processed_df = pd.concat([processed_df, result], axis=1)

        if type(encode) == list:
            require_encoding.extend(encode)
        else:
            require_encoding.append(encode)

    return processed_df, require_encoding


def apply_encoding(cols, results):
    # Encode the string columns based on the results from the GBC
    for col, val in zip(cols, results):
        if val == 2:
            continue
        else:
            cols.loc[:, col] = encode.run(cols[col], val)  # encoded column
    return cols


if __name__ == "__main__":
    # Load in the dataset
    # TODO: when we create a callable method we will probably require it to have a data param, so this won't be needed
    data = pd.read_csv('datasets/gbc_data/diamonds.csv')

    # Infer data / string type using ptype
    schema, names = inference_ptype(data)
    # names.append('string')
    datatypes = [col.type for _, col in schema.cols.items()]
    missing_vals = [col.get_na_values() for _, col in schema.cols.items()]
    print(schema.show().to_string())
    # print(missing_vals)
    print(names)

    # TODO: put the for loop inside of the handle_missing method
    for column, stringtype, missing in zip(data, datatypes, missing_vals):
        # Fill in any missing values
        if missing:
            data[column] = handle_missing(data[column].to_frame(), stringtype, missing, names)

    # Take the columns that were inferred as a string (feature)
    # string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in names]]
    unique_string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in names]]
    stringtypes = [datatypes[i] for i in range(len(datatypes)) if datatypes[i] in names]
    string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] == 'string']]
    other_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] != 'string' and datatypes[i] not in names]]

    # Process unique strings etc
    # TODO: return whether columns need to be encoded + which encoding they need
    unique_string_cols, require_encoding = apply_process_unique(unique_string_cols, stringtypes)
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
    unique_string_cols = apply_encoding(unique_string_cols, require_encoding)

    result = pd.concat([other_cols, string_cols, unique_string_cols], axis=1)
    print(string_cols['cut'])
    # Remove or repair any detected outliers
    # handle_outliers()