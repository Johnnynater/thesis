from src import infer_ptype, infer_stattype, process_stringtype, encode, handle_missing, handle_outliers
from src.gbc import heuristics
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


def handle_outlier_vals(df, dts, outliers):
    return handle_outliers.run(df, dts, outliers)


def handle_missing_vals(df, datatypes, missing_vals, names):
    any_missing = False
    for col, missing in zip(df, missing_vals):
        if missing:
            any_missing = True
            df[col].fillna(value=np.nan, inplace=True)
            df[col].replace(to_replace=missing, value=np.nan, inplace=True)
    if any_missing:
        df = handle_missing.run(df, datatypes, names)
    return df


def apply_process_unique(df, stringtypes):
    processed_df = pd.DataFrame()
    require_encoding = []
    for col, stringtype in zip(df, stringtypes):
        result, encode = process_stringtype.run(df[col].to_frame(), stringtype)
        # TODO: check if df.reset_index(drop=True, inplace=True) is necessary
        processed_df = pd.concat([processed_df, result], axis=1)

        if type(encode) == list:
            require_encoding.extend(encode)
        else:
            require_encoding.append(encode)

    return processed_df, require_encoding


def apply_encoding(df, results):
    # Encode the string columns based on the results from the GBC
    for col, val in zip(df, results):
        if val != 2:
            #df.loc[:, col] = encode.run(df[col], val)  # encoded column
            df[col] = df[col].map(encode.run(df[col], val))
    return df


if __name__ == "__main__":
    # Load in the dataset
    # TODO: when we create a callable method we will probably require it to have a data param, so this won't be needed
    data = pd.read_csv('datasets\gbc_data\diamonds.csv')  # winemag-data-130k-v2.csv
    #data = data.iloc[:1000, :]
    # Infer data / string type using ptype
    schema, names = inference_ptype(data)
    # names.append('string')
    datatypes = [col.type for _, col in schema.cols.items()]
    missing_vals = [col.get_na_values() for _, col in schema.cols.items()]
    outlier_vals = [col.get_an_values() for _, col in schema.cols.items()]
    print(schema.show().to_string())
    print(missing_vals)
    print(names)

    # Impute missing values
    data = handle_missing_vals(data, datatypes, missing_vals, names)

    # Take the columns that were inferred as a string (feature)
    # string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in names]]
    unique_string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in names]]
    string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] == 'string']]
    other_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] != 'string' and datatypes[i] not in names]]

    # Make a list of string types for each column
    unique_string_dts = [datatypes[i] for i in range(len(datatypes)) if datatypes[i] in names]
    string_dts = [datatypes[i] for i in range(len(datatypes)) if datatypes[i] == 'string']

    # Make a list of outlier values for each column
    unique_string_out = [outlier_vals[i] for i in range(len(datatypes)) if datatypes[i] in names]
    string_out = [outlier_vals[i] for i in range(len(datatypes)) if datatypes[i] == 'string']

    # Handle typos in string columns
    unique_string_cols = handle_outlier_vals(unique_string_cols, unique_string_dts, unique_string_out)
    string_cols = handle_outlier_vals(string_cols, string_dts, string_out)

    # Process unique strings etc
    unique_string_cols, require_encoding = apply_process_unique(unique_string_cols, unique_string_dts)
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
    # Remove or repair any detected outliers
    # handle_outliers()
    print(result.iloc[:10, :].to_string())
