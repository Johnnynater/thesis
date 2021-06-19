from src import infer_ptype, infer_stattype, process_stringtype, encode_data, handle_missing, handle_outliers
from src.gbc import heuristics
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


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
        result, encoding = process_stringtype.run(df[col].to_frame(), stringtype)
        # TODO: check if df.reset_index(drop=True, inplace=True) is necessary
        df.reset_index(drop=True, inplace=True)
        processed_df = pd.concat([processed_df, result], axis=1)

        if type(encoding) == list:
            require_encoding.extend(encoding)
        else:
            require_encoding.append(encoding)

    return processed_df, require_encoding


def apply_encoding(df, results, dense):
    # Encode the string columns based on the results from the GBC
    for col, val in zip(df, results):
        if val != 2 and dense or val == 0 and not dense:
            #df.loc[:, col] = encode.run(df[col], val)  # encoded column
            df[col] = df[col].map(encode_data.run(df, df[col], val, dense))
        elif val == 1 and not dense:
            df = encode_data.run(df, df[col], val, dense)
    return df


def run(data, encode=True, dense_encoding=True, verbose=True):
    # Infer data / string type using ptype
    print('> Inferring data types and string features...')
    schema, names = inference_ptype(data)

    # Store obtained data in separate variables
    datatypes = [col.type for _, col in schema.cols.items()]
    missing_vals = [col.get_na_values() for _, col in schema.cols.items()]
    outlier_vals = [col.get_an_values() for _, col in schema.cols.items()]

    # Impute missing values
    print('> Checking and handling any missing values...')
    data = handle_missing_vals(data, datatypes, missing_vals, names)

    # Take the columns that were inferred as a string or a string feature
    unique_string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in names]]
    string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] == 'string']]
    other_cols = data.iloc[
                 :,
                 [i for i in range(len(datatypes)) if datatypes[i] != 'string' and datatypes[i] not in names]
                 ]

    # Make a list of string types for each column
    unique_string_dts = [datatypes[i] for i in range(len(datatypes)) if datatypes[i] in names]
    string_dts = [datatypes[i] for i in range(len(datatypes)) if datatypes[i] == 'string']

    # Make a list of outlier values for each column
    unique_string_out = [outlier_vals[i] for i in range(len(datatypes)) if datatypes[i] in names]
    string_out = [outlier_vals[i] for i in range(len(datatypes)) if datatypes[i] == 'string']

    # Handle outliers in string columns
    print('> Checking and handling any string outliers...')
    unique_string_cols = handle_outlier_vals(unique_string_cols, unique_string_dts, unique_string_out)
    string_cols = handle_outlier_vals(string_cols, string_dts, string_out)

    # Process unique strings etc
    print('> Processing string features in the data...')
    unique_string_cols, require_encoding = apply_process_unique(unique_string_cols, unique_string_dts)

    # If cannot infer using given PFSMs, gather features and infer nominal / ordinal using GradientBoostingClassifier
    # 0 = ordinal, 1 = nominal
    print('> Predicting ordinality of string columns without string features...')
    results_heur = inference_heuristics(string_cols)
    results_gbc = inference_statistical_type(results_heur)

    if encode:
        print('> Encoding string data...')
        # Encode the string columns based on the results from the GBC
        string_cols = apply_encoding(string_cols, results_gbc, dense_encoding)
        unique_string_cols = apply_encoding(unique_string_cols, require_encoding, dense_encoding)

    if verbose:
        # Instantiate DataFrame for information per string column
        # TODO: fix more things such as encoding strategy applied?
        info = pd.DataFrame({
            'Number of unique values': [len(data[col].unique()) for col in data],
            'Type': datatypes,
            'Missing values': missing_vals,
            'Outliers': outlier_vals
        })
        info.index = list(data.columns)
        # Store obtained data in info DataFrame
        check_ord = {x: y for x, y in zip(list(string_cols.columns), ['Yes' if i == 0.0 else 'No' for i in results_gbc])}
        enc_used = {x:
                        'SimilarityEncoder' if info.at[x, 'Number of unique values'] < 30
                        else 'GapEncoder' if info.at[x, 'Number of unique values'] < 100
                        else 'MinHashEncoder' for x in list(string_cols.columns)
                    }

        for name, mapping in zip(['Ordinal?', 'Encoding'], [check_ord, enc_used]):
            mapping = pd.Series(info.index).map(mapping)
            mapping.index = list(data.columns)
            info[name] = mapping
        print(info.to_string())

    result = pd.concat([other_cols, string_cols, unique_string_cols], axis=1)
    return result


if __name__ == "__main__":
    # Load in the dataset
    dfsd = pd.read_csv('datasets\gbc_data\diamonds.csv')  # winemag-data-130k-v2.csv
    run(dfsd, encode=False).to_string()
