import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import encode_data
import handle_missing
import handle_outliers
import infer_ptype
import infer_stattype
import process_stringtype
from gbc import heuristics


# Some warnings are shown from other libraries.
warnings.filterwarnings("ignore")


def inference_ptype(df):
    """ Infer the data and/or string feature type for each column in the data.

    :param df: a pandas DataFrame consisting of unknown data types and string feature types.
    :return: a ptype schema containing information such as data or string feature types, missing values, and outliers;
             a List containing the names of the string feature types that can be explicitly inferred.
    """
    return infer_ptype.infer(df)


def inference_statistical_type(df):
    """ Predict the statistical type of each standard string column.

    :param df: a List of Floats/Integers containing results from each heuristic.
    :return: a List of Integers representing the statistical type of each standard string column.
    """
    return infer_stattype.infer(df)


def extract_features_gbc(df):
    """ Extract features from standard string columns to be used for statistical type prediction.

    :param df: a pandas DataFrame consisting of standard string columns.
    :return: a List of Lists containing Floats/Integers that represent the results from each heuristic.
    """
    return heuristics.run(df)


def handle_missing_vals(df, datatypes, missing_vals, names):
    """ Handle missing values in the data.

    :param df: a pandas DataFrame consisting of the data with potential missing values.
    :param datatypes: a List of Strings representing the data or string feature types of each column.
    :param missing_vals: a List of Lists that can contain Strings, where each String represents a missing value in a
                         column.
    :param names: a List of Strings containing all string feature types that we can infer using PFSMs.
    :return: a pandas DataFrame without missing values.
    """
    any_missing = False
    for col, missing in zip(df, missing_vals):
        if missing:
            any_missing = True
            df[col].fillna(value=np.nan, inplace=True)
            df[col].replace(to_replace=missing, value=np.nan, inplace=True)
    if any_missing:
        df = handle_missing.run(df, datatypes, names)
    return df


def handle_outlier_vals(df, datatypes, outlier_vals, names):
    """ Handle outlying data types / typos in the data.

    :param df: a pandas DataFrame consisting of data with potential outliers and/or data type outliers.
    :param datatypes: a List of Strings representing the data or string feature types of each column.
    :param outlier_vals: a List of Lists that can contain Strings, where each String represents a potential outlier in a
                         column.
    :param names: a List of Strings representing the string features that can be inferred by the PFSMs.
    :return: a pandas DataFrame without outliers; a List representing the (adjusted) datatypes.
    """
    return handle_outliers.run(df, datatypes, outlier_vals, names)


def apply_process_unique(df, stringtypes, dense):
    """ Process all string feature columns found in the data.

    :param df: a pandas DataFrame consisting of string feature columns.
    :param stringtypes: a List of Strings representing the inferred string feature of each column.
    :param dense: a Boolean indicating whether multi-dimensional encodings need to be stored in a single column or not.
    :return: a pandas DataFrame consisting of processed string feature columns; a List of Lists containing Integers
             that indicate whether/which type of encoding is required for each processed string feature column.
    """
    processed_df = pd.DataFrame()
    require_encoding = []
    for col, stringtype in zip(df, stringtypes):
        result, encoding = process_stringtype.run(df[col].to_frame(), stringtype, dense)
        df.reset_index(drop=True, inplace=True)
        processed_df = pd.concat([processed_df, result], axis=1)

        if type(encoding) == list:
            require_encoding.extend(encoding)
        else:
            require_encoding.append(encoding)

    return processed_df, require_encoding


def apply_encoding(df, y, results, dense):
    """ Encode the string columns based on the results from the GBC.

    :param df: a pandas DataFrame consisting of string columns.
    :param y: a pandas Series consisting of target values.
    :param results: a List of Integers representing whether/which encoding needs to be applied for which column.
    :param dense: a Boolean indicating whether multi-dimensional encodings need to be stored in a single column or not.
    :return: a pandas DataFrame consisting of encoded columns.
    """
    if y is not None:
        # Check target class balance. Target class is balanced if the standard deviation is smaller than 10% of the mean
        target_freq = list(y.value_counts())
        mean_tf, std_tf = np.mean(target_freq), np.std(target_freq)
        balanced = std_tf / mean_tf < 0.5
    else:
        balanced = False

    for col, val in zip(df, results):
        if val == 2:
            # No encoding is required
            continue
        else:
            if val == 1 and df[col].value_counts().count() < 30 and balanced:
                # A nominal encoding is required
                df[col] = encode_data.run(df, y, df[col], val, dense, balanced)
            elif val == 1 and not dense:
                df = encode_data.run(df, y, df[col], val, dense, balanced)
            elif (val == 1 and dense) or val == 0:
                # A nominal encoding summed up into one column or an ordinal encoding is required
                df[col] = df[col].map(encode_data.run(df, y, df[col], val, dense, balanced))
    return df


def run(data, y=None, encode=True, dense_encoding=True, display_info=True):
    """ Run the framework for automated string handling, cleaning, and encoding.

    :param data: a pandas DataFrame containing string columns.
    :param y: a pandas Series containing the target values.
    :param encode: a Boolean indicating whether the cleaned string data needs to be encoded.
    :param dense_encoding: a Boolean indicating whether multi-dimensional encodings need to be stored in a single column
                           or not.
    :param display_info: a Boolean indicating whether additional information on every column needs to be printed.
    :return: a pandas DataFrame whose string entries are cleaned based on the other parameters.
    """
    print('> Performing pre-checks...')
    for col in data:
        # Replace the different uni-code space with a regular space
        if data[col].dtype == 'object':
            data[col] = data[col].map(lambda x: x.replace(u'\xa0', ' ') if type(x) == str else x)

        if data[col].isnull().all():
            print('>> Column "{}" contains no values. Removing this column from the dataset.'.format(col))
            data = data.drop(columns=[col])

    if y is not None:
        if encode and y.dtype not in ['int64', 'float64']:
            print('>> Given target column is not encoded. Encoding using LabelEncoder...')
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=y.name)

    # Infer data / string type using ptype
    print('> Inferring data types and string features...')
    schema, names = inference_ptype(data)

    # Store obtained data in separate variables
    datatypes = [col.type for _, col in schema.cols.items()]
    missing_vals = [col.get_na_values() for _, col in schema.cols.items()]
    outlier_vals = [col.get_an_values() for _, col in schema.cols.items()]

    # FOR TESTING PROCESSING:
    # datatypes = ['string' if item == 'day' else item for item in datatypes]

    # Impute missing values
    print('> Checking and handling any missing values...')
    data = handle_missing_vals(data, datatypes, missing_vals, names)

    # Handle outliers in data
    print('> Checking and handling any string and data type outliers...')
    data, datatypes = handle_outlier_vals(data, datatypes, outlier_vals, names)

    # Separate columns based on their inferred string feature or data type
    unique_string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in names]]
    string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] == 'string']]
    bool_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in ['boolean', 'gender']]]
    date_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in ['date-iso-8601', 'date-eu']]]
    other_cols = data.iloc[
                 :,
                 [
                     i for i in range(len(datatypes))
                     if datatypes[i] != 'string'
                     and datatypes[i] not in names + ['boolean', 'gender', 'date-iso-8601', 'date-eu']
                 ]
                 ]

    # Make a list of types for each of the split columns
    unique_string_dts = [datatypes[i] for i in range(len(datatypes)) if datatypes[i] in names]
    bool_dts = [datatypes[i] for i in range(len(datatypes)) if datatypes[i] in ['boolean', 'gender']]
    date_dts = [datatypes[i] for i in range(len(datatypes)) if datatypes[i] in ['date-iso-8601', 'date-eu']]

    # Process unique strings and boolean / date types
    print('> Processing string features in the data...')
    unique_string_cols, require_enc_unique = apply_process_unique(unique_string_cols, unique_string_dts, dense_encoding)
    bool_cols, _ = apply_process_unique(bool_cols, bool_dts, dense_encoding)
    date_cols, _ = apply_process_unique(date_cols, date_dts, dense_encoding)

    # If cannot infer using given PFSMs, gather features and infer nominal / ordinal using GradientBoostingClassifier
    print('> Predicting ordinality of string columns without string features...')
    results_heur = extract_features_gbc(string_cols)

    # The GBC assigns a 0 (= ordinal) or a 1 (= nominal) for each standard string column
    results_gbc = inference_statistical_type(results_heur)

    if encode:
        print('> Encoding string data...')
        # Encode the string columns based on the results from the GBC
        string_cols = apply_encoding(string_cols, y, results_gbc, dense_encoding)
        unique_string_cols = apply_encoding(unique_string_cols, y, require_enc_unique, dense_encoding)

    if display_info:
        # Instantiate DataFrame for information per string column
        # TODO: fetch more information?
        info = pd.DataFrame({
            'Number of unique values': [len(data[col].unique()) for col in data],
            'Type': datatypes,
            'Missing values': missing_vals,
            'Outliers': outlier_vals
        })
        info.index = list(data.columns)
        # Store obtained data in info DataFrame
        check_ord = {x: y for x, y in
                     zip(list(string_cols.columns), ['Yes' if i == 0.0 else 'No' for i in results_gbc])}
        enc_used = {x:
                        'OrdinalEncoder' if check_ord[x] == 'Yes'
                        else 'SimilarityEncoder' if info.at[x, 'Number of unique values'] < 30
                        else 'GapEncoder' if info.at[x, 'Number of unique values'] < 100
                        else 'MinHashEncoder' for x in list(string_cols.columns)
                    }

        for name, mapping in zip(['Ordinal?', 'Encoding'], [check_ord, enc_used]):
            mapping = pd.Series(info.index).map(mapping)
            mapping.index = list(data.columns)
            info[name] = mapping
        print(info.to_string())

    result = pd.concat([other_cols, string_cols, bool_cols, date_cols, unique_string_cols], axis=1)

    # Return the label as well if it was given as a parameter
    if y is not None:
        return result, y
    else:
        return result


if __name__ == "__main__":
    # Load in the dataset
    test_df = pd.read_csv(
        r'C:\Users\s165399\Documents\[MSc] Data Science in Engineering\Year 2\Master thesis\Program\src\datasets\gbc_data\fifa.csv')  # winemag-data-130k-v2.csv Automobile_data.csv
    # test_df = test_df.iloc[:9999, :]
    label = test_df['Value']
    test_df = test_df.drop(columns=['Value'])
    # print(test_df.iloc[:10, :].to_string())
    print(run(test_df, y=label, dense_encoding=False, display_info=False)[0].iloc[:10, :].to_string())
