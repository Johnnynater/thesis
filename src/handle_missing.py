import math
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder


def mcar_test(df):
    """ Implementation of Little's MCAR test, created by RianneSchouten and the full code can be found at
    https://github.com/RianneSchouten/pymice/blob/master/pymice/exploration/mcar_tests.py

    :param df: a pandas DataFrame representing an incomplete dataset with samples as index and variables as columns.
    :return a float value representing the outcome of a chi-square statistical test, testing whether the null hypothesis
            'the missingness mechanism of the incomplete dataset is MCAR' can be rejected.
    """
    dataset = df.copy()
    vars = dataset.dtypes.index.values
    n_var = dataset.shape[1]

    # mean and covariance estimates
    # ideally, this is done with a maximum likelihood estimator
    gmean = dataset.mean()
    gcov = dataset.cov()

    # set up missing data patterns
    r = 1 * dataset.isnull()
    mdp = np.dot(r, list(map(lambda x: math.pow(2, x), range(n_var))))
    sorted_mdp = sorted(np.unique(mdp))
    n_pat = len(sorted_mdp)
    correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
    dataset['mdp'] = pd.Series(correct_mdp, index=dataset.index)

    # calculate statistic and df
    pj = 0
    d2 = 0
    for i in range(n_pat):
        dataset_temp = dataset.loc[dataset['mdp'] == i, vars]
        select_vars = ~dataset_temp.isnull().any()
        pj += np.sum(select_vars)
        select_vars = vars[select_vars]
        means = dataset_temp.reindex(select_vars, axis=1).mean() - gmean.reindex(select_vars, axis=1)
        select_cov = gcov.reindex(index=select_vars, columns=select_vars)
        mj = len(dataset_temp)
        parta = np.dot(means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[1])))
        d2 += mj * (np.dot(parta, means))

    deg_free = pj - n_var

    # perform test and save output
    p_value = 1 - st.chi2.cdf(d2, deg_free)

    return p_value


def calc_ratio_missing(df):
    """ Calculate the ratio of rows containing missing values vs. the total number of rows.

    :param df: a pandas DataFrame containing missing values.
    :return: a Float indicating the ratio of rows containing missing values vs. total nr. of rows.
    """
    return len(df[df.isnull().any(axis=1)]) / df.shape[0]


def delete_rows(df):
    """ Delete rows containing missing values.

    :param df: a pandas DataFrame containing missing values.
    :return: a pandas DataFrame without missing values.
    """
    return df.dropna()


def impute_mcar(df, datatypes, names):
    """ Impute missing values that are MCAR using mean/mode imputation.

    :param df: a pandas DataFrame consisting of Floats/Integers and missing values.
    :param datatypes: a List containing the string feature types that can be inferred specifically.
    :param names: a List containing the data type / string feature type of each column.
    :return: a pandas DataFrame whose missing values are imputed.
    """
    for col, val in zip(df, datatypes):
        if val in names:
            imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            df[col] = imp_freq.fit_transform(df[col])
        else:
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            df[col] = imp_mean.fit_transform(df[col])
    return df


def impute_mar_mnar(df):
    """ Impute the data using scikit-learn's IterativeImputer.

    :param df: a pandas DataFrame consisting of Floats/Integers and missing values.
    :return: a pandas DataFrame whose missing values are imputed.
    """
    imp = IterativeImputer(missing_values=np.nan)
    imputed_df = imp.fit_transform(df)
    return pd.DataFrame(np.round(imputed_df), columns=df.columns)


def encode_strings(df, datatypes, names):
    """ Ordinally-encode Strings to Floats such that the imputer can handle these columns.

    :param df: a pandas DataFrame consisting of Strings entries.
    :param datatypes: a List of Strings representing the ptype-inferred data type for each column in df.
    :param names: a List of Strings representing all string-type names that can be inferred by ptype.
    :return: a pandas DataFrame consisting of encoded Strings.
    """
    encoded_vals = {}
    for col, val in zip(df, datatypes):
        if val in names or df[col].dtype not in ['int64', 'float64']:
            replacement = df[col].dropna().map(lambda x: str(x) if type(x) != str else x).unique().reshape(-1, 1)
            enc = OrdinalEncoder()
            enc.fit(replacement)
            encoding = enc.categories_
            encoding_dict = dict(zip((encoding[0]), range(len(encoding[0]))))
            df[col] = df[col].map(encoding_dict)
            reverse_encoding_dict = dict(zip(range(len(encoding[0])), (encoding[0])))
            encoded_vals[col] = reverse_encoding_dict
    return df, encoded_vals


def decode_strings(df, datatypes, names, encoded_vals):
    """ Revert encoded Strings to Strings and give newly introduced classes a placeholder String.

    :param df: a pandas DataFrame consisting of encoded Strings.
    :param datatypes: a List of Strings representing the ptype-inferred data type for each column in df.
    :param names: a List of Strings representing all string-type names that can be inferred by ptype.
    :param encoded_vals: a Dictionary representing a mapping from encoded value to original string (Float/Int -> String)
    :return: a pandas DataFrame consisting of the original String entries.
    """
    for col, val in zip(df, datatypes):
        if val in names or df[col].dtype not in ['int64', 'float64']:
            for item in df[col].unique():
                if item not in encoded_vals[col]:
                    # Associate item with the closest key that is in the dict
                    encoded_vals[col][item] = encoded_vals[col][
                        min(encoded_vals[col].keys(), key=lambda k: abs(k-item))
                    ]
            df[col] = df[col].map(encoded_vals[col])
    return df


def run(df, datatypes, names):
    """ Determine the most fitting missing value imputer and execute accordingly.

    :param df: a pandas DataFrame consisting of missing values.
    :param datatypes: a List of Strings representing the ptype-inferred data type for each column in df.
    :param names: a List of Strings representing all string-type names that can be inferred by ptype.
    :return: a pandas DataFrame whose missing values are deleted/imputed.
    """
    if calc_ratio_missing(df) < 0.05:
        # Since less than 5% of the data is missing, removing the missing values will have no significant impact
        # on the performance
        print('>> Removed rows containing missing values')
        return delete_rows(df)

    # Encode string data so it can be imputed
    names.append('string')
    names.append('date-eu')
    df, encoded_vals = encode_strings(df, datatypes, names)

    try:
        # Test if the missing values in the DataFrame are MCAR
        if mcar_test(df) >= 0.05:
            # Missing data is probably MCAR, we can impute with mean/mode
            df = impute_mcar(df, datatypes, names)
            print('>> Missing values imputed using mean and mode imputation')
        else:
            # Missing data is probably MAR/MNAR. Since we cannot test for MNAR, We choose an imputation strategy that
            # works for both MAR/MNAR, namely a Multivariate imputer (IterativeImputer) from sklearn
            df = impute_mar_mnar(df)
            print('>> Missing values imputed using IterativeImputer')
    except:
        # Since MCAR test cannot be run on the data, we apply the IterativeImputer.
        df = impute_mar_mnar(df)
        print('>> Missing values imputed using IterativeImputer')

    # Return the data with decoded string columns
    df = decode_strings(df, datatypes, names, encoded_vals)
    names.remove('string')
    names.remove('date-eu')
    return df
