from src.unused_methods.infer_stattype import generate_clusters
import re
import numpy as np
import pandas as pd
import pickle as pkl
from itertools import zip_longest
import csv

# My features are:
# - Number of entries                                       (numerical)     (nom/ord)
# - Number of unique entries                                (numerical)     (nom/ord)
# - Ratio unique/total                                      (numerical)     (nom/ord)
# - Median distance between unique word embeddings          (numerical)     (nom/ord)
# - Check if column name is a known nominal column name     (True/False)    (nominal)
# - Check if column name is a known ordinal column name     (True/False)    (ordinal)
# - Check if unique entries contain nominal keywords        (True/False)    (nominal)
# - Check if unique entries contain ordinal keywords        (True/False)    (ordinal)
# - Check if unique values have common substring            (True/False)    (ordinal)


# TODO: come up with more names / keywords
nominal_colnames = [
    'address', 'age', 'area', 'category', 'city', 'code', 'color', 'colour', 'country', 'gender', 'group',
    'location', 'mode', 'name', 'nationality', 'place', 'post', 'race', 'role', 'sex', 'type', 'year', 'zip'
]

ordinal_colnames = [
    'stage', 'phase', 'rating', 'level', 'size', 'rank', 'order', 'grade', 'opinion', 'scale', 'degree'
]

# nominal_keywords = []

ordinal_keywords = [
    # Adjectives
    'probably', 'hardly', 'scarcely', 'minutely', 'vaguely', 'barely', 'slightly', 'moderately', 'pretty',
    'strongly', 'very', 'intensely', 'immensely', 'extremely', 'absolutely', 'infinitely', 'unlimitedly',
    'completely', 'ultimately', 'much', 'many', 'a lot', 'plenty', 'numerous',

    # Nouns
    'agree', 'neutral', 'low', 'medium', 'high', 'good', 'better', 'best', 'bad', 'worse', 'worst',
    'much', 'many', 'more', 'most', 'far', 'less', 'lesser', 'least', 'well', 'better', 'best'
]

# Since the file size of the original embeddings is too big, I have to split the dict into two separate dicts
# embeddings = pkl.load(open(f'datasets/6B.50_glove.pkl', 'rb'))
fh_emb = pkl.load(open(f'datasets/50_glove_1.pkl', 'rb'))
sh_emb = pkl.load(open(f'datasets/50_glove_2.pkl', 'rb'))
embeddings = fh_emb | sh_emb


def ratio_check(nr_unique, nr_total):
    """ Check what the ratio between unique values and the number of entries is. This can help with determining order
    in the data, as ordered data (e.g., Likert-scale data) have a limited number of unique entries.

    :param nr_unique: an Integer representing the number of unique entries.
    :param nr_total: an Integer representing the number of entries.
    :return: a Float representing the ratio between unique values and total number of values.
    """
    # ratio = nr_unique / nr_total

    # print((0.104 - 0.0001 * nr_total), (0.0002 * nr_total + 0.2), nr_total, ratio,
    #       (ratio < (0.104 - 0.0001 * nr_total)), (ratio < (0.0002 * nr_total + 0.2) and nr_unique in [3, 5, 7]),
    #       nr_unique)
    return nr_unique / nr_total
    # if nr_total <= 50:
    #     return 'Not enough data'
    # elif nr_unique == 1:
    #     return 'Single value'
    # elif nr_unique == 2:
    #     return 'Binary'
    # # Ratio scales down or up with the number of entries, depending on which distribution we are evaluating
    # elif ratio < (0.104 - 0.0001 * nr_total) and nr_unique < 10:
    #     return 'Ordinal'
    # elif ratio < (0.0002 * nr_total + 0.2) and nr_unique <= 7:
    #     return 'Categorical/Ordinal'
    # elif ratio < (0.0002 * nr_total + 0.2) and nr_unique > 7:
    #     return 'Categorical'
    # else:
    #     return 'Continuous'


def distance_check(df):
    """ Calculate the mean variance of the word embedding of all entries.

    :param df: a pandas DataFrame consisting of unique String entries.
    :return: a Float representing the mean variance of all embedded String entries.
    """
    mean_words = []
    for item, _ in df.iteritems():
        words = re.split("[ \-_]", item.lower())
        mean_word = 0
        for word in words:
            try:
                mean_word += embeddings[word]
            except KeyError:
                embeddings[word] = np.random.normal(scale=0.6, size=(50,))
                mean_word += embeddings[word]
        mean_word = mean_word / len(words)
        mean_words.append(mean_word)
    mean_all = sum(mean_words) / len(mean_words)
    var_all = sum((word - mean_all) ** 2 for word in mean_words)
    mean_var_all = np.mean(var_all)
    return mean_var_all


def name_check(column_name):
    """ Check whether the column name is a frequently occurring column name.

    Often times, it can be determined what type of encoding a column needs by solely looking at the name of the column.
    For example, names such as 'location', 'city', and 'country' often require categorical encoding.

    :param column_name: a String representing the column name.
    :return: a pair of Integers {0,1} indicating whether the column name is a known nominal / ordinal name (1)
             or not (0).
    """
    column_name = column_name.lower()
    name_nom, name_ord = 0, 0  # False, False

    # Check whether column name is (part of) known nominal / ordinal column names
    for nominal, ordinal in zip_longest(nominal_colnames, ordinal_colnames):
        if nominal and (nominal in column_name or column_name in nominal):
            name_nom = 1  # True
            break
        if ordinal and (ordinal in column_name or column_name in ordinal):
            name_ord = 1  # True
            break

    return name_nom, name_ord


def keyword_check(df):
    """ Check whether entries contain certain keywords associated to data with or without order.

    :param df: a pandas DataFrame consisting of unique String entries.
    :return: 1 if a sufficient amount of keywords are found, 0 otherwise.
    """
    # TODO: maybe don't do it based on percentages, but just whenever we see one keyword
    # keyword_nom_count, keyword_ord_count = 0, 0
    # keyword_nom, keyword_ord = False, False
    keyword_ord_count = 0
    keyword_ord = 0  # False

    # Check for nominal and ordinal keywords
    # for nominal, ordinal in zip_longest(nominal_keywords, ordinal_keywords):
    for ordinal in ordinal_keywords:
        for item, _ in df.iteritems():
            # if nominal and nominal in item.lower():
            #     keyword_nom_count += 1
            if ordinal and ordinal in item.lower():
                keyword_ord_count += 1
    # Make sure this occurs for at least 20% of the data
    # if keyword_nom_count / len(df) >= 0.2:
    #     keyword_ord = True
    if keyword_ord_count / len(df) >= 0.2:
        keyword_ord = 1  # True

    # return keyword_nom, keyword_ord
    return keyword_ord


def comstring_check(df):
    """ Check for common substrings between all unique entries.

    :param df: a pandas DataFrame consisting of unique String entries.
    :return: 1 if a substring was found, 0 otherwise.
    """
    labels = [x[0] for x in df.index.values.tolist()]
    substrings = []

    # Compare two string entries by creating an array of characters representing the string
    for i in range(len(labels)):
        word_i = [c for c in labels[i]]
        for j in range(i + 1, len(labels)):
            word_j = [c for c in labels[j]]

            # Keep track of the common substring, the index in which the substring starts,
            # and whether we are currently dealing with a common substring or not
            common_substring = ''
            counter = 0
            found_substring = False

            # Find substrings between each pair of strings
            for ci in range(len(word_i)):
                for cj in range(counter, len(word_j)):
                    if word_i[ci] == word_j[cj]:
                        common_substring += word_i[ci]
                        counter = cj + 1
                        found_substring = True
                        break

                    elif word_i[ci] != word_j[cj] and found_substring:
                        # We do not consider substrings of length 1 as having one character in
                        # common is not indicative that two strings share a common substring
                        if len(common_substring) > 3 and common_substring not in substrings:
                            return 1  # True  # substrings.append(common_substring)
                        common_substring = ''
                        found_substring = False
                        counter = 0

            if len(common_substring) > 3 and common_substring not in substrings:
                return 1  # True  # substrings.append(common_substring)
    return 0  # False  # bool(substrings)


def run(df):
    """ Run all the heuristics and save the result.

    :param df: a pandas DataFrame.
    :return: a List of Floats/Integers containing results from each heuristic.
    """
    results = []
    for column in df:
        # Eliminate any outlying entries
        # clusters = generate_clusters(df[column].str.lower().to_frame())
        unique_entries = df[column].value_counts()
        total_nr_entries = len(df)

        # Pre-check the ratio to avoid evaluating continuous data types
        ratio_type = ratio_check(unique_entries.count(), total_nr_entries)

        # Calculate the variance of the embedded words
        variance = distance_check(unique_entries)

        # Pre-check for existing column name and keywords
        name_nominal, name_ordinal = name_check(df[column].name)
        # keyword_nominal, keyword_ordinal = keyword_check(unique_entries)
        keyword_ordinal = keyword_check(unique_entries)

        if ratio_type > 0.8 or unique_entries.count() > 500:
            results.append(
                [
                    total_nr_entries,
                    unique_entries.count(),
                    ratio_type,
                    variance,
                    name_nominal,
                    name_ordinal,
                    # keyword_nominal,
                    keyword_ordinal,
                    0,
                    # 'nominal'
                ]
            )  # , column])
        else:
            # clusters = generate_clusters(df[column].str.lower().to_frame())
            # unique_entries = clusters.value_counts()
            # ratio_type = ratio_check(unique_entries.count(), len(clusters))
            # column_name_lower = clusters.columns[0].lower()
            results.append(
                [
                    total_nr_entries,
                    unique_entries.count(),
                    ratio_type,
                    variance,
                    name_nominal,
                    name_ordinal,
                    # keyword_nominal,
                    keyword_ordinal,
                    comstring_check(unique_entries),
                    # 'nominal'
                ]
            )  # , column])

    # When training the GBC, we use this statement to gather training data.
    # with open("../out_nominal.csv", "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(results)
    print(results)
    return results


# Test run
# data = pd.read_csv('datasets/gbc_data/nominal_data/winemag-data-130k-nom.csv')#, sep=';')
# # data = data[['cut', 'color', 'clarity']]
# print(data)
# run(data)
# # distance_check(data['cut'].value_counts())
