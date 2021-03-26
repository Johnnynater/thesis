import pandas as pd
from src.infer_stattype import generate_clusters


# TODO: come up with more names / keywords
COLUMN_NAMES = ['name', 'address', 'year', 'zip', 'code', 'x', 'y',
                'latitude', 'longitude', 'location', 'city', 'country', 'gender', 'sex']

KEYWORDS = ['probably', 'hardly', 'scarcely', 'minutely', 'vaguely',
            'barely', 'slightly', 'moderately', 'pretty', 'strongly',
            'very', 'intensely', 'immensely', 'extremely', 'absolutely',
            'infinitely', 'unlimitedly', 'neutral', 'low', 'medium', 'high']


def name_check(column_name):
    """ Check whether the column name is a frequently occurring column name.

    Often times, it can be determined what type of encoding a column needs by solely looking at the name of the column.
    For example, names such as 'location', 'city', and 'country' often require categorical encoding.

    :param column_name: a string containing the column name.
    :return: a string indicating whether the column name is contained in COLUMN_NAMES or not.
    """
    for name in COLUMN_NAMES:
        if name in column_name or column_name in name:
            return 'Column name is part of known column names'
    return 'Column name is not part of known column names'


# TODO: needs fine-tuning
def ratio_check(nr_unique, nr_total):
    """ Check what the ratio between unique values and the number of entries is.

    :param nr_unique: an integer value of the number of unique entries.
    :param nr_total: an integer value of the number of entries.
    :return: a string indicating whether the ratio appears to be either binary, ordinal, categorical, or continuous.
    """
    if nr_unique == 2:
        return 'Binary'
    if nr_total <= 50:
        return 'Not enough data'
    ratio = nr_unique / nr_total
    # Check whether number of unique elements are according to Likert-Scale characteristics
    print((0.104 - 0.0001 * nr_total), (0.0002 * nr_total + 0.2), nr_total, ratio, (ratio < (0.104 - 0.0001 * nr_total)), (ratio < (0.0002 * nr_total + 0.2) and nr_unique in [3, 5, 7]), nr_unique)
    if ratio < (0.104 - 0.0001 * nr_total):
        return 'Ordinal'
    elif ratio < (0.0002 * nr_total + 0.2) and nr_unique in [3, 5, 7]:
        return 'Categorical/Ordinal'
    elif ratio < (0.0002 * nr_total + 0.2) and nr_unique not in [3, 5, 7]:
        return 'Categorical'
    else:
        return 'Continuous'


def keyword_check(df):
    """ Check whether entries contain certain keywords associated to certain string types.

    :param df: a pandas DataFrame.
    :return: a string indicating whether the keywords do or do not match ordinal data.
    """
    keyword_counter = 0
    for keyword in KEYWORDS:
        for item, _ in df.iteritems():
            if keyword in item[0]:
                keyword_counter += 1
    if keyword_counter / df.size < 0.2:
        return 'Keywords do not match ordinal data'
    return 'Keywords match ordinal data'


def comstring_check(df):
    """ Check for common substrings between all unique entries.

    :param df: a pandas DataFrame containing unique entries.
    :return: a list of all common substrings.
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
                            substrings.append(common_substring)
                        common_substring = ''
                        found_substring = False
                        counter = 0

            if len(common_substring) > 3 and common_substring not in substrings:
                substrings.append(common_substring)
    return substrings


def run_heuristics(df):
    """ Run all the heuristics and save the result.

    :param df: a pandas DataFrame.
    :return: a list containing results.
    """
    # Take samples if there are too many entries
    if df.size > 1000:
        df = df.sample(1000)
    results = []
    for column in df:
        # Eliminate any outlying entries
        clusters = generate_clusters(df[column].str.lower().to_frame())
        unique_entries = clusters.value_counts()

        # Pre-check the ratio to avoid evaluating continuous data types
        ratio_type = ratio_check(unique_entries.count(), clusters.size)

        if ratio_type == 'Continuous':
            results.append([ratio_type])
        else:
            results.append([ratio_type,
                            name_check(clusters.columns[0]),
                            keyword_check(unique_entries),
                            comstring_check(unique_entries)])
    print(results)
    return results


# Test run
# data = pd.read_csv('datasets/diamonds.csv')
# data = data[['cut', 'color', 'clarity']]
# print(data)
# run_heuristics(data)
