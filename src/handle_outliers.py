import difflib


def generate_ngram(unique_vals, n=3):
    """ Generate n-gram representations for each entry in unique_vals, with n=3 by default.
        For example: 'example' -> {'exa', 'xam', 'amp', 'mpl', 'ple'}.

    :param unique_vals: a List of Strings containing unique entries.
    :param n: an Integer representing the length of each gram (default value = 3).
    :return: a List of Sets containing Strings, each represent the n-gram representation of each unique word.
    """
    ngram_vals = []
    for i in range(len(unique_vals)):
        ngram_vals.append(set([unique_vals[i][j:j + n] for j in range(len(unique_vals[i]) - n-1)]))
    return ngram_vals


def compute_most_similar(unique_vals, ngram_vals):
    """ Compute the most similar String pair using a variation of the Jaccard distance in combination with n-gram.
        The Jaccard distance (in our context the n-gram similarity) for n-grams [ngram1, ngram2] is given by:
        intersect(ngram1, ngram2) / union(ngram1, ngram2)

    :param unique_vals: a List of Strings containing unique entries.
    :param ngram_vals: a List of Sets containing Strings, each represent the n-gram representation of each unique word.
    :return: a List containing the two most-similar String entries.
    """
    suspects = []
    for i in range(len(ngram_vals)):
        best_ratio = 0
        best_pair = []
        for j in range(i + 1, len(ngram_vals)):
            # Calculate n-gram similarity:
            pair_intersect = ngram_vals[i].intersection(ngram_vals[j])
            pair_union = ngram_vals[i].union(ngram_vals[j])
            if len(pair_union) != 0:
                ratio = len(pair_intersect) / len(pair_union)
            else:
                ratio = 0

            # TODO: change this threshold?
            if ratio >= 0.6 and ratio > best_ratio:
                best_ratio = ratio
                best_pair = [unique_vals[i], unique_vals[j]]
        if best_pair:
            suspects.append(best_pair)
    return suspects


def sort_tuples(df, pair):
    """ Sort the Tuple of most-similar entries based on their frequency.

    :param df: a pandas DataFrame consisting of unique entries and their frequency.
    :param pair: a Tuple of Strings consisting of the two most-similar String entries.
    :return: a Tuple (String, Integer) sorted on the Integer value.
    """
    tup = [
        (pair[0], df.at[pair[0]]),
        (pair[1], df.at[pair[1]])
    ]
    tup.sort(key=lambda x: x[1])
    return tup


def compute_ratio(tup, threshold=0.05):
    """ Compute the ratio of two frequencies.

    :param tup: a Tuple (String, Integer)
    :param threshold: an Integer representing a threshold (default value = 0.05).
    :return: Boolean, True if the ratio is below the threshold, False otherwise.
    """
    return True if tup[0][1] / tup[1][1] < threshold else False


def compute_close_matches(df, suspect):
    """ Compute all close matches of a potential outlier using difflib's get_close_matches.

    :param df: a pandas Series consisting of String entries.
    :param suspect: a String representing the potential outlier.
    :return: a List of Sets containing all String entries that are a close match to suspect.
    """
    return list(set(difflib.get_close_matches(suspect, df)))


def confirm_close_matches(vals, pair):
    """ Confirm using difflib's get_close_matches whether the two values are indeed close matches using Gestalt
        Pattern Matching.

    :param vals: a List of Strings where each entry is unique.
    :param pair: a List consisting of a pair of Strings who are claimed to be similar to each other.
    :return: Boolean, True if the second item of pair is in the calculated close_matches, False otherwise.
    """
    tmp = vals.copy()
    tmp.remove(pair[0])
    close_matches = difflib.get_close_matches(pair[0], tmp)
    return True if pair[1] in close_matches else False


def run(df, datatypes, outliers):
    """ Run the outlier handling procedure, where each column could be treated differently based on reported outliers
        and their datatype.

    :param df: a pandas DataFrame representing all columns whose datatype are inferred as a String (feature).
    :param datatypes: a List of Strings each representing the data type of each column in df.
    :param outliers: a List of Lists consisting of Strings, where each List of Strings represents all discovered
                     outliers in that column in the df.
    :return: a pandas DataFrame with no/reduced number of outliers.
    """
    for col, dt, outlier in zip(df, datatypes, outliers):
        unique_vals = list(df[col].unique())
        unique_freq = df[col].value_counts()

        # Check the outliers detected by ptype, excluding sentences since these are mostly false negatives
        if outlier and dt != 'sentence':
            for out in outlier:
                # Check for similar entries using difflib
                # TODO: check if this df[col] should be unique_vals
                close_matches = compute_close_matches(df[col], out)

                # Take the most similar entries
                ngrams = generate_ngram(close_matches)
                most_sim = compute_most_similar(close_matches, ngrams)

                for pair in most_sim:
                    if out in pair:
                        # Sort the matches based on their frequency
                        tup = sort_tuples(unique_freq, pair)
                        if compute_ratio(tup):
                            # Replace outlying value with the most similar value
                            df[col] = df[col].replace(tup[0][0], tup[1][0])

        elif dt != 'sentence':
            # Take the most similar entries
            ngrams = generate_ngram(unique_vals)
            suspects = compute_most_similar(unique_vals, ngrams)

            for pair in suspects:
                if confirm_close_matches(unique_vals, pair):
                    # Sort the matches based on their frequency
                    tup = sort_tuples(unique_freq, pair)
                    if compute_ratio(tup):
                        # Replace outlying value with the most similar value
                        df[col] = df[col].replace(tup[0][0], tup[1][0])
    return df
