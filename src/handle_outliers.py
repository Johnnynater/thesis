# TODO: place comments in this file
import difflib


def generate_ngram(unique_vals, n=3):
    ngram_vals = []
    for i in range(len(unique_vals)):
        # Example: 'hello' becomes {'hel', 'ell', 'llo'}
        ngram_vals.append(set([unique_vals[i][j:j + n] for j in range(len(unique_vals[i]) - n-1)]))
    return ngram_vals


def compute_most_similar(unique_vals, ngram_vals):
    suspects = []
    for i in range(len(ngram_vals)):
        best_ratio = 0
        best_pair = []
        for j in range(i + 1, len(ngram_vals)):
            # n-gram similarity: intersect(ngram1, ngram2) / union(ngram1, ngram2)
            pair_intersect = ngram_vals[i].intersection(ngram_vals[j])
            pair_union = ngram_vals[i].union(ngram_vals[j])

            ratio = len(pair_intersect) / len(pair_union)

            # TODO: change this threshold?
            if ratio >= 0.5 and ratio > best_ratio:
                best_ratio = ratio
                best_pair = [unique_vals[i], unique_vals[j]]
        if best_pair:
            suspects.append(best_pair)
    return suspects


def sort_tuples(df, pair):
    tup = [
        (pair[0], df.at[pair[0]]),
        (pair[1], df.at[pair[1]])
    ]
    tup.sort(key=lambda x: x[1])
    return tup


def compute_ratio(tup, threshold=0.05):
    return True if tup[0][1] / tup[1][1] < threshold else False


def compute_close_matches(df, suspect):
    return list(set(difflib.get_close_matches(suspect, df)))


def confirm_close_matches(vals, pair):
    tmp = vals.copy()
    tmp.remove(pair[0])
    close_matches = difflib.get_close_matches(pair[0], tmp)
    return True if pair[1] in close_matches else False


def run(df, datatypes, outliers):
    for col, dt, outlier in zip(df, datatypes, outliers):
        unique_vals = list(df[col].unique())
        unique_freq = df[col].value_counts()

        # Check the outliers detected by ptype, excluding sentences since these are mostly false negatives
        if outlier and dt != 'sentence':
            for out in outlier:
                # Check for similar entries using difflib
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
