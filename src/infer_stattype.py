import pandas as pd
from difflib import SequenceMatcher


def convert_similar(df):
    # TODO: come up with better measure to detect 'outliers'
    avg = df.mean()
    above_avg, below_avg = [], []
    for label, value in df.iteritems():
        if value >= avg:
            above_avg.append(label[0])
        else:
            below_avg.append(label[0])
    for i in below_avg:
        sim_word = ''
        max_sim = 0
        for j in above_avg:
            similarity = SequenceMatcher(None, i, j).quick_ratio()
            if similarity > max_sim:
                sim_word = j
                max_sim = similarity
        above_avg.append(sim_word)
    return pd.DataFrame(above_avg)


def calculate_probabilities(df):
    return df.value_counts(normalize=True)


def simulate_distributions():
    pass


def match_distributions():
    pass


def run_inference(df):
    total = df.count()
    unique_counts = df.value_counts(normalize=True)
    print(unique_counts)
    clusters = convert_similar(unique_counts)
    probs = calculate_probabilities(clusters)
    print(probs)
    pass

# NEEDED:
# DONE - number of unique clusters/values (using fuzzy matching / similarity thresholds)
# DONE - total number of entries
# DONE - calculate probabilities
# - simulate several distributions using the probabilities
# - check best distribution match -> ordinal or categorical?


test = ['John', 'John', 'john', 'joh', 'John','Johnny','Jon', 'Stef', 'Stef', 'Stef', 'staf', 'sted', 'stf', 'stofn', 'nick', 'Nick', 'dick', 'jick', 'sick', 'Nick', 'Nick', 'Nicky', 'Niuck']
test_df = pd.DataFrame(test)
print(test_df)
run_inference(test_df)