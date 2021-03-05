import pandas as pd
from difflib import SequenceMatcher
from numpy.random import random, multinomial, dirichlet
from scipy.spatial.distance import jensenshannon


def convert_similar(df):
    # TODO: come up with better measure to detect 'outliers'
    threshold = 0.05
    above_avg, below_avg = [], []
    for label, value in df.iteritems():
        if value >= threshold:
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


def simulate_distribution(nr_sim, nr_total, probs):
    simulations = []
    for i in range(nr_sim):
        simulation = []
        for j in range(nr_total):
            rand_prob = random()
            cum_prob = 0
            idx = 0
            while cum_prob < rand_prob:
                cum_prob += probs[idx]
                idx += 1
            simulation.append(idx)
        simulations.append(simulation)
    print(simulations)
    return simulations


def match_distribution(nr_sim, nr_total, probs):
    cat, ordinal = [], []
    for i in range(nr_sim):
        cat.append(jensenshannon(probs, multinomial(nr_total, probs)))
        ordinal.append(jensenshannon(probs, dirichlet(alpha=probs)))
    avg_cat = sum(cat) / len(cat)
    avg_ord = sum(ordinal) / len(ordinal)
    return 'Categorical' if avg_cat < avg_ord else 'Ordinal'


def run_inference(df):
    total = df.count().values[0]
    print(total)
    unique_counts = df.value_counts(normalize=True)
    print(unique_counts)

    clusters = convert_similar(unique_counts)

    probs = calculate_probabilities(clusters)
    print(probs)

    simulations = simulate_distribution(10, total, probs)
    print(simulations)

    distribution = match_distribution(10, total, probs)
    print(distribution)
    pass


test = ['John', 'John', 'john', 'joh', 'John', 'Johnny', 'Jon', 'Stef', 'Stef', 'Stef', 'staf', 'sted', 'stf', 'stofn', 'nick', 'Nick', 'dick', 'jick', 'sick', 'Nick', 'Nick', 'Nicky', 'Niuck']
test_df = pd.DataFrame(test)
print(test_df)
run_inference(test_df)
