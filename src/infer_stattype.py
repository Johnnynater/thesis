import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from scipy.spatial.distance import jensenshannon


class OrdinalDistribution:
    def __init__(self, nr_unique):
        """ Fourteen different ordinal distribution from the paper: "Five-Point Likert Items:
            t test versus Mann-Whitney-Wilcoxon" by Joost de Winter and Dimitra Dodou (2010).

        The probabilities of these distributions are scaled to match a nr_unique-Likert scale.

        :param nr_unique: the number of unique clusters.
        """
        self.very_strongly_agree = np.cumsum([0, 0.01, 0.03, 0.06, 0.9])
        self.strongly_agree = np.cumsum([0.01, 0.03, 0.06, 0.3, 0.6])
        self.agree_peak = np.cumsum([0.05, 0.1, 0.2, 0.45, 0.2])
        self.agree_flat = np.cumsum([0.1, 0.15, 0.2, 0.3, 0.25])
        self.neutral_to_agree = np.cumsum([0.1, 0.2, 0.3, 0.25, 0.15])
        self.neutral_peak = np.cumsum([0, 0.2, 0.5, 0.2, 0.1])
        self.neutral_flat = np.cumsum([0.15, 0.2, 0.25, 0.2, 0.2])
        self.very_strongly_disagree = np.cumsum([0.8, 0.12, 0.04, 0.03, 0.01])
        self.strongly_disagree = np.cumsum([0.7, 0.2, 0.06, 0.03, 0.01])
        self.disagree_flat = np.cumsum([0.25, 0.35, 0.2, 0.15, 0.05])
        self.neutral_to_disagree = np.cumsum([0.1, 0.25, 0.3, 0.2, 0.15])
        self.certainly_not_disagree = np.cumsum([0.01, 0.04, 0.5, 0.3, 0.15])
        self.multimodal = np.cumsum([0.15, 0.05, 0.15, 0.25, 0.4])
        self.strong_multimodal = np.cumsum([0.45, 0.05, 0, 0.05, 0.45])

        for item in vars(self).keys():
            # Interpolate the cumulative distribution to match the number of unique entries
            interpolated_cum = np.interp(np.arange(nr_unique),
                                         [nr_unique / 5 * x for x in range(5)],
                                         vars(self)[item])

            # Calculate the probabilities of the interpolated values using cdf theory
            vars(self)[item] = [interpolated_cum[0] if i == 0
                                else interpolated_cum[i] - interpolated_cum[i - 1]
                                for i in range(nr_unique)]


def generate_clusters(df):
    """ Generate clusters of similar entries.

    This method checks whether the proportion of an entry is above or below a pre-defined threshold.
    (In our case, the threshold is set at 0.05.) After that, all entries that are below the threshold
    are replaced by above-threshold values based on their string similarity.

    :param df: a pandas DataFrame to be clustered
    :return: a pandas DataFrame containing the resulting clusters
    """
    # TODO: come up with better measure to detect 'outliers'
    threshold = 0.05
    above_thresh, below_thresh = [], []
    probabilities = df.value_counts(normalize=True, sort=False)
    for label, value in probabilities.iteritems():
        if value >= threshold:
            # Change to label[0] if required
            above_thresh.append(label[0])
        else:
            below_thresh.append(label[0])

    for i in below_thresh:
        sim_word = i
        # TODO: come up with a better way to tackle below_thresh but completely different values
        max_sim = 0.45
        for j in above_thresh:
            similarity = SequenceMatcher(None, i, j).quick_ratio()
            if similarity > max_sim:
                sim_word = j
                max_sim = similarity
        df = df.replace(to_replace=i, value=sim_word)
    return df


def match_distribution(nr_sim, nr_total, probs):
    """ Calculate which pre-defined distribution is the closest to the distribution of the given list.

    :param nr_sim: the number of times the multinomial distribution is simulated.
    :param nr_total: the number of entries in our original DataFrame.
    :param probs: a list of probabilities for each unique entry to occur.
    :return: 'Ordinal' if the smallest distance value is contained within list{ordinal}, 'Categorical' otherwise.
    """
    # Calculate the distance between the average of 10 multinomial (categorical) distributions
    cat = [jensenshannon(probs, np.random.multinomial(nr_total, probs)) for _ in range(nr_sim)]
    avg_cat = sum(cat) / len(cat)

    # Calculate the distance between 14 pre-defined distributions that characterize ordinal data
    ordinal_distributions = OrdinalDistribution(len(probs))
    ordinal = [jensenshannon(probs, x) for x in list(vars(ordinal_distributions).values())]
    print(ordinal, avg_cat)
    # Return the name of the most fitting distribution
    if min(ordinal) < avg_cat:
        return 'Ordinal'
    else:
        return 'Categorical'


def run_inference(df):
    """ Call this method to run the statistical inference steps.

    :param df: a pandas DataFrame containing the string column.
    :return: a string indicating whether the inferred distribution is categorical or ordinal.
    """
    # Retrieve the number of unique entries in the DataFrame
    total = df.count().values[0]
    print('Number of unique entries:', total)

    # Retrieve the % occurrence for each unique entry
    unique_counts = df.value_counts(normalize=True, sort=False)
    print('Probabilities for each entry:\n', unique_counts)

    # Cluster similar values (to tackle outliers)
    clusters = generate_clusters(df)
    print('after clustering:\n', clusters)

    # Recalculate the % occurrence for each unique entry
    probs = clusters.value_counts(normalize=True, sort=False)
    print('Probabilities for clustered entries:\n', probs)

    # Calculate which pre-defined distribution comes closest to the given data distribution
    distribution = match_distribution(10, total, probs)
    print('Inferred distribution:', distribution)
    return distribution


# Test run #
test = ['John', 'John', 'john', 'joh', 'John', 'Johnny', 'Jon', 'Stef', 'Stef', 'Stef', 'staf', 'sted',
        'stf', 'stofn', 'nick', 'Nick', 'dick', 'jick', 'sick', 'Nick', 'Nick', 'Nicky', 'Niuck']
# test_df = pd.DataFrame(test)
# print(test_df)
# run_inference(test_df)
