import flair
import re
import numpy as np
from dirty_cat import SimilarityEncoder, MinHashEncoder, GapEncoder
from sklearn.preprocessing import OrdinalEncoder
from nltk.corpus import wordnet as wn
# from nltk.stem.wordnet import WordNetLemmatizer


def get_word_variations(words):
    variations = {
        'antonyms': set(),
        'superlatives': set()
    }
    # Make sure to separate words if there is more than one
    words = re.split('[ \-_]', words)

    for word in words:
        # Retrieve antonyms
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    variations['antonyms'].add(lemma.antonyms()[0].name())

        # Retrieve or generate superlatives
        for wordset in superlatives:
            if word in wordset[0]:
                variations['superlatives'].add(wordset[1])
                variations['superlatives'].add(wordset[2])
            elif word in wordset[1]:
                variations['superlatives'].add(wordset[0])
                variations['superlatives'].add(wordset[2])
            elif word in wordset[2]:
                variations['superlatives'].add(wordset[0])
                variations['superlatives'].add(wordset[1])
            elif word not in wordset:
                # Disclaimer: this will return both correct words and typo words, but this is not a problem.
                if word[-1] == 'e':
                    variations['superlatives'].add(word + 'r')
                    variations['superlatives'].add(word + 'st')
                elif word[-1] == 'y':
                    variations['superlatives'].add(word[:-1] + 'ier')
                    variations['superlatives'].add(word[:-1] + 'iest')
                else:
                    variations['superlatives'].add(word + 'er')
                    variations['superlatives'].add(word + 'est')
                    variations['superlatives'].add(word + word[-1] + 'er')
                    variations['superlatives'].add(word + word[-1] + 'est')
                variations['superlatives'].add('more ' + word)
                variations['superlatives'].add('most ' + word)
    return variations


def determine_order(values):
    """ Determine the word order based on flair's LSTM.

    :param values: a pandas DataFrame consisting of the unique values of the column.
    :return: a list containing the determined order.
    """
    # Check "Notes 19-04.docx" initial idea for implementation details
    # FOR NOW: use flair to determine word order
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    results = []
    for value in values:
        # variations = get_word_variations(value)
        s = flair.data.Sentence(value)
        flair_sentiment.predict(s)
        total_sentiment = s.labels
        result = (value, -total_sentiment[0].score) \
            if total_sentiment[0].value == 'NEGATIVE' \
            else (value, total_sentiment[0].score)
        results.append(result)
    results.sort(key=lambda x: x[1])
    results = [results[i][0] for i in range(len(results))]
    return results


def run(column, encode_type):
    """ Run the heuristics on string encoding.

    :param column: a pandas DataFrame consisting of the column to be encoded.
    :param encode_type: an Integer indicating whether the column needs ordinal or nominal encoding.
    :return: a Dictionary consisting of the mappings String -> Float/List.
    """
    if encode_type == 0:
        # Ordinal encoding required. Determine the order and encode accordingly
        unique_entries = [item for item, _ in column.value_counts().iteritems()]
        # TODO: consider LSTM/heuristics hybrid
        order = determine_order(unique_entries)
        enc = OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=np.nan)
    else:
        if column.value_counts().count() < 30:
            # Data has low cardinality. Encode using SimilarityEncoder
            enc = SimilarityEncoder()
        else:
            # Data has high cardinality. Encode using GapEncoder
            enc = GapEncoder()
    return dict({
        x: (y[0] if encode_type == 0 else y) for x, y in zip(column, enc.fit_transform(column.to_frame()))
    })


# List of all types of quantifiers (split between little and large), taken from
# https://linguapress.com/grammar/quantifiers.htm
large_quantifiers = [
    'much',
    'many',
    'lots',
    'plenty',
    'numerous',
    'a large number of',
    'most',
    'enough'
]

little_quantifiers = [
    'not much',
    'not many',
    'few',
    'little',
    'a small number of'
]

# List of irregular superlatives, taken from
# https://www.curso-ingles.com/en/resources/cheat-sheets/adjectives/list-of-comparatives-and-superlatives
superlatives = [
    ['good', 'better', 'best'],
    ['bad', 'worse', 'worst'],
    ['much', 'many', 'more', 'most'],
    ['far', 'farther', 'further', 'farthest', 'furthest'],
    ['less', 'lesser', 'least'],
    ['well', 'better', 'best']
]
# import pandas as pd
# # print(determine_order(['better', 'best', 'good']))
# column = pd.Series(['<100', 'over 600', 'over 600', '100-300', '300-600'])
# cool1 = [item for item, _ in column.value_counts().iteritems()]
# cool2 = list(column.values.flatten())
# print(cool1)
# print(cool2)
# print(run(column, 1))
