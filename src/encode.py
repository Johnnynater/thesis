import re
import numpy as np
from dirty_cat import SimilarityEncoder, MinHashEncoder, GapEncoder
from sklearn.preprocessing import OrdinalEncoder
from nltk.corpus import wordnet as wn
# from nltk.stem.wordnet import WordNetLemmatizer

import flair


def get_word_variations(words):
    variations = {
        'antonyms': set(),
        'superlatives': set()
    }
    # Make sure to separate words if there is more than one
    words = re.split(' |-|_', words)

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

    :param values:
    :return: A list containing the determined order
    """
    # Check "Notes 19-04.docx" initial idea for implementation details
    # FOR NOW: use flair to determine word order
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    results = []
    for value in values:
        #variations = get_word_variations(value)
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
    if encode_type == 1:
        if column.value_counts().count() < 30:
            enc = SimilarityEncoder()
            return [x for x in enc.fit_transform(column.values.reshape((-1, 1)).astype(str))]
        else:
            enc = GapEncoder()
            return [x for x in enc.fit_transform(column.astype(str))]
    else:
        unique_entries = [item for item, _ in column.value_counts().iteritems()]
        order = determine_order(unique_entries)
        # order.append(0)
        enc = OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=np.nan)
        # column = column.fillna(value=0).to_frame()
        return enc.fit_transform(column.to_frame())


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

# print(determine_order(['better', 'best', 'good']))
