import flair
import re
import numpy as np
from dirty_cat import SimilarityEncoder, MinHashEncoder, GapEncoder
from sklearn.preprocessing import OrdinalEncoder
from nltk.corpus import wordnet as wn
# from nltk.stem.wordnet import WordNetLemmatizer


def get_word_variations(word):
    variations = {
        'antonyms': set(),
        'superlatives': set(),
        # 'large quantifiers': set(),
        # 'small quantifiers': set()
    }
    # Make sure to separate words if there is more than one
    # words = [' '.join(re.split('[ \-_]', word)) for word in words]
    # print(words)

    # for word in words:
        # Retrieve antonyms
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                variations['antonyms'].add(lemma.antonyms()[0].name())

    # Retrieve or generate superlatives
    for wordset in superlatives:
        if word in wordset:
            for item in wordset:
                if item != word:
                    variations['superlatives'].add(item)
        else:
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
    # # Generate quantifiers
    # for quant in small_quantifiers:
    #     variations['small quantifiers'].add(quant + ' ' + word)
    # for quant in large_quantifiers:
    #     variations['large quantifiers'].add(quant + ' ' + word)
    return variations


def determine_order_flair(values):
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


def determine_order_heur(values):
    order, ant_order = [], []
    ant = ''
    values_clean = [' '.join(re.split('[ \-_]', val)) for val in values]
    val_mapping = {x: y for x, y in zip(values_clean, values)}
    for val in values_clean:
        variations = get_word_variations(val)
        print(variations)
        order.append(val)
        values_clean.remove(val)
        for antonym in variations['antonyms']:
            if antonym in values_clean:
                ant = antonym
                ant_order.append(antonym)
                values_clean.remove(antonym)
        for superlative in variations['superlatives']:
            if superlative in values_clean:
                order.append(superlative)
                values_clean.remove(superlative)
        for quant in small_quantifiers:
            quant_val = '{} {}'.format(quant, val)
            if quant_val in values_clean:
                order.append(quant_val)
                values_clean.remove(quant_val)
            quant_ant = '{} {}'.format(quant, ant)
            if quant_ant in values_clean:
                ant_order.insert(0, quant_ant)
                values_clean.remove(quant_ant)
        for quant in large_quantifiers:
            quant_val = '{} {}'.format(quant, val)
            if quant_val in values_clean:
                order.insert(0, quant_val)
                values_clean.remove(quant_val)
            quant_ant = '{} {}'.format(quant, ant)
            if quant_ant in values_clean:
                ant_order.append(quant_ant)
                values_clean.remove(quant_ant)
    ant_order.extend(order)
    final_order = [val_mapping[x] for x in ant_order]
    return final_order


def run(df, column, encode_type, dense):
    """ Run the heuristics on string encoding.

    :param column: a pandas DataFrame consisting of the column to be encoded.
    :param encode_type: an Integer indicating whether the column needs ordinal or nominal encoding.
    :param dense: a Boolean indicating whether the encoded values are fitted in one column or in multiple.
    :return: a Dictionary consisting of the mappings String -> Float/List.
    """
    if encode_type == 0:
        # Ordinal encoding required. Determine the order and encode accordingly
        unique_entries = [item for item, _ in column.value_counts().iteritems()]
        # TODO: consider LSTM/heuristics hybrid
        order = determine_order_flair(unique_entries)
        enc = OrdinalEncoder(categories=[order])
    else:
        if column.value_counts().count() < 30:
            # Data has low cardinality. Encode using SimilarityEncoder
            enc = SimilarityEncoder()
        elif column.value_counts().count() < 100:
            enc = MinHashEncoder()
        else:
            # Data has high cardinality. Encode using GapEncoder
            enc = GapEncoder()

    if dense or encode_type == 0:
        return dict({
            x: (y[0] if encode_type == 0 else y) for x, y in zip(column, enc.fit_transform(column.to_frame()))
        })
    else:
        # Create a column for each dimension and return the resulting DataFrame
        encoding = enc.fit_transform(np.array(column).reshape((-1, 1)))
        df = df.drop(columns=column.name)
        dim_len = len(encoding[0])
        encoding = pd.DataFrame([encoding[i] for i in range(len(encoding))])
        encoding = encoding.rename(columns={i: column.name + str(i) for i in range(dim_len)})
        return pd.concat([df, encoding], axis=1)


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
    'enough',
    'very',
    'extremely'
]

small_quantifiers = [
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
import pandas as pd
# print(determine_order_heur(['very good', 'good', 'bad', 'very bad']))
# column = pd.Series(['<100', 'over 600', 'over 600', '100-300', '300-600'])
# cool1 = [item for item, _ in column.value_counts().iteritems()]
# cool2 = list(column.values.flatten())
# print(cool1)
# print(cool2)
# print(run(column, 1))
