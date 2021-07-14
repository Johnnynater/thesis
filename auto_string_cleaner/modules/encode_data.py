import flair
import numpy as np
import pandas as pd
from dirty_cat import SimilarityEncoder, MinHashEncoder, GapEncoder
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder


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


def run(df, y, column, encode_type, dense, balanced):
    """ Run the heuristics on string encoding.

    :param df: a pandas DataFrame consisting of the data.
    :param y: a pandas DataFrame consisting of target values.
    :param column: a pandas DataFrame consisting of the column to be encoded.
    :param encode_type: an Integer indicating whether the column needs ordinal or nominal encoding.
    :param dense: a Boolean indicating whether the encoded values are fitted in one column or in multiple.
    :param balanced: a Boolean indicating whether the target classes are balanced.
    :return: a Dictionary consisting of the mappings String -> Float/List.
    """
    if encode_type == 0:
        # Ordinal encoding required. Determine the order and encode accordingly
        unique_entries = [item for item, _ in column.value_counts().iteritems()]
        order = determine_order_flair(unique_entries)
        enc = OrdinalEncoder(categories=[order])
    else:
        if column.value_counts().count() < 30:
            # Data has low cardinality. Check the class balance
            if balanced:
                # Encode using TargetEncoder
                enc = TargetEncoder()
            else:
                # Encode using SimilarityEncoder
                enc = SimilarityEncoder()
        elif column.value_counts().count() < 100:
            # Data has medium to high cardinality. Encode using GapEncoder
            enc = GapEncoder()
        else:
            # Data has high cardinality. Encode using MinHashEncoder
            enc = MinHashEncoder()

    if encode_type == 1 and column.value_counts().count() < 30 and balanced:
        return enc.fit_transform(column, y)
    elif encode_type == 1 and not dense:
        # Create a column for each dimension and return the resulting DataFrame
        encoding = enc.fit_transform(np.array(column).reshape((-1, 1)))
        df = df.drop(columns=column.name)
        dim_len = len(encoding[0])
        encoding = pd.DataFrame([encoding[i] for i in range(len(encoding))])
        encoding = encoding.rename(columns={i: column.name + str(i) for i in range(dim_len)})
        return pd.concat([df, encoding], axis=1)
    else:
        return dict({
            x: (val[0] if encode_type == 0 else val) for x, val in zip(column, enc.fit_transform(column.to_frame()))
        })
