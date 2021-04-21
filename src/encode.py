from dirty_cat import SimilarityEncoder, MinHashEncoder, GapEncoder
from sklearn.preprocessing import OrdinalEncoder


def run(column, encode_type):
    if encode_type == 1:
        if column.value_counts().count() < 30:
            enc = SimilarityEncoder()
            return enc.fit_transform(column.values.reshape((-1, 1)))
        else:
            enc = GapEncoder()
            return enc.fit_transform(column)
    else:
        return OrdinalEncoder(column)
