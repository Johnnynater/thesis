import pickle as pkl
import os


def infer(data):
    """ Run statistical type inference using a pre-trained GradientBoostingClassifier.

    :param data: a List of Lists, where each List contains eight features extracted from extract_features.py of each
                 standard string column.
    :return: a List of predicted statistical types for each standard string column.
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'gbc', 'trained_gbc.sav')
    model = pkl.load(open(filename, 'rb'))
    return model.predict(data)
