import pickle as pkl


def infer(data):
    """ Run statistical type inference using a pre-trained GradientBoostingClassifier.

    :param data: a List of Lists, where each List contains eight features extracted from heuristics.py of each
                 standard string column.
    :return: a List of predicted statistical types for each standard string column.
    """
    model = pkl.load(open('gbc/trained_gbc.sav', 'rb'))
    return model.predict(data)
