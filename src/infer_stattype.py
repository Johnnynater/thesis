import pickle as pkl


def infer(data):
    model = pkl.load(open('gbc/trained_gbc.sav', 'rb'))
    return model.predict(data)
