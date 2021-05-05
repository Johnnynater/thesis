from ptype.Ptype import Ptype
import pickle as pkl


def infer(data):
    ptype = Ptype()
    names = ['coordinate', 'day', 'email', 'filepath', 'month', 'numerical', 'ordinal', 'sentence', 'url', 'zipcode']
    for name in names:
        machine = pkl.load(open('pfsms/trained_machines/%s.obj' % name, 'rb'))
        ptype.types.append(name)
        ptype.machines.forType[name] = machine
    return ptype.schema_fit(data), names
