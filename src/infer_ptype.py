from ptype.Ptype import Ptype
import pickle as pkl


def infer_ptype(data):
    ptype = Ptype()
    names = ['email', 'sentence', 'coordinate', 'day', 'filepath', 'month', 'ordinal', 'url']
    for name in names:
        machine = pkl.load(open('pfsms/trained_machines/%s.obj' % name, 'rb'))
        ptype.types.append(name)
        ptype.machines.forType[name] = machine
    return ptype.schema_fit(data), names
