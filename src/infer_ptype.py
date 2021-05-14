from ptype.Ptype import Ptype
import pickle as pkl


def infer(data):
    ptype = Ptype()
    # Adjust the Strings pfsm s.t. it can tolerate account names starting with '@' and tags starting with '#'
    ptype.machines.forType['string'].initialize(reg_exp="[@#]*[a-zA-Z0-9 .,\-_%:;]+ ?")
    names = ['coordinate', 'day', 'email', 'filepath', 'month', 'numerical', 'ordinal', 'sentence', 'url', 'zipcode']
    for name in names:
        machine = pkl.load(open('pfsms/trained_machines/%s.obj' % name, 'rb'))
        ptype.types.append(name)
        ptype.machines.forType[name] = machine
    return ptype.schema_fit(data), names
