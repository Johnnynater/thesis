from ptype.Ptype import Ptype
from .pfsms import create_pfsm


def infer(df):
    """ Infer data and string feature types for a given dataset using ptype.

    :param df: a pandas DataFrame.
    :return: a ptype schema containing information such as data or string feature types, missing values, and outliers;
             a List containing the names of the string feature types that can be explicitly inferred.
    """
    ptype = Ptype()
    # Adjust the Strings pfsm s.t. it can tolerate account names starting with '@' and tags starting with '#'
    ptype.machines.forType['string'].initialize(reg_exp="[@#]*[a-zA-Z0-9 .,\\\-_%:;&]+ ?")
    names = ['coordinate', 'day', 'email', 'filepath', 'month', 'numerical', 'sentence', 'url', 'zipcode']
    machines = [
        create_pfsm.Coordinate(), create_pfsm.Day(), create_pfsm.Email(),
        create_pfsm.Filepath(), create_pfsm.Month(), create_pfsm.Numerical(),
        create_pfsm.Sentence(), create_pfsm.URL(), create_pfsm.Zipcode(),
    ]
    for name, machine in zip(names, machines):
        ptype.types.append(name)
        ptype.machines.forType[name] = machine
    return ptype.schema_fit(df), names
