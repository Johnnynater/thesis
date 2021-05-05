import pandas as pd
import pickle as pkl
from src.pfsms import create_pfsm
from ptype.Trainer import Trainer
from ptype.Ptype import Ptype


def load_data(names):
    """ Load train and test data from .csv files.

    :param names: a list of substrings of each file.
    :return: lists train_data, test_data, and column_labels.
    """
    train_data, test_data, column_labels = [], [], []
    for name in names:
        # Applying temp and removing the last char is needed to remove \n for each entry
        tmp = list(open('datasets/%s_train.csv' % name))
        train_data.append([tmp[i][:-1] for i in range(len(tmp))])

        tmp = list(open('datasets/%s_test.csv' % name))
        test_data.append([tmp[i][:-1] for i in range(len(tmp))])

        column_labels.append([name])
    return train_data, test_data, column_labels


def load_machines(names):
    """ Load a (set of) machine(s) from one or more .obj files.

    :param names: a list of names of each Machine class to be loaded.
    :return: a list of Machine classes.
    """
    loaded_machines = []
    for name in names:
        loaded_machines.append(pkl.load(open('pfsms/trained_machines/%s.obj' % name, 'rb')))
    return loaded_machines


def save_machines(names):
    """ Save the trained machines in a separate .obj file.

    :param names: a list of names of each Machine class to be saved.
    :return:
    """
    for name in names:
        file_machine = open('pfsms/trained_machines/%s.obj' % name, 'wb')
        pkl.dump(ptype.machines.forType[name], file_machine)


def print_params(uniformly, initial, final):
    print("\nuniformly is", uniformly)

    for i_machine, f_machine in zip(initial.machines, final.machines):
        if i_machine.states != []:

            if i_machine.I != f_machine.I:
                print("\tMachine is", i_machine)
                print("\tInitial I", i_machine.I)
                print("\tFinal I  ", f_machine.I, '\n')

            if i_machine.T != f_machine.T:
                print("\tMachine is", i_machine)
                print("\tT's are not the same (omitted as it's quite large)")

            if i_machine.F != f_machine.F:
                print("\tMachine is", i_machine)
                print("\tInitial F", i_machine.F)
                print("\tFinal F  ", f_machine.F, '\n')


def setup_machines(machines, names):
    """ Append the custom machine classes and their names into the ptype class.

    :param machines: a list of Machine classes from create_pfsm.
    :param names: a list of names of each machine.
    :return:
    """
    for machine, name in zip(machines, names):
        ptype.types.append(name)
        ptype.machines.forType[name] = machine
    print(ptype.machines.forType)


def train_machines(dataset, columns, epochs):
    """ Train (custom) machines using the Trainer in the ptype library.

    :param dataset: a list of data columns to train the machines on.
    :param columns: a list of correct labels of each data column in dataset.
    :param epochs: the number of iterations performed by the Trainer.
    :return:
    """
    for data, column in zip(dataset, columns):
        df_trainings, y_trains = [], []

        # Convert the training data into a pandas DataFrame object
        df_training = pd.DataFrame(data, dtype='str', columns=column)
        y_train = temp = [key + 1 for key, value in enumerate(ptype.types) if value == 'integer']

        # Append the training data and label to a list of lists
        df_trainings.append(df_training)
        y_trains.append(y_train)

        # Use the Trainer class from the ptype library to train our machines on the data
        trainer = Trainer(ptype.machines, df_trainings, y_trains)
        initial, final, training_error = trainer.train(epochs)

        # See whether the Trainer changed any transition probabilities of each trained machine
        print_params(False, initial, final)


def test_machines(data):
    """ Test whether (custom) machines can correctly infer the column types in the test set.

    :param data: a list of test data used for evaluation.
    :return:
    """
    test_set = pd.DataFrame({i: data[i] for i in range(len(data))})
    schema = ptype.schema_fit(test_set)
    print(schema.show().to_string())


# Instantiate Ptype class
ptype = Ptype()

# Load required data
names = ['numerical']#'coordinate', 'day', 'email', 'filepath', 'month', 'numerical', 'ordinal', 'sentence', 'url', 'zipcode']
train, test, columns = load_data(names)

# Create the machines to train/test
machines = [
    #create_pfsm.Coordinate(), create_pfsm.Day(),
    #create_pfsm.Email(), create_pfsm.Filepath(),
    #create_pfsm.Month(), create_pfsm.Numerical(),
    #create_pfsm.OrdinalNumbers(), create_pfsm.Sentence(),
    #create_pfsm.URL(), create_pfsm.Zipcode(),
create_pfsm.Numerical()
]

setup_machines(machines, names)

# In case an existing machine causes trouble while testing custom machines: uncomment these lines
# ptype.machines.types.remove('string')
# del ptype.machines.forType['string']

# train_machines(train, columns, 200)
test_machines(test)
save_machines(names)
