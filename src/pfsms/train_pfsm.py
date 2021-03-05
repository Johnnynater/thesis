import pandas as pd
import pickle as pkl
from src.pfsms import create_pfsm
from ptype.Trainer import Trainer
from ptype.Ptype import Ptype

train_email = [
    r'email@example.com',
    r'firstname.lastname@example.com',
    r'email@subdomain.example.com',
    r'firstname+lastname@example.com',
    r'email@123.123.123.123',
    r'email@[123.123.123.123]',
    r'"email"@example.com',
    r'1234567890@example.com',
    r'email@example-one.com',
    r'_______@example.com',
    r'email@example.name',
    r'email@example.museum',
    r'email@example.co.jp',
    r'firstname-lastname@example.com',
    r'much.”more\ unusual”@example.com',
    r'very.unusual.”@”.unusual.com@example.com',
    r'very.”(),:;<>[]”.VERY.”very@\\ "very”.unusual@strange.example.com'
]
column_email = ['email']
test_email = ['Johnny1250@hotmail.com', 'jlith1997@gmail.com', 'nickh2olaat@goog.de', 'stefgoogle.com', 'NA', 'NA']

train_sentence = [
    r"An Irishman named O'Malley went to his doctor after a long illness.",
    r"the doctor, after a lengthy examination, sighed and looked O'Malley in the eye and said,",
    r"I've some bad news for you. You have cancer, and it can't be cured, you'd best put your affairs in order.",
    r"oMalley was shocked and saddened; but of solid character, he managed to compose himself and walk from the doctor's office into the waiting room. ",
    r"To his son who had been waiting, O'Malley said, ",
    r"well son. We Irish celebrate when things are good, and we celebrate when things don't go so well. In this case, things aren't so well. I have cancer.",
    r" Let's head for the pub and have a few pints ",
    r"after 3 or 4 pints, the two were feeling a little less somber. There were some laughs and more beers.",
    r"They were eventually approached by some of O'Malley's old friends who asked what the two were celebrating.",
    r"o'Malley told them that the Irish celebrate the good and the bad",
    r"He went on to tell them that they were drinking to his impending end",
    r"he told his friends, I have been diagnosed with AIDS.",
    r"The friends gave O'Malley their condolences, and they had a couple more beers. ",
    r"ffter his friends left, O'Malley's son whispered his confusion."
]
column_sentence = ['sentence']

test_sentence = [
    "just came across a counterparty that needs to be flipped to the FT-US/CAND-ERMS book for all deals in Houston books.",
    "ECC has now signed a Master Swap Agreement with Aquila Canada Corp., and all deals traded with ENA",
    "and Aquila Canada Corp. need to be settled with ECC in the P& book.",
    " These deals are showing up in TAGG as settling with ENA, could you all please keep an eye out for deals with this company and flip them right away.",
    "Please let me know if you have any questions on this.",
    "Thanks"
]


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
    print(schema.show())


def save_machines(names):
    """ Save the trained machines in a separate .obj file.

    :param names: a list of names of each Machine class to be saved.
    :return:
    """
    for name in names:
        file_machine = open('pfsms/trained_machines/%s.obj' % name, 'wb')
        pkl.dump(ptype.machines.forType[name], file_machine)


def load_machines(names):
    """ Load a (set of) machine(s) from one or more .obj files.

    :param names: a list of names of each Machine class to be loaded.
    :return: a list of Machine classes
    """
    loaded_machines = []
    for name in names:
        loaded_machines.append(pkl.load(open('pfsms/trained_machines/%s.obj' % name, 'rb')))
    return loaded_machines

# TODO: Coordinate, Day, Filepath, Month, OrdinalNumbers, URL
# Instantiate Ptype class
ptype = Ptype()

# Load up the machines to train/test
machines = [create_pfsm.Email(), create_pfsm.Sentence()]
names = ['email', 'sentence']

setup_machines(machines, names)

# In case an existing machine causes trouble while testing custom machines: uncomment these lines
# ptype.machines.types.remove('string')
# del ptype.machines.forType['string']

train_machines([train_email, train_sentence], [column_email, column_sentence], 20)
test_machines([test_email, test_sentence])
# save_machines(names)
