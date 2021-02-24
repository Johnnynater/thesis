import pandas as pd
import numpy as np
from ptype.Machine import Machine
from ptype.Machines import Machines
from ptype.Trainer import Trainer
from ptype.Ptype import Ptype


class Email(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)')
        self.create_T_new()
        self.copy_to_z()
        print(self.T)


class Text(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'([A-Za-z&])\w*[.,?!|\s]+')
        self.create_T_new()
        self.copy_to_z()
        print(self.T)


x = [
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
column = 'email'


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

ptype = Ptype()
ptype.types.append("email")
ptype.machines.forType["email"] = Email()
print(ptype.types)
df_trainings, y_trains = [], []

df_training = pd.DataFrame(x, dtype='str', columns=[column])
y_train = temp = [key + 1 for key, value in enumerate(ptype.types) if value == 'integer']

df_trainings.append(df_training)
y_trains.append(y_train)

email = Email()
print(y_train)
print(df_training)

trainer = Trainer(ptype.machines, df_trainings, y_trains)
initial, final, training_error = trainer.train(200)
print_params(False, initial, final)
