import numpy as np
from greenery.lego import parse


class PFSM:
    def __init__(self, regex):
        # self.regex = regex
        # self.states = states
        # self.init = init
        # self.final = final
        # self.trans = trans
        # self.alphabet = alphabet
        self.regex = regex
        self.states = []
        self.init = {}
        self.final = {}
        self.trans = {}
        self.alphabet = []
        self.stop_prob = 0.2

    def create_pfsm(self):
        # Use greenery to convert regex into a finite-state machine
        fsm = parse(self.regex).to_fsm()
        print(fsm)
        # Add the FSM alphabet to the PFSM alphabet. Remove 'anything_else' because this string is not desired
        self.alphabet = sorted([str(i) for i in list(fsm.alphabet) if str(i) != 'anything_else'])
        print('Alphabet:', self.alphabet)

        # Add the FSM state space to the PFSM state/transition space
        for state in list(fsm.states):
            if state not in self.states:
                self.states.append(state)
                self.trans[state] = {}
        print('States:', self.states)

        # Set the initial state(s)
        init_list = [np.log(1) if state == fsm.initial else -1e150 for state in self.states]
        self.init = {state: i for state, i in zip(self.states, init_list)}
        print('Initials:', self.init)

        # Set the final state(s)
        final_list = [np.log(1e-2) if state in list(fsm.finals) else -1e150 for state in self.states]
        self.final = {state: i for state, i in zip(self.states, final_list)}
        print('Finals:', self.final)

        # Set the probabilistic transition(s)
        for st_tr in fsm.map:
            transition = {
                symbol: v for symbol, v in fsm.map[st_tr].items() if str(symbol) != 'anything_else'
            }
            tr_values = np.array(list(transition.values()))
            if len(tr_values) == 0:
                self.final[st_tr] = 0.0
            else:
                tr_keys = np.array(list(transition.keys()))
                dividend = 1.0 if self.final[st_tr] == -1e150 else 1.0 - np.exp(self.final[st_tr])
                probabilities = np.array([dividend / len(tr_keys) for _ in tr_keys])

                # For each state, construct the probabilistic transitions
                for val in np.unique(tr_values):
                    index = np.where(tr_values == val)[0]
                    self.add_transitions(st_tr, val, list(tr_keys[index]), list(probabilities[index]), self.final)

        print(self.trans)

    def add_transitions(self, i, j, obs, probs, finals):
        for obs, prob in zip(obs, probs):
            if obs not in self.trans[i]:
                self.trans[i][obs] = {}
            # This turns it into entropy: np.log(prob)
            # TODO: fix for any final state instead of i == 5
            if i == 5:
                self.trans[i][obs][j] = (1 - self.stop_prob) / 66
            else:
                self.trans[i][obs][j] = np.log(prob)

            # for faster search later
            if obs not in self.alphabet:
                self.alphabet.append(obs)

    def train_pfsm(self):
        pass

    def save_pfsm(self):
        pass
