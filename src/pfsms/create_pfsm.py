from ptype.Machine import Machine


class Coordinate(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'[NSZ]([0-8][0-9](\.[0-5]\d){2}|90(\.00){2})\040[EOW]((0\d\d|1[0-7]\d)(\.[0-5]\d){2}|180(\.00){2})')
        self.create_T_new()
        self.copy_to_z()
        # print(self.T)


class Day(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Mo|Tue|Tu|Wed|We|Thu|Th|Fri|Fr|Sat|Sa|Sun|Su)')
        self.create_T_new()
        self.copy_to_z()
        # print(self.T)


class Email(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)')
        self.create_T_new()
        self.copy_to_z()
        # print(self.T)


class Filepath(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'((?:[^/]*/){1})(.*)')
        self.create_T_new()
        self.copy_to_z()
        # print(self.T)


class Month(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'(January|February|March|April|May|June|July|August|September|October|November|December|'
                           r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)')
        self.create_T_new()
        self.copy_to_z()
        # print(self.T)


class OrdinalNumbers(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'(?i)[^ ]+(st|nd|rd|th)\b')
        self.create_T_new()
        self.copy_to_z()
        # print(self.T)


class Sentence(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'\s*[\w&.,?!/\-|\s]+[\s+|.,?!]+[\w&.,?!/\-|\s]*')
        self.create_T_new()
        self.copy_to_z()
        # print(self.T)


class URL(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'((http|https|ftp)\://){0,1}[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}'
                           r'(:[a-zA-Z0-9]*)?/?([a-zA-Z0-9\-\._\?\,\'/\\\+&amp;%\$#\=~])*')
        self.create_T_new()
        self.copy_to_z()
        # print(self.T)
