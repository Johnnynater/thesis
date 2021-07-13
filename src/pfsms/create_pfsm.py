from ptype.Machine import Machine
from greenery.lego import escapes


# Add extra escaped characters to the regex library
escapes['\('] = '\\('
escapes['\)'] = '\\)'
escapes['\''] = "\\'"

# Define more special characters otherwise there will be many anomalies
spec_char = 'áéíóúýàèìòùäëïöüÿâêîôûãñõÁÉÍÓÚÝÀÈÌÒÙÄËÏÖÜŸÂÊÎÔÛÃÑÕŠšŽž$¢£€¥¤©®™—–’“”`~@#%\'"¡¿'


# Coordinate PFSM inspired by https://www.sunearthtools.com/dp/tools/conversion.php
class Coordinate(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(
            r'((([NSZ](([0-8][0-9]|[0-9])((\.|:)[0-5]\d){2}|90((\.|:)00){2}))?\s?'
            r'([EOW](([\d]{1,2}|(0\d\d|1[0-7]\d))((\.|:)[0-5]\d){2}|180((\.|:)00){2}))?){1})|'      # N89.49.00
            r'(( *\d{1,2}((\.|,)\d+)?(\*|°|º|º|\\?\'|"|´|:|d|m|s) *){3}[NSZEOW] *){1,2}'            # 20º 19' 2.333" W
        )
        self.create_T_new()
        self.copy_to_z()


class Day(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(
            r'(((M|m)on|(T|t)ues|(W|w)ednes|(T|t)hurs|(F|f)ri|(S|s)atur|(S|s)un)(day)?|'
            r'(M|m)o|(T|t)ue?|(W|w)ed?|(T|t)hu?|(F|f)r|(S|s)at?|(S|s)u)'
        )
        self.create_T_new()
        self.copy_to_z()


class Email(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)')
        self.create_T_new()
        self.copy_to_z()


class Filepath(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(
            r'(([a-zA-Z]:(\\|/))|(\\|/))?'
            r'((\\|/)[\w.\-_\\/\[\]\(\) ]+|[\w.\-_\\/\[\]\(\) ]+(\\|/))+'
            r'([\w.\-_\\/\[\]\(\) ]+(\.\w+)?)?'
        )
        self.create_T_new()
        self.copy_to_z()


class Month(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(
            r'((\'?\d{1,4})[,.\-_ ]*)?((J|j)an(uary)?|(F|f)eb(ruary)?|(M|m)ar(ch)?|'
            r'(A|a)pr(il)?|(M|m)ay|(J|j)une?|(J|j)uly?|(A|a)ug(ust)?|(S|s)ep(tember)?|'
            r'(O|o)ct(ober)?|((N|n)ov|(D|d)ec)(ember)?)([,.\-_ \']*(\d{1,4}))?(, (\d{1,4}))?'
        )
        self.create_T_new()
        self.copy_to_z()


# PFSM to handle integers including special characters, e.g.: 100-500, <10, >10, 100+, $100
class Numerical(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(
            r"([0-9]+ ?[\-+_/:;&'(to)] ?[0-9]+)|"
            r"(([<>$#@%=]+|((L|l)ess|(L|l)ower|(G|g)reater|(H|h)igher) than"
            r"|((U|u)nder|(B|b)elow|(O|o)ver|(A|a)bove))[ \-]?[0-9]+)|([0-9]+ ?[<>+$%=]+)"
        )
        self.create_T_new()
        self.copy_to_z()


class OrdinalNumbers(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(r'[\s\w\-]+(st|nd|rd|th)')
        self.create_T_new()
        self.copy_to_z()


class Sentence(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(
            r'.*\s*([\w&.,;:?!/\-\s\(\){}]+[\s+.,;:?!%—–]+){{5,}}'
            r'([\w&.,;:?!/\-\(\){}]*|\s)*.*'.format(spec_char, spec_char)
        )
        self.create_T_new()
        self.copy_to_z()


class URL(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(
            r'((http|https|ftp)://){0,1}[a-zA-Z0-9\-.]+\.[a-zA-Z]{2,3}'
            r'(:[a-zA-Z0-9]*)?/?([a-zA-Z0-9\-._?,/\\+&amp;%$#=~])*'
        )
        self.create_T_new()
        self.copy_to_z()


# Zipcode pfsm. See https://stackoverflow.com/questions/578406/what-is-the-ultimate-postal-code-and-zip-regex
class Zipcode(Machine):
    def __init__(self):
        super().__init__()
        self.STOP_P = 1e-4
        self.pfsm_from_fsm(
            r'(GIR[ ]?0AA|((AB|AL|B|BA|BB|BD|BH|BL|BN|BR|BS|BT|CA|CB|CF|CH|CM|CO|CR|CT|CV|CW|DA|DD|DE'
            r'|DG|DH|DL|DN|DT|DY|E|EC|EH|EN|EX|FK|FY|G|GL|GY|GU|HA|HD|HG|HP|HR|HS|HU|HX|IG|IM|IP|IV|JE'
            r'|KA|KT|KW|KY|L|LA|LD|LE|LL|LN|LS|LU|M|ME|MK|ML|N|NE|NG|NN|NP|NR|NW|OL|OX|PA|PE|PH|PL|PO'
            r'|PR|RG|RH|RM|S|SA|SE|SG|SK|SL|SM|SN|SO|SP|SR|SS|ST|SW|SY|TA|TD|TF|TN|TQ|TR|TS|TW|UB|W|WA'
            r'|WC|WD|WF|WN|WR|WS|WV|YO|ZE)(\d[\dA-Z]?[ ]?\d[ABD-HJLN-UW-Z]{2}))|BFPO[ ]?\d{1,4})|'
            r'((JE|GY|IM)\d[\dA-Z]?[ ]?\d[ABD-HJLN-UW-Z]{2})|'
            r'([ABCEGHJKLMNPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ ]?\d[ABCEGHJ-NPRSTV-Z]\d)|'
            r'(\d{4}[ ]?[A-Z]{2})|'
            r'(AD\d{3})|'
            r'(([A-HJ-NP-Z])?\d{4}([A-Z]{3})?)|'
            r'(BB\d{5})|'
            r'([A-Z]{2}[ ]?[A-Z0-9]{2})|'
            r'(BBND 1ZZ)|'
            r'([A-Z]{2}[ ]?\d{4})|'
            r'([A-Z]\d{4}[A-Z]|(?:[A-Z]{2})\d{6})|'
            r'([A-Z]{3}[ ]?\d{2,4})|'
            r'((PC )?\d{3})|'
            r'(FIQQ 1ZZ)|'
            r'(SIQQ 1ZZ)|'
            r'(PCRN 1ZZ)|'
            r'((ASCN|STHL) 1ZZ)|'
            r'([HLMS]\d{3})|'
            r'(TKCA 1ZZ)|'
            r'(\d{3}[A-Z]{2}\d{3})'
        )
        self.create_T_new()
        self.copy_to_z()
