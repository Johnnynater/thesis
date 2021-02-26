from src import selfmade_pfsm


def load_dataset():
    pass


def inference_ptype():
    pass


def inference_string_features():
    pass


def inference_statistical_type():
    pass


def inference_heuristics():
    pass


def handle_outliers():
    pass


def handle_missing():
    pass


def apply_encoding():
    pass


def output_dataset():
    pass


if __name__ == "__main__":
    # Selftest: create pfsm
    pfsm = selfmade_pfsm.PFSM(r'([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)')
    pfsm.create_pfsm()

    # Load in the dataset
    load_dataset()

    # Infer data type
    inference_ptype()

    # Infer string feature
    # First infer using constructed PFSMs
    inference_string_features()

    # If cannot infer using given PFSMs, infer categorical / ordinal / continuous
    inference_statistical_type()

    # Additionally: try to infer the type using heuristics
    inference_heuristics()

    # Remove or repair any detected outliers
    handle_outliers()

    # Fill in any missing values
    handle_missing()

    # Encode (if desired)
    apply_encoding()

    # Output the cleaned dataset
    output_dataset()
