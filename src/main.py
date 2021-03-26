from src import infer_ptype, infer_stattype, heuristics
import pandas as pd


def load_dataset():
    pass


def inference_ptype(data):
    return infer_ptype.infer_ptype(data)


def inference_statistical_type(data):
    result = []
    for col in data:
        result.append(infer_stattype.run_inference(data[col].to_frame()))
    return result


def inference_heuristics(data):
    return heuristics.run_heuristics(data)


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
    # pfsm = src.selfmade_pfsm.PFSM(r'([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)')
    # pfsm.create_pfsm()

    # Load in the dataset
    # TODO: when we create a callable method we will probably require it to have a data param, so this won't be needed
    load_dataset()
    data = pd.read_csv('datasets/aug_train.csv')

    # Infer data type
    schema, names = inference_ptype(data)
    names.append('string')
    datatypes = [col.type for _, col in schema.cols.items()]
    print(schema.show().to_string())
    print(names)

    # Take the columns that were inferred as a string (feature)
    string_cols = data.iloc[:, [i for i in range(len(datatypes)) if datatypes[i] in names]]
    print(string_cols)

    # If cannot infer using given PFSMs, infer categorical / ordinal / continuous
    print(inference_statistical_type(string_cols))

    # Additionally: try to infer the type using heuristics
    inference_heuristics(string_cols)

    # Remove or repair any detected outliers
    handle_outliers()

    # Fill in any missing values
    handle_missing()

    # Encode (if desired)
    apply_encoding()

    # Output the cleaned dataset
    output_dataset()
