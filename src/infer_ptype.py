from ptype.Ptype import Ptype


def infer_ptype(data):
    ptype = Ptype()
    schema = ptype.schema_fit(data)
    return schema.transform(schema)
