from ptype.Ptype import Ptype

# TODO: check if we need this at all
def infer_ptype(data):
    ptype = Ptype()
    schema = ptype.schema_fit(data)
    return schema.transform(schema)
