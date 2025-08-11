import json, sys
from jsonschema import validate, Draft202012Validator
schema = json.load(open("data/schemas/manifest.schema.json"))
doc    = json.load(open(sys.argv[1]))
Draft202012Validator.check_schema(schema)
validate(instance=doc, schema=schema)
print("OK", sys.argv[1])
