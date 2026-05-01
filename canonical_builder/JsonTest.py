from utils.JsonParser import parse_llm_json

from utils.MultiFunctionalJsonParser import parse_list_json

content = r"""

}]"""

json_text, json_obj = parse_list_json(content)

print("json_text")

print(json_text)

print("json_obj")

print(json_obj)
