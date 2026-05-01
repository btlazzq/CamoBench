from label_text.label_prompt import DrugsPrompt

from label_word.origin_word.LabelWordList import DRUGS_LIST

from utils.JsonParser import parse_llm_json

def getText(client, model, prompt_number, information):

    if isinstance(DRUGS_LIST, dict):

        parts = []

        for category, sub_list in DRUGS_LIST.items():

            part = f"{category}：" + "、".join(sub_list)

            parts.append(part)

        words = "；".join(parts)

    else:

        words_list = DRUGS_LIST

        words = "、".join(words_list)

    if prompt_number == 5:

        user_prompt = getattr(DrugsPrompt, f"user_prompt_{prompt_number}").format(words=words, **information)

    else:

        user_prompt = getattr(DrugsPrompt, f"user_prompt_{prompt_number}").format(words=words)

    messages = [

        {"role": "system", "content": getattr(DrugsPrompt, f"system_prompt_{prompt_number}")},

        {"role": "user", "content": user_prompt}

    ]

    response = client.chat.completions.create(

        model=model,

        messages=messages

    )

    return parse_llm_json(response.choices[0].message.content)
