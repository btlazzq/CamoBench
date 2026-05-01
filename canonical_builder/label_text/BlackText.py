from label_text.label_prompt import BlackPrompt

from label_word.origin_word.LabelWordList import BLACK_LIST

from utils.JsonParser import parse_llm_json

def getText(client, model, prompt_number, information):

    if isinstance(BLACK_LIST, dict):

        parts = []

        for category, sub_list in BLACK_LIST.items():

            part = f"{category}：" + "、".join(sub_list)

            parts.append(part)

        words = "；".join(parts)

    else:

        words_list = BLACK_LIST

        words = "、".join(words_list)

    if prompt_number == 5:

        user_prompt = getattr(BlackPrompt, f"user_prompt_{prompt_number}").format(words=words, **information)

    else:

        user_prompt = getattr(BlackPrompt, f"user_prompt_{prompt_number}").format(words=words)

    messages = [

        {"role": "system", "content": getattr(BlackPrompt, f"system_prompt_{prompt_number}")},

        {"role": "user", "content": user_prompt}

    ]

    response = client.chat.completions.create(

        model=model,

        messages=messages

    )

    return parse_llm_json(response.choices[0].message.content)
