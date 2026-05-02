import model_api.ModelParam as ModelParam
from openai import OpenAI
import random
from label_text.DrugsText import getText as getDrugs
from label_text.GamblingText import getText as getGambling
from label_text.PornText import getText as getPorn
from label_text.BlackText import getText as getBlack
from label_text.SwindleText import getText as getSwindle
from label_text.CurseText import getText as getCurse
from label_text.NoneText import getText as getNone
from label_word.origin_word import RandomSceneList as getInformation

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def createClient(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key)

def safe_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return "内容非法，拒绝生成！"

def getDrugsText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getDrugs, client, config["model"], prompt_number, information)

def getGamblingText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getGambling, client, config["model"], prompt_number, information)

def getPornText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getPorn, client, config["model"], prompt_number, information)

def getBlackText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getBlack, client, config["model"], prompt_number, information)

def getSwindleText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getSwindle, client, config["model"], prompt_number, information)

def getCurseText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getCurse, client, config["model"], prompt_number, information)

def getNoneText(model_name, prompt_number, information):
    config = getattr(ModelParam, model_name, None)
    client = createClient(config["base_url"], config["api_key"])
    return safe_call(getNone, client, config["model"], prompt_number, information)

def getRandomInformation():
    return {
        "sex": random.choice(getInformation.SEX),
        "age": random.choice(getInformation.AGE),
        "profession": random.choice(getInformation.PROFESSION),
        "education": random.choice(getInformation.EDUCATION),
        "platform": random.choice(getInformation.PLATFORM),
        "angle": random.choice(getInformation.ANGLE),
        "length": random.choice(getInformation.LENGTH)
    }

if __name__ == "__main__":
    model_name = "DEEPSEEK"
    prompt_number = 5

    for i in range(5):
        randomInformation = getRandomInformation()
        print(randomInformation)
        print(getDrugsText(model_name, prompt_number, randomInformation)[0])
        print(getGamblingText(model_name, prompt_number, randomInformation)[0])
        print(getPornText(model_name, prompt_number, randomInformation)[0])
        print(getBlackText(model_name, prompt_number, randomInformation)[0])
        print(getSwindleText(model_name, prompt_number, randomInformation)[0])
        print(getCurseText(model_name, prompt_number, randomInformation)[0])
        print(getNoneText(model_name, prompt_number, randomInformation)[0])