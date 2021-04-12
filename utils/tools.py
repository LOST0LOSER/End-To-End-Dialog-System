import zipfile
import ujson
from typing import get_type_hints
import jieba
import sys
import re

##############辅助函数##############
def readZippedJSON(filepath, filename) -> dict:
    archive = zipfile.ZipFile(filepath, 'r')
    return ujson.load(archive.open(filename))


def entityList(kb_path, task_id):
    type_dict = get_type_hints(kb_path, dstc2=(task_id == 6))
    entity_list = []
    for key in type_dict.keys():
        for value in type_dict[key]:
            entity_list.append(value)
    return entity_list


def delexicalize_da(da):
    delexicalized_da = []
    counter = {}
    for intent, domain, slot, value in da:
        if intent in ['Inform', 'Recommend']:
            key = '+'.join([intent, domain, slot])
            counter.setdefault(key, 0)
            counter[key] += 1
            delexicalized_da.append(key+'+'+str(counter[key]))
        else:
            delexicalized_da.append('+'.join([intent, domain, slot, value]))

    return delexicalized_da


def hasNumbers(inputString:str):
    return any(char.isdigit() for char in inputString)


def splitSentence(inputSentence:str) -> list:
    return jieba.lcut(inputSentence)

def addWordsToJieba(wordlist:list):
    for each in wordlist:
        jieba.add_word(each)

def removeBianDian(sentence):
    cleanSentence = re.sub("[\.\!\/_,$%^*+\"\']+|[+——！，。?？、~@·#￥%……&*（：）\-]+", "",sentence) 
    return cleanSentence


#进度条
def progress_bar(start,end,lossMsg=None):
    from math import floor
    fix = end/100
    fix*=2

    print("\r", end="")
    print("{} trainning progress: {}%: ".format(lossMsg,100*start//end), "▋ " * floor(start/fix), end="")
    sys.stdout.flush()
