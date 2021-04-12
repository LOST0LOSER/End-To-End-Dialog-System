from logging import log
from os.path import join
from random import choice
from re import A

from numpy import True_, append, diag, number
from utils.config import *
import torch
from torch import Tensor
import torch.nn as NN
import torch.nn.functional as Fun
import torch.utils.data as Data
import torch.optim as OP
from torch.autograd import Variable
from torch.utils import data
from typing import Dict, ItemsView, Tuple, get_type_hints
import zipfile
import ujson
from math import floor
from utils.Dataset import *
from utils.DataBase import OBJECT_INDEX, SUBJECTS, PREDICATES, ENTITIES, SUBJECT_INDEX, generateAllSPO, getAttrsByEntity, getEntitesBySubject, patternEntity, patternSubject

import jieba
import jieba.posseg as pseg  # 词性标注
import jieba.analyse as anls  # 关键词提取

from utils.Dataset import Dataset, FastDataloader
from utils.tools import *
from utils.LanguageProcessUnit import LanguageProcessUnit


##############核心函数##############


def createNecessaryInputData(Tasks: dict, lang:LanguageProcessUnit ,entity=None) -> Tuple[list, int, int]:
    """
    return firstLayerProcessedData:list, max_len:int, max_ptr_len:int
    firstLayerProcessedData details:
    [0]: userInput splitWords
    [1]: sysReply sentence
    [2]: index_ptr_list (markup the wordIndex that the sysReply mentioned the word which user said)
    [3]: gate list
    max_len: the max input sequence length
    max_ptr_len:the max ptr list length 
    """
    firstLayerProcessedData = []
    dialog_counter = 0

    for taskID, item in Tasks.items():
        dialog_counter += 1
        # get item content
        user_wordlists = item['user']  # list
        sys_wordlists = item['sys']  # list
        request_KB_list = item['request_KB']  # dict
        inform_KB_list = item['inform_KB']  # dict

        STATEFIRST=1
        STATESECOND=2
        ents = []

        for userWordsList, sysWordsList in zip(user_wordlists, sys_wordlists):
            user_time, user_split_words = userWordsList
            sys_time, sys_sentence = sysWordsList

            # init
            KB_dict = {'request_KB': [], 'inform_KB': []}
            KB_dict['request_KB'] += request_KB_list[user_time]
            KB_dict['inform_KB'] += inform_KB_list[sys_time]
            sysSeq_ptr_list = []
            gate = []
            # ent = []
            conversation_arr = []

            # PAD 填充
            # for i, word in enumerate(user_split_words):
            #     user_split_words[i] = lang.fillPAD([word])
            # 
            # 这个还可以保持1维,深拷贝
            origin_user_split_words = list(user_split_words)
            splitSysSeqList = splitSentence(sys_sentence)

            user_split_words = lang.fillPADForInitWordsList(user_split_words)
            # SPO_list = generateAllSPO(user_split_words,sys_sentence)
            # user_split_words = SPO_list+user_split_words # list concat                

            # filling pad
            # for i,word in enumerate(userWordsList):
            #     userWordsList[i] = lang.fillPAD([word])
            ents = None
            end_index = len(user_split_words)

            
            # state 1
            subject = patternSubject(origin_user_split_words)
            if subject is not None:
                # ents will use for KB
                ents = getEntitesBySubject(subject)
                ents = [[subject,PAD_token,ent] for ent in ents]
                user_split_words = ents + user_split_words # [B;X]
                user_split_words_start_index = len(ents) # 连接后的起点
                for key in splitSysSeqList:
                    index_ptr_list = [index for index,val in enumerate(ents) if val[OBJECT_INDEX]==key]
                    if index_ptr_list:
                        index_ptr_list = max(index_ptr_list)
                        gate.append(1)
                    else:
                        index_ptr_list = [index for index, val in enumerate(user_split_words) if index >= user_split_words_start_index and val[SUBJECT_INDEX] == key]
                        if index_ptr_list:
                            index_ptr_list = max(index_ptr_list)
                            gate.append(1)
                        else:
                            index_ptr_list = end_index
                            gate.append(0)

                    sysSeq_ptr_list.append(index_ptr_list)
            
            else:
                # state 2
                ent = patternEntity(origin_user_split_words)
                if ent is not None:
                    attrs = getAttrsByEntity(ent)
                    # kb = [[ent,key,val] for key, val in attrs.items()]
                    kb = []
                    for key,val in attrs.items():
                        if isinstance(val,list): # info为数组情况
                            for v in val:
                                kb.append([ent,key,v])
                        elif val is not None:
                            kb.append([ent,key,val])
                    user_split_words = kb+user_split_words

                    user_split_words_start_index = len(kb)  # 起始地址
                    for key in splitSysSeqList:
                        index_ptr_list = [index for index,val in enumerate(kb) if val[OBJECT_INDEX]==key]
                        if index_ptr_list:
                            index_ptr_list = max(index_ptr_list)
                            gate.append(1)
                        else:
                            index_ptr_list = [index for index,
                                              val in enumerate(user_split_words) if index >= user_split_words_start_index and val[SUBJECT_INDEX] == key]
                            if index_ptr_list:
                                index_ptr_list = max(index_ptr_list)
                                gate.append(1)
                            else:
                                index_ptr_list = end_index
                                gate.append(0)

                        sysSeq_ptr_list.append(index_ptr_list)
            
            # 如此一来就能加载知识库了,并且输出部分为        

            # for key in splitSentence(sys_sentence):
            #     # situation first and second
            #     if not ENTPTR or (ENTPTR and (key in ENTITIES)):
            #         index_ptr_list = [index for index, val in enumerate(
            #             user_split_words) if val[SUBJECT_INDEX] == key]
            #         if index_ptr_list:  # 1&2
            #             index_ptr_list = max(index_ptr_list)
            #             gate.append(1)
            #         elif ENTPTR:  # 2
            #             index_ptr_list = len(user_split_words)
            #         else:  # 1
            #             index_ptr_list = len(user_split_words)
            #             gate.append(0)
            #     else:  # 2
            #         #entptr and key not in entity
            #         index_ptr_list = len(user_split_words)
            #         gate.append(0)


                # ent op
                # if(key in entity):
                #     ent.append(key)

                
            # END
            user_split_words.append([EOS_token for i in range(MEM_TOKEN_SIZE)])

            firstLayerProcessedData.append(
                [user_split_words, sys_sentence, sysSeq_ptr_list, gate, KB_dict])

    # maximum userinputSeq length
    max_len = max([len(d[0]) for d in firstLayerProcessedData])
    # maximum sysSeq_ptr length
    max_ptr_len = max([len(d[2]) for d in firstLayerProcessedData])

    return firstLayerProcessedData, max_len, max_ptr_len


def get_seq(pairs: list, lang: LanguageProcessUnit, batch_size: int, type, max_len: int) -> Data.dataloader:
    x_seq = []
    y_seq = []
    ptr_seq = []
    gate_seq = []
    conv_seq = []
    ent = []
    ID = []
    kb_arr = []  # 知识库数组
    for pair in pairs:
        x_seq.append(pair[0])  # 带词标注序列
        y_seq.append(pair[1])  # 原句子
        ptr_seq.append(pair[2])  # 指针序列
        gate_seq.append(pair[3])  # 门序列
        kb_arr.append(pair[4])  # 知识库数组
        # conv_seq.append(pair[4])  # 对话内容词标注序列
        # ent.append(pair[5])
        # ID.append(pair[6])  # 对话ID号
        if(type):
            lang.index_words(pair[0])
            lang.index_words(pair[1], trg=True)

    # data[index]
    dataset = Dataset(x_seq, y_seq, ptr_seq, gate_seq, lang.word2index,
                      lang.word2index, max_len, kb_arr)
    # data_loader = FastDataloader(dataset=dataset,
    #                              batch_size=batch_size,
    #                              shuffle=type,
    #                             #  num_workers=1,
    #                              pin_memory=True,
    #                              drop_last=True,
    #                              collate_fn=collate_fn)
    return dataset


def prepareLang(user_word_lists,sys_sentences)->LanguageProcessUnit:
    lang = LanguageProcessUnit()
    lang.index_words(user_word_lists)
    lang.index_words(sys_sentences,trg=True)
    return lang

def getKB_Info(dialog_act: list, conv_stamp_str: str, requestFlag=False, informFlag=False) -> list:
    """
    each item in dialog_act:

    item[0]:'General','Request','Inform' , General is not necessary
    item[1]:'topic'
    item[2]:'entity'
    item[3]:'information'
    """
    kb_arr = None
    if requestFlag:
        choice = 'Request'
    elif informFlag:
        choice = 'Inform'

    for item in dialog_act:
        if item[0] == 'General':
            continue
        elif item[0] == choice:
            # 3-tuple by (topic-entity-information) addition conv_stamp
            kb_arr = [item[1], item[2], item[3], conv_stamp_str]
            # kb_arr.append([item[1], item[2], item[3], conv_stamp_str])

    return kb_arr

##############核心函数##############


dataPath = os.path.join(
    os.getcwd(), "data\\crosswoz\\train.json.zip").replace('\\', '/')
fileName = "train.json"

##############读取文件函数##############


def GetDataSetTokenlizeContent(dataSetPath, limitTasks=300) -> dict:
    data = readZippedJSON(filepath=dataSetPath, filename=fileName)
    Tasks = {}

    data = data.items()
    for taskID, item in data:
        # limit
        if Tasks.__len__() >= limitTasks:
            break

        # reset
        userDataVoc, sysDataVoc, request_KB_dict, inform_KB_dict = [], [], {}, {}
        messages = {'user': userDataVoc, 'sys': sysDataVoc,
                    'request_KB': request_KB_dict, 'inform_KB': inform_KB_dict}

        conv_stamp = 1
        for index, turn in enumerate(item['messages']):
            conv_stamp = index//2 + 1
            if turn['role'] == 'usr':
                userDataVoc.append(
                    ["t{}".format(conv_stamp), splitSentence(removeBianDian(turn['content']))])
            else:
                sysDataVoc.append(
                    ["t{}".format(conv_stamp), removeBianDian(turn['content'])])
            # dataVoc[index] = []
            # for da in delexicalize_da(turn['dialog_act']):
            #     dataVoc[index].append(da)

            # 装载知识库内容
            conv_stamp_str = "t{}".format(conv_stamp)
            if not request_KB_dict.__contains__(conv_stamp_str):
                request_KB_dict.setdefault("t{}".format(conv_stamp), [getKB_Info(
                    turn['dialog_act'], conv_stamp_str="t{}".format(conv_stamp), requestFlag=True)])
            else:
                request_KB_dict[conv_stamp_str].append(getKB_Info(
                    turn['dialog_act'], conv_stamp_str="t{}".format(conv_stamp), requestFlag=True))
            if not inform_KB_dict.__contains__(conv_stamp_str):
                inform_KB_dict.setdefault("t{}".format(conv_stamp), [getKB_Info(
                    turn['dialog_act'], conv_stamp_str="t{}".format(conv_stamp), informFlag=True)])
            else:
                inform_KB_dict[conv_stamp_str].append(getKB_Info(
                    turn['dialog_act'], conv_stamp_str="t{}".format(conv_stamp), informFlag=True))

        Tasks.setdefault(taskID, messages)

    return Tasks
    # , dataVoc


def getDataInToDataloader(dataSetPath, lang: LanguageProcessUnit, limitTasks=1000):
    data = readZippedJSON(filepath=dataSetPath, filename=fileName)
    data = data.items()
    limit_counter = 0

    user_words_lists, sys_sentences, request_KB_list, inform_KB_list = [], [], [], []
    for taskID, item in data:
        # limit
        limit_counter += 1
        if limit_counter >= limitTasks:
            break

        conv_stamp = 1
        each_req_list, each_inform_list = [], []
        for index, turn in enumerate(item['messages']):
            conv_stamp = index//2 + 1

            if turn['role'] == 'usr':
                user_words_lists.append(splitSentence(turn['content']))
                each_req = getKB_Info(turn['dialog_act'],
                                      "t{}".format(conv_stamp), requestFlag=True)
                each_inform = getKB_Info(turn['dialog_act'],
                                         "t{}".format(conv_stamp), informFlag=True)
                each_req_list.append(
                    each_req) if each_req is not None else None
                each_inform_list.append(
                    each_inform) if each_inform is not None else None
            else:
                sys_sentences.append(turn['content'])
                each_inform = getKB_Info(turn['dialog_act'],
                                         "t{}".format(conv_stamp), informFlag=True)
                each_inform_list.append(
                    each_inform) if each_inform is not None else None

            if (index+1) % 2 == 0:
                request_KB_list.append(each_req_list)
                inform_KB_list.append(each_inform_list)
                each_req_list, each_inform_list = [], []

    entity = None
    sysSeq_ptr_lists = []
    gates = []
    ents = []
    conversation_arrs = []
    # lang = LanguageProcessUnit()
    for userWordsList, sysSentence in zip(user_words_lists, sys_sentences):
        sysSeq_ptr_list = []
        gate = []
        ent = []
        conversation_arr = []

        lang.index_words(userWordsList)
        lang.index_words(sysSentence, trg=True)

        for key in splitSentence(sysSentence):
            # situation first and second
            if not ENTPTR or (ENTPTR and (key in entity)):
                index_ptr_list = [index for index, val in enumerate(
                    sysSentence) if val == key]
                if index_ptr_list:  # 1&2
                    index_ptr_list = max(index_ptr_list)
                    gate.append(1)
                elif ENTPTR:  # 2
                    index_ptr_list = len(sysSentence)
                else:  # 1
                    index_ptr_list = len(userWordsList)
                    gate.append(0)
            else:  # 2
                #entptr and key not in entity
                index_ptr_list = len(userWordsList)
                gate.append(0)

            sysSeq_ptr_list.append(index_ptr_list)
        sysSeq_ptr_lists.append(sysSeq_ptr_list)
        gates.append(gate)

    max_len = max([len(d) for d in user_words_lists])
    max_ptr_len = max([len(d) for d in sysSeq_ptr_lists])

    dataset = Dataset(user_words_lists, sys_sentences,
                      sysSeq_ptr_lists, gates, lang.word2index, lang.word2index, max_len, list(zip(request_KB_list, inform_KB_list)))

    data_loader = FastDataloader(dataset=dataset,
                                 batch_size=BATCHSIZE,
                                 shuffle=type,
                                 num_workers=2,
                                 drop_last=True,
                                 collate_fn=collate_fn)
    return data_loader, max_len, max_ptr_len


def storeJsonFile(name, dataSet, jsonObj):
    jsonfilepath = os.path.join("data", dataSet, name+".json")
    if not os.path.exists(jsonfilepath):
        with open(jsonfilepath, "w+") as f:
            # f.writelines(jsonObj)
            ujson.dump(jsonObj, f)
            logging.log("save success")


def tokenlizeContent(tasks):
    tasks_items = tasks.items()
    for key, cur_task in tasks_items:
        sysTasks = cur_task['sys'].items()
        userTasks = cur_task['user'].items()
        for index, sentence in sysTasks:
            sentence = splitSentence(sentence)
        for index, sentence in userTasks:
            sentence = splitSentence(sentence)
    return tasks
##############读取文件函数##############


# def inputWordsListAddKBInfo(wordslist:list):
    
