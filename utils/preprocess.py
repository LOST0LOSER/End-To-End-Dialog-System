from os.path import join
from re import A

from numpy import diag
from utils.config import *
import torch
import torch.nn as NN
import torch.nn.functional as Fun
import torch.utils.data as Data
import torch.optim as OP
from torch.autograd import Variable
from torch.utils import data
from typing import ItemsView, get_type_hints
import zipfile
import json

import jieba
import jieba.posseg as pseg  # 词性标注
import jieba.analyse as anls  # 关键词提取


##############辅助函数##############
def readZippedJSON(filepath, filename)->dict:
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


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


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def splitSentence(inputSentence) -> list:
    return jieba.lcut(inputSentence)


MEM_TOKEN_SIZE = 3

##############辅助函数##############


##############核心函数##############
def createNecessaryInputData(Tasks:dict,user_split_words: list, sys_sentence: str, entity, sysSeq_ptr_list: list, gate: list, firstLayerProcessedData: list):
    sysSeq_ptr_list = []
    gate = []
    ent = []
    KB_arr = []
    firstLayerProcessedData = []
    conversation_arr = []
    dialog_counter = 0
    for taskID,item in Tasks.items():
        user_wordlists = item['user']
        sys_wordlists = item['sys']
        for userWordsList, sysWordsList in zip(user_wordlists, sys_wordlists):
            user_time, user_split_words = userWordsList
            sys_time,sys_sentence = sysWordsList

            for key in splitSentence(sys_sentence):
                # situation first and second
                if not ENTPTR or (ENTPTR and (key in entity)):
                    index_list = [index for index, val in enumerate(
                        user_split_words) if val == key]
                    if (index_list):  # 1&2
                        index_list = max(index_list)
                        gate.append(1)
                    elif ENTPTR:  # 2
                        index_list = len(user_split_words)
                    else:  # 1
                        index_list = len(user_split_words)
                        gate.append(0)
                else:  # 2
                    #entptr and key not in entity
                    index_list = len(user_split_words)
                    gate.append(0)

                sysSeq_ptr_list.append(index_list)

                # ent op
                if(key in entity):
                    ent.append(key)
            firstLayerProcessedData.append([user_split_words,sys_sentence,sysSeq_ptr_list,gate,list(conversation_arr),ent,dialog_counter,KB_arr])
    
    max_len = max([len(d[0]) for d in firstLayerProcessedData])# maximum userinputSeq length
    max_ptr_len = max([len(d[2]) for d in firstLayerProcessedData]) #maximum sysSeq_ptr length

    return firstLayerProcessedData,max_len,max_ptr_len   


def collate_fn(data):
    def merge(sequences, max_len):
        lengths = [len(seq) for seq in sequences]
        if (max_len):
            # 填充的序列，对话内容转换为张量
            padded_seqs = torch.ones(len(sequences), max(
                lengths), MEM_TOKEN_SIZE).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end, :] = seq[:end]
        else:
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate source and target sequences
    src_seqs, trg_seqs, ind_seqs, gete_s, max_len, src_plain, trg_plain, conv_seq, ent, ID, kb_arr = zip(
        *data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs, max_len)
    trg_seqs, trg_lengths = merge(trg_seqs, None)
    ind_seqs, _ = merge(ind_seqs, None)
    gete_s, _ = merge(gete_s, None)
    conv_seqs, conv_lengths = merge(conv_seq, max_len)

    src_seqs = Variable(src_seqs).transpose(0, 1)
    trg_seqs = Variable(trg_seqs).transpose(0, 1)
    ind_seqs = Variable(ind_seqs).transpose(0, 1)
    gete_s = Variable(gete_s).transpose(0, 1)
    conv_seqs = Variable(conv_seqs).transpose(0, 1)

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        ind_seqs = ind_seqs.cuda()
        gete_s = gete_s.cuda()
        conv_seqs = conv_seqs.cuda()
    return src_seqs, src_lengths, trg_seqs, trg_lengths, ind_seqs, gete_s, src_plain, trg_plain, conv_seqs, conv_lengths, ent, ID, kb_arr


def get_seq(pairs, lang, batch_size, type, max_len):
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
        conv_seq.append(pair[4])  # 对话内容词标注序列
        ent.append(pair[5])
        ID.append(pair[6])  # 对话ID号
        kb_arr.append(pair[7])  # 知识库数组
        if(type):
            lang.index_words(pair[0])
            lang.index_words(pair[1], trg=True)

    # data[index]
    dataset = Dataset(x_seq, y_seq, ptr_seq, gate_seq, lang.word2index,
                      lang.word2index, max_len, conv_seq, ent, ID, kb_arr)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=collate_fn)
    return data_loader
##############核心函数##############


dataPath = os.path.join(
    os.getcwd(), "data\\crosswoz\\train.json.zip").replace('\\', '/')
fileName = "train.json"

##############读取文件函数##############


def GetDataSetTokenlizeContent(dataSetPath) -> dict:
    data = readZippedJSON(filepath=dataSetPath, filename=fileName)
    Tasks = {}
    Tasks = {}
    userDataVoc, sysDataVoc = [], []
    messages = {'user': userDataVoc, 'sys': sysDataVoc}
    dataVoc = {}
    for taskID, item in data.items():
        for index, turn in enumerate(item['messages']):
            if turn['role'] == 'usr':
                userDataVoc.append(["t{}".format(index), splitSentence(turn['content'])])
            else:
                sysDataVoc.setdefault(
                    ["t{}".format(index), turn['content']])
            dataVoc[index] = []
            for da in delexicalize_da(turn['dialog_act']):
                dataVoc[index].append(da)
        Tasks.setdefault(taskID, messages)
    return Tasks, dataVoc


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


class LanguageProcessUnit:
    def __init__(self) -> None:
        # UNK: unkown_token
        # PAD: fill the batch to satisfy the batch size
        # EOS:end of sentence
        # SOS: the beginning identifier of the sentence in decoder
        self.word2index = {}
        self.wordCounts = {}
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD",
                           EOS_token: "EOS",  SOS_token: "SOS"}
        self.wordsAmount = 4  # Count default tokens

    def indexWords(self, dialog):
        for word in dialog.split(' '):
            self.indexWord(word)

    def indexWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.wordsAmount
            self.index2word[self.wordsAmount] = word
            self.wordCounts[word] = 1
            self.wordsAmount += 1
        else:
            self.wordCounts[word] += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, src_seq, trg_seq, index_seq, trg_plain, src_word2id, trg_word2id, max_len):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.index_seqs = index_seq
        self.trg_plain = trg_plain
        self.src_plain = src_seq
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.max_len = max_len

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        index_s = self.index_seqs[index]
        trg_plain = self.trg_plain[index]
        src_plain = self.src_plain[index]
        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)
        index_s = self.preprocess_inde(index_s, src_seq)
        # gete_s  = self.preprocess_gate(gete_s)

        return src_seq, trg_seq, index_s, trg_plain, self.max_len, src_plain

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        sequence = [word2id[word] if word in word2id else UNK_token for word in sequence.split(
            ' ')] + [EOS_token]
        sequence = torch.Tensor(sequence)
        return sequence

    def preprocess_inde(self, sequence, src_seq):
        """Converts words to ids."""
        sequence = sequence + [len(src_seq)-1]
        sequence = torch.Tensor(sequence)
        return sequence

    def preprocess_gate(self, sequence):
        """Converts words to ids."""
        sequence = sequence + [0]
        sequence = torch.Tensor(sequence)
        return
