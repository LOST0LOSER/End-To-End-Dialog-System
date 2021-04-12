from utils.tools import splitSentence
from torch import load as torchLoad
from os import path

from torch.autograd.grad_mode import F
from utils.config import *
from torch import Tensor
from utils.DataBase import *

UNK_token_index = 0
PAD_token_index = 1
EOS_token_index = 2
SOS_token_index = 3
comma_token_index = 4
full_stop_token_index = 5


UNK_token = '<UNK>'
PAD_token = '<PAD>'
EOS_token = '$'
SOS_token = '<SOS>'
comma_token = '，'
full_stop_token = '。'


class LanguageProcessUnit:
    @classmethod
    def load_lang(self, modeldirpath):
        if path.isdir(modeldirpath):
            langpath = path.join(modeldirpath, 'lang.th')
            if path.isfile(langpath):
                return torchLoad(langpath)
        return None

    def __init__(self) -> None:
        super().__init__()
        self.word2index = {}
        self.word2count = {}
        self.index2word = {UNK_token_index: UNK_token, PAD_token_index: PAD_token,
                           EOS_token_index: EOS_token,  SOS_token_index: SOS_token,
                           comma_token_index: comma_token, full_stop_token_index: full_stop_token}
        self.wordsAmount = 6  # Count default tokens
        self.MEM_TOKEN_SIZE = MEM_TOKEN_SIZE  # set from config

    def index_words(self, story, trg=False):
        if trg:
            for word in splitSentence(story):
                self.index_word(word)
        else:
            for words in story:
                for word in words:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.wordsAmount
            self.word2count[word] = 1
            self.index2word[self.wordsAmount] = word
            self.wordsAmount += 1
        else:
            self.word2count[word] += 1

    def tensorTheInputWords(self, input_split_list, trg=False) -> Tensor:
        """Converts words to ids to tensor."""
        # 一开始输入句子是原生一个一维列表[word,word,...]
        # 先获取SPO三元组，作为知识库
        # SPO_list = generateAllSPO(inputSeq)
        # inputSeq = self.fillPADForInitWordsList(inputSeq)
        # # 第二步concat KB SPO_list
        # inputSeq = SPO_list+inputSeq

        # # 若为机器人输出的句子
        # inputSeq = self.addEOS([[self.word2index[word] if word in self.word2index else UNK_token_index for word in words ] for words in
        #                         inputSeq])
        # inputSeq = Tensor(inputSeq).to(device=DEVICE).long()

        # 利用这个还可以保持1维来操作 #深拷贝
        origin_user_split_words = list(input_split_list)
        # 第一步fillPadding
        input_split_list = self.fillPADForInitWordsList(input_split_list)
        ents = None

        # state 1
        # 加载知识库并与输入进行连接
        subject = patternSubject(origin_user_split_words)
        if subject is not None:
            # ents will use for KB
            ents = getEntitesBySubject(subject)
            ents = [[subject, PAD_token, ent] for ent in ents]
            input_split_list = ents + input_split_list  # [B;X]
        else:
            # state 2
            ent = patternEntity(origin_user_split_words)
            if ent is not None:
                attrs = getAttrsByEntity(ent)
                # generate kb
                kb = []
                for key,val in attrs.items():
                    if isinstance(val,list):
                        for v in val:
                            kb.append([ent,key,v])
                    elif val is not None:
                        kb = [ent, key, val]
                
                input_split_list = kb + input_split_list

        input_split_list = self.addEOS([[self.word2index[word] if word in self.word2index else UNK_token_index for word in words] for words in
                                        input_split_list])
        input_split_list = Tensor(input_split_list).to(device=DEVICE).long()

        return input_split_list

    def save_lang(self, word):
        pass

    def fillPADForInitWordsList(self, inputList) -> list:
        """为一个分词列表填充内存单元，1维"""
        for i, word in enumerate(inputList):
            inputList[i] = [word]+[PAD_token for i in range(self.MEM_TOKEN_SIZE-1)]
        return inputList

    def addEOS(self, sentList) -> list:
        """设置句子结束符"""
        sentList.append([EOS_token_index for i in range(MEM_TOKEN_SIZE)])
        return sentList
