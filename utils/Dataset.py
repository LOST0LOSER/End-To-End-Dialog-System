from typing import Counter, Tuple

import torch
# from torch.functional import Tensor

from torch import Tensor
from torch.utils import data
from utils.config import *

from utils.tools import splitSentence
from utils.LanguageProcessUnit import *
from torch.autograd import Variable
from prefetch_generator import BackgroundGenerator


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, src_seq, trg_seq, index_seq, gate_seq, src_word2id,
                 trg_word2id, max_user_input_len, kb_arr):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.index_seqs = index_seq
        # self.src_plain = src_seq
        # self.trg_plain = trg_plain
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.max_user_input_len = max_user_input_len
        self.gate_seq = gate_seq

        # 这是一个以 每一对对话句子的知识库dict 为item的list [{}，{}]
        self.kb_arr = kb_arr

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        """
        src_seq
        trg_seq
        index_s
        gate_s
        self.max_user_input_len
        user_dialog_words
        sys_dialog_sentences
        request_KB_arr
        inform_KB_arr
        """
        user_dialog_words = self.src_seqs[index]
        sys_dialog_sentences = self.trg_seqs[index]
        index_s = self.index_seqs[index]
        gate_seq = self.gate_seq[index]
        kb_dict = self.kb_arr[index]

        # 从每一对对话句中提取其知识库
        request_KB_arr = kb_dict['request_KB']
        inform_KB_arr = kb_dict['inform_KB']
        # request_KB_arr = kb_tuple[0]
        # inform_KB_arr = kb_tuple[1]

        # kb_tuple = self.kb_arr[index]
        # trg_plain = self.trg_plain[index]
        # src_plain = self.src_plain[index]
        # SPO_arr = self.generateSPO(request_KB_arr, inform_KB_arr)
        # SPO_arr = self.SPO_word2index(SPO_arr, self.src_word2id)

        src_seq = self.sequence_word2index(user_dialog_words, self.src_word2id, trg=False)
        trg_seq = self.sequence_word2index(sys_dialog_sentences, self.trg_word2id)

        # src_seq = self.sequence_padding(src_seq)
        # trg_seq = self.sequence_padding(trg_seq)

        # src_seq = self.concat_SPO_sequence(SPO_arr,src_seq)

        # 他俩本身就是数字list
        index_s = self.preprocess_inde(index_s, src_seq)
        gate_s = self.preprocess_gate(gate_seq)

        return src_seq, trg_seq, index_s, gate_s, self.max_user_input_len, user_dialog_words, sys_dialog_sentences, request_KB_arr, inform_KB_arr

    def __len__(self) -> int:
        return self.num_total_seqs

    # def sequence_padding(self, sequence):
    #     "padding by copy index"
    #     sequence = [
    #         [index for i in range(MEM_TOKEN_SIZE)] for index in sequence
    #     ]

    def sequence_word2index(self, sequence, word2id, trg=True) -> Tensor:
        """Converts words to ids."""
        if trg:
            # 系统输出为句子
            # MEM_TOKEN PADDING by copy
            sequence = [word2id[word] if word in word2id else UNK_token_index
                    for word in splitSentence(sequence)]+[EOS_token_index]
            sequence = torch.Tensor(sequence).to(device=DEVICE)
        else:
            #contain SPO MEMTOKENSIZE
            sequence = [[word2id[word] if word in word2id else UNK_token_index for word in words] 
                    for words in sequence]
            sequence = torch.Tensor(sequence).to(device=DEVICE)
        return sequence

    def preprocess_inde(self, sequence, src_seq) -> Tensor:
        """Converts words to ids."""
        sequence = sequence + [len(src_seq)-1]
        sequence = torch.Tensor(sequence).to(device=DEVICE)
        return sequence

    def preprocess_gate(self, sequence) -> Tensor:
        """Converts words to ids."""
        sequence = sequence + [0]
        sequence = torch.Tensor(sequence).to(device=DEVICE)
        return sequence

    def generateTensor(self, sequence):
        return torch.Tensor(sequence).to(device=DEVICE)

    # def generateSPO(self, request_KB_arr, inform_KB_arr):
    #     # (subject-predicate-object)
    #     SPO_arr = []
    #     subject = None
    #     predicate = None
    #     object = None
    #     for req in request_KB_arr:
    #         predicate = req[1]
    #         for inf in inform_KB_arr:
    #             if req[0] == inf[0] and req[1] != inf[1]:
    #                 subject = inf[2]
    #             elif req[0] == inf[0] and req[1] == inf[1]:
    #                 object = inf[2]
    #         if subject and object:
    #             SPO_arr.append((subject, predicate, object))
    #     return SPO_arr

    # def SPO_word2index(SPO_arr, word2id):
    #     "SPO word2index"
    #     return [word2id[word] for word in SPO_arr]

    # def concat_SPO_sequence(SPO_ids, sequence_padded_ids):
    #     "[B;X] a concat sequences"
    #     return SPO_ids+sequence_padded_ids


class FastDataloader(data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# 读取从dataset获取到的数据并进行转置，然后利用迭代器选取每一个句子对内容


def collate_fn(data):
    """
    :src_seqs:Tensor 
    src_lengths:list
    trg_seqs:Tensor 
    trg_lengths:list
    ind_seqs:Tensor
    gete_s:Tensor
    src_plain:list
    trg_plain:list
    request_KB_arr:list
    inform_KB_arr:list
    """
    def merge(sequences, max_len):
        lengths = [len(seq) for seq in sequences]
        if (max_len):
            # 填充batches
            padded_seqs = torch.Tensor([PAD_token_index]).to(device=DEVICE).repeat(
                                            len(sequences), max(lengths),
                                            MEM_TOKEN_SIZE).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end, :] = seq[:end]
        else:
            padded_seqs = torch.Tensor([
                PAD_token_index
            ]).to(device=DEVICE).repeat(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate source and target sequences
    # src_plain = user input splits words
    # trg_plain = sys reply sentence
    src_seqs, trg_seqs, ind_seqs, gete_s, max_len, src_plain, trg_plain, request_KB_arr, inform_KB_arr = zip(
        *data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs, max_len)
    trg_seqs, trg_lengths = merge(trg_seqs, None)
    ind_seqs, _ = merge(ind_seqs, None)
    gete_s, _ = merge(gete_s, None)
    # conv_seqs, conv_lengths = merge(conv_seq, max_len)

    src_seqs = Variable(src_seqs).transpose(0, 1)
    trg_seqs = Variable(trg_seqs).transpose(0, 1)
    ind_seqs = Variable(ind_seqs).transpose(0, 1)
    gete_s = Variable(gete_s).transpose(0, 1)
    # conv_seqs = Variable(conv_seqs).transpose(0, 1)

    # if USE_CUDA:
    #     src_seqs = src_seqs.cuda()
    #     trg_seqs = trg_seqs.cuda()
    #     ind_seqs = ind_seqs.cuda()
    #     gete_s = gete_s.cuda()
    # conv_seqs = conv_seqs.cuda()
    return src_seqs, src_lengths, trg_seqs, trg_lengths, ind_seqs, gete_s, src_plain, trg_plain, request_KB_arr, inform_KB_arr
    # del conv_seqs, conv_lengths, ent, ID,


class Data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.totolLength = len(loader)
        # self.stream = torch.cuda.Stream()
        self.counter = 0
        self.is_end = False
        if USE_CUDA:
            self.stream = torch.cuda.Stream(device=DEVICE)
        self.preload()

    def preload(self):
        try:
            self.next_src_seqs, self.next_src_lengths, self.next_trg_seqs, self.next_trg_lengths, self.next_index_s, self.next_gate_s, self.next_src_plain, self.next_trg_plain, self.next_request_KB_arr, self.next_inform_KB_arr = next(
                self.loader)
        except StopIteration:
            self.is_end = True

            self.next_src_seqs = None
            self.next_src_lengths = None
            self.next_trg_seqs = None
            self.next_trg_lengths = None
            self.next_index_s = None
            self.next_gate_s = None
            self.next_src_plain = None
            self.next_trg_plain = None
            self.next_request_KB_arr = None
            self.next_inform_KB_arr = None
            return
        with torch.cuda.stream(self.stream):
            self.next_src_seqs = self.next_src_seqs.cuda(non_blocking=True)
            self.next_trg_seqs = self.next_trg_seqs.cuda(non_blocking=True)
            self.next_index_s = self.next_index_s.cuda(non_blocking=True)
            self.next_gate_s = self.next_gate_s.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_src_seqs
        input_lengths = self.next_src_lengths
        target = self.next_trg_seqs
        target_lengths = self.next_trg_lengths
        index_s = self.next_index_s
        gate_s = self.next_gate_s
        src_plain = self.next_src_plain
        trg_plain = self.next_trg_plain
        request_KB_arr = self.next_request_KB_arr
        inform_KB_arr = self.next_inform_KB_arr

        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if index_s is not None:
            index_s.record_stream(torch.cuda.current_stream())
        if gate_s is not None:
            gate_s.record_stream(torch.cuda.current_stream())
        self.preload()
        self.counter += 1
        return input, input_lengths, target, target_lengths, index_s, gate_s, src_plain, trg_plain, request_KB_arr, inform_KB_arr
