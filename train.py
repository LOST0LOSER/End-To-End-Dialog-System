# import numpy as np
# import torch
from multiprocessing import process
from typing import Dict

from torch.utils.data import dataloader
from utils.Dataset import *

import jieba
from models.Mem2Seq import Mem2Seq
import numpy
import torch
from torch.random import seed
# import os
# print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
# print("PATH:", os.environ.get('PATH'))
# if __name__ == "__main__":
#     pass
from utils import preprocess
from utils.config import *
from utils.tools import *
from utils.LanguageProcessUnit import LanguageProcessUnit
from models.Mem2Seq import Mem2Seq
import ijson
import ujson
import json
import torch.multiprocessing as mp

torch.cuda.manual_seed(seed())
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 读取crossWOZ数据集
# if(os.path.isfile(TRAINJSONPATH)):
#     with open(TRAINJSONPATH) as f:
#         processContent = f.readlines()
# else:
processContent = preprocess.GetDataSetTokenlizeContent(
    preprocess.dataPath)

# save extract trainning data
# fail dumps too slow and file too large
# json.dump(processContent,)
# jsonObj = ujson.dumps(processContent)
# preprocess.storeJsonFile('train', 'crossWOZ', processContent)


# # 语言处理标记单元
lang = LanguageProcessUnit()

# 第一层处理文本数据,处理成方便转化成张量的数据，并设置好指针网络的初始化
firstProcessedData, max_len, max_ptr_len = preprocess.createNecessaryInputData(
    processContent,lang)

# 预备语言处理标记单元，输入对话数据集来标记数据，word2index
# lang = preprocess.prepareLang(user_word_lists=firstProcessedData[0],sys_sentences=firstProcessedData[1])

# 封装torch数据集

# dataset = FastDataloader()

# torch数据加载器
dataset = preprocess.get_seq(
    pairs=firstProcessedData, lang=lang, batch_size=BATCHSIZE, type=True, max_len=max_len)

data_loader = FastDataloader(dataset=dataset,
                                batch_size=BATCHSIZE,
                                shuffle=type,
                                # num_workers=1,
                                # pin_memory=True,
                                drop_last=True,
                                collate_fn=collate_fn)

# data_loader, max_len, max_ptr_len = preprocess.getDataInToDataloader(
#     preprocess.dataPath, lang)


# def train(model, prefetcher):
#     input_batches, input_lengths, target_batches, target_lengths, target_index, target_gate, src_plain, trg_plain, request_KB_arr, inform_KB_arr = prefetcher.next()
#     while input_batches is not None:
#         model.train_batch(input_batches, input_lengths, target_batches,
#                           target_lengths, target_index, target_gate, batch_size=BATCHSIZE, clip=10.0, teacher_forcing_ratio=0.5)

#         print(model.print_loss())

#         input_batches, input_lengths, target_batches, target_lengths, target_index, target_gate, src_plain, trg_plain, request_KB_arr, inform_KB_arr = prefetcher.next()

if __name__ == '__main__':

    if(args['decoder'] == "Mem2Seq"):
        model = Mem2Seq(hiddenSize=int(args['hidden']),
                        max_input=int(max_len),
                        max_response=int(max_ptr_len),
                        path=args['path'],
                        lang=lang,
                        task=int(args['task']),
                        learningRate=float(args['learn']),
                        n_layers=int(args['layer']),
                        dropout=float(args['drop']),
                        unk_mask=bool(int(args['unk_mask'])),
                        batchsize=BATCHSIZE)
        print(param for param in model.parameters())


    model.share_memory()

    if USE_CUDA:
        for epoch in range(300):
            prefetcher = Data_prefetcher(data_loader)
            input_batches, input_lengths, target_batches, target_lengths, target_index, target_gate, src_plain, trg_plain, request_KB_arr, inform_KB_arr = prefetcher.next()
            while prefetcher.is_end is False:
                model.train_batch(input_batches, input_lengths, target_batches,
                                  target_lengths, target_index, target_gate,batch_size=BATCHSIZE,clip=10.0,teacher_forcing_ratio=0.5)
                # print((prefetcher.counter,prefetcher.totolLength))
                # print(model.print_loss(),end=None)
                progress_bar(prefetcher.counter, prefetcher.totolLength, lossMsg=model.print_loss())
                input_batches, input_lengths, target_batches, target_lengths, target_index, target_gate, src_plain, trg_plain, request_KB_arr, inform_KB_arr = prefetcher.next()

            if((epoch+1) % int(args['evalp']) == 0):
                inputSeq = splitSentence((removeBianDian("请问颐和园的地址在哪？")))
                inputTensor = lang.tensorTheInputWords(inputSeq).transpose(0,1).unsqueeze(1)
                sentence = model.outputSentence(inputTensor=inputTensor, input_src_words=inputSeq)
    
    else:
        for epoch in range(300):
            pbar = tqdm(enumerate(data_loader), total=len(data_loader))
            for i, data in pbar:
                # input_batches, input_lengths, target_batches, target_lengths, target_ptr_indexs, target_gate, src_content, trg_content, request_KB_arr, inform_KB_arr = zip(*data)
                # clip = 10.0
                # teacher_forcing_ratio = 0.5
                # reset = i==0 #在最开始初始化重新训练
                model.train_batch(input_batches=data[0], input_lengths=data[1],
                                target_batches=data[2], target_lengths=data[3], target_index=data[4],
                                    target_gate=data[5], batch_size=BATCHSIZE, clip=10.0, teacher_forcing_ratio=0.5)

                pbar.set_description(model.print_loss())
            if((epoch+1) % int(args['evalp']) == 0):
                inputSeq = jieba.lcut("你好，请问故宫的地址在哪？")
                inputTensor = lang.tensorTheInputWords(inputSeq)
                sentence = model.outputSentence(inputTensor=inputTensor, input_src_words=inputSeq)
                # logging.log("输出句子为:"+model.outputSentence(
                #     inputTensor=inputTensor, input_src_words=inputSeq))

    # Configure models
    avg_best, cnt, acc = 0.0, 0, 0.0
    cnt_1 = 0
    # LOAD DATA
    # 读出转换成tensor张量的内容
