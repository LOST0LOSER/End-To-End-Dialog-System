# import numpy as np
# import torch
from models.mem2Seq import Mem2Seq
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
from models.Mem2Seq import Mem2Seq

torch.cuda.manual_seed(seed())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # 读取crossWOZ数据集
    processContent, dataVoc = preprocess.GetDataSetContent(preprocess.dataPath)
    processedContent = preprocess.tokenlizeContent(processContent)

    # 第一层处理文本数据,处理成方便转化成张量的数据，并设置好指针网络的初始化
    firstProcessedData, max_len, max_ptr_len = preprocess.createNecessaryInputData(
        processContent)
    # 语言处理标记单元
    lang = preprocess.LanguageProcessUnit()

    # torch数据加载器
    dataloader = preprocess.get_seq(
        pairs=firstProcessedData, lang=lang, batch_size=args['batch'], type=True, max_len=max_len)

    if(args['decoder'] == "Mem2Seq"):
        model = Mem2Seq(int(args['hidden']),
                        max_len, max_ptr_len, lang, args['path'], args['task'],
                        lr=float(args['learn']),
                        n_layers=int(args['layer']),
                        dropout=float(args['drop']),
                        unk_mask=bool(int(args['unk_mask'])))

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_batches, input_lengths, target_batches, target_lengths, target_ptr_indexs, target_gate = zip(
            data)
        batch_size = args['batch']
        # clip = 10.0
        # teacher_forcing_ratio = 0.5
        # reset = i==0 #在最开始初始化重新训练
        model.train_batch(input_batches, input_lengths,
                          target_batches, target_lengths, target_ptr_indexs, target_gate,clip=10.0,teacher_forcing_ratio=0.5,reset= i==0)

# Configure models
avg_best, cnt, acc = 0.0, 0, 0.0
cnt_1 = 0
# LOAD DATA
# 读出转换成tensor张量的内容
