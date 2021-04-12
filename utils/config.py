import os
import logging
import argparse
from os import path
import torch
from tqdm import tqdm
import sys
import re


# if os.cpu_count() <= 4:
#     USE_CUDA = True
# else:
#     USE_CUDA = False
USE_CUDA = torch.cuda.is_available()

MAX_LENGTH = 10

#单个输入词的词性标注空间
MEM_TOKEN_SIZE = 3

DEVICE = torch.device(
    'cuda:0' if USE_CUDA else 'cpu')


runfilepath = sys.argv[0]
name = runfilepath.split('\\')[-1]

if re.match(r'train',name):
    # 运行模型加载参数
    parser = argparse.ArgumentParser(
        description='Mem2Seq Dialog System Train by CrossWOZ DataSet')
    # parser.add_argument('-istrn', '--istrain',
    #                     help='trainng confirm', required=False, default=False)

    parser.add_argument('-ds', '--dataset',
                        help='dataset, CrossWOZ', required=False,default=None)
    parser.add_argument('-t', '--task', help='Task Number', required=False,default=None)
    parser.add_argument('-dec', '--decoder',
                        help='decoder model, Mem2Seq', required=False)
    parser.add_argument('-hdd', '--hidden',
                        help='Hidden size, default: 256', default=256, required=False)
    parser.add_argument('-bsz', '--batch',
                        help='Batch_size, default: 256', required=False)
    parser.add_argument('-lr', '--learn', help='Learning Rate',
                        default=0.0001, required=False)
    parser.add_argument('-dr', '--drop', help='Drop Out',
                        default=0.3, required=False)
    parser.add_argument('-um', '--unk_mask',
                        help='mask out input token to UNK', required=False, default=1)
    parser.add_argument('-layer', '--layer', help='Layer Number',
                        default=3, required=False)
    parser.add_argument('-lm', '--limit', help='Word Limit',
                        required=False, default=-10000)
    parser.add_argument('-path', '--path',
                        help='path of the file to load', required=False)
    parser.add_argument('-test', '--test', help='Testing mode', required=False)
    parser.add_argument('-sample', '--sample',
                        help='Number of Samples', required=False, default=None)
    parser.add_argument('-useKB', '--useKB',
                        help='Put KnowledgeBase in the input or not', required=False, default=True)
    parser.add_argument(
        '-ep', '--entPtr', help='Restrict Ptr only point to entity', required=False, default=0)
    parser.add_argument('-evalp', '--evalp',
                        help='evaluation period', required=False, default=2)
    parser.add_argument('-an', '--addName',
                        help='An add name for the save folder', required=False, default='')
    parser.add_argument('-trnf', '--trainFileName',
                        help='the name of the json file which includes training data', required=False, default=None)

    args = vars(parser.parse_args())

    name = str(args['task'])+str(args['decoder'])+str(args['hidden'])+str(args['batch']) + \
        str(args['learn'])+str(args['drop'])+str(args['layer'])+str(args['limit'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')  # ,filename='save/logs/{}.log'.format(str(name)))

    LIMIT = int(args["limit"])
    USEKB = bool(args["useKB"])
    ENTPTR = int(args["entPtr"])
    ADDNAME = args["addName"]
    BATCHSIZE = int(args["batch"])
    DATASET = args['dataset']
    LAYER = int(args["layer"])
    MODELPATH = args["path"]
    TRAINJSONPATH = os.path.join(os.getcwd(),"data",str(args['dataset']),
                                str(args['trainFileName'])+".json")

else:
    DATASET = "crossWOZ"
    MODELPATH = path.join(os.getcwd(),"save/model","")
