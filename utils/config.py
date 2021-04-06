import os
import logging
import argparse
from tqdm import tqdm



UNK_token = 0
PAD_token = 1
EOS_token = 2
SOS_token = 3

if os.cpu_count()<=4:
    USE_CUDA = True
else:
    USE_CUDA = False

MAX_LENGTH = 10

#运行模型加载参数
parser = argparse.ArgumentParser(description='Mem2Seq Dialog System Train by CrossWOZ DataSet')
parser.add_argument('-ds','--dataset', help='dataset, CrossWOZ', required=False)
parser.add_argument('-t','--task', help='Task Number', required=False)
parser.add_argument('-dec','--decoder', help='decoder model, Mem2Seq', required=False)
parser.add_argument('-hdd','--hidden', help='Hidden size, default: 256', default=256, required=False)
parser.add_argument('-bsz','--batch', help='Batch_size, default: 256', required=False)
parser.add_argument('-lr','--learn', help='Learning Rate',default=0.0001 ,required=False)
parser.add_argument('-dr','--drop', help='Drop Out',default=0.3, required=False)
parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', required=False, default=1)
parser.add_argument('-layer','--layer', help='Layer Number', default=3 ,required=False)
parser.add_argument('-lm','--limit', help='Word Limit', required=False,default=-10000)
parser.add_argument('-path','--path', help='path of the file to load', required=False)
parser.add_argument('-test','--test', help='Testing mode', required=False)
parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
parser.add_argument('-useKB','--useKB', help='Put KnowledgeBase in the input or not', required=False, default=True)
parser.add_argument('-ep','--entPtr', help='Restrict Ptr only point to entity', required=False, default=0)
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=2)
parser.add_argument('-an','--addName', help='An add name for the save folder', required=False, default='')


args = vars(parser.parse_args())

name = str(args['task'])+str(args['decoder'])+str(args['hidden'])+str(args['batch'])+str(args['learn'])+str(args['drop'])+str(args['layer'])+str(args['limit'])
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))

LIMIT = int(args["limit"])
USEKB = bool(args["useKB"])
ENTPTR = int(args["entPtr"])
ADDNAME = args["addName"]

LAYER = int(args["layer"])
MODELPATH = args["path"]



