from pickle import NONE
from utils.LanguageProcessUnit import LanguageProcessUnit
import torch
from models.Mem2Seq import Mem2Seq
from utils.LanguageProcessUnit import LanguageProcessUnit
import os.path as path

def loadMem2SeqModel(modeldirpath)->Mem2Seq:
    if path.isdir(modeldirpath):
        modelpath = path.join(modeldirpath,'model.th')
        if path.isfile(modelpath):
            return torch.load(modelpath)
    return None

def loadLanguageProcessUnit(modeldirpath)->LanguageProcessUnit:
    if path.isdir(modeldirpath):
        langpath = path.join(modeldirpath,'lang.th')
        if path.isfile(langpath):
            return torch.load(langpath)
    return None