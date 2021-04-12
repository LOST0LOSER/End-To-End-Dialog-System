from os import cpu_count
from typing import Dict, Optional
import fastapi
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware import cors
import torch
import utils
import json
from utils.tools import *
from utils.config import MODELPATH
from models.Mem2Seq import Mem2Seq
from utils.LanguageProcessUnit import LanguageProcessUnit
import ModelLoder
# import sys
# sys.setdefaultencoding("utf-8")

app = fastapi.FastAPI()


origins = [
    "http://localhost",
    "http://localhost:5000",
    "http://localhost:8080"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# os != 'windows NT'
# jieba.enable_parallel(cpu_count())

class reqBody(BaseModel):
    text:str


Model = None

# if args["decoder"] == "Mem2Seq":
#     Model = Mem2Seq(int(args['hidden']),
#                     max_len, max_r, lang, args['path'], args['task'],
#                     lr=float(args['learn']),
#                     n_layers=int(args['layer']),
#                     dropout=float(args['drop']),
#                     unk_mask=bool(int(args['unk_mask'])))

# lang:LanguageProcessUnit = LanguageProcessUnit.load_lang(MODELPATH)

Model = ModelLoder.loadMem2SeqModel(MODELPATH)


@app.post("/dialog")
async def processDialog(inputBody:reqBody) -> JSONResponse:
    words = splitSentence(inputBody.text)
    words_batches = Model.lang.tensorTheInputWords(words)
    
    
    # result = dict()
    # result["text"] = "result"

    return {"text":words}
