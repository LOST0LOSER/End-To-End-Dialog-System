from os import cpu_count
from typing import Dict, Optional
import fastapi
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic.main import Model
import torch
import utils
import json
from utils.preprocess import splitSentence
from utils.config import *
from models import Mem2Seq

# import sys
# sys.setdefaultencoding("utf-8")

app = fastapi.FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5000",
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

if args["decoder"] == "Mem2Seq":
    model = Mem2Seq(int(args['hidden']),
                    max_len, max_r, lang, args['path'], args['task'],
                    lr=float(args['learn']),
                    n_layers=int(args['layer']),
                    dropout=float(args['drop']),
                    unk_mask=bool(int(args['unk_mask'])))

@app.post("/dialog")
async def processDialog(inputBody:reqBody) -> JSONResponse:
    words = splitSentence(inputBody.text)
    words_batches = torch.tensor(words)
    
    
    # result = dict()
    # result["text"] = "result"

    result = {"text":"result"}
    return result
