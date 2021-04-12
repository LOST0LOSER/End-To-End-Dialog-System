from logging import setLoggerClass
from typing import Tuple, Union
from numpy.core.fromnumeric import transpose
import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.sparse import Embedding
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
# from utils.masked_cross_entropy import *
# from utils.config import *
from utils.LanguageProcessUnit import *
import random
import numpy as np
import datetime
from utils.measures import wer, moses_multi_bleu
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import os
# from sklearn.metrics import f1_score
import json
# from utils.until_temp import entityList
# from utils.config import USE_CUDA
from utils.config import *
from math import floor

from utils.masked_cross_entropy import *


class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hops, dropout, unk_mask):
        super().__init__()
        self.num_vocab = vocab
        self.max_hops = hops
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        for hop in range(self.max_hops+1):
            embeddedUnit = nn.Embedding(
                self.num_vocab, embedding_dim, padding_idx=PAD_token_index)
            embeddedUnit.weight.data.normal_(0, 0.1)
            self.add_module("embeddedUnit_{}".format(hop), embeddedUnit)
        self.embeddedUnit = AttrProxy(self, "embeddedUnit_")
        self.softmax = nn.Softmax(dim=1)

    def get_state(self, batchsize):
        # if USE_CUDA:
        #     return Variable(torch.zeros(batchsize, self.embedding_dim)).cuda()
        # else:
        return Variable(torch.zeros(batchsize, self.embedding_dim).to(device=DEVICE))

    def forward(self, story):
        story = story.transpose(0, 1)
        story_size = story.size()
        if self.unk_mask:
            if self.training:
                ones = np.ones((story_size[0], story_size[1], story_size[2]))
                rand_mask = np.random.binomial(
                    [np.ones((story_size[0], story_size[1]))], 1-self.dropout)[0]
                ones[:, :, 0] = ones[:, :, 0]*rand_mask
                cur_tensor = Variable(torch.Tensor(ones).to(device=DEVICE))
                # if USE_CUDA:
                #     cur_tensor = cur_tensor.cuda()
                story = story*cur_tensor.long()
        u = [self.get_state(story.size(0))]
        for hop in range(self.max_hops):
            # k represent the hops
            embed_A = self.embeddedUnit[hop](story.contiguous().view(
                story.size(0), -1).long())  # b * (m * s) * e
            embed_A = embed_A.view(
                story_size+(embed_A.size(-1),)) #tensor.size()返回一个tuple的子类，可以与tuple相加  # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2)  # b* m * e  C_i^k

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)  # (q^k)^T
            prob = self.softmax(torch.sum(m_A*u_temp, 2))  # p_i^k

            embed_C = self.embeddedUnit[hop +
                                        1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),))  # Ci
            m_C = torch.sum(embed_C, 2).squeeze(2)  # zip the batch to 1 dim

            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C*prob, 1)  # o^k = pi^k*Ci^(k+1)
            u_k = u[-1] + o_k   # q^(k+1) = q^k+o^k
            u.append(u_k)  # accumulate q^n
        return u_k


class DecoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hops, dropout, unk_mask):
        super(DecoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hops
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim,
                             padding_idx=PAD_token_index)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.W = nn.Linear(embedding_dim, 1)
        self.W1 = nn.Linear(2*embedding_dim, self.num_vocab)
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)

    def load_memory(self, story):
        story_size = story.size()  # b * m * 3
        if self.unk_mask:
            if(self.training):
                ones = np.ones((story_size[0], story_size[1], story_size[2]))
                rand_mask = np.random.binomial(
                    [np.ones((story_size[0], story_size[1]))], 1-self.dropout)[0]
                ones[:, :, 0] = ones[:, :, 0] * rand_mask
                a = Variable(torch.Tensor(ones).to(device=DEVICE))
                # if USE_CUDA:
                #     a = a.cuda()
                story = story*a.long()
        self.m_story = []
        # 利用多跳注意力的方式
        for hop in range(self.max_hops):
            # .long()) # b * (m * s) * e
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1))
            embed_A = embed_A.view(
                story_size+(embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            m_A = embed_A
            embed_C = self.C[hop +
                             1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    """

    """

    def ptrMemDecoder(self, enc_query: torch.Tensor, last_hidden: torch.Tensor):
        embed_q = self.C[0](enc_query)  # b * e
        output, hidden = self.gru(embed_q.unsqueeze(0), last_hidden)
        temp = []
        u = [hidden[0].squeeze()]
        # 从多跳注意力存储内容的取
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            # used for bathcsize = 1.
            if(len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A*u_temp, 2)
            prob_ = self.softmax(prob_lg)
            m_C = self.m_story[hop+1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C*prob, 1)
            if (hop == 0):
                # vocabulary distribution
                p_vocab: torch.Tensor = self.W1(torch.cat((u[0], o_k), 1))
            u_k = u[-1] + o_k  # q^(k+1) = q_i^k+o_i^k
            u.append(u_k)
        p_ptr = prob_lg  # memory distribution
        return p_ptr, p_vocab, hidden


class Mem2Seq(nn.Module):

    @classmethod
    def loadModel(self, modeldirpath):
        if path.isdir(modeldirpath):
            modelpath = path.join(modeldirpath, 'model.th')
            if path.isfile(modelpath):
                return torch.load(modelpath)
        return None

    def __init__(self, hiddenSize: int, max_input: int, max_response: int, path: str, lang: LanguageProcessUnit, task: int, learningRate: float, n_layers: int, dropout: float, unk_mask: bool,batchsize:int):
        super(Mem2Seq, self).__init__()
        self.name = "Mem2Seq"
        self.task = task
        self.input_size = lang.wordsAmount
        self.output_size = lang.wordsAmount
        self.hidden_size = hiddenSize
        self.max_input = max_input  # max input len 最大输入了多少个词
        self.max_response = max_response  # max response len 最大输出了多少个词
        self.lang = lang
        self.lr = learningRate
        self.n_layers = n_layers
        self.dropout = dropout
        self.unk_mask = unk_mask

        # 加载模型
        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(
                    str(path)+'/enc.th', lambda storage, loc: storage)
                self.decoder = torch.load(
                    str(path)+'/dec.th', lambda storage, loc: storage)
        else:
            self.encoder = EncoderMemNN(
                lang.wordsAmount, self.hidden_size, self.n_layers, self.dropout, self.unk_mask)
            self.decoder = DecoderMemNN(
                lang.wordsAmount, self.hidden_size, self.n_layers, self.dropout, self.unk_mask)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=self.lr)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to selected DEVICE
        # if USE_CUDA:
        self.encoder.to(device=DEVICE)
        self.decoder.to(device=DEVICE)

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_vac = self.loss_vac / self.print_every
        self.print_every += 1
        return 'LossAvg:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg, print_loss_vac, print_loss_ptr)

    def save_model(self, dec_type):
        name_data = "crossWOZ/"
        directory = os.path.join('save/mem2seq', name_data, str(self.task), 'HDD', str(self.hidden_size), 'BSZ', str(
            args['batch']), 'DR', str(self.dropout), 'L', str(self.n_layers), 'lr', str(self.lr), str(dec_type))
        if not os.path.exists(directory):
            os.makedirs(directory)
        infoPath = directory+'/info.json'
        with open(infoPath, "w+", encoding="utf8") as f:
            jsonInfo = {'hidden': self.hidden_size,
                        'max_input': self.max_input, 'max_response': self.max_response, }
            f.writelines()

        torch.save(self, directory+'/model.th')
        # torch.save(self.encoder, directory+'/enc.th')
        # torch.save(self.decoder, directory+'/dec.th')

    # def responseUser(self, input_batches,input_lengths) -> str:

    def outputSentence(self, inputTensor,input_src_words:list):

        self.encoder.train(False)
        self.decoder.train(False)

        # init
        inputTensor = inputTensor.cuda()
        inputTensor_size = inputTensor.size()
        
        input_src_plain = [input_src_words for i in range(self.batch_size)]
        
        input_length = inputTensor.size(0)
        # FILLing & transform satisfied through the neural network
        inputTensor = inputTensor.expand(MEM_TOKEN_SIZE,self.batch_size,-1).transpose(0,2)
        decoder_hidden = self.encoder(inputTensor).unsqueeze(0)
        self.decoder.load_memory(inputTensor.transpose(0, 1))
        # init decoder_input by start token
        decoder_input = Variable(torch.LongTensor(
            [SOS_token_index]*self.batch_size).to(device=DEVICE))

        # first_max = {
        #     input_length: self.max_response, 
        #     floor(0.2*self.max_response): floor(0.4*self.max_response),
        #     }
        max_target_length = self.max_response

        all_decoder_outputs_vocab = Variable(torch.zeros(
            max_target_length, self.batch_size, self.output_size).to(device=DEVICE))
        all_decoder_outputs_ptr = Variable(torch.zeros(
            max_target_length, self.batch_size, inputTensor.size(0)).to(device=DEVICE))

        # Move new Variables to CUDA
        # if USE_CUDA:
        #     all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
        #     all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
        #     decoder_input = decoder_input.cuda()
        
        max_ptr = [0 for i in range(self.batch_size)] #用来判断与回答的关联程度
        decoded_words= []
        for t in range(max_target_length):
            decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(
                decoder_input, decoder_hidden)
            _, toppi = decoder_ptr.data.topk(1)  # 指针网络大根堆第一个Tensor
            _, topvi = decoder_vacab.data.topk(1)
            all_decoder_outputs_vocab[t] = decoder_vacab
            all_decoder_outputs_ptr[t] = decoder_ptr
            # get the correspective word in input
            top_ptr_i = torch.gather(inputTensor[:,:, 0], 0, Variable(
                toppi.view(1, -1))).transpose(0, 1)
            next_in = [top_ptr_i[i].item() if (toppi[i].item(
            ) < input_length-1) else topvi[i].item() for i in range(self.batch_size)]

            # Chosen word is next input
            decoder_input = Variable(
                torch.LongTensor(next_in).to(device=DEVICE))

            # if USE_CUDA:
            #     decoder_input = decoder_input.cuda()

            each_out=[]
            for i in range(self.batch_size):
                if(toppi[i].item() < len(input_src_plain[i])-1):
                    # 根据指针网络分布选择词
                    each_out.append(input_src_plain[i][toppi[i].item()])
                    max_ptr[i]+=1
                else:
                    # 根据词库分布选择词
                    ind = topvi[i].item()
                    each_out.append(self.lang.index2word[ind])
            decoded_words.append(each_out)

        self.encoder.train(True)
        self.decoder.train(True)

        # decoded_words = np.array(decoded_words,dtype=))
        # decoded_words = transpose(decoded_words)
        output = None
        decoded_words = list(map(list,zip(*decoded_words)))
        if len(decoded_words) > 0 and len(decoded_words[0])>0:
            output = decoded_words[0]
            print(output)
        else:
            print("nothing")

        return output
            


    def reset(self):
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1

    def train_batch(self, input_batches, input_lengths, target_batches,
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio):

        # if reset:
        #     self.loss = 0
        #     self.loss_ptr = 0
        #     self.loss_vac = 0
        #     self.print_every = 1

        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab, loss_Ptr = 0, 0
        

        # Run words through encoder
        # get the hidden processed by encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        # init the first decoder_input
        decoder_input = Variable(torch.LongTensor(
            [SOS_token_index] * batch_size).to(device=DEVICE))

        # self.output_size 指输入数据集中总共容纳的词汇,利用词概率分布来选择最佳输出词
        max_target_length = max(target_lengths)
        self.max_response = max(max_target_length,self.max_response)
        all_decoder_outputs_vocab = Variable(torch.zeros(
            max_target_length, batch_size, self.output_size).to(device=DEVICE))

        # input_batches.size(0)是指输入句子的分词长度，用于指针根据用户输入关键词引入回复句子中
        all_decoder_outputs_ptr = Variable(torch.zeros(
            max_target_length, batch_size, input_batches.size(0)).to(device=DEVICE))

        # Move new Variables to CUDA
        # if USE_CUDA:
        #     all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
        #     all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
        #     decoder_input = decoder_input.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        if use_teacher_forcing:
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(
                    decoder_input, decoder_hidden)
                # 对于每一个输出的词存储其对应的预测分布
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                decoder_input = target_batches[t]  # Chosen word is next input
                decoder_input.to(device=DEVICE)
                # if USE_CUDA:
                #     decoder_input = decoder_input.cuda()
        else:
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(
                    decoder_input, decoder_hidden)
                _, toppi = decoder_ptr.data.topk(1)  # 指针网络大根堆第一个Tensor
                _, topvi = decoder_vacab.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                # get the correspective word in input
                top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(
                    toppi.view(1, -1))).transpose(0, 1)
                next_in = [top_ptr_i[i].item() if (toppi[i].item(
                ) < input_lengths[i]-1) else topvi[i].item() for i in range(batch_size)]

                # Chosen word is next input
                decoder_input = Variable(torch.LongTensor(next_in).to(device=DEVICE))
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()

        # Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(
                0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(
                0, 1).contiguous(),  # -> batch x seq
            target_index.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab + loss_Ptr
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_ptr += loss_Ptr.item()
        self.loss_vac += loss_Vocab.item()

    def evaluate_batch(self, batch_size, input_batches, input_lengths, target_batches, target_lengths, target_index, target_gate, src_plain):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(
            0)  # encoder forward result
        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor(
            [SOS_token_index] * batch_size))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(
            self.max_response, batch_size, self.output_size).to(device=DEVICE))
        all_decoder_outputs_ptr = Variable(torch.zeros(
            self.max_response, batch_size, input_batches.size(0)).to(device=DEVICE))
        # all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        # if USE_CUDA:
        #     all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
        #     all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
        #     # all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
        #     decoder_input = decoder_input.cuda()

        p = []
        for elm in src_plain:
            elm_temp = [word_triple[0] for word_triple in elm]
            p.append(elm_temp)

        self.from_whichs = []
        acc_gate, acc_ptr, acc_vac = 0.0, 0.0, 0.0
        # Run through decoder one time step at a time
        # 利用网络指针的思想在内存中选择需要的词
        for t in range(self.max_response):
            decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(
                decoder_input, decoder_hidden)
            all_decoder_outputs_vocab[t] = decoder_vacab
            topv, topvi = decoder_vacab.data.topk(
                1)  # top_vocabulary,top_vacab_index
            all_decoder_outputs_ptr[t] = decoder_ptr
            topp, toppi = decoder_ptr.data.topk(1)  # top_ptr,top_ptr_index
            top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(
                toppi.view(1, -1))).transpose(0, 1)
            next_in = [top_ptr_i[i].item() if (toppi[i].item() < input_lengths[i]-1)
                       else topvi[i].item() for i in range(batch_size)]

            # Chosen word is next input，将选好的词作为下一个词的输入，回瞥
            decoder_input = Variable(
                torch.LongTensor(next_in).to(device=DEVICE))
            # if USE_CUDA:
            #     decoder_input = decoder_input.cuda()

            temp = []
            from_which = []
            for i in range(batch_size):
                if(toppi[i].item() < len(p[i])-1):
                    temp.append(p[i][toppi[i].item()])
                    from_which.append('p')
                else:
                    ind = topvi[i].item()
                    if ind == EOS_token_index:
                        temp.append('<EOS>')
                    else:
                        # 根据指针选择对应的词
                        temp.append(self.lang.index2word[ind])
                    from_which.append('v')
            decoded_words.append(temp)
            self.from_whichs.append(from_which)
        self.from_whichs = np.array(self.from_whichs)

        # indices = torch.LongTensor(range(target_gate.size(0)))
        # if USE_CUDA: indices = indices.cuda()

        # ## acc pointer
        # y_ptr_hat = all_decoder_outputs_ptr.topk(1)[1].squeeze()
        # y_ptr_hat = torch.index_select(y_ptr_hat, 0, indices)
        # y_ptr = target_index
        # acc_ptr = y_ptr.eq(y_ptr_hat).sum()
        # acc_ptr = acc_ptr.data[0]/(y_ptr_hat.size(0)*y_ptr_hat.size(1))
        # ## acc vocab
        # y_vac_hat = all_decoder_outputs_vocab.topk(1)[1].squeeze()
        # y_vac_hat = torch.index_select(y_vac_hat, 0, indices)
        # y_vac = target_batches
        # acc_vac = y_vac.eq(y_vac_hat).sum()
        # acc_vac = acc_vac.data[0]/(y_vac_hat.size(0)*y_vac_hat.size(1))

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words  # , acc_ptr, acc_vac

    def evaluate(self, dev, avg_best, BLEU=False):
        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        microF1_PRED, microF1_PRED_cal, microF1_PRED_nav, microF1_PRED_wet = 0, 0, 0, 0
        microF1_TRUE, microF1_TRUE_cal, microF1_TRUE_nav, microF1_TRUE_wet = 0, 0, 0, 0
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        dialog_acc_dict = {}

        if args['dataset'] == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_')
                                               for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_')
                                                   for k in item.keys()]
                global_entity_list = list(set(global_entity_list))

        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            if args['dataset'] == 'crossWOZ':
                words = self.evaluate_batch(batch_size=len(data_dev[1]), input_batches=data_dev[0], input_lengths=data_dev[1],
                                            target_batches=data_dev[2], target_lengths=data_dev[3], target_index=data_dev[4], target_gate=data_dev[5], src_plain=data_dev[6])
            else:
                words = self.evaluate_batch(batch_size=len(data_dev[1]), input_batches=data_dev[0], input_lengths=data_dev[1],
                                            target_batches=data_dev[2], target_lengths=data_dev[3], target_index=data_dev[4], target_gate=data_dev[5], src_plain=data_dev[6])

            acc = 0
            w = 0
            temp_gen = []

            for i, row in enumerate(np.transpose(words)):
                statement = ''
                for each_word in row:
                    if each_word == '<EOS>':
                        break
                    else:
                        statement += each_word + ' '
                temp_gen.append(statement)
                correct = data_dev[7][i]
                # compute F1 SCORE
                statement = statement.lstrip().rstrip()
                correct = correct.lstrip().rstrip()
                if args['dataset'] == 'kvr':
                    f1_true, count = self.compute_prf(
                        data_dev[8][i], statement.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE += f1_true
                    microF1_PRED += count
                    f1_true, count = self.compute_prf(
                        data_dev[9][i], statement.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE_cal += f1_true
                    microF1_PRED_cal += count
                    f1_true, count = self.compute_prf(
                        data_dev[10][i], statement.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE_nav += f1_true
                    microF1_PRED_nav += count
                    f1_true, count = self.compute_prf(
                        data_dev[11][i], statement.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE_wet += f1_true
                    microF1_PRED_wet += count
                elif args['dataset'] == 'babi' and int(args["task"]) == 6:
                    f1_true, count = self.compute_prf(
                        data_dev[10][i], statement.split(), global_entity_list, data_dev[12][i])
                    microF1_TRUE += f1_true
                    microF1_PRED += count

                if args['dataset'] == 'babi':
                    if data_dev[11][i] not in dialog_acc_dict.keys():
                        dialog_acc_dict[data_dev[11][i]] = []
                    if (correct == statement):  # 成功输出正确的句子
                        acc += 1
                        dialog_acc_dict[data_dev[11][i]].append(
                            1)  # 对话矩阵one-hot编码，1代表正确回答的选项
                    else:
                        dialog_acc_dict[data_dev[11][i]].append(0)
                else:
                    if (correct == statement):
                        acc += 1
                #    print("Correct:"+str(correct))
                #    print("\tPredict:"+str(st))
                #    print("\tFrom:"+str(self.from_whichs[:,i]))

                w += wer(correct, statement)
                ref.append(str(correct))
                hyp.append(str(statement))
                ref_s += str(correct) + "\n"
                hyp_s += str(statement) + "\n"

            acc_avg += acc/float(len(data_dev[1]))
            wer_avg += w/float(len(data_dev[1]))
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg/float(len(dev)),
                                                            wer_avg/float(len(dev))))

        # dialog accuracy
        if args['dataset'] == 'babi':
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            logging.info("Dialog Accuracy:\t" +
                         str(dia_acc*1.0/len(dialog_acc_dict.keys())))

        if args['dataset'] == 'kvr':
            logging.info("F1 SCORE:\t{}".format(
                microF1_TRUE/float(microF1_PRED)))
            logging.info("\tCAL F1:\t{}".format(
                microF1_TRUE_cal/float(microF1_PRED_cal)))
            logging.info("\tWET F1:\t{}".format(
                microF1_TRUE_wet/float(microF1_PRED_wet)))
            logging.info("\tNAV F1:\t{}".format(
                microF1_TRUE_nav/float(microF1_PRED_nav)))
        elif args['dataset'] == 'babi' and int(args["task"]) == 6:
            logging.info("F1 SCORE:\t{}".format(
                microF1_TRUE/float(microF1_PRED)))

        # bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        # logging.info("BLEU SCORE:"+str(bleu_score))
        bleu_score = 0
        if (BLEU):
            if (bleu_score >= avg_best):
                self.save_model(str(self.name)+str(bleu_score))
                logging.info("MODEL SAVED")
            return bleu_score
        else:
            acc_avg = acc_avg/float(len(dev))
            if (acc_avg >= avg_best):
                self.save_model(str(self.name)+str(acc_avg))
                logging.info("MODEL SAVED")
            return acc_avg

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
            recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
            F1 = 2 * precision * recall / \
                float(precision + recall) if (precision+recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


def loadModel(modeldir) -> Mem2Seq:
    fulldirpath = os.path.join(os.getcwd(), "save/Mem2Seq", modeldir)
    if os.path.isdir(fulldirpath):
        modelpath = os.path.join(fulldirpath, 'model.th')
        if USE_CUDA:
            loadModel = torch.load(loadModel)
        else:
            loadModel = torch.load(loadModel, lambda storage, loc: storage)
        return loadModel
    else:
        return None
