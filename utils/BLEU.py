# from typing import Counter
# import numpy as np
# import math
# def n_gram_generator(sentence,n= 2,n_gram= False):
#     '''
#     N-Gram generator with parameters sentence
#     n is for number of n_grams
#     The n_gram parameter removes repeating n_grams 
#     '''
#     sentence = sentence.lower() # converting to lower case
#     sent_arr = np.array(sentence.split()) # split to string arrays
#     length = len(sent_arr)

#     word_list = []
#     for i in range(length+1):
#         if i < n:
#             continue
#         word_range = list(range(i-n,i))
#         s_list = sent_arr[word_range]
#         string = ' '.join(s_list) # converting list to strings
#         word_list.append(string) # append to word_list
#         if n_gram:
#             word_list = list(set(word_list))
#     return word_list
# def bleu_score(original,machine_translated):
#     '''
#     Bleu score function given a orginal and a machine translated sentences
#     '''
#     mt_length = len(machine_translated.split())
#     o_length = len(original.split())

#     # Brevity Penalty 
#     if mt_length>o_length:
#         BP=1
#     else:
#         penality=1-(mt_length/o_length)
#         BP=np.exp(penality)

#     # Clipped precision
#     clipped_precision_score = []
#     for i in range(1, 5):
#         original_n_gram = Counter(n_gram_generator(original,i))
#         machine_n_gram = Counter(n_gram_generator(machine_translated,i))

#         c = sum(machine_n_gram.values())
#         for j in machine_n_gram:
#             if j in original_n_gram:
#                 if machine_n_gram[j] > original_n_gram[j]:
#                     machine_n_gram[j] = original_n_gram[j]
#             else:
#                 machine_n_gram[j] = 0

#         #print (sum(machine_n_gram.values()), c)
#         clipped_precision_score.append(sum(machine_n_gram.values())/c)

#     #print (clipped_precision_score)

#     weights =[0.25]*4

#     s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, clipped_precision_score))
#     s = BP * math.exp(math.fsum(s))
#     return s

# original = "It is a guide to action which ensures that the military alwasy obeys the command of the party"
# machine_translated = "It is the guiding principle which guarantees the military forces alwasy being under the command of the party"

# print (bleu_score(original, machine_translated))
# print (sentence_bleu([original.split()], machine_translated.split()))


from bleu import list_bleu
ref = ['it is a white cat .',
             'wow , this dog is huge .']
ref1 = ['This cat is white .',
             'wow , this is a huge dog .']
hyp = ['it is a white kitten .',
            'wowww , the dog is huge !']
hyp1 = ["it 's a white kitten .",
             'wow , this dog is huge !']
list_bleu([ref], hyp)
list_bleu([ref, ref1], hyp1)