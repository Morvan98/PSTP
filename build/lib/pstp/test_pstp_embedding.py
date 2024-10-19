from tqdm import tqdm 
import numpy as np
from tools_nn_related import * 



'''
Example
'''
seqs = [
    'GRGDSPYS'*25,
    'ARADSPYS'*25,
    'SRSDSPYS'*25, 
    'GRGDVPYS'*25, 
    'GQGNSPYS'*25,
    'GRGNSPYS'*25, 
    'GRGNSPWS'*25, 
    'GRGNSPFS'*25,
    'GRGNSPAS'*25,
    'GKGNSPYS'*25, 
    'GRGASPYA'*25,
]
''' 
get pstp embedding
'''
seq_matrix_lst = seq2matrix_lst(seqs)
print(seq_matrix_lst)