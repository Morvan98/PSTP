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
saps_seqwide_scores_lst,\
    pdps_seqwide_scores_lst,mix_seqwide_scores_lst = [],[],[]
saps_py_lst,pdps_py_lst,\
    mix_py_lst = [],[],[]
''' 
get pstp prediction
'''
for matrix_ in tqdm(seq_matrix_lst):
    ### prediction using pstp block
    res_p,seq_p, = predict_by_saps_models(matrix_) ### seq_p: sequence level prediction/ ### res_p: residue level score
    saps_seqwide_scores_lst.append(res_p)
    res_p,seq_p, = predict_by_pdps_models(matrix_)
    pdps_seqwide_scores_lst.append(res_p)
    res_p,seq_p, = predict_by_mix_models(matrix_)
    mix_seqwide_scores_lst.append(res_p)
    ### prediction using kernel
    p = saps_kernel(matrix_)
    saps_py_lst.append(p)
    p = pdps_kernel(matrix_)
    pdps_py_lst.append(p)
    p = mix_kernel(matrix_)
    mix_py_lst.append(p)
print('prediction completed')
''' 
residue-level scores
'''
print('residue_level_scores',mix_seqwide_scores_lst)
'''
predicted values
'''
print('mix py lst',mix_py_lst)
print(np.average(mix_py_lst))
print('saps py lst',saps_py_lst)
print(np.average(saps_py_lst))
print('pdps py lst',pdps_py_lst)
print(np.average(pdps_py_lst))