from pstp.pstp_collections import pstp_scan_saps_prediction,pstp_scan_pdps_prediction,pstp_scan_mix_prediction
from pstp.pstp_collections import pstp_embedding_by_batch
from pstp.pstp_collections import *
''' 
### predict single sequence
### saps: self-assembly ps model
### pdps: partner-dependent ps model
### mix: mix-dataset ps model
'''
test_seq = 'GRGDSPYS'*25
saps_residue_level_scores, saps_scan_predicted, saps_scan_kernel_predicted = pstp_scan_saps_prediction(
    test_seq)
print(len(saps_residue_level_scores),len(test_seq))
pdps_residue_level_scores, pdps_scan_predicted, pdps_scan_kernel_predicted = pstp_scan_pdps_prediction(
    test_seq)
mix_residue_level_scores, mix_scan_predicted, mix_scan_kernel_predicted = pstp_scan_mix_prediction(
    test_seq)

##########################################
##########################################
##########################################
### batch embedding and prediction
seqs = ['GRGDSPYS'*25,
        'ARADSPYS'*25,
        'SRSDSPYS'*25,
        'GRGDSPYS'*24,
        'GRGDSPYS'*23,
        'GRGDSPYS'*20,
        ]
seq_matrix_lst = pstp_embedding_by_batch(seqs)


''' 
residue level prediction and sequence level prediction by PSTP-Scan
res_p: residue level prediction
'''
saps_py_lst,pdps_py_lst,\
    mix_py_lst = [],[],[]
for matrix_ in seq_matrix_lst:
    res_p,p = predict_by_saps_models(matrix_)
    saps_py_lst.append(p)
    res_p,p = predict_by_pdps_models(matrix_)
    pdps_py_lst.append(p)
    res_p,p = predict_by_mix_models(matrix_)
    mix_py_lst.append(p)

print(saps_py_lst,pdps_py_lst,mix_py_lst)



''' 
residue level prediction and sequence level prediction by PSTP-Scan kernel
'''
saps_py_lst,pdps_py_lst,\
    mix_py_lst = [],[],[]

for matrix_ in seq_matrix_lst:
    p = saps_kernel(matrix_)
    saps_py_lst.append(p)
    p = pdps_kernel(matrix_)
    pdps_py_lst.append(p)
    p = mix_kernel(matrix_)
    mix_py_lst.append(p)

print(saps_py_lst,pdps_py_lst,mix_py_lst)