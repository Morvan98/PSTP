from .tools_nn_related import * 

RESIDUES_SET = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
def _check_seq_tool(seq):
    ''' 
    input: seq\n
    checking illegal amino acid characters
    '''
    seq = str(seq)
    if set(list(seq)) | RESIDUES_SET == RESIDUES_SET:
        return
    else:
        raise Exception('Illegal residue character(s)!')

# check_seq_tool('ACDEFG')
# check_seq_tool('ABCDEFGHIJK')


def pstp_embedding(seq):
    seq = str(seq)
    # check illegal character
    _check_seq_tool(seq)
    return seq2matrix_lst([seq])


def pstp_embedding_by_batch(seq_lst):
    seq_lst_ = [str(s) for s in seq_lst]
    for s in seq_lst:
        _check_seq_tool(s)
    return seq2matrix_lst(seq_lst_)

def pstp_scan_saps_prediction(seq):
    '''
    input: seq\n
    output:\n
    1.sequence-length residue-level values\n
    2.sequence-level propensity by PSTP-Scan\n
    3.sequence-level propensity by Scan-Kernel\n
    '''
    seq = str(seq)
    # check illegal character
    _check_seq_tool(seq)
    seqmatrixlst = seq2matrix_lst([seq])
    res_p,seq_p, = predict_by_saps_models(seqmatrixlst[0])
    kernel_p = saps_kernel(seqmatrixlst[0])
    return res_p,seq_p,kernel_p

def pstp_scan_pdps_prediction(seq):
    '''
    input: seq\n
    output:\n
    1.sequence-length residue-level values\n
    2.sequence-level propensity by PSTP-Scan\n
    3.sequence-level propensity by Scan-Kernel\n
    '''
    seq = str(seq)
    # check illegal character
    _check_seq_tool(seq)
    seqmatrixlst = seq2matrix_lst([seq])
    res_p,seq_p, = predict_by_pdps_models(seqmatrixlst[0])
    kernel_p = pdps_kernel(seqmatrixlst[0])
    return res_p,seq_p,kernel_p

def pstp_scan_mix_prediction(seq):
    '''
    input: seq\n
    output:\n
    1.sequence-length residue-level values\n
    2.sequence-level propensity by PSTP-Scan\n
    3.sequence-level propensity by Scan-Kernel\n
    '''
    seq = str(seq)
    # check illegal character
    _check_seq_tool(seq)
    seqmatrixlst = seq2matrix_lst([seq])
    res_p,seq_p, = predict_by_mix_models(seqmatrixlst[0])
    kernel_p = mix_kernel(seqmatrixlst[0])
    return res_p,seq_p,kernel_p

# print(pstp_embedding('ACDEFG'))

    


# # test_seqs = [
# # 'RRQWRPRRFS'*20,
# # 'RRRRRERRDL'*20,
# # 'KRKKQRSRRN'*20,
# # 'KNRKAKAKPV'*20,
# # 'RRQRKSRRTI'*20,
# # 'KRKKQRSRRN'*20,
# # 'RKKKALRIHS'*20,
# # 'SLSKMLKKRS'*20,
# # 'RRRRRERRDL'*20,
# # 'KIYDLHKKRS'*20,
# # 'RRRRARLRFM'*20,
# # 'KKRIKPIVWP'*20,
# # 'RRQWRPRRFS'*20,
# # 'KVGFFKRNLK'*20,]

# test_seqs = [
# 'RRQWRPRRFS',
# 'RRRRRERRDL',
# 'KRKKQRSRRN',
# 'KNRKAKAKPV',
# 'RRQRKSRRTI',
# 'KRKKQRSRRN',
# 'RKKKALRIHS',
# 'SLSKMLKKRS',
# 'RRRRRERRDL',
# 'KIYDLHKKRS',
# 'RRRRARLRFM',
# 'KKRIKPIVWP',
# 'RRQWRPRRFS',
# 'KVGFFKRNLK',]



# for i,s in enumerate(test_seqs):
#     print(s[:10],i)
#     print(pstp_scan_saps_prediction(s)[2])
#     print(pstp_scan_pdps_prediction(s)[2])
#     print(pstp_scan_mix_prediction(s)[2])
