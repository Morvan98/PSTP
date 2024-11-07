print('---loading ESM2 model---')
import torch
import esm
import numpy as np
from Bio import SeqIO
import time
from tqdm import tqdm
import pandas as pd
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

print('---loading or downloading esm2_t6_8M_UR50D model')
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
tokens_list = alphabet.all_toks
print('---loading or downloading esm2_t6_8M_UR50D model completed')
batch_converter = alphabet.get_batch_converter()
model.eval()

dev1ce = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dev1ce = 'cpu'

model = model.to(dev1ce)
if torch.cuda.device_count() > 0:
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(dev1ce)
else:
    model.cpu()

# window_size = 1024
window_size = 256 
chunk_size_ = 10
def seq_list_2_embedding(seq_list_fixlen,chunk_size=chunk_size_):
    seq_list_original = seq_list_fixlen.copy()
    seq_list = list(set(seq_list_original))
    data_ = [[f'seq_{idx+1}',seq_] for idx,seq_ in enumerate(seq_list)]
    matrix_output = []
    len_chunk_1 = False
    for chunk_i in range(0,len(data_),chunk_size):
        data_chunk = data_[chunk_i:chunk_i+chunk_size]
        if len(data_chunk) == 1:
            len_chunk_1 = True
            data_chunk = data_chunk + data_chunk
        _,_,batch_tokens = batch_converter(data_chunk)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(dev1ce)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_representations = results["representations"][6]
        sequence_representations = []
        for seq_idx_, tokens_len in enumerate(batch_lens):
            matrix_ = token_representations[seq_idx_, 1 : tokens_len - 1].mean(0).cpu().numpy()
            sequence_representations.append(matrix_)
        if len_chunk_1: 
            matrix_output += [sequence_representations[0]]
        else:
            matrix_output += sequence_representations
    seq2matrix = dict(zip(seq_list,matrix_output))
    matrix_output_original = [seq2matrix[seq] for seq in seq_list_original]
    return matrix_output_original

def _seq_2_embedding_pos_matrix_maxsize_window(seq_list_fixlen,chunk_size=chunk_size_):
    ''' 
    input a list of seq, return pos-wide embedding os these seqs
    '''
    seq_list = seq_list_fixlen.copy()
    data_ = [[f'seq_{idx+1}',seq_] for idx,seq_ in enumerate(seq_list)]
    embedding_output = []
    for chunk_i in range(0,len(data_),chunk_size):
        data_chunk = data_[chunk_i:chunk_i+chunk_size]
        len_chunk_1 = False
        if len(data_chunk) == 1:
            len_chunk_1 = True
            data_chunk = data_chunk + data_chunk
        _,_,batch_tokens = batch_converter(data_chunk)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(dev1ce)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_representations = results["representations"][6]
        embeddings = []
        for seq_idx_, tokens_len in enumerate(batch_lens):
            matrix_ = token_representations[seq_idx_, 1 : tokens_len - 1].cpu().numpy()
            embeddings.append(matrix_)
        if len_chunk_1:
            embedding_output += [embeddings[0]]
        else:
            embedding_output += embeddings
    return embedding_output


def _generate_sliding_sequences(sequence, window_size=window_size, step_size=50):
    sequence_length = len(sequence)
    if sequence_length < window_size:
        return [sequence]
    sequences = []
    for start in range(0, sequence_length - window_size + step_size, step_size):
        end = min(sequence_length,start + window_size)
        sequences.append(sequence[start:end])
    return sequences

RESIDUES_SET = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}

def check_seq_token(seq_to_check):
    if not all(res in RESIDUES_SET for res in seq_to_check):
        return ''.join([res if res in RESIDUES_SET else 'L' for res in seq_to_check])
    else:
        return seq_to_check
    
def get_seq_embedding(seq_list,winsize=256):
    window_size = winsize
    ### seq to list of subseqs
    seq_idx_lst,short_seqs,long_seqs = [],[],[]
    for seq_idx,seq in enumerate(seq_list):
        seq_idx_lst.append(seq_idx)
        seq = check_seq_token(seq)
        if len(seq) <= window_size:
            short_seqs.append((seq_idx,seq))
        else:
            long_seqs.append((seq_idx,seq))
    # print(f'embedding seq len < {window_size}')
    short_seqs_idx_lst,short_seqs_lst = [i[0] for i in short_seqs],[i[1] for i in short_seqs]
    short_seqs_embeddings = seq_list_2_embedding(short_seqs_lst)
    idx2embeddings = dict(zip(short_seqs_idx_lst,short_seqs_embeddings))
    # print(f'embedding seq len > {window_size}')
    long_seqs_embeddings = []
    for item in long_seqs:
        long_seq = item[1]
        sub_seqs = _generate_sliding_sequences(long_seq,window_size=window_size)
        embeddings = np.average(seq_list_2_embedding(sub_seqs,len(sub_seqs)),0)
        long_seqs_embeddings.append(embeddings)
         
    long_seqs_idx_lst = [i[0] for i in long_seqs]
    long_seq_idx2embeddings = dict(zip(long_seqs_idx_lst,long_seqs_embeddings))
    idx2embeddings.update(long_seq_idx2embeddings)
    return [idx2embeddings[idx] for idx in seq_idx_lst]

def _get_seq_embedding_matrix_long_seq_input(seq,step_size=50,window_size=window_size):
    # each time one sequence as input
    sequence_length = len(seq)
    idxs = [_ for _ in range(sequence_length)]
    assert len(seq) > window_size
    sequences = []
    position_split_dict_list = []
    for start in range(0, sequence_length - window_size + step_size, step_size):
        end = min(sequence_length,start + window_size)
        sub_seq = seq[start:end]
        sequences.append(sub_seq)
        position_split_dict_list.append(dict(zip(idxs[start:end],[_ for _ in range(len(sub_seq))])))
    embeddings = _seq_2_embedding_pos_matrix_maxsize_window(sequences)
    position_wide_embedding_for_lst = []
    for idx in idxs:
        ### get deepest position embeddings
        mapped_pos_dict = []
        mapped_shortest_dist = []
        for pos_dict in position_split_dict_list:
            if idx in pos_dict.keys():
                mapped_pos_dict.append(pos_dict)
                relative_pos = pos_dict[idx]
                mapped_shortest_dist.append(min(relative_pos,window_size-relative_pos))
            else:
                mapped_pos_dict.append({})
                mapped_shortest_dist.append(-1)
        best_subseq_idx = np.argmax(mapped_shortest_dist)
        best_subseq_pos_dict = mapped_pos_dict[best_subseq_idx]
        best_subseq_embedding = embeddings[best_subseq_idx]
        position_embedding = best_subseq_embedding[best_subseq_pos_dict[idx]]
        position_wide_embedding_for_lst.append(position_embedding)
    return np.array(position_wide_embedding_for_lst)

def get_seq_poswide_embedding_matrix(seq_list):
    '''
    return embedding with size (sample_num,seq_length,320)
    '''
     ### seq to list of subseqs
    seq_idx_lst,short_seqs,long_seqs = [],[],[]
    for seq_idx,seq in enumerate(seq_list):
        seq_idx_lst.append(seq_idx)
        seq = check_seq_token(seq)
        if len(seq) <= window_size:
            short_seqs.append((seq_idx,seq))
        else:
            long_seqs.append((seq_idx,seq))
    # print(f'embedding seq len < {window_size}')
    short_seqs_idx_lst,short_seqs_lst = [i[0] for i in short_seqs],[i[1] for i in short_seqs]
    short_seqs_embeddings = _seq_2_embedding_pos_matrix_maxsize_window(short_seqs_lst)
    idx2embeddings = dict(zip(short_seqs_idx_lst,short_seqs_embeddings))
    # print(f'embedding seq len > {window_size}')
    long_seqs_embeddings = []
    for item in long_seqs:
        long_seq = item[1]
        embeddings = _get_seq_embedding_matrix_long_seq_input(long_seq)
        long_seqs_embeddings.append(embeddings)
         
    long_seqs_idx_lst = [i[0] for i in long_seqs]
    long_seq_idx2embeddings = dict(zip(long_seqs_idx_lst,long_seqs_embeddings))
    idx2embeddings.update(long_seq_idx2embeddings)
    return [idx2embeddings[idx] for idx in seq_idx_lst]
        # return a list of matrix, each matrix is of shape (seq_len,320)
    
