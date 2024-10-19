'''
pip install cython
pip install git+https://git@github.com/idptools/sparrow.git
'''
from tqdm import tqdm 
import numpy as np
import os
import sys

'''
Demo from github 
##############################################
##############################################
##############################################
START
'''
# from tqdm import tqdm
# from sparrow import Protein
# from sparrow.predictors import batch_predict
# import numpy as np
# my_cool_protein = Protein('THISISAMTHISISTHISISAMINAACIDSEQWENCEAMINAACIDSEQWENCEINAACIDSEQWENCTHISISAMINAACIDSEQWENCEE')
# print(my_cool_protein.FCR)
# print(my_cool_protein.NCPR)
# print(my_cool_protein.hydrophobicity)
# seq = 'MLSISAMTHISIMLSISAMTHISIMLSISAMTHMLSISAMTHISIMLSISAMTHMLSISAMTHISIMLSISAMTHMLSISAMTHISIMLSISAMTH'
# P = Protein(seq)
# print('asphericity',P.predictor.asphericity())
# print('radius_of_gyration',P.predictor.radius_of_gyration(use_scaled=False))
# # print('radius_of_gyration_scaled',P.predictor.radius_of_gyration(use_scaled=True))
# re = P.predictor.radius_of_gyration(use_scaled=True)
# print(re/np.sqrt(len(seq)))

# print('end_to_end_distance_scaled',P.predictor.end_to_end_distance(use_scaled=True))
# print('end_to_end_distance',P.predictor.end_to_end_distance(use_scaled=False))
# print(P.predictor.end_to_end_distance(use_scaled=True)/np.sqrt(len(seq)))
# print('scaling_exponent',P.predictor.scaling_exponent())
# print('prefactor',P.predictor.prefactor())
# ####dictionary with one sequence, but in general, you'd probably
# ###### want to pass in many...xs
# input_seqs = {1:P}
# # run batch prediction
# return_dict = batch_predict.batch_predict({1:P}, network='re')
# exit()
'''
END
Demo from github end
##############################################
##############################################
##############################################
'''


import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from parrot import encode_sequence
import sparrow
import torch
import numpy as np
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
class BRNN_MtO_modify(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(BRNN_MtO_modify, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=hidden_size*2,  
                            out_features=num_classes)
        
    def forward(self, x):
        x = x.to(self.device)
        h0 = torch.zeros(self.num_layers*2,     
                         x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2,
                         x.size(0), self.hidden_size).to(self.device)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        final_outs = torch.cat((h_n[:, :, :][-2, :], h_n[:, :, :][-1, :]), -1)
        fc_out = self.fc(final_outs)
        return fc_out,final_outs,out,h_n

class ALBATROSS_Predictor_modify():
    def __init__(self,modelweight_inner_pth,version=None):
        saved_weights = sparrow.get_data(modelweight_inner_pth)
        loaded_model = torch.load(saved_weights, map_location=device)
        num_layers = 0
        while True:
            s = f'lstm.weight_ih_l{num_layers}'
            try:
                temp = loaded_model[s]
                num_layers += 1
            except KeyError:
                break                
        number_of_classes = np.shape(loaded_model['fc.bias'])[0] 
        input_size = 20 
        hidden_vector_size = int(np.shape(loaded_model['lstm.weight_ih_l0'])[0] / 4)
        self.number_of_classes = number_of_classes
        self.input_size = input_size
        self.number_of_layers = num_layers
        self.hidden_vector_size = hidden_vector_size
        self.network = BRNN_MtO_modify(input_size, hidden_vector_size, num_layers, 
                                       number_of_classes, device)
        self.network.load_state_dict(loaded_model)
        self.network.to(device)

    def predict_property(self, seq):
        ### single seq input
        seq = seq.upper()
        seq_vector = encode_sequence.one_hot(seq)
        seq_vector = seq_vector.view(1, len(seq_vector), -1) 
        prediction,_,_,_ = self.network(seq_vector.float())
        prediction = prediction.cpu().detach().numpy().flatten()[0]
        return prediction
    
    def predict_property_batch(self,seq_batch_matrix):
        prediction,_,_,_ = self.network(seq_batch_matrix)
        prediction = prediction.cpu().detach().numpy().flatten()
        return prediction
    
    def get_pos_embedding(self,seq):
        ### single seq input
        seq = seq.upper()
        seq_vector = encode_sequence.one_hot(seq)
        seq_vector = seq_vector.view(1, len(seq_vector), -1) 
        _,_,out,_ = self.network(seq_vector.float())
        return out.cpu().detach().numpy()[0] # (seq_len, 110 or 70 or 140)
    
    def get_pos_embedding_batch(self,seq_batch_matrix):
        _,_,out,_ = self.network(seq_batch_matrix)
        return out.cpu().detach().numpy() 
    
    def get_seq_embedding(self,seq):
        ### single seq input
        seq = seq.upper()
        seq_vector = encode_sequence.one_hot(seq)
        seq_vector = seq_vector.view(1, len(seq_vector), -1) 
        _,final_out,_,_= self.network(seq_vector.float())
        return final_out.cpu().detach().numpy()[0]
    
    def get_seq_embedding_batch(self,seq_batch_matrix):
        _,final_out,_,_ = self.network(seq_batch_matrix)
        return final_out.cpu().detach().numpy()


'''
TEST SEQ EMBEDDINGS
#############################
START

'''
ALBATROSS_model_weight_path_dict = {
        'asphericity':'networks/asphericity/asphericity_network_v2.pt',
        'rg':'networks/rg/rg_network_v2.pt',
        'scaled_rg':'networks/scaled_rg/scaled_rg_network_v2.pt',
        're':'networks/re/re_network_v2.pt',
        'scaled_re':'networks/scaled_re/scaled_re_network_v2.pt',
        'scaling_exponent':'networks/scaling_exponent/scaling_exponent_network_v2.pt',
        'prefactor':'networks/prefactor/prefactor_network_v2.pt'
    }

asphericity_pred = ALBATROSS_Predictor_modify(
    ALBATROSS_model_weight_path_dict['asphericity'])
rg_pred = ALBATROSS_Predictor_modify(
    ALBATROSS_model_weight_path_dict['scaled_rg']) ### 使用scaled rg, 排除长度影响
re_pred = ALBATROSS_Predictor_modify(
    ALBATROSS_model_weight_path_dict['scaled_re']) ### 使用scaled re, 排除长度影响
scaling_exponent_pred = ALBATROSS_Predictor_modify(
    ALBATROSS_model_weight_path_dict['scaling_exponent'])
prefactor_pred = ALBATROSS_Predictor_modify(
    ALBATROSS_model_weight_path_dict['prefactor'])

def asphericity_pred_seqlist(listofseqs):
    output_ = [asphericity_pred.predict_property(seq_) for seq_ in listofseqs]
    return output_

def rg_pred_seqlist(listofseqs):
    output_ = [rg_pred.predict_property(seq_) for seq_ in listofseqs]
    return output_

def re_pred_seqlist(listofseqs):
    output_ = [re_pred.predict_property(seq_) for seq_ in listofseqs]
    return output_

def scaling_exponent_pred_seqlist(listofseqs):
    output_ = [scaling_exponent_pred.predict_property(seq_) for seq_ in listofseqs]
    return output_

def prefactor_pred_seqlist(listofseqs):
    output_ = [prefactor_pred.predict_property(seq_) for seq_ in listofseqs]
    return output_

def predict_property_by_batch(listofseqs,model_name):
    '''
    model_name should be in ['asphericity','scaled_rg','scaled_re',
    'scaling_exponent','prefactor']
    '''
    if model_name not in ['asphericity','scaled_rg','scaled_re','scaling_exponent','prefactor']:
        print(f'{model_name} not in asphericity,scaled_rg,scaled_re,scaling_exponent,prefactor,')
        exit()
    
    
    '''
    group seqs according to length
    '''
    listofseqs_array = np.array(listofseqs)
    ori_seq_idx_list = [i for i in range(len(listofseqs))]
    seq_len_idx_collection = {}
    for seq,seq_idx in zip(listofseqs,ori_seq_idx_list):
        len_seq = len(seq)
        if len_seq not in seq_len_idx_collection.keys():
            seq_len_idx_collection[len_seq] = [seq_idx]
        else:
            seq_len_idx_collection[len_seq].append(seq_idx)
    seqidx2result = {}
    model_ = ALBATROSS_Predictor_modify(    
    ALBATROSS_model_weight_path_dict[model_name])
    '''
    prediction according to batches
    '''
    for len_seq in seq_len_idx_collection.keys():
        lenseq_oriseqidx_lst = seq_len_idx_collection[len_seq]
        seqs = listofseqs_array[lenseq_oriseqidx_lst]
        numofseqs = len(seqs)
        if numofseqs > 1:
            for batch_num in range(0,numofseqs,4096):
                oriidx_batch = lenseq_oriseqidx_lst[batch_num:batch_num+4096]
                seqsbatch = seqs[batch_num:batch_num+4096]
                seqs_padded = pad_sequence([encode_sequence.one_hot(seq).float() for seq in seqsbatch], batch_first=True)
                seqs_padded = seqs_padded.to(device)
                results = model_.predict_property_batch(seqs_padded)
                for i_,pred_ in zip(oriidx_batch,results):
                    seqidx2result[i_] = pred_ 
        elif len(seqs) == 1:
            seqidx2result[lenseq_oriseqidx_lst[0]] = model_.predict_property(seqs[0])
    
    return [seqidx2result[idx] for idx in ori_seq_idx_list]

def _get_pos_wide_embedding_by_batch(listofseqs,model_name):
    '''
    part of get_albatross_pos_embedding function
    return size (len(seqs),matrix) each matrix size(len(seq), 110/70/140)
    model should be in ['asphericity','scaled_rg','scaled_re',
    'scaling_exponent','prefactor']
    '''
    if model_name not in ['asphericity','scaled_rg','scaled_re','scaling_exponent','prefactor']:
        print(f'{model_name} not in asphericity,scaled_rg,scaled_re,scaling_exponent,prefactor,')
        exit()
    
    '''
    group seqs according to length
    '''
    listofseqs_array = np.array(listofseqs)
    ori_seq_idx_list = [i for i in range(len(listofseqs))]
    seq_len_idx_collection = {}
    for seq,seq_idx in zip(listofseqs,ori_seq_idx_list):
        len_seq = len(seq)
        if len_seq not in seq_len_idx_collection.keys():
            seq_len_idx_collection[len_seq] = [seq_idx]
        else:
            seq_len_idx_collection[len_seq].append(seq_idx)
    seqidx2result = {}
    model_ = ALBATROSS_Predictor_modify(    
    ALBATROSS_model_weight_path_dict[model_name])
    
    '''
    embedding according to batches
    '''
    for len_seq in seq_len_idx_collection.keys():
        lenseq_oriseqidx_lst = seq_len_idx_collection[len_seq]
        seqs = listofseqs_array[lenseq_oriseqidx_lst]
        numofseqs = len(seqs)
        if numofseqs > 1:
            for batch_num in range(0,numofseqs,4096):
                oriidx_batch = lenseq_oriseqidx_lst[batch_num:batch_num+4096]
                seqsbatch = seqs[batch_num:batch_num+4096]
                seqs_padded = pad_sequence([encode_sequence.one_hot(seq).float() for seq in seqsbatch], batch_first=True)
                seqs_padded = seqs_padded.to(device)
                results = model_.get_pos_embedding_batch(seqs_padded)
                for i_,embed_ in zip(oriidx_batch,results):
                    seqidx2result[i_] = embed_ 
        elif len(seqs) == 1:
            seqidx2result[lenseq_oriseqidx_lst[0]] = model_.get_pos_embedding(seqs[0])
    
    return [seqidx2result[idx] for idx in ori_seq_idx_list]

def get_albatross_pos_embedding(listofseqs,listofpositions,modelname):
    '''
    return size (n,110 or 70 or 140) 
    model should be in ['asphericity','scaled_rg','scaled_re',
    'scaling_exponent','prefactor']
    '''
    embeddings = _get_pos_wide_embedding_by_batch(listofseqs,modelname)
    pos_embeddings_list = []
    for emb,pos_ in zip(embeddings,listofpositions):
        pos_embedding = emb[pos_-1,:]
        pos_embeddings_list.append(pos_embedding)
    return pos_embeddings_list

def get_seqwide_embedding_by_batch(listofseqs,model_name):
    '''
    return size (len(seqs),100)
    model should be in ['asphericity','scaled_rg','scaled_re',
    'scaling_exponent','prefactor']
    '''
    if model_name not in ['asphericity','scaled_rg','scaled_re','scaling_exponent','prefactor']:
        print(f'{model_name} not in asphericity,scaled_rg,scaled_re,scaling_exponent,prefactor,')
        exit()
    
    '''
    group seqs according to length
    '''
    listofseqs_array = np.array(listofseqs)
    ori_seq_idx_list = [i for i in range(len(listofseqs))]
    seq_len_idx_collection = {}
    for seq,seq_idx in zip(listofseqs,ori_seq_idx_list):
        len_seq = len(seq)
        if len_seq not in seq_len_idx_collection.keys():
            seq_len_idx_collection[len_seq] = [seq_idx]
        else:
            seq_len_idx_collection[len_seq].append(seq_idx)
    seqidx2result = {}
    model_ = ALBATROSS_Predictor_modify(    
    ALBATROSS_model_weight_path_dict[model_name])
    '''
    embedding according to batches
    '''
    for len_seq in seq_len_idx_collection.keys():
        lenseq_oriseqidx_lst = seq_len_idx_collection[len_seq]
        seqs = listofseqs_array[lenseq_oriseqidx_lst]
        numofseqs = len(seqs)
        if numofseqs > 1:
            for batch_num in range(0,numofseqs,4096):
                oriidx_batch = lenseq_oriseqidx_lst[batch_num:batch_num+4096]
                seqsbatch = seqs[batch_num:batch_num+4096]
                seqs_padded = pad_sequence([encode_sequence.one_hot(seq).float() for seq in seqsbatch], batch_first=True)
                seqs_padded = seqs_padded.to(device)
                results = model_.get_seq_embedding_batch(seqs_padded)
                for i_,pred_ in zip(oriidx_batch,results):
                    seqidx2result[i_] = pred_ 
        elif len(seqs) == 1:
            seqidx2result[lenseq_oriseqidx_lst[0]] = model_.get_seq_embedding(seqs[0])
    return [seqidx2result[idx] for idx in ori_seq_idx_list]

print('Importing ALBATROSS completed')