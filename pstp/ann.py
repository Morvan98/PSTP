import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


class sliding_nn(nn.Module):
    def __init__(self,dim):
        super(sliding_nn, self).__init__()
        self.layer1 = nn.Linear(dim, 20)
        self.layer2 = nn.Linear(20, 5)
        self.layer3 = nn.Linear(5, 1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
         
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
                
    def slide_net(self,x,kernel_size,padding_size):
        ### x shape (1,max_len,650)
        # seq_len,ker_siz,pad= x.shape[1],129,64
        seq_len,ker_siz,pad= x.shape[1],kernel_size,padding_size
        x = x.permute(0, 2, 1)
        x_meaned = F.avg_pool1d(x,ker_siz,padding=pad,stride=1)
        x_meaned = x_meaned.permute(0, 2, 1)
        scaler_ = compute_fix_avgpool_scaler(
                seq_len,kernel_size,padding_size)
        scaler_ = torch.tensor([scaler_]).unsqueeze(-1)
        x_meaned = scaler_*x_meaned
        x_1 = self.layer1(x_meaned)
        x_1 = self.relu(x_1)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_out = self.sigmoid(x_3)
        x_out_max,_ = torch.max(x_out,dim=1,keepdim=True)
        return x_out_max.squeeze(),x_out.squeeze()
                
    def forward(self, x):
        '''
        current best
        x1,_ = self.slide_net(x,33,16)
        x2,_ = self.slide_net(x,129,64)
        x3,_ = self.slide_net(x,257,128)
        return x1*0.1+x2*0.4+x3*0.5
        '''
        x1,_ = self.slide_net(x,33,16)
        x2,_ = self.slide_net(x,129,64)
        x3,_ = self.slide_net(x,257,128)
        return (x1+x2+x3)/3
        # return x1*0.1+x2*0.4+x3*0.5
    
    def get_window_score(self, x, kernel_size=65):
        padding = kernel_size // 2
        _,window_score = self.slide_net(x,kernel_size,padding)
        return window_score.squeeze().detach().numpy()

    
def compute_fix_avgpool_scaler(len_seq,kernel_size,padding_size):
    ### convoluation will cause bias on both end
    ### use a scaler to fix it
    assert kernel_size==padding_size*2+1
    return [kernel_size/(min(len_seq,min(kernel_size,padding_size+1+min(i,len_seq-i-1))))
            for i in range(len_seq)]    


class sliding_nn_fixed(sliding_nn):
    def __init__(self,dim):
        super(sliding_nn_fixed, self).__init__(dim)
    
    def slide_net(self,x,win_size,scale=False):
        seq_len = x.shape[1]
        ker_siz = win_size
        x = x.permute(0, 2, 1)
        
        if scale == True:
            pad = (win_size-1)//2
            x_meaned = F.avg_pool1d(x,ker_siz,padding=pad,stride=1)
            x_meaned = x_meaned.permute(0, 2, 1)
            scaler_ = compute_fix_avgpool_scaler(
                    seq_len,win_size,pad)
            scaler_ = torch.tensor([scaler_]).unsqueeze(-1)
            x_meaned = scaler_*x_meaned
        else:
            x_meaned = F.avg_pool1d(x,ker_siz,padding=0,stride=1)
            x_meaned = x_meaned.permute(0, 2, 1)
        x_1 = self.layer1(x_meaned)
        x_1 = self.relu(x_1)
        # x_1 = self.dropout(x_1)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_out = self.sigmoid(x_3)
        x_out_max,_ = torch.max(x_out,dim=1,keepdim=True)
        return x_out_max.squeeze(),x_out.squeeze()
                
    def forward(self, x, win_size=513,
                scale=True):
        x,_ = self.slide_net(x,win_size,
                             scale=scale)
        return x



class kernel_only(sliding_nn):
    def __init__(self,dim):
        self.dim = dim
        super(kernel_only, self).__init__(dim)
    
    def slide_net(self,x):
        assert x.shape[-1] == self.dim 
        x_1 = self.layer1(x)
        x_1 = self.relu(x_1)
        # x_1 = self.dropout(x_1)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_out = self.sigmoid(x_3)
        return x_out.squeeze()
                
    def forward(self, x):
        x = self.slide_net(x)
        return x

# ###### testing
# net = kernel_only(650)
# # state_dict = torch.load('temp/slide_nn_model_weights/dense_650_20_5_1_weights_0.pth')
# # print(type(state_dict))
# # print(state_dict.keys())
# # net.load_state_dict(state_dict)

# for parm in net.layer1.parameters():
#     print(parm)
# print(net.layer1.weight)
# x = torch.randn(5, 650)
# net.eval()
# output = net(x)
# print(output.detach().cpu().numpy())

# # window_score = net.get_window_score(x)
# # print(window_score)
# exit()


class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, pos_weight=2, neg_weight=1):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy(input, target, reduction='none')
        weights = target * self.pos_weight + (1 - target) * self.neg_weight
        weighted_bce_loss = bce_loss * weights
        return weighted_bce_loss.mean()

def list_to_gfs_matrix(feature_group_list):
    '''
    feature group list can be [5,10,3],the first 5 as a group, next 10 as a group,
    last 3 as a group
    '''
    matrix = [[0] * sum(feature_group_list) for _ in range(len(feature_group_list))]
    start = 0
    for i, val in enumerate(feature_group_list):
        for j in range(start,start+val):
            matrix[i][j] = 1
        start += val
    
    return torch.tensor(matrix)

def gfs_regularizer(param_matrix,gfs_matrix,alpha=0.003):
    '''
    gfs matrix should be like: 
    [[1,0,0,0,0,0]
    [0,1,1,0,0,0]
    [0,0,0,1,0,0]
    [0,0,0,0,1,1]] 
    '''
    gfs_matrix = torch.tensor(gfs_matrix).to(torch.float)
    x = torch.matmul(gfs_matrix,(torch.t(param_matrix))**2)
    x = torch.sum(x,dim=1)
    x = torch.sqrt(x)
    gfs_loss = torch.sum(x)
    return alpha*gfs_loss


def train_slidenn_model(m,train_x_lst,train_y,iter=20,l1 = 0,
                        sample_neg=False,gfs=False):
    gfs_matrix_ = list_to_gfs_matrix([320,330])
    layer1_regularizer = partial(gfs_regularizer,gfs_matrix = gfs_matrix_)
    m = m.to(device)
    loss_fn = WeightedBinaryCrossEntropy(1,1)
    optimizer = optim.Adam(m.parameters(), lr=0.003,)
    original_idx = np.array([_ for _ in range(len(train_x_lst))])
    positive_idx = original_idx[np.where(train_y==1)]
    negative_idx = original_idx[np.where(train_y==0)]
    last_iter_train_pred_y, last_iter_train_y = [],[]
    print('-----training network-----')
    for i__ in tqdm(range(iter)):
        if not sample_neg:
            idx_lst = original_idx
        else:
            neg_sub_idx = np.random.choice(negative_idx,
                     int(len(positive_idx)),replace=False)
            idx_lst = np.array(list(neg_sub_idx)+list(positive_idx))
        np.random.shuffle(idx_lst)
        for sample_idx in idx_lst:
            x = torch.tensor(train_x_lst[sample_idx],dtype=torch.float).unsqueeze(0)
            x = x.to(device)
            m.train()
            optimizer.zero_grad()
            output = m.forward(x)
            target = torch.tensor(np.array(train_y[sample_idx]),dtype=torch.float).to(device)
            if not gfs:
                l1_norm = sum(p.abs().sum() for p in m.parameters())
            else:
                l1_norm = 0
                l1_norm += layer1_regularizer(m.layer1.weight)
                l1_norm += sum(p.abs().sum() for p in m.layer1.bias)
                l1_norm += sum(p.abs().sum() for p in m.layer2.parameters())
                l1_norm += sum(p.abs().sum() for p in m.layer3.parameters())
            loss = loss_fn(output,target)
            loss += l1*l1_norm + loss
            loss.backward()
            optimizer.step()
            if i__ == iter-1:
                last_iter_train_pred_y.append(output.detach().cpu().numpy())
                last_iter_train_y.append(train_y[sample_idx])
    print('training performances AUC/AUPR:')
    print(compute_aupr(last_iter_train_y,last_iter_train_pred_y))
    print(roc_auc_score(last_iter_train_y,last_iter_train_pred_y))



def slidenn_prediction(m,test_x_lst):
    m = m.to(device)
    predictions = []
    m.eval()
    for x in test_x_lst:
        x = torch.tensor(x).unsqueeze(0)
        x = x.to(device)
        output = m.forward(x)
        predictions.append(output.detach().cpu().numpy())
    return np.array(predictions)

