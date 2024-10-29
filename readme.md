## Python requirements
python >= 3.7; python <= 3.9
## Please install the following packages first:
pip install cython  
pip install numpy  
pip install git+https://git@github.com/idptools/sparrow.git  
pip install idptools-parrot[optimize]  
pip install "fair-esm[esmfold]"  
## Installation
pip install git+https://git@github.com/Morvan98/PSTP.git
## Usage and examples
### single sequence prediction
```python
from pstp.pstp_collections import pstp_scan_saps_prediction # saps: self-assembly ps model
from pstp.pstp_collections import pstp_scan_pdps_prediction # pdps: partner-dependent ps model
from pstp.pstp_collections import pstp_scan_mix_prediction # mix: mix-dataset ps model
test_seq = 'GRGDSPYS'*25
saps_residue_level_scores, saps_scan_predicted, saps_scan_kernel_predicted = pstp_scan_saps_prediction(
    test_seq)
print(len(saps_residue_level_scores),len(test_seq))
pdps_residue_level_scores, pdps_scan_predicted, pdps_scan_kernel_predicted = pstp_scan_pdps_prediction(
    test_seq)
mix_residue_level_scores, mix_scan_predicted, mix_scan_kernel_predicted = pstp_scan_mix_prediction(
    test_seq)
```

### batch embedding and prediction
```python
from pstp.pstp_collections import pstp_embedding_by_batch
from pstp.pstp_collections import predict_by_saps_models,predict_by_pdps_models,predict_by_mix_models # PSTP-Scan models
from pstp.pstp_collections import saps_kernel,pdps_kernel,mix_kernel # trained MLP kernels of PSTP-Scan
seqs = ['GRGDSPYS'*25,
        'ARADSPYS'*25,
        'SRSDSPYS'*25,
        'GRGDSPYS'*24,
        'GRGDSPYS'*23,
        'GRGDSPYS'*20,
        ]
seq_matrix_lst = pstp_embedding_by_batch(seqs)
for matrix in seq_matrix_lst:
    print(matrix.shape)

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
saps_py_lst,pdps_py_lst,mix_py_lst = [],[],[]
for matrix_ in seq_matrix_lst:
    p = saps_kernel(matrix_)
    saps_py_lst.append(p)
    p = pdps_kernel(matrix_)
    pdps_py_lst.append(p)
    p = mix_kernel(matrix_)
    mix_py_lst.append(p)

print(saps_py_lst,pdps_py_lst,mix_py_lst)
```
