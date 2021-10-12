import torch
import pickle
import numpy as np

weights = 'mcfairmot_dla34.pth'
target_name = 'mcfairmot_dla34_pd.pdparams'
src = torch.load(weights)['state_dict']
dst = {}
with open('parameters_name.pkl', 'rb') as f:
    paddle_name = pickle.load(f)
i = 0
torch2paddle = {}
for k, v in src.items():
    if 'num_batches_tracked' in k:
        continue
    if True in [k.startswith(n) for n in ['classifier', 'hm', 'wh', 'id', 'reg']]:
        torch2paddle[k] = k
    elif k == 's_det':
        torch2paddle[k] = 'fair_mot_loss_0.w_0'
    elif k == 's_id':
        torch2paddle[k] = 'fair_mot_loss_0.w_1'
    else:
        torch2paddle[k] = paddle_name[i]
    print(k, '=====', torch2paddle[k])
    i += 1
for k, v in src.items():
    if 'num_batches_tracked' in k:
        continue
    if k == 'classifier.weight':
        dst[torch2paddle[k]] = np.array(src[k].T.cpu())  ###
    else:
        dst[torch2paddle[k]] = np.array(src[k].cpu())
pickle.dump(dst, open(target_name, 'wb'), protocol=2)


