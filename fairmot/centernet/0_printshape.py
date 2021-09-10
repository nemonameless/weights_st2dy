import torch
f = open('model_dict_ct_hardnet.txt','a+')
model_dict = torch.load('centernet_hardnet85_coco_608.pth')['state_dict']
not_keys = ['num_batches_tracked'] #, 'running_mean', 'running_var']

for key in model_dict.keys():
    write_flag = 1
    for nk in not_keys:
        if nk in key:
            write_flag=0
            break
    if write_flag:
        shape = model_dict[key].shape
        if len(shape)==4:
            line = '{} [{},{},{},{}]\n'.format(key, shape[0], shape[1], shape[2], shape[3])
        elif len(shape)==2:
            line = '{} [{},{}]\n'.format(key, shape[0], shape[1])
        else:
            line = '{} [{}]\n'.format(key, shape[0])
        #print(line)
        f.write(line)
f.close()
