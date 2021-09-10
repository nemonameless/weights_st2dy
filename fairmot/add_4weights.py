import paddle
a=paddle.load('')
b=paddle.load('')
a.update({'reid.classifier.weight':b['reid.classifier.weight']})
a.update({'reid.classifier.bias':b['reid.classifier.bias']})
a.update({'loss.det_weight':b['loss.det_weight']})
a.update({'loss.reid_weight':b['loss.reid_weight']})
paddle.save(a, 'new_fairmot_hd85.pdparams')

