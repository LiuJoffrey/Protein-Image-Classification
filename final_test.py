#!/usr/bin/env python -W ignore::DeprecationWarning
import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")


from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt

print("import finish")
PATH = "./"
TRAIN = '/media/cnrgntu11/left PC/r06942141/ML_final/train/'
TEST = '/media/cnrgntu11/left PC/r06942141/ML_final/test/'
LABELS = '/media/cnrgntu11/left PC/r06942141/ML_final/train.csv'
SAMPLE = '//media/cnrgntu11/left PC/r06942141/ML_final/sample_submission.csv'
MODEL = '/media/cnrgntu11/left PC/r06942141/ML_final/'

name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',   
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',   
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',   
14:  'Microtubules',
15:  'Microtubule ends',  
16:  'Cytokinetic bridge',   
17:  'Mitotic spindle',
18:  'Microtubule organizing center',  
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',   
22:  'Cell junctions', 
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',   
27:  'Rods & rings' }

nw = 2   #number of workers for data loader
arch = resnet34 #specify target architecture

train_names = list({f[:36] for f in os.listdir(TRAIN)})
test_names = list({f[:36] for f in os.listdir(TEST)})
tr_n, val_n = train_test_split(train_names, test_size=0.1, random_state=42)
#tr_n = train_names

labels = pd.read_csv(LABELS).set_index('Id')
name = labels.index.tolist()


group1_c = [0,1,2,3,4,5]
group2_c = [6,7,11,12,13,14,15,16,17]
group3_c = [8,9,10,18,19,20,23,27]
group4_c = [21,22,24,25,26]
group0485_c = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
group2_n = []
gropu2_total_c = []
count = 0
for i in range(len(name)):
    s = labels.loc[name[i]]['Target']
    l = [int(i) for i in s.split()]
    for c in group1_c:
        if c in l:
            #print(l)
            gropu2_total_c.append(l)
            group2_n.append(name[i])
            #print(labels.loc[name[i]]['Target'])
            count += 1

            break
#print(labels.loc[group1_n[0]]['Target'])
print(len(gropu2_total_c), " ", len(group2_n))
print(count)

# tr_n = group2_n
# val_n = group2_n
name_label_dict = {
0:  'Peroxisomes',
1:  'Endosomes',
2:  'Lysosomes',   
3:  'Microtubule organizing center',
4:  'Centrosome',
5:  'Lipid droplets',
6:  'Mitochondria',
7:  'Rods & rings'}

def open_rgby(path,id): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags).astype(np.float32)/255 for color in colors]
    return np.stack(img, axis=-1)




class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.labels = pd.read_csv(LABELS).set_index('Id')
        self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        img = open_rgby(self.path,self.fnames[i])
        if self.sz == 512: return img 
        else: return cv2.resize(img, (self.sz, self.sz),cv2.INTER_AREA)
    
    def get_y(self, i):
        if(self.path == TEST): return np.zeros(len(name_label_dict),dtype=np.int)
        else:
            labels = self.labels.loc[self.fnames[i]]['Target']

            new_label = []
            for i in range(len(labels)):
                if labels[i] in group_c:
                    new_label.append(group_c.index(labels[i]))
            
            #print(labels, " ", new_label, " ",np.eye(len(name_label_dict),dtype=np.float)[new_label].sum(axis=0))
            return np.eye(len(name_label_dict),dtype=np.float)[new_label].sum(axis=0)
        
    @property
    def is_multi(self): return True
    @property
    def is_reg(self):return True
    #this flag is set to remove the output sigmoid that allows log(sigmoid) optimization
    #of the numerical stability of the loss function
    
    def get_c(self): return len(name_label_dict) #number of classes
    
    

def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(30, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    #mean and std in of each channel in the train set
    stats = A([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])
    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.NO, 
                aug_tfms=aug_tfms)
    if len(tr_n)%bs != 0:				
        ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN), 
                    (val_n,TRAIN), tfms, test=(test_names,TEST))
    else:
        ds = ImageData.get_ds(pdFilesDataset, (tr_n,TRAIN), 
                    (val_n,TRAIN), tfms, test=(test_names,TEST)) 
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    return md
    
    

bs = 16
sz = 512
"""
md = get_data(sz,bs)

x,y = next(iter(md.trn_dl))
x.shape, y.shape
"""

def display_imgs(x):
    columns = 4
    bs = x.shape[0]
    rows = min((bs+3)//4,4)
    fig=plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        for j in range(columns):
            idx = i+j*columns
            fig.add_subplot(rows, columns, idx+1)
            plt.axis('off')
            plt.imshow((x[idx,:,:,:3]*255).astype(np.int))
    plt.show()
    


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()





def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

class ConvnetBuilder_custom():
    def __init__(self, f, c, is_multi, is_reg, ps=None, xtra_fc=None, xtra_cut=0, 
                 custom_head=None, pretrained=True):
        self.f,self.c,self.is_multi,self.is_reg,self.xtra_cut = f,c,is_multi,is_reg,xtra_cut
        if xtra_fc is None: xtra_fc = [512]
        if ps is None: ps = [0.25]*len(xtra_fc) + [0.5]
        self.ps,self.xtra_fc = ps,xtra_fc

        if f in model_meta: cut,self.lr_cut = model_meta[f]
        else: cut,self.lr_cut = 0,0
        cut-=xtra_cut
        layers = cut_model(f(pretrained), cut)
        
        #replace first convolutional layer by 4->64 while keeping corresponding weights
        #and initializing new weights with zeros
        w = layers[0].weight
        layers[0] = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
        layers[0].weight = torch.nn.Parameter(torch.cat((w,(w[:,1:2,:,:]+w[:,2:3,:,:])/2),dim=1))
        
        self.nf = model_features[f] if f in model_features else (num_features(layers)*2)
        if not custom_head: layers += [AdaptiveConcatPool2d(), Flatten()]
        self.top_model = nn.Sequential(*layers)

        n_fc = len(self.xtra_fc)+1
        if not isinstance(self.ps, list): self.ps = [self.ps]*n_fc

        if custom_head: fc_layers = [custom_head]
        else: fc_layers = self.get_fc_layers()
        self.n_fc = len(fc_layers)
        self.fc_model = to_gpu(nn.Sequential(*fc_layers))
        if not custom_head: apply_init(self.fc_model, kaiming_normal)
        self.model = to_gpu(nn.Sequential(*(layers+fc_layers)))

    @property
    def name(self): return f'{self.f.__name__}_{self.xtra_cut}'

    def create_fc_layer(self, ni, nf, p, actn=None):
        res=[nn.BatchNorm1d(num_features=ni)]
        if p: res.append(nn.Dropout(p=p))
        res.append(nn.Linear(in_features=ni, out_features=nf))
        if actn: res.append(actn)
        return res

    def get_fc_layers(self):
        res=[]
        ni=self.nf
        for i,nf in enumerate(self.xtra_fc):
            res += self.create_fc_layer(ni, nf, p=self.ps[i], actn=nn.ReLU())
            ni=nf
        final_actn = nn.Sigmoid() if self.is_multi else nn.LogSoftmax()
        if self.is_reg: final_actn = None
        res += self.create_fc_layer(ni, self.c, p=self.ps[-1], actn=final_actn)
        return res

    def get_layer_groups(self, do_fc=False):
        if do_fc:
            return [self.fc_model]
        idxs = [self.lr_cut]
        c = children(self.top_model)
        if len(c)==3: c = children(c[0])+c[1:]
        lgs = list(split_by_idxs(c,idxs))
        return lgs+[self.fc_model]
    
class ConvLearner(Learner):
    def __init__(self, data, models, precompute=False, **kwargs):
        self.precompute = False
        super().__init__(data, models, **kwargs)
        if hasattr(data, 'is_multi') and not data.is_reg and self.metrics is None:
            self.metrics = [accuracy_thresh(0.5)] if self.data.is_multi else [accuracy]
        if precompute: self.save_fc1()
        self.freeze()
        self.precompute = precompute

    def _get_crit(self, data):
        if not hasattr(data, 'is_multi'): return super()._get_crit(data)

        return F.l1_loss if data.is_reg else F.binary_cross_entropy if data.is_multi else F.nll_loss

    @classmethod
    def pretrained(cls, f, data, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, precompute=False,
                   pretrained=True, **kwargs):
        models = ConvnetBuilder_custom(f, data.c, data.is_multi, data.is_reg,
            ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut, custom_head=custom_head, pretrained=pretrained)
        return cls(data, models, precompute, **kwargs)

    @classmethod
    def lsuv_learner(cls, f, data, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, precompute=False,
                  needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=False, **kwargs):
        models = ConvnetBuilder(f, data.c, data.is_multi, data.is_reg,
            ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut, custom_head=custom_head, pretrained=False)
        convlearn=cls(data, models, precompute, **kwargs)
        convlearn.lsuv_init()
        return convlearn
    
    @property
    def model(self): return self.models.fc_model if self.precompute else self.models.model
    
    def half(self):
        if self.fp16: return
        self.fp16 = True
        if type(self.model) != FP16: self.models.model = FP16(self.model)
        if not isinstance(self.models.fc_model, FP16): self.models.fc_model = FP16(self.models.fc_model)
    def float(self):
        if not self.fp16: return
        self.fp16 = False
        if type(self.models.model) == FP16: self.models.model = self.model.module.float()
        if type(self.models.fc_model) == FP16: self.models.fc_model = self.models.fc_model.module.float()

    @property
    def data(self): return self.fc_data if self.precompute else self.data_

    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0,n), np.float32), chunklen=1, mode='w', rootdir=name)

    def set_data(self, data, precompute=False):
        super().set_data(data)
        if precompute:
            self.unfreeze()
            self.save_fc1()
            self.freeze()
            self.precompute = True
        else:
            self.freeze()

    def get_layer_groups(self):
        return self.models.get_layer_groups(self.precompute)

    def summary(self):
        precompute = self.precompute
        self.precompute = False
        res = super().summary()
        self.precompute = precompute
        return res

    def get_activations(self, force=False):
        tmpl = f'_{self.models.name}_{self.data.sz}.bc'
        # TODO: Somehow check that directory names haven't changed (e.g. added test set)
        names = [os.path.join(self.tmp_path, p+tmpl) for p in ('x_act', 'x_act_val', 'x_act_test')]
        if os.path.exists(names[0]) and not force:
            self.activations = [bcolz.open(p) for p in names]
        else:
            self.activations = [self.create_empty_bcolz(self.models.nf,n) for n in names]

    def save_fc1(self):
        self.get_activations()
        act, val_act, test_act = self.activations
        m=self.models.top_model
        if len(self.activations[0])!=len(self.data.trn_ds):
            predict_to_bcolz(m, self.data.fix_dl, act)
        if len(self.activations[1])!=len(self.data.val_ds):
            predict_to_bcolz(m, self.data.val_dl, val_act)
        if self.data.test_dl and (len(self.activations[2])!=len(self.data.test_ds)):
            if self.data.test_dl: predict_to_bcolz(m, self.data.test_dl, test_act)

        self.fc_data = ImageClassifierData.from_arrays(self.data.path,
                (act, self.data.trn_y), (val_act, self.data.val_y), self.data.bs, classes=self.data.classes,
                test = test_act if self.data.test_dl else None, num_workers=8)

    def freeze(self):
        self.freeze_to(-1)

    def unfreeze(self):
        self.freeze_to(0)
        self.precompute = False

    def predict_array(self, arr):
        precompute = self.precompute
        self.precompute = False
        pred = super().predict_array(arr)
        self.precompute = precompute
        return pred

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

def F1_soft(preds,targs,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    return score

def fit_val(x,y):
    params = 0.5*np.ones(len(name_label_dict))
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x,y,p) - 1.0,
                                      wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p

sz = 512 #image size
bs = 64  #batch size



group1_c = [0,1,2,3,4,5]
group2_c = [6,7,11,12,13,14,15,16,17]
group3_c = [8,9,10,18,19,20,23,27]
group4_c = [21,22,24,25,26]
group0485_c = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

#group1

group_c = group1_c
print(group_c, "group1 start")
name_label_dict = {0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',   
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies'}

md = get_data(sz,bs)
learner = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%

learner.load(os.path.join("./",'ResNet34_group1'))
preds_t1, y_t1 = learner.TTA(n_aug=16, is_test=True)
preds_t1 = np.stack(preds_t1, axis=-1)
preds_t1 = sigmoid_np(preds_t1)
np.save("./group1_pre.npy", preds_t1)
np.save("./group1_y.npy", y_t1)
print("group1 end")

#group2
group_c = group2_c
print(group_c, "group2 start")
name_label_dict = {0: 'Endoplasmic reticulum', 1:'Golgi apparatus', 2:'Intermediate filaments', 3:'Actin filaments',4:'Focal adhesion sites',
                   5: 'Microtubules', 6:'Microtubule ends',  7:'Cytokinetic bridge',  8:'Mitotic spindle'}

md = get_data(sz,bs)
learner = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%

learner.load(os.path.join("./",'ResNet34_group2'))
preds_t2,y_t2 = learner.TTA(n_aug=16, is_test=True)
preds_t2 = np.stack(preds_t2, axis=-1)
preds_t2 = sigmoid_np(preds_t2)
np.save("./group2_pre.npy", preds_t2)
np.save("./group2_y.npy", y_t2)
print("group2 end")

#group3
group_c = group3_c
print(group_c, "group3 start")
name_label_dict = {
0:  'Peroxisomes',
1:  'Endosomes',
2:  'Lysosomes',   
3:  'Microtubule organizing center',
4:  'Centrosome',
5:  'Lipid droplets',
6:  'Mitochondria',
7:  'Rods & rings'}
md = get_data(sz,bs)
learner = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%

learner.load(os.path.join("./",'ResNet34_group3'))
preds_t3,y_t3 = learner.TTA(n_aug=16, is_test=True)
preds_t3 = np.stack(preds_t3, axis=-1)
preds_t3 = sigmoid_np(preds_t3)
np.save("./group3_pre.npy", preds_t3)
np.save("./group3_y.npy", y_t3)
print("group3 end")

#group4
group_c = group4_c
print(group_c, "group4 start")
name_label_dict = {
0:  'Plasma membrane',
1:  'Cell junctions',
2:  'Aggresome',   
3:  'Cytosol',
4:  'Cytoplasmic bodies'}

md = get_data(sz,bs)
learner = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%

learner.load(os.path.join("./",'ResNet34_group4'))
preds_t4,y_t4 = learner.TTA(n_aug=16, is_test=True)
preds_t4 = np.stack(preds_t4, axis=-1)
preds_t4 = sigmoid_np(preds_t4)
np.save("./group4_pre.npy", preds_t4)
np.save("./group4_y.npy", y_t4)
print("group4 end")

#model 0.485
group_c = group0485_c
print(group_c, "model 0.485 start")
name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',   
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',   
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',   
14:  'Microtubules',
15:  'Microtubule ends',  
16:  'Cytokinetic bridge',   
17:  'Mitotic spindle',
18:  'Microtubule organizing center',  
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',   
22:  'Cell junctions', 
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',   
27:  'Rods & rings' }
md = get_data(sz,bs)
learner = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%

learner.load(os.path.join("./",'ResNet34_0.485'))
preds_t0485,y_t0485 = learner.TTA(n_aug=16,is_test=True)
preds_t0485 = np.stack(preds_t0485, axis=-1)
preds_t0485 = sigmoid_np(preds_t0485)
np.save("./model_485.npy", preds_t0485)
np.save("./485_y.npy", y_t0485)
print("model_0.485 end")
exit()

def ensemble_pred():
    preds_t = np.load('model_0.485.npz.npy')
    preds_t1 = np.load('group1_pre.npz.npy')
    preds_t2 = np.load('group2_pre.npz.npy')
    preds_t3 = np.load('group3_pre.npz.npy')
    preds_t4 = np.load('group4_pre.npz.npy')
    print(preds_t.shape)
    print(preds_t1.shape)
    print(preds_t2.shape)
    print(preds_t3.shape)
    print(preds_t4.shape)

    min_shape = min(preds_t1.shape[0], preds_t2.shape[0], preds_t3.shape[0], preds_t4.shape[0])
    print(min_shape)
    preds_t = np.zeros((min_shape, 28, 20))

    preds_t = preds_t.max(axis=-1)
    preds_t1 = preds_t1.max(axis=-1)
    preds_t2 = preds_t2.max(axis=-1)
    preds_t3 = preds_t3.max(axis=-1)
    preds_t4 = preds_t4.max(axis=-1)

    
    #group1
    preds_t[:, 0] = preds_t1[:min_shape,0]
    preds_t[:, 1] = preds_t1[:min_shape,1]
    preds_t[:, 2] = preds_t1[:min_shape,2]
    preds_t[:, 3] = preds_t1[:min_shape,3]
    preds_t[:, 4] = preds_t1[:min_shape,4]
    preds_t[:, 5] = preds_t1[:min_shape,5]
    #group2
    preds_t[:, 6] = preds_t2[:min_shape,0]
    preds_t[:, 7] = preds_t2[:min_shape,1]
    preds_t[:, 11] = preds_t2[:min_shape,2]
    preds_t[:, 12] = preds_t2[:min_shape,3]
    preds_t[:, 13] = preds_t2[:min_shape,4]
    preds_t[:, 14] = preds_t2[:min_shape,5]
    preds_t[:, 15] = preds_t2[:min_shape,6]
    preds_t[:, 16] = preds_t2[:min_shape,7]
    preds_t[:, 17] = preds_t2[:min_shape,8]
    #group3
    preds_t[:, 8] = preds_t3[:min_shape,0]
    preds_t[:, 9] = preds_t3[:min_shape,1]
    preds_t[:, 10] = preds_t3[:min_shape,2]
    preds_t[:, 18] = preds_t3[:min_shape,3]
    preds_t[:, 19] = preds_t3[:min_shape,4]
    preds_t[:, 20] = preds_t3[:min_shape,5]
    preds_t[:, 23] = preds_t3[:min_shape,6]
    preds_t[:, 27] = preds_t3[:min_shape,7]
    #group4
    preds_t[:, 21] = preds_t4[:,0]
    preds_t[:, 22] = preds_t4[:,1]
    preds_t[:, 24] = preds_t4[:,2]
    preds_t[:, 25] = preds_t4[:,3]
    preds_t[:, 26] = preds_t4[:,4]

    print(preds_t1[0])
    print(preds_t2[0])
    print(preds_t3[0])
    print(preds_t4[0])
    print(preds_t[0])

ensemble_pred()
th = fit_val(pred,y)
th[th<0.1] = 0.1
print('Thresholds: ',th)
print('F1 macro: ',f1_score(y, pred>th, average='macro'))
print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
print('F1 micro: ',f1_score(y, pred>th, average='micro'))
print('Fractions: ',(pred > th).mean(axis=0))
print('Fractions (true): ',(y > th).mean(axis=0))
exit()



preds_t = preds_t1
np.save("./raw_pre.npz", preds_t)


pred_t = preds_t.max(axis=-1) #max works better for F1 macro score
print(pred_t.shape)
np.save("./max_pre.npz", pred_t)
print(preds_t1.shape)
def save_pred(pred, th=0.5, fname='protein_classification_multi.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>th)[0]]))
        pred_list.append(s)
        
    sample_df = pd.read_csv(SAMPLE)
    sample_list = list(sample_df.Id)
    pred_dic = dict((key, value) for (key, value) 
                in zip(learner.data.test_ds.fnames,pred_list))
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id':sample_list,'Predicted':pred_list_cor})
    df.to_csv(fname, header=True, index=False)



th_t = np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])
print('Fractions: ',(pred_t > th_t).mean(axis=0))
save_pred(pred_t,th_t)



'''
learner.opt_fn = optim.Adam
learner.clip = 1.0 #gradient clipping
learner.crit = FocalLoss() 
#learner.crit = FocalLossMultiLabel()
learner.metrics = [acc]

learner.summary

lr = 2e-2
learner.fit(lr,1)

learner.unfreeze()
lrs=np.array([lr/10,lr/3,lr])


learner.fit(lrs/4,4,cycle_len=2,use_clr=(10,20))
learner.fit(lrs/4,2,cycle_len=4,use_clr=(10,20))
learner.fit(lrs/16,30,cycle_len=8,use_clr=(5,20))

learner.save('ResNet34_256_1_multi')
'''



"""
th = fit_val(pred,y)
th[th<0.1] = 0.1
print('Thresholds: ',th)
print('F1 macro: ',f1_score(y, pred>th, average='macro'))
print('F1 macro (th = 0.5): ',f1_score(y, pred>0.5, average='macro'))
print('F1 micro: ',f1_score(y, pred>th, average='micro'))
print('Fractions: ',(pred > th).mean(axis=0))
print('Fractions (true): ',(y > th).mean(axis=0))
"""





