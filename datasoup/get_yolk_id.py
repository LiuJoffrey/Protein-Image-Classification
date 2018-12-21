# -*-encoding:utf8-*-
import pandas as pd
import numpy as np

label_names = {
    0  : 'Nucleoplasm' ,
    1  : 'Nuclear membrane' ,
    2  : 'Nucleoli' ,
    3  : 'Nucleoli fibrillar center' ,
    4  : 'Nuclear speckles' ,
    5  : 'Nuclear bodies' ,
    6  : 'Endoplasmic reticulum' ,
    7  : 'Golgi apparatus' ,
    8  : 'Peroxisomes' ,
    9  : 'Endosomes' ,
    10 : 'Lysosomes' ,
    11 : 'Intermediate filaments' ,
    12 : 'Actin filaments' ,
    13 : 'Focal adhesion sites' ,
    14 : 'Microtubules' ,
    15 : 'Microtubule ends' ,
    16 : 'Cytokinetic bridge' ,
    17 : 'Mitotic spindle' ,
    18 : 'Microtubule organizing center' ,
    19 : 'Centrosome' ,
    20 : 'Lipid droplets' ,
    21 : 'Plasma membrane' ,
    22 : 'Cell junctions' ,
    23 : 'Mitochondria' ,
    24 : 'Aggresome' ,
    25 : 'Cytosol' ,
    26 : 'Cytoplasmic bodies' ,
    27 : 'Rods & rings'}

reverse_train_labels = dict ((v , k) for k , v in label_names.items ())

train = pd.read_csv ('train.csv')


def fill_targets (row) :
    row ['Target'] = np.array (row ['Target'].split (' ')).astype (np.int)
    for num in row ['Target'] :
        name = label_names [int (num)]
        row.loc [name] = 1
    return row


for key in label_names.keys () :
    train [label_names [key]] = 0

train = train.apply (fill_targets , axis = 1)

df = train [train.columns [2 :]].astype (np.int)


df [df.columns [6 :]] *= -100
mask = df.sum (axis = 1)
mask = mask > 0
yolk = train ['Id'] [mask]

print (train ['Target'].loc [yolk.index])
yolk.to_csv ('yolk_id.csv')
