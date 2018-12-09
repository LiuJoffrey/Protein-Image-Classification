
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import cv2

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, Conv2D, BatchNormalization, Reshape, Lambda
# from keras_applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Concatenate#, ReLU, LeakyReLU

import warnings
warnings.filterwarnings("ignore")

#PATH = './'
TRAIN = '/media/cnrg-ntu2/HDD1TB/r07921052/ml_final/all/train/'
TEST = '/media/cnrg-ntu2/HDD1TB/r07921052/ml_final/all/test/'
#LABELS = 'input/train.csv'
#SAMPLE = '../input/sample_submission.csv'


def main():
    input_shape = (256,256,4)
    datagen, train_dataset_info = load_train_data()
    # show_img(datagen)

    #train(input_shape, train_dataset_info)
    predict(input_shape)
class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            
            
            yield batch_images, batch_labels
            
    
    def load_image(path, shape):
        image_red_ch = skimage.io.imread(path+'_red.png')/255.0
        image_yellow_ch = skimage.io.imread(path+'_yellow.png')/255.0
        image_green_ch = skimage.io.imread(path+'_green.png')/255.0
        image_blue_ch = skimage.io.imread(path+'_blue.png')/255.0

        #image_red_ch += (image_yellow_ch/2).astype(np.uint8) 
        #image_blue_ch += (image_yellow_ch/2).astype(np.uint8)

        image = np.stack((
            image_red_ch, 
            image_green_ch, 
            image_blue_ch,
            image_yellow_ch
        ), -1)
        image = resize(image, (shape[0], shape[1]), mode='reflect')
        return image
                
            
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug

def load_train_data():
    path_to_train = '/media/cnrg-ntu2/HDD1TB/r07921052/ml_final/all/train/'
    data = pd.read_csv('/media/cnrg-ntu2/HDD1TB/r07921052/ml_final/all/train.csv')

    train_dataset_info = []
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        train_dataset_info.append({
            'path':os.path.join(path_to_train, name),
            'labels':np.array([int(label) for label in labels])})
    train_dataset_info = np.array(train_dataset_info)

    input_shape = (256,256,4)

    # create train datagen
    train_datagen = data_generator.create_train(
        train_dataset_info, 5, input_shape, augument=True)
    
    return train_datagen, train_dataset_info

def show_img(datagen):
    images, labels = next(datagen)

    fig, ax = plt.subplots(1,5,figsize=(25,5))
    for i in range(5):
        ax[i].imshow(images[i])
    fig.savefig('fff.png')
    print('min: {0}, max: {1}'.format(images.min(), images.max()))

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
#     y_true = Lambda(K.argmax, arguments={'axis':1})(y_true)
#     y_true = Lambda(K.cast, arguments={'dtype':'float32'})(y_true)
    
#     y_pred = Lambda(K.argmax, arguments={'axis':1})(y_pred)
#     y_pred = Lambda(K.cast, arguments={'dtype':'float32'})(y_pred)
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)

def reduce(x):
    return K.argmax(x, axis=1)

def cast(x):
    return K.cast(x, 'float32')

def create_model(input_shape, n_out):
    inp = Input(input_shape)
    """
    pretrain_model = MobileNetV2(include_top=False, weights=None, input_tensor=inp)
    #x = pretrain_model.get_layer(name="block_13_expand_relu").output
    x = pretrain_model.output
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(n_out, activation="relu")(x)
    
    for layer in pretrain_model.layers:
        layer.trainable = True
    """
    
    dropRate = 0.25
    x = BatchNormalization(axis=-1)(inp)
    #x = BatchNormalization()(init)
    x = Conv2D(8, (3, 3))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(8, (3, 3))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (3, 3))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    c1 = Conv2D(16, (3, 3), padding='same')(x)
    c1 = Activation('relu')(c1)
    c2 = Conv2D(16, (5, 5), padding='same')(x)
    c2 = Activation('relu')(c2)
    c3 = Conv2D(16, (7, 7), padding='same')(x)
    c3 = Activation('relu')(c3)
    c4 = Conv2D(16, (1, 1), padding='same')(x)
    c4 = Activation('relu')(c4)
    x = Concatenate()([c1, c2, c3, c4])
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(128, (3, 3))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    #x = Conv2D(256, (1, 1), activation='relu')(x)
    #x = BatchNormalization(axis=-1)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(28)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.1)(x)
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)




        
    return Model(inp, x)

def train(input_shape, train_dataset_info):
    keras.backend.clear_session()

    model = create_model(input_shape=input_shape, n_out=28)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['acc', f1])
    model.summary()

    epochs = 70 ; batch_size = 64
    checkpointer = ModelCheckpoint('./InceptionResNetV2.model.h5', verbose=2, 
        save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1)

    # split and suffle data 
    np.random.seed(2018)
    indexes = np.arange(train_dataset_info.shape[0])
    np.random.shuffle(indexes)
    train_indexes = indexes[:27500]
    valid_indexes = indexes[27500:]

    train_steps = len(train_indexes)//batch_size
    #train_steps = 2
    valid_steps = len(valid_indexes)//batch_size

    # create train and valid datagens
    train_generator = data_generator.create_train(train_dataset_info[train_indexes], batch_size, input_shape, augument=True)
    validation_generator = data_generator.create_train(train_dataset_info[valid_indexes], 100, input_shape, augument=False)

    # train model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=next(validation_generator),
        validation_steps=valid_steps, 
        epochs=epochs,workers=4, max_queue_size = 15, use_multiprocessing=True, 
        verbose=1,
        callbacks=[checkpointer, reduce_lr])
    
    set_history(history)

def set_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    plt.savefig('predict.png')

def predict(input_shape):
    th = [0.5, 0.04, 0.12, 0.05, 0.06,  0.08, 0.03, 0.09, 0.01, 0.005, 0.005, 0.04, 0.03,
        0.02,   0.03,  0.005, 0.02, 0.01, 0.03, 0.05,  0.01, 0.13 ,0.03, 0.1 ,0.01, 0.26,
        0.01, 0.005]



    submit = pd.read_csv('/media/cnrg-ntu2/HDD1TB/r07921052/ml_final/all/sample_submission.csv')
    model = load_model('./InceptionResNetV2.model.h5', custom_objects={'f1': f1})
    predicted = []
    for name in tqdm(submit['Id']):
        path = os.path.join('/media/cnrg-ntu2/HDD1TB/r07921052/ml_final/all/test/', name)
        image = data_generator.load_image(path, input_shape)
        score_predict = model.predict(image[np.newaxis])[0]

        index = []
        for idx in range(len(th)):
            if score_predict[idx]> th[idx]:
                index.append(True)
            else:
                index.append(False)


        #label_predict = np.arange(28)[score_predict>=0.5]
        label_predict = np.arange(28)[index]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)
    submit['Predicted'] = predicted
    submit.to_csv('submission.csv', index=False)
    print("finish")

main()