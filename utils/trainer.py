import pathlib
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import gc


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras import backend as K


import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
try:
    from semantic_segm_models import semantic_segm_models
except:
    from utils.semantic_segm_models import semantic_segm_models

import pandas as pd
from sklearn.model_selection import train_test_split

## Seeding 
seed = 41
random.seed = seed
np.random.seed = seed
tf.seed = seed

def load_segm_mask_data(self, X_path, y_path, test_size=0.1):
    # load images and mask from numpy array saved files
    X = np.load(X_path)
    y = np.load(y_path)
    print('Full Dataset Shapes: ', X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print('Dataset split: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print('Train dataset size: ', len(X_train))
    print('Test dataset size: ', len(X_test))
    return X_train, X_test, y_train, y_test


class trainer:
    def __init__(self, images_npy_path, masks_npy_path, test_size, model_name, input_height, input_width, output_channel, saved_model_path='saved_model/', pretrained_model_path=''):
        self.images_npy_path = images_npy_path
        self.masks_npy_path = masks_npy_path
        self.output_channel = output_channel
        self.test_size = test_size
        self.model_name = model_name
        self.input_height = input_height
        self.input_width = input_width
        self.saved_model_path = saved_model_path

        # Load Dataset
        self.X_train, self.X_test, self.y_train, self.y_test = load_segm_mask_data(self, images_npy_path, masks_npy_path, test_size=0.1)

        # LOAD MODEL            
        ss_models = semantic_segm_models(input_height=input_height, input_width=input_width)
        if model_name == 'vggunet' and pretrained_model_path != '':
            # Load the pre-trained model
            print('Loading Pretrained Model')
            self.model = ss_models.load_model(model_name=model_name, pretrained_wights=pretrained_model_path)
        else:
            self.model = ss_models.load_model(model_name=model_name)

    def save_loss_accuracy_curve(self, results):
        # Plot curves
        plt.figure(figsize = (15,6))
        plt.title("Learning curve")
        plt.plot(results.history["loss"], label="loss")
        plt.plot(results.history["val_loss"], label="val_loss")
        plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()

        plt.figure(figsize = (15,6))
        plt.title("Learning curve")
        plt.plot(results.history["accuracy"], label="Accuracy")
        plt.plot(results.history["val_accuracy"], label="val_Accuracy")
        plt.plot(np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.imsave(f'{self.saved_model_path}/loss_accuracy_curve.png')

    def train(self, batch_size=2, epochs=10, min_lr=0.00001, patience=10, save_best_only=True, save_weights_only=True, verbose=1):
        K.clear_session()
        # Configur Metrics
        metrics = ["accuracy", 
                tf.keras.metrics.AUC(), 
                tf.keras.metrics.SensitivityAtSpecificity(0.5), 
                tf.keras.metrics.SpecificityAtSensitivity(0.5)]
        
        # Configur Model Compiler
        self.model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=metrics)
        print(gc.collect())

        callbacks = [
            EarlyStopping(patience=patience, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=patience, min_lr=min_lr, verbose=verbose),
            ModelCheckpoint(f'{self.saved_model_path}/model-{self.model_name}.h5', verbose=verbose, save_best_only=save_best_only, save_weights_only=save_weights_only),
            CSVLogger(f"{self.saved_model_path}/data_{self.model_name}.csv"),
            TensorBoard(log_dir=f'{self.saved_model_path}/logs')
        ]
        results = self.model.fit(self.X_train, 
                                self.y_train, 
                                batch_size=batch_size, 
                                epochs=epochs, 
                                callbacks=callbacks, 
                                validation_data=(self.X_test, self.y_test), 
                                use_multiprocessing=True)
        # print(results)
        # print(results.history)
        # df_result = pd.DataFrame(results.history)
        # df_result.sort_values('val_loss', ascending=True, inplace = True)
        # self.save_loss_accuracy_curve(df_result)

    def test_accuracy(self,):
        eval_matrics = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print('Accuracy: ', eval_matrics[1])
        print('sensitivity_at_specificity: ', eval_matrics[3])
        print('specificity_at_sensitivity: ', eval_matrics[4])

def main(opt):
    saved_model_path = opt.saved_model_path
    pathlib.Path(saved_model_path).mkdir(parents=True, exist_ok=True)
    ss_trainer = trainer(images_npy_path=opt.images_npy_path, 
                        masks_npy_path=opt.masks_npy_path, 
                        test_size=opt.test_size, 
                        model_name=opt.model_name, 
                        input_height=opt.input_height, 
                        input_width=opt.input_width,
                        output_channel=opt.output_channel,
                        saved_model_path = saved_model_path,
                        pretrained_model_path = opt.pretrained_model_path,
                        )
    ss_trainer.train(batch_size=opt.batch_size, 
                    epochs=opt.epochs, 
                    min_lr=opt.min_lr,
                    patience=opt.patience,
                    save_best_only=True, 
                    save_weights_only=True, 
                    verbose=1)
    ss_trainer.test_accuracy()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-npy-path', type=str, default='', help='Path to x.npy') 
    parser.add_argument('--masks-npy-path', type=str, default='', help='Path to y.npy') 
    parser.add_argument('--test-size', type=float, default=0.1, help='')
    parser.add_argument('--model-name', type=str, default='', help='Supported: unet, vggunet')
    parser.add_argument('--input-height', type=int, default=64, help='')
    parser.add_argument('--input-width', type=int, default=64, help='')
    parser.add_argument('--output-channel', type=int, default=1, help='')
    parser.add_argument('--epochs', type=int, default=10, help='')
    parser.add_argument('--batch-size', type=int, default=2, help='')
    parser.add_argument('--patience', type=int, default=10, help='')
    parser.add_argument('--min-lr', type=float, default=0.0001, help='')
    parser.add_argument('--saved-model-path', type=str, default='saved_model/', help='')
    parser.add_argument('--pretrained-model-path', type=str, default='', help='')
    # parser.add_argument('--test-size', type=float, default=0.1, help='')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    # main(**vars(opt))
 
# !python --images-npy-path 'X_64x64.npy' --masks-npy-path 'y_64x64.npy' --test-size 0.1 --model-name 'unet' --input-height 64 --input-width 64 --min-lr 0.0001 --epochs 5