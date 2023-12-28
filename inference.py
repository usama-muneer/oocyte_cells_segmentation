
from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K

  
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.callbacks import *
from semantic_segm_models import VGGUnet, DenseUNet, Unet
import copy

def load_model(model_name='DenseUnet', input_size=304, model_path=f'models\\model-DenseUnet.h5'):
    # input_img = Input((h, w, 3), name='img')
    if model_name == 'denseunet':
        model = DenseUNet(image_size = input_size) 
    elif model_name == 'unet':
        # Input((h, w, 3), name='img')
        input_img = Input((input_size, input_size, 3), name='img')
        model = Unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    elif model_name == 'vggunet':
        model = VGGUnet(image_size = input_size)
    model.load_weights(model_path)
    return model

class semantic_segm_inference:
    def __init__(self, model_name, input_size, model_path):
        self.input_size = input_size
        self.model = load_model(model_name, input_size, model_path)
    
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
    
    def mask2bounding_box(self, image_mask):
        lbl_0 = label(image_mask) 
        props = regionprops(lbl_0)
        bboxes = []
        for prop in props:
            bboxes.append([prop.bbox[0], prop.bbox[1], prop.bbox[2], prop.bbox[3]])
        return bboxes
    
    def draw_polygon(self):
        for poly in self.polygons:
            cv2.polylines(self.input_object, [np.array(poly).reshape((-1, 1, 2))], True, (255, 0, 0), 1)       
        plt.imshow(self.input_object) 
        plt.show()
    
    def draw_predictions(self,):
        has_mask = self.preds_val_t.max() > 0
        fig, ax = plt.subplots(1, 3, figsize=(25, 15))
        ax[0].imshow(self.input_object, cmap='gray')
        if has_mask:
            ax[0].contour(self.preds_val_t.squeeze(), colors='k', levels=[0.1])
        ax[0].set_title('Original Image')
        ax[0].set_axis_off()

        ax[1].imshow(self.preds_val.squeeze(), vmin=0, vmax=1)
        ax[1].set_title('Predicted Image')
        ax[1].set_axis_off()

        ax[2].imshow(self.preds_val_t.squeeze(), vmin=0, vmax=1)
        ax[2].set_title('Predicted binary Mask Image')
        ax[2].set_axis_off()   
        plt.show()

    def inference(self, input_object, threshold=0.5, clas='', output_dict={}):
        self.input_object = input_object
        h = self.input_object.shape
        new_output_dict = copy.deepcopy(output_dict)
        # img = load_img(self.input_object)
        # x_img = img_to_array(img)
        x_img = resize(self.input_object, (self.input_size, self.input_size, 3), mode = 'constant', preserve_range = True)
        X = np.zeros((1, self.input_size, self.input_size, 3), dtype=np.float32)
        X[0] = x_img/255.0
        self.preds_val = self.model.predict(X, verbose=1)
        self.preds_val_t = (self.preds_val > threshold).astype(np.uint8)

        self.preds_val = cv2.resize(self.preds_val[0], (self.input_object.shape[1], self.input_object.shape[0]))
        self.preds_val_t = cv2.resize(self.preds_val_t[0], (self.input_object.shape[1], self.input_object.shape[0]))

        contours, _ = cv2.findContours(self.preds_val_t, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.pred_masks = []
        self.polygons = []
        self.pred_bboxes = []
        self.pred_classes = []
        # self.pred_bboxes = self.mask2bounding_box(self.preds_val_t)
        for object in contours:
            x_coords = []
            y_coords = []
            for point in object:
                x_coords.append(int(point[0][0]))
                y_coords.append(int(point[0][1]))
            new_output_dict['pred_masks'].append([x_coords, y_coords])
            new_output_dict['pred_bboxes'].append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
            new_output_dict['pred_polygons'].append((object))
            new_output_dict['pred_classes'].append(clas)
        new_output_dict['pred_values'].append(self.preds_val)
        new_output_dict['pred_mask_threshold'].append(self.preds_val_t)
        return new_output_dict


