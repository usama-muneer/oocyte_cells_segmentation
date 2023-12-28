import cv2
import glob
import json
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from pycocotools import mask as cocomask
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from skimage import measure
import argparse
import os

def mask2polygon(mask):
    '''
    Return pycocotools library
    input -> coco annotation with RLE and crowd=1
    output -> mask image (Numpy array)
    '''
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    draw_coords = []
    for obj in contours:
        coords = []
        draw_coords = []
        for point in obj:
            draw_coords.append([int(point[0][0]), int(point[0][1])])
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))
        polygons.append(coords)
    return polygons, draw_coords

def rle2polygon(ann, draw_polygon=False):
    '''
    convert coco rle encoding to coco polygon
    input -> encoded RLE
    output -> coco segmentation
    '''
    compressed_rle = mask_utils.frPyObjects(ann['segmentation'], ann['segmentation']['size'][0], ann['segmentation']['size'][1])
    _mask = mask_utils.decode(compressed_rle)
    poly, draw_coords = mask2polygon(_mask)
    ann['segmentation'] = poly
    ann['iscrowd'] = 0
    if draw_polygon == True:
        p = Polygon(draw_coords, facecolor = 'r')
        fig,ax = plt.subplots()
        ax.add_patch(p)
        ax.imshow(_mask)
        plt.show()
    return ann
def coco_roi_2_mask(annotation_path='', output_path='', min_pixel_in_segm=10):
    # Load Annotations
    coco_annotations = COCO(annotation_path)
    with open(annotation_path, 'r') as fr:
        data = json.load(fr)

    # ID to Categories and Categories to ID dictionary
    categories = data['categories']
    cat_id_2_name = {}
    cat_name_2_id = {}
    for cat in data['categories']:
        cat_id_2_name[cat['id']] = cat['name']
        cat_name_2_id[cat['name']] = cat['id']

    print('Total annotations: ', len(data['annotations']))

    annotations = []
    for ann in data['annotations']:
        if type(ann['segmentation']) == dict:
            annotations.append(rle2polygon(ann, draw_polygon=False))
        else:
            annotations.append(ann)

    data['annotations'] = annotations
    print('total annotation after preprocessing: ', len(annotations))
    # cat_id_2_name, cat_name_2_id

    # Save the updated COCO annotation to a new JSON file
    with open(f'{output_path}/up_annotations.json', 'w') as f:
        json.dump(data, f)
    
    return True

def split_train_test(annotation_path='', images_path='', output_path='', included_categories=[]):

    all_images = [i.rsplit("/", 1)[-1] for i in glob.glob(f"{images_path}/*.jpg")]
    # all_images = [i.rsplit("\\", 1)[-1] for i in glob.glob(f"{images_path}\\*.jpg")]
    print('Total images: ', len(all_images))

    with open(annotation_path, 'r') as fr:
        data = json.load(fr)

    id_2_cat = dict()
    cat_2_id = {}
    train_data = data.copy()
    test_data = data.copy()
    cat_id_2_name = {}
    cat_name_2_id = {}
    categories = []

    for idx, cat in enumerate(data['categories']):
        if len(included_categories) == 0:
            cat_id_2_name[cat['id']] = cat['name']
            cat_name_2_id[cat['name']] = cat['id']
        elif cat['name'] in included_categories:
            id_2_cat[cat['id']] = cat['name']
            cat_2_id[cat['name']] = cat['id']
            categories.append({'id': cat['id'], 'name': cat['name'], 'supercategory': 'root'})
    
    if len(included_categories) > 0:
        print('Updated Categories: ', categories)
    else:
        categories = data['categories']

    train_images = []
    train_annotations = []

    test_images = []
    test_annotations = []
    train_unique_images = []

    dataset_info = {'total_org_images': len(data['images']),
                    'total_org_ann': len(data['annotations']),
                    }
    for idx, image_row in enumerate(data['images']):
        if image_row['file_name'] in all_images:
            for ann in data['annotations']:
                if ann['image_id'] == image_row['id'] and ann['category_id'] in list(id_2_cat.keys()):
                    # print('Before: ', ann['category_id'],  id_2_cat[ann['category_id']])
                    ann['category_id'] = [i['id'] for i in categories if id_2_cat[ann['category_id']] == id_2_cat[i['id']]][0]
                    # print('After: ', ann['category_id'])
                    if idx%10 != 0:
                        train_images.append(image_row)
                        train_annotations.append(ann)
                    else:
                        test_images.append(image_row)
                        test_annotations.append(ann)

    for x in train_images:
        if x not in train_unique_images:
            train_unique_images.append(x)
    train_data['images'] = train_unique_images
    train_data['annotations'] = train_annotations
    train_data['categories'] = categories

    train_annotation_path = f'{output_path}/train_annotations.json'
    print('Train annotation path: ', train_annotation_path)
    with open(train_annotation_path, 'w') as fp:
        json.dump(train_data, fp)

    test_unique_images = []
    for x in test_images:
        if x not in test_unique_images:
            test_unique_images.append(x)
    test_data['images'] = test_unique_images
    test_data['annotations'] = test_annotations
    test_data['categories'] = categories

    test_annotations_path = f'{output_path}/test_annotations.json'
    print('Test annotation path: ', test_annotations_path)
    with open(test_annotations_path, 'w') as fp:
        json.dump(test_data, fp)

    dataset_info['train_images'] = len(train_data['images'])
    dataset_info['train_ann'] = len(train_data['annotations'])
    dataset_info['test_images'] = len(test_data['images'])
    dataset_info['test_ann'] = len(test_data['annotations'])
    print(dataset_info)
    return dataset_info

def main(annotation_path, images_path, output_path, included_categories, min_pixel_in_segm, roi_2_mask_flag=False):
    if roi_2_mask_flag:
        coco_roi_2_mask(annotation_path=annotation_path, 
                        output_path=output_path, 
                        min_pixel_in_segm=min_pixel_in_segm)
        annotation_path = f'{output_path}/up_annotations.json'

    split_train_test(annotation_path=annotation_path,
                     images_path=images_path,
                     output_path=output_path, 
                     included_categories=included_categories)
    
    print(f'processed data is saved inside {output_path} directory') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process COCO annotations and images.')
    parser.add_argument('--annotation_path', type=str, default='dataset/annotations.json',
                        help='Path to the COCO annotations JSON file.')
    parser.add_argument('--images_path', type=str, default='dataset/images',
                        help='Path to the directory containing images.')
    parser.add_argument('--output_path', type=str, default='dataset',
                        help='Path to the output directory where processed data will be saved.')
    parser.add_argument('--included_categories', type=str, nargs='+', default=['Cytoplasm', 'Oolemma', 'Zona Pellucida'],
                        help='List of included categories for data splitting.')
    parser.add_argument('--min_pixel_in_segm', type=int, default=10,
                        help='Minimum number of pixels in a segmentation.')
    parser.add_argument('--roi_2_mask_flag', action='store_true',
                        help='Flag to convert ROI annotations to masks.')

    args = parser.parse_args()

    main(args.annotation_path, args.images_path, args.output_path, args.included_categories, args.min_pixel_in_segm, args.roi_2_mask_flag)
# python script_name.py --annotation_path 'dataset/annotations.json' --images_path 'dataset/images' --output_path 'dataset' --included_categories 'Cytoplasm' 'Oolemma' 'Zona Pellucida' --min_pixel_in_segm 10
