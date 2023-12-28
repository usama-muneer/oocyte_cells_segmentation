import glob
import json
import cv2
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from pycocotools import mask as coco_mask

try:
    from config import classes, colors
    from oocyte_visualizer import check_hollow_circle, annToMask
except:
    from code.config import classes, colors
    from code.oocyte_visualizer import check_hollow_circle, annToMask

def main(annotation_path, image_dir, dataset_save_dir, number_of_images):
  
  output_images_path = dataset_save_dir + '/images'
  output_masks_path = dataset_save_dir + '/masks'
  Path(output_images_path).mkdir(parents=True, exist_ok=True)
  Path(output_masks_path).mkdir(parents=True, exist_ok=True)

  all_imgs = glob.glob(f'{image_dir}/*.jpg')
  all_dir_images = [i.split('/')[-1].split('\\')[-1] for i in all_imgs]
  print('all_dir_images: ', all_dir_images[:4])

  with open(annotation_path, 'r') as rf:
    data = json.load(rf)
  coco = COCO(annotation_path)

  cat_id_2_name = dict()
  for cat in data['categories']:
      cat_id_2_name[cat['id']] = cat['name']

  print('Total images in annotations: ', len(data['images']))
  print('Total images in images directory: ', len(all_imgs))
  print('Total annotations: ', len(data['annotations']))
  print('Total categories: ', len(data['categories']))

  for idx, img_row in enumerate(data['images'][:]):
    if  idx == number_of_images:  
      break

    if img_row['file_name'] in all_dir_images:
      img_path = image_dir + '/' + img_row['file_name']
      image = cv2.imread(img_path)
      try: 
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      except:
          print('issue with image: ', img_row['file_name'])
      # plt.imshow(image)
      # plt.axis('off')
      # plt.show

      cat_ids = coco.getCatIds()
      anns_ids = coco.getAnnIds(imgIds=img_row['id'], catIds=cat_ids, iscrowd=None)
      annotations = coco.loadAnns(anns_ids)
      
      # caries_anns = [ann for ann in anns]
      overlay_mask = np.zeros_like(image)
      if len(annotations) > 0:
        # mask = coco.annToMask(annotations[0])
        mask = overlay_mask[:,:,1].copy()
        # print(mask.shape)
        for annotation in annotations:
          try:
            if len(annotation['segmentation']) == 2 and len(annotation['segmentation'][0]) > 10 and len(annotation['segmentation'][1]) > 10:
              # print(len(annotation['segmentation'][0]), len(annotation['segmentation'][1]))
              polygons, overlay_mask = check_hollow_circle(annotation['segmentation'], overlay_mask)
              mask += overlay_mask[:,:,1]
            else:
              if len(annotation['segmentation']) == 2:
                print(len(annotation['segmentation'][0]), len(annotation['segmentation'][1]))
                for segm in annotation['segmentation']:
                  if len(segm) > 50:
                    tmp_ann = annotation.copy()
                    tmp_ann['segmentation'] = [segm]
                    mask += coco.annToMask(tmp_ann)
              else:
                mask += coco.annToMask(annotation)
          except:
            pass
                    
        image_name = f'{img_row["file_name"].split(".")[0]}.jpg'
        cv2.imwrite(f'{dataset_save_dir}/images/{image_name}', image)
        image_mask_name = f'{img_row["file_name"].split(".")[0]}.jpg'
        mask[mask > 0] = 1
        plt.imsave(f'{dataset_save_dir}/masks/{image_mask_name}', mask, cmap=cm.gray)
        # if len(caries_anns) > 2:
        # print(img_row['file_name'])
        # print('Total caries: ', len(annotations))
        # plt.imshow(mask, cmap=cm.gray)
        # plt.show()
        # break

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-path', type=str, default='coco.json', help='') 
    parser.add_argument('--image-dir', type=str, default='', help='images/') 
    parser.add_argument('--dataset-save-dir', type=str, default='dataset/', help='')
    parser.add_argument('--number-of-images', type=int, default=40, help='')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(**vars(opt))

# pbw_semantic_seg_data_pred(ann_path, img_dir, save_dataset_dir)
# python pbw_semantic_seg_data_preparation.py --annotation-path cvat/coco_caries_manifest_ann/Labeled-annotation-CVAT/coco_json_anns/combined_annotations.json --image-dir pbw/all-pbw-04-12-2022/images --dataset-save-dir caries_segm/dataset