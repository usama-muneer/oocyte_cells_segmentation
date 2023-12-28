from inference import semantic_segm_inference
import matplotlib.pyplot as plt
import glob, json
import cv2, random
from oocyte_visualizer import draw_circles_intersection2, is_inner_polygon, is_circular_shape
from shapely.geometry import Polygon
import numpy as np
from config import colors

from machine_format import convert_machine_format

def extract_center_bbox(image):
    """
    Extracts the bounding box coordinates for the center region of the image.

    Args:
    - image: Input image

    Returns:
    - List representing [x_min, y_min, x_max, y_max] of the center bounding box
    """
    margin_h, margin_w = int(image.shape[0]/3), int(image.shape[1]/5)
    x_min, y_min, x_max, y_max = int((image.shape[1]/2)-margin_w), int((image.shape[0]/2)-margin_h), int((image.shape[1]/2)+margin_w), int((image.shape[0]/2)+margin_h)
    return [x_min, y_min, x_max, y_max]

def extract_global_bounding_box(pred_bboxes, margin=50):
    """
    Extracts the global bounding box that encompasses all individual bounding boxes.

    Args:
    - pred_bboxes: List of bounding boxes [x_min, y_min, x_max, y_max]
    - margin: Margin to extend the global bounding box

    Returns:
    - List representing [x_min, y_min, x_max, y_max] of the global bounding box
    """
    if not pred_bboxes:
        return None
    x_min_global = int(min(box[0] for box in pred_bboxes)) - margin
    y_min_global = int(min(box[1] for box in pred_bboxes)) - margin
    x_max_global = int(max(box[2] for box in pred_bboxes)) + margin
    y_max_global = int(max(box[3] for box in pred_bboxes)) + margin
    global_bounding_box = [x_min_global, y_min_global, x_max_global, y_max_global]
    return global_bounding_box

def single_cell_filter(image, pred_dict, compare_bbox=None, compare_margin=30):
    """
    Filters predictions based on a center bounding box and optional comparison bounding box.

    Args:
    - image: Input image
    - pred_dict: Dictionary containing prediction information
    - compare_bbox: Bounding box for comparison
    - compare_margin: Margin for comparison

    Returns:
    - Filtered dictionary containing prediction information
    """
    temp_dict = {
        'pred_polygons': [],
        'pred_masks': [],
        'pred_bboxes': [],
        'pred_classes': [],
    }
    center_box = extract_center_bbox(image)
    for poly, mask, bbox, clas in zip(pred_dict['pred_polygons'], pred_dict['pred_masks'], pred_dict['pred_bboxes'], pred_dict['pred_classes']):
        x_min, y_min, x_max, y_max = bbox
        if compare_bbox is not None:
            if (compare_bbox[0] < bbox[0] + compare_margin) & (compare_bbox[1] < bbox[1] + compare_margin) & (compare_bbox[2] > bbox[2] - compare_margin) & (compare_bbox[3] > bbox[3] - compare_margin):
                pass
            else:
                continue
        if (x_min - center_box[0] > 0) & (y_min - center_box[1] > 0) & (center_box[2] - x_max > 0) & (center_box[3] - y_max > 0):
            temp_dict['pred_polygons'].append(poly)
            temp_dict['pred_masks'].append(mask)
            temp_dict['pred_bboxes'].append(bbox)
            temp_dict['pred_classes'].append(clas)
    return temp_dict

def draw_predictions(image, pred_dict, clas='cytoplasm', color=None):
    """
    Draws predictions on the image based on the prediction dictionary.

    Args:
    - image: Input image
    - pred_dict: Dictionary containing prediction information
    - clas: Class name for drawing
    - color: Color for drawing

    Returns:
    - Image with drawn predictions
    """
    circles_list = []
    for poly, mask, bbox in zip(pred_dict['pred_polygons'], pred_dict['pred_polygons'], pred_dict['pred_bboxes']):
        if is_circular_shape(poly, threshold=0.95) and clas != 'cytoplasm':
            if len(circles_list) > 0:
                for circle in circles_list:
                    if is_inner_polygon(circle, poly) and not np.all(poly == circles_list):
                        draw_circles_intersection2(circle, poly, image, color=color, plot_flag=False)
                    elif is_inner_polygon(poly, circle) and not np.all(poly == circles_list):
                        draw_circles_intersection2(poly, circle, image, color=color, plot_flag=False)
                else:
                    circles_list.append(poly)
            else:
                circles_list.append(poly)
        else:
            cv2.polylines(image, [np.array(poly).reshape((-1, 1, 2))], True, color, 4)
            cv2.fillPoly(image, [np.array(poly).reshape((-1, 1, 2))], color)
    return image

def oocyte_prediction(test_annotations, images_path, cytoplasm_model_path, zona_pellucida_model_path, oolemma_model_path, number_of_images, cytoplasm_threshold=0.5, zona_pellucida_threshold=0.5, oolemma_threshold=0.3):
    """
    Performs oocyte predictions on a set of images.

    Args:
    - test_annotations: Path to the test annotations file (JSON format)
    - images_path: Path to the directory containing images
    - cytoplasm_model_path: Path to the cytoplasm segmentation model
    - zona_pellucida_model_path: Path to the zona pellucida segmentation model
    - oolemma_model_path: Path to the oolemma segmentation model
    - number_of_images: Number of images to process
    - cytoplasm_threshold: Threshold for cytoplasm segmentation
    - zona_pellucida_threshold: Threshold for zona pellucida segmentation
    - oolemma_threshold: Threshold for oolemma segmentation
    """
    # LOAD MODELS
    cytoplasm_predictor = semantic_segm_inference(model_name='vggunet', input_size=128, model_path=cytoplasm_model_path)
    zp_predictor = semantic_segm_inference(model_name='vggunet', input_size=128, model_path=zona_pellucida_model_path)
    oolemma_predictor = semantic_segm_inference(model_name='vggunet', input_size=128, model_path=oolemma_model_path)

    with open(test_annotations, 'r') as fr:
        data = json.load(fr)

    for img_path in [images_path + '/' + img['file_name'] for img in data['images']][0:number_of_images]:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_image = image.copy()
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = random_color

        output_dict = {
            'pred_values': [],
            'pred_mask_threshold': [],
            'pred_classes': [],
            'pred_masks': [],
            'pred_bboxes': [],
            'pred_polygons': [],
        }

        cytoplasm_pred_dict = cytoplasm_predictor.inference(input_object=image, threshold=cytoplasm_threshold, output_dict=output_dict)
        cytoplasm_pred_dict = single_cell_filter(image, cytoplasm_pred_dict)
        cytoplasm_bbox = extract_global_bounding_box(cytoplasm_pred_dict['pred_bboxes'], margin=50)
        pred_image = draw_predictions(pred_image, cytoplasm_pred_dict, clas='cytoplasm', color=colors[0])

        zp_pred_dict = zp_predictor.inference(input_object=image, threshold=zona_pellucida_threshold, output_dict=output_dict)
        zp_pred_dict = single_cell_filter(image, zp_pred_dict)
        pred_image = draw_predictions(pred_image, zp_pred_dict, clas='zona pellucida', color=colors[1])

        oolemma_pred_dict = oolemma_predictor.inference(input_object=image, threshold=oolemma_threshold, output_dict=output_dict)
        oolemma_pred_dict = single_cell_filter(image, oolemma_pred_dict, cytoplasm_bbox)
        pred_image = draw_predictions(pred_image, oolemma_pred_dict, clas='oolemma', color=colors[2])

        plt.imshow(pred_image)
        # plt.plot(cytoplasm_bbox[0], cytoplasm_bbox[1], "og", markersize=5)
        # plt.plot(cytoplasm_bbox[2], cytoplasm_bbox[1], "og", markersize=5)
        # plt.plot(cytoplasm_bbox[0], cytoplasm_bbox[3], "og", markersize=5)
        # plt.plot(cytoplasm_bbox[2], cytoplasm_bbox[3], "og", markersize=5)
        plt.axis('off')
        plt.show()

def oocyte_single_prediction(img_path, fill_mask, cytoplasm_model_path, zona_pellucida_model_path, oolemma_model_path, cytoplasm_threshold=0.5, zona_pellucida_threshold=0.5, oolemma_threshold=0.3):
    """
    Performs oocyte predictions on single image.

    Args:
    - imag_path: Path to the image
    - cytoplasm_model_path: Path to the cytoplasm segmentation model
    - zona_pellucida_model_path: Path to the zona pellucida segmentation model
    - oolemma_model_path: Path to the oolemma segmentation model
    - cytoplasm_threshold: Threshold for cytoplasm segmentation
    - zona_pellucida_threshold: Threshold for zona pellucida segmentation
    - oolemma_threshold: Threshold for oolemma segmentation
    """
    # LOAD MODELS
    cytoplasm_predictor = semantic_segm_inference(model_name='vggunet', input_size=128, model_path=cytoplasm_model_path)
    zp_predictor = semantic_segm_inference(model_name='vggunet', input_size=128, model_path=zona_pellucida_model_path)
    oolemma_predictor = semantic_segm_inference(model_name='vggunet', input_size=128, model_path=oolemma_model_path)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred_image = image.copy()
    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color = random_color

    output_dict = {
        'pred_values': [],
        'pred_mask_threshold': [],
        'pred_classes': [],
        'pred_masks': [],
        'pred_bboxes': [],
        'pred_polygons': [],
    }

    cytoplasm_pred_dict = cytoplasm_predictor.inference(input_object=image, threshold=cytoplasm_threshold, output_dict=output_dict)
    cytoplasm_pred_dict = single_cell_filter(image, cytoplasm_pred_dict)
    cytoplasm_bbox = extract_global_bounding_box(cytoplasm_pred_dict['pred_bboxes'], margin=50)
    pred_image = draw_predictions(pred_image, cytoplasm_pred_dict, clas='cytoplasm', color=colors[0])

    zp_pred_dict = zp_predictor.inference(input_object=image, threshold=zona_pellucida_threshold, output_dict=output_dict)
    zp_pred_dict = single_cell_filter(image, zp_pred_dict)
    pred_image = draw_predictions(pred_image, zp_pred_dict, clas='zona pellucida', color=colors[1])

    oolemma_pred_dict = oolemma_predictor.inference(input_object=image, threshold=oolemma_threshold, output_dict=output_dict)
    oolemma_pred_dict = single_cell_filter(image, oolemma_pred_dict, cytoplasm_bbox)
    pred_image = draw_predictions(pred_image, oolemma_pred_dict, clas='oolemma', color=colors[2])
    
    if len(cytoplasm_pred_dict['pred_classes']) > 0 and (len(zp_pred_dict['pred_classes']) > 0 or len(oolemma_pred_dict['pred_classes']) > 0):
        plt.imshow(pred_image)
        # plt.plot(cytoplasm_bbox[0], cytoplasm_bbox[1], "og", markersize=5)
        # plt.plot(cytoplasm_bbox[2], cytoplasm_bbox[1], "og", markersize=5)
        # plt.plot(cytoplasm_bbox[0], cytoplasm_bbox[3], "og", markersize=5)
        # plt.plot(cytoplasm_bbox[2], cytoplasm_bbox[3], "og", markersize=5)
        plt.axis('off')
        plt.show()
        
        cytoplasm = convert_machine_format(cytoplasm_pred_dict['pred_masks'], image_height=image.shape[0], fill_mask=fill_mask, sort_t2b_l2r=True, flip_y=True)
        oolemma = convert_machine_format(oolemma_pred_dict['pred_masks'], image_height=image.shape[0], fill_mask=fill_mask, sort_t2b_l2r=True, flip_y=True)
        zona_pellucida = convert_machine_format(zp_pred_dict['pred_masks'], image_height=image.shape[0], fill_mask=fill_mask, sort_t2b_l2r=True, flip_y=True)

        return {"Cytoplasm": cytoplasm,
                "Oolemma": oolemma,
                "Zona Pellucida": zona_pellucida}
    else:
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        return "No cell in the middle"

def visualize_mask(images_path, pred_mask):
    img = plt.imread(images_path)
    image_height, image_width = img.shape[0], img.shape[1]

    print(img.shape, len(pred_mask))

    plt.imshow(img)
    for points in pred_mask:  
        plt.plot(points[0], points[1], "or", markersize=1)
    plt.show()