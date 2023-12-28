import cv2
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_utils
from shapely.geometry import Polygon
try:
    from config import classes, colors 
except:
    from utils.config import classes, colors

def is_inner_polygon(polygon1, polygon2):
    if type(polygon1) != list:
        polygon1 = [point[0] for point in polygon1.tolist()]
        polygon2 = [point[0] for point in polygon2.tolist()] 
        # print('----',type(polygon1[0:2]), polygon1[0:3], polygon2[0:3])
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)

    if poly1.contains(poly2):
        return True
    else:
        return False

def is_circular_shape(polygon_vertices, threshold=0.95):
    # Create a blank mask and draw the polygon on it
    polygon_vertices = np.array(polygon_vertices)
    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_vertices], 255)
    
    # Find the convex hull of the polygon
    hull = cv2.convexHull(polygon_vertices)
    
    # Calculate areas of the polygon and its convex hull
    polygon_area = cv2.contourArea(polygon_vertices)
    hull_area = cv2.contourArea(hull)
    
    # Check if the polygon is circular based on area comparison
    is_circular = np.isclose(polygon_area, hull_area, rtol=1 - threshold)
    
    return is_circular

def annToRLE(mask, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    # t = self.imgs[ann['image_id']]
    # h, w = t['height'], t['width']
    segm = mask
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = mask_utils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = mask
    return rle

def annToMask(mask, overlay_mask):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(mask, overlay_mask.shape[0], overlay_mask.shape[1])
    m = mask_utils.decode(rle)
    return m

def draw_circles_intersection(circle1_polygon, circle2_polygon, overlay_mask, color='', plot_flag=True):
    """
        circle1_polygon -> Polygon of Circle 1 in coco format
        circle2_polygon -> Polygon of Circle 2 in coco format
        image -> Original loaded image object defined in coco annotation

        describe: this function computer the area between two circle and draw on the image
    """
    if color != '':
        color = (14,112, 241)
    outer_circle = np.array(circle1_polygon)
    inner_circle = np.array(circle2_polygon)
    # Create a mask for the area between the circles
    mask = np.zeros(overlay_mask.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [outer_circle], 255)
    cv2.fillPoly(mask, [inner_circle], 5)

    # Fill the area between the circles with a color
    overlay_mask[mask == 255] = color
    if plot_flag:
        # # Display the circles on the image
        plt.imshow(overlay_mask)
        plt.show()
        return True
    return overlay_mask
    
def draw_circles_intersection2(circle1_polygon, circle2_polygon, overlay_mask, color=None, plot_flag=True):
    """
    circle1_polygon -> Polygon of Circle 1 in coco format
    circle2_polygon -> Polygon of Circle 2 in coco format
    overlay_mask -> Original image or mask where the circles will be drawn
    color -> RGB color tuple (e.g., (0, 0, 255)) for the area between the circles
    plot_flag -> Whether to display the image with Matplotlib

    describe: This function computes the area between two circles and draws it on the image or mask.
    """
    if color is None:
        color = (14, 112, 241)  # Default color (e.g., blue)

    outer_circle = np.array(circle1_polygon)
    inner_circle = np.array(circle2_polygon)

    # Create a mask for the area between the circles
    mask = np.zeros(overlay_mask.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [outer_circle], 255)
    cv2.fillPoly(mask, [inner_circle], 105)

    # Fill the area between the circles with the specified color
    overlay_mask[mask == 255] = color

    if plot_flag:
        # Display the result using Matplotlib
        plt.imshow(cv2.cvtColor(overlay_mask, cv2.COLOR_BGR2RGB))
        plt.title("Circles Intersection")
        plt.axis('off')
        plt.show()
    
    return overlay_mask

def overlay_image(mask, clas, overlay_mask):
    """
        coco -> COCO Loaded Annotations
        annotation -> single annotation of an image
        overlay_mask -> np.zeros_like(image)

        describe: overlay coco annotation mask over image and return the overlay that will be use in ploting
    """

    mask = annToMask(mask, overlay_mask)
            
    # Create a binary mask
    binary_mask = np.array(mask > 0, dtype=np.uint8)
    
    # Set the color for the overlay mask (red in BGR format)
    if clas == classes[0]:
        color = colors[0]
    elif clas == classes[1]:
        color = colors[1]
        # color = (random.randint(0,255),random.randint(0,255), random.randint(0,255))
    elif clas == classes[2]:
        color = colors[2]
    else:
        color = (0,55, 0)

    # Overlay the binary mask on the image
    overlay_mask[:, :, 0] = np.where(binary_mask, color[0], overlay_mask[:, :, 0])
    overlay_mask[:, :, 1] = np.where(binary_mask, color[1], overlay_mask[:, :, 1])
    overlay_mask[:, :, 2] = np.where(binary_mask, color[2], overlay_mask[:, :, 2])
    return overlay_mask

def check_hollow_circle(mask, overlay_mask=None):
    polygons = []
    for tmp_mask in mask:
        plygn = []
        idx = 0
        for idx, x in enumerate(tmp_mask):
            if (idx%2) == 0:
                plygn.append([tmp_mask[idx], tmp_mask[idx+1]])
                idx += 1
        polygons.append(plygn)
    if type(overlay_mask) != None:
        outer_circle = np.array(polygons[0])
        inner_circle = np.array(polygons[1])
        overlay_mask = draw_circles_intersection2(outer_circle, inner_circle, overlay_mask, plot_flag=False)
        return polygons, overlay_mask
    return polygons

def load_annotation(coco_data, image_info, image_dir, file_name=''):
    id_2_category = {}
    for cat in coco_data['categories']:
        id_2_category[cat['id']] = cat['name']

    if file_name != '':
        image_info = [image_row for image_row in coco_data['images'] if image_row['id'] == file_name][0]
    # Load image information
    image_path = os.path.join(image_dir, image_info['file_name'])
    image = cv2.imread(image_path)
    # Load segmentation masks for the image
    masks = [ann['segmentation'] for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
    bbox = [ann['bbox'] for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
    cls = [id_2_category[ann['category_id']] for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
    return image, masks, bbox, cls
       
def plot_segmentation(image, masks, bboxes, classes, image_name='Sample'):
    # Create an empty mask to overlay on the image
    overlay_mask = np.zeros_like(image)
    # print('Total Annotations: --- ', len(masks))
    for mask, bbox, clas in zip(masks, bboxes, classes):
        # Get segmentation mask
        if len(mask) == 2 and clas == 'Zona Pellucida' and (len(mask[0]) > 50 and len(mask[1]) > 50):
            # print("annotation['segmentation']: ", len(annotation['segmentation']))
            polygons, overlay_mask = check_hollow_circle(mask, overlay_mask)
            # draw_circles_intersection(circle1_polygon=np.array(polygons[0]), circle2_polygon=np.array(polygons[1]), overlay_mask=overlay_mask)
        else:
            overlay_mask = overlay_image(mask, clas, overlay_mask)
    # Add the overlay mask to the original image
    overlayed_image = cv2.addWeighted(image, 1, overlay_mask, 0.3, 0)
    # Plot the image with segmented areas
    plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image ID: {image_name}")
    plt.axis('off')
    plt.show()

def main(coco_annotation_file='Oolemma-Zona.json', image_dir='images/'):
    # Load COCO annotation file
    with open(coco_annotation_file, 'r') as fr:
        coco_data = json.load(fr)
    # Get all image IDs from the COCO dataset
    for idx, image_info in enumerate(coco_data['images']):
        # if idx == 8:
            # break
        idx += 1
        image, masks, bboxes, classes = load_annotation(coco_data, image_info, image_dir, image_name=image_info['file_name'])
        # print(len(bbox), len(cls), len(image), len(segm))
        # print(bbox, cls, image, segm)
        plot_segmentation(image, masks, bboxes, classes, image_info['file_name'])