import numpy as np
import matplotlib.path as mpath

def create_coordinates(mask_polygons):
    """
    Create a list of coordinates from predicted mask of polygon of x and y values.

    Args:
    - mask_polygons: predicted mask of polygon

    Returns:
    - List of coordinates as [[x1, y1], [x2, y2], ..., [xn, yn]]
    """
    coordinates = []

    for contour in mask_polygons:
        x_coords, y_coords = contour

        # Combine x and y coordinates
        contour = np.column_stack((x_coords, y_coords))

        # Append to the result list
        coordinates.extend(contour)

    return coordinates

def flip_y_coordinates(coordinates, image_height):
    """
    Flip the y-coordinates in a list of coordinates.

    Args:
    - coordinates: List of coordinates as [[x1, y1], [x2, y2], ..., [xn, yn]]
    - image_height: Height of the image

    Returns:
    - List of flipped coordinates as [[x1, flipped_y1], [x2, flipped_y2], ..., [xn, flipped_yn]]
    """
    flipped_coordinates = np.array(coordinates)
    flipped_coordinates[:, 1] = image_height - flipped_coordinates[:, 1]
    return flipped_coordinates.tolist()

def sort_bleft_tright_np(coordinates):
    # Sort coordinates from left to right and bottom to top
    sorted_indices = np.lexsort((-coordinates[:, 1], coordinates[:, 0]))
    sorted_coordinates = coordinates[sorted_indices]
    return sorted_coordinates.tolist()

def sort_bleft_tright(coordinates):
    # Sort coordinates from left to right and bottom to top
    sorted_coordinates = sorted(coordinates, key=lambda point: (point[1], point[0]))
    return sorted_coordinates

def fill_polygon(coordinates):
    """
    Fill the area inside a polygon defined by a list of (x, y) coordinates.

    Args:
    - coordinates: List of (x, y) coordinates of the polygon as [[x1, y1], [x2, y2], ..., [xn, yn]]

    Returns:
    - List of filled area coordinates as [[x1, y1], [x2, y2], ..., [xn, yn]]
    """
    path = mpath.Path(coordinates)
    x_min, y_min = np.min(coordinates, axis=0)
    x_max, y_max = np.max(coordinates, axis=0)
    
    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    
    mask = path.contains_points(points)
    filled_area = points[mask]
    return filled_area.tolist()

def convert_machine_format(pred_mask, image_height, fill_mask, sort_t2b_l2r, flip_y):
    coords = create_coordinates(pred_mask)
    
    if fill_mask:
        coords = fill_polygon(coords)

    if sort_t2b_l2r:
        coords = sort_bleft_tright(np.array(coords))

    if flip_y:
        coords = flip_y_coordinates(coords, image_height)
    
    return coords