import os
from pdf2image import convert_from_path
from PIL import ImageDraw


def ConvertToImage(pdf_path, output_path):
    '''
    Converts the pdf file into images.
    '''
    pdf_name = list(pdf_path.split('/'))[-1][:-4]
    images = convert_from_path(pdf_path, dpi=300, poppler_path= r'C:\Program Files (x86)\poppler-24.08.0\Library\bin')  # 300 DPI gives high quality

    for i, image in enumerate(images):
        path = f'{output_path}/{pdf_name}_page_{i+1}.png'
        if not os.path.exists(path):
            image.save(path, 'PNG')

def expand_bbox_relative(bbox, margin_ratio=0.05):
    '''
    This function expands the original bounding box for better structure recognition.
    Returns: list of new coordinates [x0, y0, x1, y1]
    '''
    try:
        x0, y0, x1, y1 = bbox

        # Calculate width and height of bbox
        width = x1 - x0
        height = y1 - y0

        # Calculate margins
        width_margin = width * margin_ratio
        height_margin = height * margin_ratio * 2

        # Expand bbox
        new_x0 = x0 - width_margin
        new_y0 = y0 - height_margin
        new_x1 = x1 + width_margin
        new_y1 = y1 + height_margin

        return [int(new_x0), int(new_y0), int(new_x1), int(new_y1)]

    except:
        print("Skipping this due to bbox 0-D tensor")


def visualize_structure(image, rows, columns):
    """
    Draw bounding boxes on the image to visualize detected structure
    """
    draw = ImageDraw.Draw(image)
    # Draw rows in red
    for i, row in enumerate(rows):
        bbox = row['bbox']
        draw.rectangle(bbox, outline='red', width=2)
        draw.text((bbox[0], bbox[1]-15), f"Row {i+1}", fill='red')
    # Draw columns in blue
    for i, col in enumerate(columns):
        bbox = col['bbox']
        draw.rectangle(bbox, outline='blue', width=2)
        draw.text((bbox[0], bbox[1]-15), f"Col {i+1}", fill='blue')
    
    return image

def generate_cells(x, y):
    '''
    This function takes list of rows and columns.
    returns : list of cells
    '''
    grid_cells = []
    for r_idx, r in enumerate(y):
        for c_idx, c in enumerate(x):
            x0 = max(r[0], c[0])
            y0 = max(r[1], c[1])
            x1 = min(r[2], c[2])
            y1 = min(r[3], c[3])
            grid_cells.append({
                "row": r_idx,
                "column": c_idx,
                "bbox": [x0, y0, x1, y1]
            })
    return grid_cells