from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image
from utils import *
import numpy as np

def table_detection_model(file_path):
    '''
    This function detects tables in an image.
    returns : list of table coordinates
    '''
    image = Image.open(file_path).convert("RGB")

    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    all_boxes = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
        all_boxes.append(box)

    return all_boxes 



def table_structure_recognition(file_path, detection_dir, page):
    image = Image.open(file_path).convert("RGB")
    box = table_detection_model(file_path)
    if(len(box)):
        for i in range(len(box)):
            path = detection_dir + '/' + page[:-4] + '_' + str(i) + '.png'
            if not os.path.exists(path):
                single_box = expand_bbox_relative(box[i])
                if not single_box:
                    continue  # Skip on invalid bbox

                image_array = np.array(image)
                xmin, ymin, xmax, ymax = [int(coords) for coords in single_box]

                table_region = image_array[ymin:ymax, xmin:xmax]
                table_image = Image.fromarray(table_region)

                print(f"\nExtracted table region: {table_image.size}")

                print("\n=== LOADING STRUCTURE RECOGNITION MODEL ===")


                structure_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
                structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")


                structure_inputs = structure_processor(images=table_image, return_tensors="pt")
                structure_outputs = structure_model(**structure_inputs)

                target_sizes_structure = torch.tensor([table_image.size[::-1]])
                structure_results = structure_processor.post_process_object_detection(
                    structure_outputs,
                    threshold=0.9,
                    target_sizes=target_sizes_structure
                )[0]

                print("\n=== STRUCTURE RECOGNITION RESULTS ===")

                rows = []
                columns = []

                for score, label, box in zip(structure_results["scores"], structure_results["labels"], structure_results["boxes"]):
                    box_coords = [round(i, 2) for i in box.tolist()]
                    element_type = structure_model.config.id2label[label.item()]
                    confidence = round(score.item(), 3)

                    # Categorize elements
                    if 'row' in element_type.lower():
                        rows.append({
                            'type': element_type,
                            'confidence': confidence,
                            'bbox': box_coords,
                            'y_center': (box_coords[1] + box_coords[3]) / 2
                        })
                    elif 'column' in element_type.lower():
                        columns.append({
                            'type': element_type,
                            'confidence': confidence,
                            'bbox': box_coords,
                            'x_center': (box_coords[0] + box_coords[2]) / 2
                        })

                # Sort rows by vertical position and columns by horizontal position
                rows.sort(key=lambda x: x['y_center'])
                columns.sort(key=lambda x: x['x_center'])

                print(f"\nStructure Summary:")
                print(f"- Detected {len(rows)} rows")
                print(f"- Detected {len(columns)} columns")

                # Create visualization
                table_image_copy = table_image.copy()
                visualized_image = visualize_structure(table_image_copy, rows, columns)

                #store the table to output directory
                if not os.path.exists(detection_dir):
                    os.mkdir(detection_dir)

                visualized_image.save(path, 'PNG')

    else:
        return 