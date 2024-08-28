import torch
from torchvision import transforms
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import numpy as np
import csv
import easyocr
from tqdm.auto import tqdm
import os
import pandas as pd

# Custom transform
class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        return resized_image

# Utility functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def visualize_detected_tables(img, det_tables, out_path=None):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(30, 30)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 4
            hatch='//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 4
            hatch='//////'
        else:
            continue

        offset = 1.5
        shifted_bbox = [coord - offset for coord in bbox[:2]]
        rect = patches.Rectangle(shifted_bbox, bbox[2]-bbox[0] + 2*offset, bbox[3]-bbox[1] + 2*offset, linewidth=linewidth,
                                 edgecolor='none',facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(shifted_bbox, bbox[2]-bbox[0] + 2*offset, bbox[3]-bbox[1] + 2*offset, linewidth=linewidth,
                                 edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(shifted_bbox, bbox[2]-bbox[0] + 2*offset, bbox[3]-bbox[1] + 2*offset, linewidth=0,
                                 edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                             label='Table', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                             label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
               fontsize=10, ncol=2)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', dpi=150)

    return fig

def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops

def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        row_cells.sort(key=lambda x: x['column'][0])

        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates

def apply_ocr(cell_coordinates, cropped_table, reader):
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            result = reader.readtext(np.array(cell_image))
            if len(result) > 0:
                text = " ".join([x[1] for x in result])
                row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    print("Max number of columns:", max_num_columns)

    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data

def main(image_path, output_dir):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
    model.to(device)

    structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
    structure_model.to(device)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    detection_transform = transforms.Compose([
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    # Detect tables
    with torch.no_grad():
        outputs = model(pixel_values)

    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    objects = outputs_to_objects(outputs, image.size, id2label)

    # Visualize detected tables
    fig = visualize_detected_tables(image, objects)
    visualized_image = fig2img(fig)
    visualized_image.save(os.path.join(output_dir, "detected_tables.jpg"))

    # Crop tables
    tokens = []
    detection_class_thresholds = {
        "table": 0.5,
        "table rotated": 0.5,
        "no object": 10
    }
    tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=0)

    # Save cropped tables
    for i, table_crop in enumerate(tables_crops):
        cropped_table_image = table_crop['image'].convert("RGB")
        filename = f"table_{i+1}.jpg"
        full_path = os.path.join(output_dir, filename)
        cropped_table_image.save(full_path)
        print(f"Saved cropped image of table {i+1} at: {full_path}")

    # Process first table (you can extend this to process all tables if needed)
    cropped_table = tables_crops[0]['image'].convert("RGB")

    # Prepare image for structure recognition
    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pixel_values = structure_transform(cropped_table).unsqueeze(0).to(device)

    # Recognize table structure
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"
    cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)

    # Visualize cells
    cropped_table_visualized = cropped_table.copy()
    draw = ImageDraw.Draw(cropped_table_visualized)
    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")
    cropped_table_visualized.save(os.path.join(output_dir, "table_structure.jpg"))

    # Get cell coordinates
    cell_coordinates = get_cell_coordinates_by_row(cells)

    # Apply OCR
    reader = easyocr.Reader(['en'])
    data = apply_ocr(cell_coordinates, cropped_table, reader)

    # Save as CSV
    csv_path = os.path.join(output_dir, 'output.csv')
    with open(csv_path, 'w', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        for row, row_text in data.items():
            wr.writerow(row_text)

    # Display as DataFrame
    df = pd.read_csv(csv_path)
    print(df)

if __name__ == "__main__":
    image_path = "/path/to/your/image.png"  # Replace with your image path
    output_dir = "/path/to/output/directory"  # Replace with your desired output directory
    main(image_path, output_dir)
