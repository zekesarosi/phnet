import cv2
import numpy as np
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Function to draw bounding boxes (mesh grid)
def draw_mesh_grid(image, boxes, class_ids, metadata, scores, threshold):
    for box, cls, score in zip(boxes, class_ids, scores):
        if score >= threshold:  # Filter by score threshold
            # Convert tensor to numpy array and flatten
            box = box.numpy().astype(int)

            # Unpack the box coordinates
            x1, y1, x2, y2 = box

            color = (255, 0, 0)  # Green color for the mesh grid
            thickness = 1

            # Draw rectangle around each hold
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            """
            # Draw label text on the image
            label = metadata.thing_classes[cls]
            label_text = f"{label} ({score:.2f})"
            font_scale = 1
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Calculate text size to position it correctly
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            x, y = x1, y1 - 10  # Position the text

            # Ensure coordinates are integers
            x, y = int(x), int(y)

            # Draw the text
            cv2.putText(image, label_text, (x, y), font, font_scale, color, thickness)
            """
    return image

# Function to draw masks (mesh grid)
def draw_mesh_grid_from_masks(image, masks, class_ids, metadata, scores, threshold):
    for mask, cls, score in zip(masks, class_ids, scores):
        if score >= threshold:  # Filter by score threshold
            mask = mask.numpy().astype(np.uint8)
            color = (0, 0, 255)  # Red color for the mesh grid

            # Create a colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask == 1] = color

            # Overlay mask on the original image
            image = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)

            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(image, [contour], -1, color, 2)
            """
            # Draw label text on the image
            label = metadata.thing_classes[cls]
            label_text = f"Hold Segmentation"
            font_scale = 1
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Calculate text size to position it correctly
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            x, y = 10, 30 + cls * 30  # Position the text

            # Ensure coordinates are integers
            x, y = int(x), int(y)

            # Draw the text
            cv2.putText(image, label_text, (x, y), font, font_scale, color, thickness)
            """
    return image


bounding_boxes = []
meshes = []

def split_image(image_path, output_prefix, predictor, score_threshold=.5):
    # Load the image using cv2
    img = cv2.imread(image_path)
    
    # Get the dimensions of the image
    height, width, _ = img.shape
    
    # Calculate the size of each smaller image
    new_width = width // 2
    new_height = height // 2
    
    # Define the bounding boxes for the 4 smaller images
    parts = [
        img[0:new_height, 0:new_width],           # Top-left
        img[0:new_height, new_width:width],       # Top-right
        img[new_height:height, 0:new_width],      # Bottom-left
        img[new_height:height, new_width:width]   # Bottom-right
    ]
    
    # Process each smaller image
    for i, part in enumerate(parts):
        original_height, original_width = part.shape[:2]

        # Make predictions using the model
        outputs = predictor(part)

        # Extract predictions
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        classes = instances.pred_classes if instances.has("pred_classes") else None
        scores = instances.scores if instances.has("scores") else None

        # Draw and save bounding boxes
        img_with_mesh = draw_mesh_grid(part.copy(), boxes, classes, metadata, scores, score_threshold)
        bounding_boxes.append(img_with_mesh)

        # Draw and save masks
        masks = instances.pred_masks if instances.has("pred_masks") else None
        img_with_mesh_from_masks = draw_mesh_grid_from_masks(part.copy(), masks, classes, metadata, scores, score_threshold)
        meshes.append(img_with_mesh_from_masks)
    
    # Stitch the images back together
    top_row = np.hstack((bounding_boxes[0], bounding_boxes[1]))
    bottom_row = np.hstack((bounding_boxes[2], bounding_boxes[3]))
    stitched_image_boxes = np.vstack((top_row, bottom_row))

    top_row_masks = np.hstack((meshes[0], meshes[1]))
    bottom_row_masks = np.hstack((meshes[2], meshes[3]))
    stitched_image_masks = np.vstack((top_row_masks, bottom_row_masks))

    # Save or display the final stitched images
    cv2.imwrite(f'{output_prefix}_stitched_boxes.png', stitched_image_boxes)
    cv2.imwrite(f'{output_prefix}_stitched_masks.png', stitched_image_masks)


    # Save the bounding box coordinates in a json file
   

# Define file paths
MODEL_DIRECTORY = "/home/dtron2_user/demo/model/911model"
SAMPLE_IMAGE = "/home/dtron2_user/demo/phnom night.jpg"

# Set up configuration for the model
cfg = get_cfg()
cfg.set_new_allowed(True)
cfg.merge_from_file(os.path.join(MODEL_DIRECTORY, "config.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(MODEL_DIRECTORY, "model_final.pth")
cfg.MODEL.DEVICE = 'cpu'

# Setup metadata
metadata = MetadataCatalog.get("meta")
metadata.thing_classes = ["hold"]

# Create a predictor object
predictor = DefaultPredictor(cfg)


output_prefix = "output"

split_image(SAMPLE_IMAGE, output_prefix, predictor)
