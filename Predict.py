import argparse
import json
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

###### This predict dental disease semantic segmentation script was written by Ryan Banks ######

# Default class IDs
ANTERIOR_TOOTH_CLASS = 0
INFLAMMATION_CLASS = 1
NON_INFLAMMATION_CLASS = 2
PLAQUE_CLASS = 3
POSTERIOR_TOOTH_CLASS = 4
def result_to_json(result, image_percentages):
    """
    Convert one YOLO segmentation result into a fully JSON-serializable dict.

    Includes:
    - source image path
    - bounding boxes
    - segmentation masks as polygons
    - plaque and gingivitis percentages
    """

    prediction = {
        "source": str(result.path),
        "boxes": [],
        "masks": [],
        "percentages": {
            "plaque_percent": float(image_percentages["plaque_percent"]),
            "gingivitis_percent": float(image_percentages["gingivitis_percent"]),
        },
    }

    # Boxes
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy()

        for coords, confidence, class_id in zip(xyxy, conf, cls):
            prediction["boxes"].append({
                "xyxy": [float(v) for v in coords],
                "confidence": float(confidence),
                "class": int(class_id),
            })

    # Masks as polygons
    # result.masks.xy contains one polygon array per detected
    # segmentation instance in original image coordinates.
    if result.masks is not None and len(result.masks.xy) > 0:

        classes = result.boxes.cls.detach().cpu().numpy() if result.boxes is not None else []
        confidences = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else []

        for index, polygon in enumerate(result.masks.xy):
            prediction["masks"].append({
                "class": int(classes[index]) if len(classes) > index else None,
                "confidence": float(confidences[index]) if len(confidences) > index else None,
                "polygon": [[float(point[0]), float(point[1])] for point in polygon],
            })

    return prediction


def _mask_for_class(result, class_id):
    """Merge all YOLO instance masks belonging to one class into one boolean mask."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None

    masks = result.masks.data
    classes = result.boxes.cls
    indices = torch.where(classes == class_id)[0]

    if len(indices) == 0:
        return torch.zeros(masks.shape[-2:], dtype=torch.bool, device=masks.device)

    return torch.any(masks[indices].bool(), dim=0)


def _union_masks(masks):
    """
    Return the pixel-wise union of a collection of boolean masks.

    Overlapping pixels are counted only once.
    """
    valid_masks = [mask for mask in masks if mask is not None and torch.any(mask)]

    if not valid_masks:
        return None

    union = valid_masks[0].clone().bool()

    for mask in valid_masks[1:]:
        union |= mask.bool()

    return union


def calculate_plaque_percent(result):
    """
    Plaque percentage using the entire predicted plaque, anterior tooth,
    and posterior tooth areas:

        plaque area / (plaque area + anterior area + posterior area)

    The denominator uses the pixel-wise union of plaque, anterior, and
    posterior masks so overlapping pixels are counted only once.

    Returns a percentage in the range 0-100.
    """
    plaque_mask = _mask_for_class(result, PLAQUE_CLASS)
    anterior_mask = _mask_for_class(result, ANTERIOR_TOOTH_CLASS)
    posterior_mask = _mask_for_class(result, POSTERIOR_TOOTH_CLASS)

    if plaque_mask is None or not torch.any(plaque_mask):
        return 0.0

    total_area_mask = _union_masks([plaque_mask, anterior_mask, posterior_mask])

    if total_area_mask is None:
        return 0.0

    plaque_pixels = int(torch.sum(plaque_mask).item())
    total_pixels = int(torch.sum(total_area_mask).item())

    if total_pixels == 0:
        return 0.0

    return 100.0 * plaque_pixels / total_pixels


def calculate_gingivitis_percent(result):
    """
    Gingivitis percentage using the entire predicted inflamed and healthy
    gingiva areas:

        inflamed gingiva / (inflamed gingiva + healthy gingiva)

    The denominator uses the pixel-wise union of inflamed and healthy
    masks so overlapping pixels are counted only once.

    Returns a percentage in the range 0-100.
    """
    inflammation_mask = _mask_for_class(result, INFLAMMATION_CLASS)
    healthy_mask = _mask_for_class(result, NON_INFLAMMATION_CLASS)

    total_gingiva_mask = _union_masks([inflammation_mask, healthy_mask])

    if total_gingiva_mask is None:
        return 0.0

    inflamed_pixels = int(torch.sum(inflammation_mask).item()) if inflammation_mask is not None else 0
    total_gingiva_pixels = int(torch.sum(total_gingiva_mask).item())

    if total_gingiva_pixels == 0:
        return 0.0

    return 100.0 * inflamed_pixels / total_gingiva_pixels


def calculate_disease_percentages(result, source_path):
    """Calculate the two percentages requested for one prediction."""
    plaque_percent = calculate_plaque_percent(result)
    gingivitis_percent = calculate_gingivitis_percent(result)

    return {
        "source": str(source_path),
        "plaque_percent": float(plaque_percent),
        "gingivitis_percent": float(gingivitis_percent),
    }


def _image_output_dir(save_root, source_path):
    """Create a dedicated output folder for one input image."""
    image_name = os.path.splitext(os.path.basename(str(source_path)))[0]
    out_dir = os.path.join(save_root, image_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _mask_for_class_resized(result, class_id, width, height):
    """Get one merged class mask resized to the original image size."""
    mask = _mask_for_class(result, class_id)

    if mask is None:
        return np.zeros((height, width), dtype=bool)

    mask_np = mask.detach().cpu().numpy().astype(np.uint8)

    if mask_np.shape != (height, width):
        mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)

    return mask_np.astype(bool)


def save_mask_visualization(result, image_percentages, save_root):
    """Save the original image with segmentation masks and percentage labels only."""
    source_path = image_percentages["source"]
    out_dir = _image_output_dir(save_root, source_path)

    image = result.orig_img.copy()
    height, width = image.shape[:2]
    overlay = image.copy()

    # Fixed BGR colors for each segmentation class. No boxes are drawn.
    class_colors = {
        ANTERIOR_TOOTH_CLASS: (255, 200, 0),
        INFLAMMATION_CLASS: (0, 0, 255),
        NON_INFLAMMATION_CLASS: (0, 255, 0),
        PLAQUE_CLASS: (0, 255, 255),
        POSTERIOR_TOOTH_CLASS: (255, 0, 255),
    }

    for class_id, color in class_colors.items():
        class_mask = _mask_for_class_resized(result, class_id, width, height)

        if np.any(class_mask):
            overlay[class_mask] = color

    # Blend masks onto the original image; this intentionally contains no boxes.
    visual = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)

    # Place the percentages in the top-left corner rather than in a selected tooth area.
    x, y = 20, 35

    text_lines = [
        f"Plaque: {image_percentages['plaque_percent']:.2f}%",
        f"Gingivitis: {image_percentages['gingivitis_percent']:.2f}%",
    ]

    # Keep labels inside the image bounds.
    x = max(5, min(x, max(5, width - 300)))
    y = max(30, min(y, max(30, height - 45)))

    for line_index, line in enumerate(text_lines):
        text_y = y + line_index * 30

        cv2.putText(visual, line, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(visual, line, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    image_name = os.path.splitext(os.path.basename(str(source_path)))[0]
    visual_path = os.path.join(out_dir, f"{image_name}_masks_percentages.png")

    cv2.imwrite(visual_path, visual)

    return visual_path


def save_prediction_data(result, image_percentages, save_root):
    """Save boxes, masks, and percentages in one dedicated image folder."""
    source_path = image_percentages["source"]
    out_dir = _image_output_dir(save_root, source_path)
    image_name = os.path.splitext(os.path.basename(str(source_path)))[0]

    # Save boxes in a JSON-friendly format.
    boxes_path = os.path.join(out_dir, f"{image_name}_boxes.json")
    boxes_data = []

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.detach().cpu().tolist()
        conf = result.boxes.conf.detach().cpu().tolist()
        cls = result.boxes.cls.detach().cpu().tolist()

        for coords, confidence, class_id in zip(xyxy, conf, cls):
            boxes_data.append({
                "xyxy": [float(v) for v in coords],
                "confidence": float(confidence),
                "class": int(class_id),
            })

    with open(boxes_path, "w") as f:
        json.dump(boxes_data, f, indent=2)

    # Save segmentation polygons as plain JSON.
    masks_path = os.path.join(out_dir, f"{image_name}_masks.json")

    if result.masks is None:
        masks_data = []
    else:
        masks_data = [polygon.tolist() for polygon in result.masks.xy]

    with open(masks_path, "w") as f:
        json.dump(masks_data, f, indent=2)

    percentages_path = os.path.join(out_dir, f"{image_name}_percentages.json")

    with open(percentages_path, "w") as f:
        json.dump(image_percentages, f, indent=2)

    return {
        "folder": out_dir,
        "boxes": boxes_path,
        "masks": masks_path,
        "percentages": percentages_path,
    }


def predict(args, model):
    # Handles data location input.
    if os.path.exists(args.data):
        data_loc = args.data
    else:
        data_loc = os.path.join(os.getcwd(), args.data)

        if not os.path.exists(data_loc):
            raise FileNotFoundError(f"Data path {data_loc} does not exist.")

    # Resolve save directory.
    save_dir = None

    if args.save is not None:
        save_dir = args.save if os.path.isabs(args.save) else os.path.join(os.getcwd(), args.save)
        os.makedirs(save_dir, exist_ok=True)

    if (args.save_image or args.save_pred) and save_dir is None:
        raise ValueError("--save is required when using --save-image or --save-pred")

    # Run segmentation prediction.
    results = model.predict(source=data_loc, save=False, task="segment")

    # Final JSON output.
    output = {"predictions": []}

    # Process each image result.
    for result in results:

        # Calculate disease percentages.
        image_percentages = calculate_disease_percentages(result, result.path)

        # Convert prediction to JSON-compatible data.
        prediction_data = result_to_json(result, image_percentages)

        # Add to final output.
        output["predictions"].append(prediction_data)

        # Optional visualization.
        if args.save_image:
            visual_path = save_mask_visualization(result, image_percentages, save_dir)
            print(f"Saved mask visualization: {visual_path}")

        # Optional JSON prediction file.
        if args.save_pred:
            saved_paths = save_prediction_data(result, image_percentages, save_dir)
            print(f"Saved prediction JSON: {saved_paths}")

        # Optional verbose display.
        if args.verbose:
            result.show()
            print(json.dumps(output, indent=2))

    return json.dumps(output, indent=2)


# this loads the model
def load_model(args):

    # Handles weights location.
    if os.path.exists(args.weights):
        weights = args.weights
    else:
        weights = os.path.join(os.getcwd(), args.weights)

        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights path {weights} does not exist.")

    # Load YOLO model.
    model = YOLO(weights)
    return model


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation Prediction")
    parser.add_argument("--data", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--save", type=str, default=None, help="Directory to save results")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to model weights")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print detailed results")
    parser.add_argument("--save-image", action="store_true", help="Save an image with segmentation masks and plaque/gingivitis percentages")
    parser.add_argument("--save-pred", action="store_true", help="Save boxes, masks, and percentage results for each image")
    args = parser.parse_args()

    # load model and weights
    model = load_model(args)

    # Run prediction and get output
    prediction_output = predict(args, model)


    return prediction_output


if __name__ == "__main__":
    main()