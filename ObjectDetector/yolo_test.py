#!/usr/bin/env python3
import cv2
import numpy as np
import degirum as dg
import json

##############################################################################
# Utility Functions
##############################################################################

def letterbox_frame(frame, target_shape, padding_value=(0, 0, 0)):
    """
    Resize a frame with letterboxing to fit the target size while preserving the aspect ratio.
    
    Parameters:
        frame (ndarray): Input image in RGB format.
        target_shape (tuple): Target shape in NHWC format (e.g., (1, 640, 640, 3)).
        padding_value (tuple): RGB values for padding (default is black).
        
    Returns:
        final_image (ndarray): The letterboxed image as a UINT8 array with shape target_shape.
        scale (float): Scaling ratio applied to the original image.
        pad_top (int): Padding applied to the top.
        pad_left (int): Padding applied to the left.
    """
    h, w, c = frame.shape
    target_height, target_width = target_shape[1], target_shape[2]
    scale_x = target_width / w
    scale_y = target_height / h
    scale = min(scale_x, scale_y)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    letterboxed_image = np.full((target_height, target_width, c), padding_value, dtype=np.uint8)
    
    pad_top = (target_height - new_h) // 2
    pad_left = (target_width - new_w) // 2
    letterboxed_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_frame
    
    final_image = np.expand_dims(letterboxed_image, axis=0)
    return final_image, scale, pad_top, pad_left

def postprocess_detection_results(
    detection_output, 
    input_shape, 
    num_classes, 
    label_dictionary, 
    confidence_threshold=0.3,
    scale=1.0, 
    pad_top=0, 
    pad_left=0, 
    original_width=0, 
    original_height=0
):
    """
    Process the raw output tensor from the model to produce formatted detection results.
    Adjust bounding boxes from letterboxed coordinates back to the original image coordinates.
    """
    batch, input_height, input_width, _ = input_shape
    new_inference_results = []
    output_array = detection_output.reshape(-1)
    index = 0
    
    for class_id in range(num_classes):
        num_detections = int(output_array[index])
        index += 1
        if num_detections == 0:
            continue
        for _ in range(num_detections):
            if index + 5 > len(output_array):
                break
            score = float(output_array[index + 4])
            y_min, x_min, y_max, x_max = map(float, output_array[index:index+4])
            index += 5
            if score < confidence_threshold:
                continue
            
            # Convert normalized coords in [0..1] to letterbox absolute coords
            lx_min = x_min * input_width
            ly_min = y_min * input_height
            lx_max = x_max * input_width
            ly_max = y_max * input_height
            
            # Subtract padding and scale back to original
            ox_min = (lx_min - pad_left) / scale
            oy_min = (ly_min - pad_top) / scale
            ox_max = (lx_max - pad_left) / scale
            oy_max = (ly_max - pad_top) / scale
            
            # Clamp to original image dimensions
            ox_min = max(0, min(ox_min, original_width))
            oy_min = max(0, min(oy_min, original_height))
            ox_max = max(0, min(ox_max, original_width))
            oy_max = max(0, min(oy_max, original_height))
            
            result = {
                "bbox": [ox_min, oy_min, ox_max, oy_max],
                "score": score,
                "category_id": class_id,
                "label": label_dictionary.get(str(class_id), f"class_{class_id}")
            }
            new_inference_results.append(result)
        
        if index >= len(output_array) or all(v == 0 for v in output_array[index:]):
            break
    return new_inference_results

def overlay_bboxes_and_labels(image_rgb, annotations, color=(0, 255, 0), font_scale=1, thickness=2):
    """
    Overlay bounding boxes and labels on an RGB image.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for annotation in annotations:
        bbox = annotation['bbox']
        label = annotation['label']
        x1, y1, x2, y2 = map(int, map(round, bbox))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image_bgr, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

##############################################################################
# Main
##############################################################################

def main():
    video_path = "/home/chinmay/Downloads/challenge_video.mp4"
    labels_file = "/home/chinmay/hailo_models/coco_labels.json"
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Load the YOLOv7 model once
    engine = dg.load_model(
        model_name="yolov7",
        inference_host_address="@local",
        zoo_url="/home/chinmay/hailo_models"
    )
    
    # Load label dictionary from JSON
    with open(labels_file, "r") as json_file:
        label_dictionary = json.load(json_file)
    
    # Read frames in a loop
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, _ = frame_rgb.shape
        
        # Letterbox the frame to [1, 640, 640, 3]
        input_image, scale, pad_top, pad_left = letterbox_frame(frame_rgb, (1, 640, 640, 3))
        
        # Run inference
        inference_result = engine(input_image)
        
        # Post-process detection results (map back to original frame coords)
        detection_results = postprocess_detection_results(
            inference_result.results[0]['data'],
            engine.input_shape[0],
            num_classes=80,
            label_dictionary=label_dictionary,
            confidence_threshold=0.3,
            scale=scale,
            pad_top=pad_top,
            pad_left=pad_left,
            original_width=orig_w,
            original_height=orig_h
        )
        
        # Overlay bounding boxes and labels on the original frame (RGB)
        overlayed_rgb = overlay_bboxes_and_labels(frame_rgb, detection_results)
        
        # Convert back to BGR for display with OpenCV
        overlayed_bgr = cv2.cvtColor(overlayed_rgb, cv2.COLOR_RGB2BGR)
        
        # Show the resulting frame
        cv2.imshow("Detections", overlayed_bgr)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
