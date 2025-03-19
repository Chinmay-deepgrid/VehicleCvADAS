import os
import cv2
import random
import logging
import numpy as np
import degirum as dg
from typing import *

try:
    import sys
    from utils import ObjectModelType, hex_to_rgb, NMS, Scaler
    from core import ObjectDetectBase, RectInfo
    sys.path.append("..")
except:
    from .utils import ObjectModelType, hex_to_rgb, NMS, Scaler
    from .core import ObjectDetectBase, RectInfo

##############################################################################
# Utility function for letterboxing
##############################################################################
def letterbox_frame(frame_rgb: np.ndarray, target_shape: tuple, padding_value=(0, 0, 0)):
    """
    Resize a frame with letterboxing to fit the target size while preserving aspect ratio.
    
    :param frame_rgb: RGB image as a NumPy array (H,W,3)
    :param target_shape: Target shape in NHWC format (e.g., (1, 640, 640, 3))
    :param padding_value: RGB tuple for padding
    :return: (final_image, scale, pad_top, pad_left)
    """
    h, w, c = frame_rgb.shape
    _, target_h, target_w, _ = target_shape

    scale_x = target_w / w
    scale_y = target_h / h
    scale = min(scale_x, scale_y)

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    letterboxed = np.full((target_h, target_w, c), padding_value, dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    letterboxed[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    # Add batch dimension: (1, H, W, 3)
    final_image = np.expand_dims(letterboxed, axis=0)
    return final_image, scale, pad_top, pad_left

##############################################################################
# YOLO-specific parameter helper
##############################################################################
class YoloLiteParameters:
    def __init__(self, model_type, input_shape, num_classes):
        self.lite = False
        if model_type == ObjectModelType.YOLOV5_LITE:
            self.lite = True
        anchors = [[10, 13, 16, 30, 33, 23],
                   [30, 61, 62, 45, 59, 119],
                   [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.input_shape = input_shape[-2:]

##############################################################################
# Main Detector Class
##############################################################################
class YoloDetector(ObjectDetectBase, YoloLiteParameters):
    _defaults = {
        "model_path": './models/yolov7.hef',
        "model_type": ObjectModelType.YOLOV5,  # or YOLOV7 if you have that enumerated
        "classes_path": '/home/chinmay/hailo_models/coco_label.txt',  # Using a text file for labels
        "box_score": 0.4,
        "box_nms_iou": 0.45
    }

    def __init__(self, logger=None, **kwargs):
        ObjectDetectBase.__init__(self, logger)
        self.__dict__.update(kwargs)  # Update with user overrides

        self._initialize_class(self.classes_path)
        self._initialize_model(self.model_path)

        # Now that self.input_shapes is defined, pass it to YoloLiteParameters
        YoloLiteParameters.__init__(self, self.model_type, self.input_shapes, len(self.class_names))

    def _initialize_model(self, model_path: str) -> None:
        model_path = os.path.expanduser(model_path)
        if self.logger:
            self.logger.debug(f"Model path: {model_path}")

        if model_path.endswith('.hef'):
            if self.logger:
                self.logger.info(f"Loading HEF model from {model_path}")
            try:
                self.engine = dg.load_model(
                    model_name="yolov7",  # Must match your .json or model config
                    inference_host_address="localhost",
                    zoo_url=""
                )
                # YOLOv7 typically expects (1, 640, 640, 3)
                self.input_shapes = (1, 640, 640, 3)
                if self.logger:
                    self.logger.info(f"HEF model loaded successfully: {self.engine}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"HEF model loading error: {e}")
                raise
        else:
            raise Exception("Only HEF models are supported in this configuration.")

    def _initialize_class(self, classes_path: str) -> None:
        """
        Read a plain text file (one label per line).
        """
        classes_path = os.path.expanduser(classes_path)
        if self.logger:
            self.logger.debug(f"Class path: {classes_path}")
        assert os.path.isfile(classes_path), f"{classes_path} does not exist."

        with open(classes_path, "r") as f:
            class_names = f.readlines()
        self.class_names = [c.strip() for c in class_names]

        # Create a random color for each class
        get_colors = list(map(lambda i: hex_to_rgb("#" + "%06x" % random.randint(0, 0xFFFFFF)), range(len(self.class_names))))
        self.colors_dict = dict(zip(self.class_names, get_colors))

    def DetectFrame(self, srcimg: cv2.Mat) -> None:
        """
        Reads a single frame (BGR), applies letterboxing, runs inference, and postprocesses.
        Detections are stored in self._object_info.
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        self.orig_h, self.orig_w, _ = img_rgb.shape

        # Letterbox the frame to match input shape (1, 640, 640, 3)
        input_image, self.scale, self.pad_top, self.pad_left = letterbox_frame(
            img_rgb, self.input_shapes, padding_value=(0, 0, 0)
        )

        # Run inference using DeGirum PySDK
        output_from_network = self.engine(input_image)

        # Get the raw detection tensor from the model output
        detection_output = output_from_network.results[0]['data']

        # Process detections: map letterboxed coordinates back to original image
        _raw_boxes, _raw_class_ids, _raw_class_confs, _raw_kpss = self._process_output(detection_output)

        # Apply Non-Maximum Suppression (NMS)
        self._object_info = self.get_nms_results(_raw_boxes, _raw_class_confs, _raw_class_ids, _raw_kpss)

    def _process_output(self, detection_output: np.ndarray) -> Tuple[np.ndarray, list, list, list]:
        """
        Converts raw YOLO output into bounding boxes, class IDs, and confidences.
        Uses letterbox mapping to return boxes in [xmin, ymin, xmax, ymax].
        """
        postprocessed = self._postprocess_detection_results(detection_output)

        _raw_boxes = []
        _raw_class_ids = []
        _raw_class_confs = []

        for det in postprocessed:
            bbox = det['bbox']
            score = det['score']
            cls_id = det['category_id']

            _raw_boxes.append(bbox)
            _raw_class_ids.append(cls_id)
            _raw_class_confs.append(score)

        return np.array(_raw_boxes), _raw_class_ids, _raw_class_confs, []

    def _postprocess_detection_results(self, detection_output: np.ndarray, confidence_threshold=0.3) -> List[dict]:
        """
        Maps YOLO detections from letterbox coords -> original image coords.
        IMPORTANT: This assumes the YOLO model returns bounding boxes in the order:
                   (x_min, y_min, x_max, y_max).

        Returns:
            [
              {
                "bbox": [xmin, ymin, xmax, ymax],
                "score": float,
                "category_id": int,
                "label": str
              },
              ...
            ]
        """
        output_array = detection_output.reshape(-1)
        batch, in_h, in_w, _ = self.input_shapes
        num_classes = len(self.class_names)

        results = []
        idx = 0
        for class_id in range(num_classes):
            num_detections = int(output_array[idx])
            idx += 1
            if num_detections == 0:
                continue
            for _ in range(num_detections):
                if idx + 5 > len(output_array):
                    break
                x_min, y_min, x_max, y_max = map(float, output_array[idx:idx+4])
                score = float(output_array[idx + 4])
                idx += 5

                if score < confidence_threshold:
                    continue

                # Convert normalized coords -> letterbox absolute coords
                lx_min = x_min * in_w
                ly_min = y_min * in_h
                lx_max = x_max * in_w
                ly_max = y_max * in_h

                # Map letterbox coords back to original image
                ox_min = (lx_min - self.pad_left) / self.scale
                oy_min = (ly_min - self.pad_top) / self.scale
                ox_max = (lx_max - self.pad_left) / self.scale
                oy_max = (ly_max - self.pad_top) / self.scale

                # Clamp to original image
                ox_min = max(0, min(ox_min, self.orig_w))
                oy_min = max(0, min(oy_min, self.orig_h))
                ox_max = max(0, min(ox_max, self.orig_w))
                oy_max = max(0, min(oy_max, self.orig_h))

                results.append({
                    "bbox": [ox_min, oy_min, ox_max, oy_max],
                    "score": score,
                    "category_id": class_id,
                    "label": self.class_names[class_id]
                })

            if idx >= len(output_array) or all(v == 0 for v in output_array[idx:]):
                break
        return results

    def get_nms_results(self, boxes: list, class_confs: list, class_ids: list, kpss: list) -> List[RectInfo]:
        """
        Applies NMS and returns a list of RectInfo objects with final bounding boxes.
        """
        results = []
        # If your NMS expects xywh, you must convert from xyxy to xywh. Adjust as needed.
        # If it can handle xyxy directly, remove or adapt 'dets_type="xywh"'.
        nms_results = NMS.fast_soft_nms(np.array(boxes), class_confs, self.box_nms_iou, dets_type="xywh")
        if len(nms_results) > 0:
            for i in nms_results:
                try:
                    predicted_class = self.class_names[class_ids[i]]
                except:
                    predicted_class = "unknown"
                conf = class_confs[i]
                bbox = boxes[i]  # [xmin, ymin, xmax, ymax]
                results.append(
                    RectInfo(
                        *bbox,  # (xmin, ymin, xmax, ymax)
                        conf=conf,  # still stored if you need it internally
                        label=predicted_class,
                        kpss=[]
                    )
                )
        return results

    def DrawDetectedOnFrame(self, frame_show: cv2.Mat) -> None:
        """
        Draw bounding boxes and ONLY the object name on the frame.
        """
        tl = 3  # line thickness
        for _info in self._object_info:
            xmin, ymin, xmax, ymax = _info.tolist()  # [xmin, ymin, xmax, ymax]
            label = _info.label
            color = self.colors_dict.get(label, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(frame_show, (xmin, ymin), (xmax, ymax), color, thickness=tl)

            # Draw ONLY the class label above the box
            label_pos = (xmin, max(ymin - 10, 0))  # ensure we don't go off-screen
            cv2.putText(
                frame_show,
                label,  # Only the object name
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

##############################################################################
# Example Main
##############################################################################
if __name__ == "__main__":
    import time
    capture = cv2.VideoCapture(r"./temp/test.avi")  # Example video input
    config = {
        "model_path": '/home/chinmay/hailo_models/yolov7.hef',  
        "model_type": ObjectModelType.YOLOV5,
        "classes_path": '/home/chinmay/hailo_models/coco_label.txt',  # Text file with labels
        "box_score": 0.4,
        "box_nms_iou": 0.45,
    }
    YoloDetector.set_defaults(config)
    network = YoloDetector()

    fps = 0
    frame_count = 0
    start = time.time()

    while True:
        ret, frame = capture.read()
        if not ret or frame is None:
            print("End of stream.")
            break

        # Run detection
        network.DetectFrame(frame)

        # Draw results
        network.DrawDetectedOnFrame(frame)

        frame_count += 1
        if frame_count >= 30:
            end = time.time()
            fps = frame_count / (end - start)
            frame_count = 0
            start = time.time()

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("output", frame)

        if cv2.waitKey(1) == 27:  # press 'ESC' to quit
            break

    capture.release()
    cv2.destroyAllWindows()
