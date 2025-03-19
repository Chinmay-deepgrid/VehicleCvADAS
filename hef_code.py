import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import cv2
import os
import numpy as np
import hailo
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

def load_hailo_model(model_path):
    """Load the Hailo model."""
    device = hailo.Device()
    hef = hailo.Hef(model_path)
    network_group = device.create_network_group(hef)
    return device, network_group

def preprocess_frame(frame, input_shape):
    """Resize and normalize the input frame."""
    resized = cv2.resize(frame, (input_shape[1], input_shape[0]))
    normalized = resized.astype(np.float32) / 255.0  # Normalize pixel values
    return np.expand_dims(normalized, axis=0)

def postprocess_output(output):
    """Process the output tensor to extract object detections."""
    return output

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class:
    def __init__(self):
        self.new_variable = 42
        self.frame_count = 0
        self.use_frame = True
        self.frame = None

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count

    def set_frame(self, frame):
        self.frame = frame

    def new_function(self):
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    """Callback function for object detection pipeline."""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    if frame is not None:
        input_shape = (640, 640, 3)
        input_data = preprocess_frame(frame, input_shape)
        
        # Replace with your model path
        model_path = "./ObjectDetector/models/yolov5m6_fp16.hef"  # Correct the path
        device, network_group = load_hailo_model(model_path)
        output = network_group.infer(input_data)
        detections = postprocess_output(output)

        detection_count = 0
        for detection in detections:
            bbox = detection.get_bbox()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            detection_count += 1

        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

def process_video(video_file):
    """Process video frames and pass them to the inference pipeline."""
    if not os.path.exists(video_file):
        print(f"Error: {video_file} not found!")
        return

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        user_data = user_app_callback_class()
        app = GStreamerDetectionApp(app_callback, user_data)
        app.run()

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your video file path
    video_file = "./assets/challenge_video.mp4"  # Correct the path to your video file
    process_video(video_file)
