import cv2
import numpy as np
import hailo
import argparse

# Load the Hailo Model
def load_hailo_model(model_path):
    device = hailo.Device()
    hef = hailo.Hef(model_path)
    network_group = device.create_network_group(hef)
    return device, network_group

# Preprocess frame
def preprocess_frame(frame, input_shape):
    resized = cv2.resize(frame, (input_shape[1], input_shape[0]))
    normalized = resized.astype(np.float32) / 255.0  # Normalize pixel values
    return np.expand_dims(normalized, axis=0)

# Postprocess model output
def postprocess_output(output):
    return output  # Modify based on actual output format

# Custom callback class
class UserAppCallback:
    def __init__(self):
        self.frame_count = 0
        self.use_frame = True

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count

# Process video and perform inference
def process_video(video_file, model_path):
    if not model_path or not video_file:
        print("Error: Model path or video file is missing!")
        return

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    device, network_group = load_hailo_model(model_path)
    user_data = UserAppCallback()
    input_shape = (640, 640, 3)  # Adjust based on the model

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        user_data.increment()
        input_data = preprocess_frame(frame, input_shape)
        output = network_group.infer(input_data)
        detections = postprocess_output(output)

        detection_count = 0
        for detection in detections:
            bbox = detection.get_bbox()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            detection_count += 1

        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame Count: {user_data.get_count()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on video using Hailo model")
    parser.add_argument("--model", type=str, required=True, help="Path to HEF model file")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    args = parser.parse_args()

    process_video(args.video, args.model)
