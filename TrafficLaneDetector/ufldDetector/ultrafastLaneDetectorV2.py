import cv2
import numpy as np
import degirum as dg
import os
from typing import Tuple

try:
    from ufldDetector.utils import LaneModelType, OffsetType
    from TrafficLaneDetector.ufldDetector.core import LaneDetectBase
except:
    import sys
    from .utils import LaneModelType, OffsetType
    from .core import LaneDetectBase
    sys.path.append("..")


def _softmax(x, axis=-1):
    """
    Numerically stable softmax.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def fit_lane_polynomial(lane_points, degree=2, num_samples=50):
    """
    Fits a polynomial (default quadratic) to lane_points and returns a smoothed set of (x, y).
    """
    if len(lane_points) < degree + 1:
        # Not enough points to fit the chosen polynomial
        return lane_points

    xs = np.array([pt[0] for pt in lane_points], dtype=np.float32)
    ys = np.array([pt[1] for pt in lane_points], dtype=np.float32)

    # Sort points by y in descending order so we start from the bottom
    sort_idx = np.argsort(ys)[::-1]
    xs = xs[sort_idx]
    ys = ys[sort_idx]

    # Fit polynomial: x = f(y)
    coeffs = np.polyfit(ys, xs, deg=degree)
    poly_func = np.poly1d(coeffs)

    # Resample from max(y) to min(y) to ensure we start from the bottom
    y_max, y_min = ys[0], ys[-1]
    y_new = np.linspace(y_max, y_min, num_samples)
    x_new = poly_func(y_new)

    return [(int(xn), int(yn)) for xn, yn in zip(x_new, y_new)]


class ModelConfig:
    """
    Holds input dimensions, row anchors, etc., depending on the chosen UFLD model variant.
    """
    def __init__(self, model_type):
        if model_type == LaneModelType.UFLDV2_TUSIMPLE:
            self.init_tusimple_config()
        elif model_type == LaneModelType.UFLDV2_CURVELANES:
            self.init_curvelanes_config()
        else:
            self.init_culane_config()
        self.num_lanes = 4

    def init_tusimple_config(self):
        # Typically 800 x 320 for TuSimple
        self.img_w = 800
        self.img_h = 320
        self.griding_num = 100

        # No cropping => we keep the entire image
        self.crop_ratio = 1

        # Row anchors from the top (0) to near-bottom (710) in 56 steps
        self.row_anchor = np.linspace(0, 710, 56) / 720

        self.col_anchor = np.linspace(0, 1, 41)

    def init_curvelanes_config(self):
        self.img_w = 1600
        self.img_h = 800
        self.griding_num = 200
        self.crop_ratio = 0.8
        self.row_anchor = np.linspace(0.4, 1, 72)
        self.col_anchor = np.linspace(0, 1, 81)

    def init_culane_config(self):
        self.img_w = 800
        self.img_h = 320
        self.griding_num = 200
        self.crop_ratio = 0.6
        self.row_anchor = np.linspace(0.42, 1, 72)
        self.col_anchor = np.linspace(0, 1, 81)


class UltrafastLaneDetectorV2(LaneDetectBase):
    """
    Example class that loads a UFLDv2 HEF model and performs lane detection.
    Integrates coordinate scaling, polynomial fitting, and an optional horizontal narrowing.
    """
    _defaults = {
        "model_path": "models/ufld_v2.hef",
        "model_type": LaneModelType.UFLDV2_TUSIMPLE,
    }

    def __init__(self, model_path: str = None, model_type: LaneModelType = None, logger=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Base class init, which presumably sets up self.lane_info, etc.
        LaneDetectBase.__init__(self, logger)

        self.model_path = model_path if model_path else self._defaults["model_path"]
        self.model_type = model_type if model_type else self._defaults["model_type"]

        # Only TuSimple or CULane supported in this example
        if self.model_type not in [LaneModelType.UFLDV2_TUSIMPLE, LaneModelType.UFLDV2_CULANE]:
            msg = f"UltrafastLaneDetectorV2 can't use {self.model_type.name} type."
            if self.logger:
                self.logger.error(msg)
            raise Exception(msg)

        # Build config
        self.cfg = ModelConfig(self.model_type)

        # Load model
        self._initialize_model(self.model_path)

    def set_input_details(self):
        """
        Tells our code the required input dimensions for this model.
        """
        self.input_height = self.cfg.img_h
        self.input_width = self.cfg.img_w
        self.input_channels = 3
        self.input_types = np.uint8

    def set_output_details(self, engine):
        """
        If needed, specify output names or shapes for your inference engine.
        """
        self.output_names = ['output']
        self.output_shapes = [[1, 1, 1, 39576]]

    def _initialize_model(self, model_path: str) -> None:
        """
        Loads the HEF model via DeGirum's API.
        """
        if not model_path.endswith('.hef'):
            raise Exception("Only HEF models are supported in this configuration.")

        if self.logger:
            self.logger.info(f"Loading HEF model from {model_path}")

        try:
            self.engine = dg.load_model(
                model_name="ufld_v2",
                inference_host_address="localhost",
                zoo_url=""
            )
            self.set_input_details()

            if self.logger:
                self.logger.info(f"HEF model loaded successfully: {self.engine}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"HEF model loading error: {e}")
            raise

    def __prepare_input(self, image: cv2.Mat) -> np.ndarray:
        """
        1) Convert BGR to RGB
        2) Resize to (self.input_width, new_height)
        3) Bottom-crop to get (self.input_width, self.input_height)
        4) Expand dims => (1, H, W, 3)
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        new_height = int(self.input_height / self.cfg.crop_ratio)
        img_resized = cv2.resize(img, (self.input_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Keep the bottom self.input_height rows
        img_cropped = img_resized[-self.input_height:, :, :]
        input_tensor = np.expand_dims(img_cropped, axis=0)
        return input_tensor.astype(self.input_types)

    def __process_output(self, output, cfg: ModelConfig, original_image: cv2.Mat) -> Tuple[np.ndarray, list]:
        """
        - Dequantize
        - Slice the raw output into row-loc, row-exist
        - Convert to lane coordinates in the *original* frame
        - Apply polynomial smoothing
        - Return lane points + which lanes exist
        """
        # 1) Dequantize
        output_entry = output.results[0]
        raw_output = np.array(output_entry["data"], dtype=np.uint8)
        scale = output_entry["quantization"]["scale"][0]
        zero = output_entry["quantization"]["zero"][0]
        dequant_output = (raw_output.astype(np.float32) - zero) * scale

        flat_output = dequant_output.flatten()

        # 2) Slicing logic
        num_cell_row = 100
        num_cell_col = 100
        num_row = 56
        num_col = 41
        num_lanes = 4

        dim1 = num_cell_row * num_row * num_lanes
        dim2 = num_cell_col * num_col * num_lanes
        dim3 = 2 * num_row * num_lanes
        dim4 = 2 * num_col * num_lanes
        total_size = dim1 + dim2 + dim3 + dim4

        if flat_output.size != total_size:
            raise ValueError(f"Expected {total_size} elements, got {flat_output.size}")

        slice1 = flat_output[:dim1]
        # slice2 = flat_output[dim1 : dim1 + dim2]  # not used in this snippet
        slice3 = flat_output[dim1 + dim2 : dim1 + dim2 + dim3]
        # slice4 = flat_output[dim1 + dim2 + dim3 : ]  # not used

        loc_row = np.reshape(slice1, (1, num_cell_row, num_row, num_lanes))
        exist_row = np.reshape(slice3, (1, 2, num_row, num_lanes))

        # 3) Convert to lane coordinates
        exist_threshold = 0.7  # Increased threshold for better accuracy
        local_width = 2

        H, W, _ = original_image.shape
        coords = []

        # Use a horizontal narrowing factor < 1.0 to pull lines inward
        horizontal_narrow_factor = 0.8
        cx = W / 2.0

        for lane_idx in range(num_lanes):
            lane_points = []
            for k in range(num_row):
                row_logits = exist_row[0, :, k, lane_idx]  # shape (2,)
                row_probs = _softmax(row_logits, axis=0)
                if row_probs[1] > exist_threshold:
                    loc_row_logits = loc_row[0, :, k, lane_idx]
                    loc_row_argmax = np.argmax(loc_row_logits)
                    all_ind = np.arange(
                        max(0, loc_row_argmax - local_width),
                        min(num_cell_row - 1, loc_row_argmax + local_width) + 1
                    )
                    row_softmax_vals = _softmax(loc_row_logits[all_ind], axis=0)
                    weighted_index = np.sum(row_softmax_vals * all_ind)

                    # Coordinates in the *input* 800x320 space
                    x_coord_input = weighted_index / (num_cell_row - 1) * cfg.img_w
                    y_coord_input = cfg.row_anchor[k] * cfg.img_h

                    # Scale up to the original frame dimensions
                    scale_x = W / cfg.img_w
                    scale_y = H / cfg.img_h
                    x_coord = x_coord_input * scale_x
                    y_coord = y_coord_input * scale_y

                    # Move x_coord closer to the center by factor < 1
                    x_coord = cx + horizontal_narrow_factor * (x_coord - cx)

                    # Round to int for final drawing
                    x_coord = int(x_coord)
                    y_coord = int(y_coord)

                    lane_points.append((x_coord, y_coord))
            coords.append(lane_points)

        # 4) Polynomial fit & filtering
        MIN_POINTS = 5
        POLY_DEGREE = 2
        final_lanes = []
        lanes_detected = []

        for lane_pts in coords:
            if len(lane_pts) < MIN_POINTS:
                final_lanes.append([])
                lanes_detected.append(False)
                continue
            smoothed = fit_lane_polynomial(lane_pts, degree=POLY_DEGREE, num_samples=50)
            final_lanes.append(smoothed)
            lanes_detected.append(True)

        return np.array(final_lanes, dtype="object"), lanes_detected

    def DetectFrame(self, image: cv2.Mat, adjust_lanes: bool = True) -> None:
        """
        Main entry:
        - Preprocess
        - Inference
        - Postprocess => self.lane_info.lanes_points, self.lane_info.lanes_status
        """
        input_tensor = self.__prepare_input(image)
        output = self.engine.predict(input_tensor)

        self.lane_info.lanes_points, self.lane_info.lanes_status = self.__process_output(
            output, self.cfg, image
        )
        self.adjust_lanes = adjust_lanes

    def DrawDetectedOnFrame(self, image: cv2.Mat, type: OffsetType = OffsetType.UNKNOWN, alpha: float = 0.3) -> None:
        """
        Draw the  lane points as green circles on top of the image, with alpha blending.
        Then, if lanes #1 and #2 are detected, fill the region between them in green.
        """
        overlay = image.copy()
        
        # 1) Draw each lane's points in green circles
        for lane_pts in self.lane_info.lanes_points:
            for (x, y) in lane_pts:
                cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)

        # 2) Fill area between lane 1 and lane 2 if both exist
        #    Adjust indices if your main-lane boundaries differ
        if len(self.lane_info.lanes_points) > 2:
            if self.lane_info.lanes_status[1] and self.lane_info.lanes_status[2]:
                lane_left = self.lane_info.lanes_points[1]
                lane_right = self.lane_info.lanes_points[2]

                # Sort by ascending y so the polygon is built properly
                lane_left_sorted = sorted(lane_left, key=lambda p: p[1])
                lane_right_sorted = sorted(lane_right, key=lambda p: p[1], reverse=True)

                # Combine points: up the left boundary, then back down the right boundary
                polygon_pts = np.array(lane_left_sorted + lane_right_sorted, dtype=np.int32)

                fill_overlay = image.copy()
                # Fill in solid green
                cv2.fillPoly(fill_overlay, [polygon_pts], color=(0, 255, 0))
                # Blend with the original image
                cv2.addWeighted(fill_overlay, 0.4, image, 0.6, 0, dst=image)


    def DrawAreaOnFrame(self, image: cv2.Mat, color: tuple = (255, 191, 0), alpha: float = 0.85) -> None:
        """
        If your pipeline computes a polygon or area between lanes (self.lane_info.area_points),
        you can fill it here as well.
        """
        H, W, _ = image.shape
        if getattr(self.lane_info, 'area_status', False):
            lane_segment_img = image.copy()
            cv2.fillPoly(lane_segment_img, pts=[self.lane_info.area_points], color=color)
            cv2.addWeighted(lane_segment_img, 1 - alpha, image, alpha, 0, dst=image)
