import cv2
import numpy as np
import degirum as dg


def fit_lane_polynomial(lane_points, degree=2, num_samples=50):
    """
    Fits a polynomial (degree=2 by default) to lane_points and returns a smoothed set of (x, y).
    """
    if len(lane_points) < degree + 1:
        # Not enough points to fit the chosen polynomial
        return lane_points

    xs = np.array([pt[0] for pt in lane_points], dtype=np.float32)
    ys = np.array([pt[1] for pt in lane_points], dtype=np.float32)

    # Sort points by y so we can do x = f(y)
    sort_idx = np.argsort(ys)
    xs = xs[sort_idx]
    ys = ys[sort_idx]

    # Fit polynomial: x = f(y)
    coeffs = np.polyfit(ys, xs, deg=degree)
    poly_func = np.poly1d(coeffs)

    # Resample from min(y) to max(y)
    y_min, y_max = ys[0], ys[-1]
    y_new = np.linspace(y_min, y_max, num_samples)
    x_new = poly_func(y_new)

    return [(int(xn), int(yn)) for xn, yn in zip(x_new, y_new)]


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class UFLDProcessing:
    def __init__(
        self,
        num_cell_row,
        num_cell_col,
        num_row,
        num_col,
        num_lanes,
        crop_ratio,
        original_frame_width,
        original_frame_height,
        input_height,
        input_width
    ):

        self.num_cell_row = num_cell_row
        self.num_cell_col = num_cell_col
        self.num_row = num_row
        self.num_col = num_col
        self.num_lanes = num_lanes
        self.crop_ratio = crop_ratio
        self.original_frame_width = original_frame_width
        self.original_frame_height = original_frame_height
        self.input_height = input_height
        self.input_width = input_width

    def resize(self, image):

        new_height = int(self.input_height / self.crop_ratio)
        image_resized = cv2.resize(image, (self.input_width, new_height), interpolation=cv2.INTER_CUBIC)
        # Crop the bottom portion so the effective height is exactly input_height
        return image_resized[-self.input_height:, :, :]

    def _slice_and_reshape(self, output):

        dim1 = self.num_cell_row * self.num_row * self.num_lanes  # e.g. 22400
        dim2 = self.num_cell_col * self.num_col * self.num_lanes  # e.g. 16400
        dim3 = 2 * self.num_row * self.num_lanes                  # e.g. 448
        dim4 = 2 * self.num_col * self.num_lanes                  # e.g. 328
        total_size = dim1 + dim2 + dim3 + dim4

        if output.size != total_size:
            raise ValueError(f"Expected {total_size}, got {output.size}")

        slice1 = output[:, :dim1]
        slice2 = output[:, dim1 : dim1 + dim2]
        slice3 = output[:, dim1 + dim2 : dim1 + dim2 + dim3]
        slice4 = output[:, dim1 + dim2 + dim3 : ]

        loc_row = np.reshape(slice1, (-1, self.num_cell_row, self.num_row, self.num_lanes))
        loc_col = np.reshape(slice2, (-1, self.num_cell_col, self.num_col, self.num_lanes))
        exist_row = np.reshape(slice3, (-1, 2, self.num_row, self.num_lanes))
        exist_col = np.reshape(slice4, (-1, 2, self.num_col, self.num_lanes))

        return loc_row, loc_col, exist_row, exist_col

    def _pred2coords(self, loc_row, exist_row, local_width=2, exist_threshold=0.5):

        row_anchor = np.linspace(170, 710, self.num_row) / 720  # skip top ~300px
        coords = []

        for lane_idx in range(self.num_lanes):
            lane_points = []
            for k in range(self.num_row):
                # Existence
                row_logits = exist_row[0, :, k, lane_idx]  # shape (2,)
                row_probs = _softmax(row_logits, axis=0)
                if row_probs[1] > exist_threshold:
                    # Localization
                    loc_row_logits = loc_row[0, :, k, lane_idx]  # shape (num_cell_row,)
                    loc_row_argmax = np.argmax(loc_row_logits)
                    all_ind = np.arange(
                        max(0, loc_row_argmax - local_width),
                        min(self.num_cell_row - 1, loc_row_argmax + local_width) + 1
                    )
                    row_softmax_vals = _softmax(loc_row_logits[all_ind], axis=0)
                    weighted_index = np.sum(row_softmax_vals * all_ind)

                    # Scale x => original frame
                    x = int(weighted_index / (self.num_cell_row - 1) * self.original_frame_width)
                    # Scale y => original frame
                    y = int(row_anchor[k] * self.original_frame_height)
                    lane_points.append((x, y))
            coords.append(lane_points)

        return coords

    def get_coordinates(self, endnodes, exist_threshold=0.5):
        loc_row, loc_col, exist_row, exist_col = self._slice_and_reshape(endnodes)
        # We only pass row-loc & row-exist to _pred2coords
        return self._pred2coords(loc_row, exist_row, exist_threshold=exist_threshold)


if __name__ == "__main__":
    VIDEO_PATH = "/home/chinmay/Downloads/input_video.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Typical UFLD input
    input_height = 320
    input_width = 800

    # Create UFLDProcessing
    ufld_processing = UFLDProcessing(
        num_cell_row=100,
        num_cell_col=100,
        num_row=56,
        num_col=41,
        num_lanes=4,
        crop_ratio=0.8,
        original_frame_width=original_width,
        original_frame_height=original_height,
        input_height=input_height,
        input_width=input_width
    )

    # Load the model
    engine = dg.load_model(
        model_name="ufld_v2",
        inference_host_address="@local",
        zoo_url="/home/chinmay/hailo_models"
    )
    print("[Main] Model loaded.")

    # Hyperparams
    EXIST_THRESHOLD = 0.5
    MIN_POINTS = 5
    POLY_DEGREE = 2  # Quadratic polynomial

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Resize + bottom-crop
        resized = ufld_processing.resize(frame)
        input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)

        # 2) Inference
        output = engine.predict(input_tensor)
        output_entry = output.results[0]

        # 3) Dequantize
        raw_output = np.array(output_entry["data"], dtype=np.uint8)
        scale = output_entry["quantization"]["scale"][0]
        zero = output_entry["quantization"]["zero"][0]
        dequant_output = (raw_output.astype(np.float32) - zero) * scale
        dequant_output = np.squeeze(dequant_output)
        if dequant_output.ndim == 1:
            dequant_output = np.expand_dims(dequant_output, axis=0)

        # 4) Postprocess -> row-lane coords
        lanes = ufld_processing.get_coordinates(dequant_output, exist_threshold=EXIST_THRESHOLD)

        # 5) Filter & polynomial fit
        final_lanes = []
        for lane_pts in lanes:
            if len(lane_pts) < MIN_POINTS:
                final_lanes.append([])
                continue
            smoothed = fit_lane_polynomial(lane_pts, degree=POLY_DEGREE, num_samples=50)
            final_lanes.append(smoothed)

        # 6) Draw
        for lane_pts in final_lanes:
            for (x, y) in lane_pts:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("Lane Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
