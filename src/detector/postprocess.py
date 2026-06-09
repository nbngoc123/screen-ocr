import cv2
import numpy as np
from .types import BoundingBox
from .polygon_utils import expand_polygon, calculate_polygon_score

def postprocess_dbnet(
    prob_map: np.ndarray,
    orig_shape: tuple[int, int],
    thresh: float = 0.3,
    unclip_ratio: float = 2.0,
    min_area: int = 10,
    min_padding_x: int = 4,
    min_padding_y: int = 4,
    box_score_threshold: float = 0.5,
) -> list[BoundingBox]:
    orig_h, orig_w = orig_shape
    pred_h, pred_w = prob_map.shape

    bitmap = (prob_map > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    scale_x = orig_w / pred_w
    scale_y = orig_h / pred_h
    boxes: list[BoundingBox] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        pts = contour.squeeze(1)
        if pts.ndim != 2 or len(pts) < 3:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        distance = area * unclip_ratio / perimeter
        expanded_contour = expand_polygon(pts, distance)
        if expanded_contour is None:
            continue

        score = calculate_polygon_score(prob_map, contour)
        if score < box_score_threshold:
            continue

        x, y, w, h = cv2.boundingRect(expanded_contour)

        x1 = max(0, int(x * scale_x) - min_padding_x)
        y1 = max(0, int(y * scale_y) - min_padding_y)
        x2 = min(orig_w, int((x + w) * scale_x) + min_padding_x)
        y2 = min(orig_h, int((y + h) * scale_y) + min_padding_y)

        boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=score))

    return boxes
