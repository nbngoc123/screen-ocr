import cv2
import numpy as np
import pyclipper

def expand_polygon(pts: np.ndarray, distance: float) -> np.ndarray | None:
    """Nới rộng đa giác bằng pyclipper theo thuật toán Vatti."""
    try:
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(
            pts.tolist(),
            pyclipper.JT_ROUND,
            pyclipper.ET_CLOSEDPOLYGON
        )
        expanded = offset.Execute(distance)
        
        if not expanded:
            return None
            
        largest_poly = max(
            expanded,
            key=lambda p: cv2.contourArea(np.array(p, dtype=np.int32))
        )
        
        return np.array(largest_poly, dtype=np.int32).reshape(-1, 1, 2)
    except Exception:
        return None

def calculate_polygon_score(prob_map: np.ndarray, contour: np.ndarray) -> float:
    """Tính điểm confidence trung bình bằng cách tạo mask từ contour gốc."""
    mask = np.zeros(prob_map.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 1)
    
    scores = prob_map[mask == 1]
    if scores.size == 0:
        return 0.0
        
    return float(scores.mean())
