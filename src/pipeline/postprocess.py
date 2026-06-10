"""
postprocess.py — Hậu xử lý cho OCR Pipeline.

Bao gồm thuật toán gom cụm (Clustering) các bounding box thành từng dòng ngang 
và sắp xếp chúng từ trái qua phải, trên xuống dưới.
"""
from __future__ import annotations

from src.detector.types import BoundingBox

def sort_boxes_to_lines(boxes: list[BoundingBox]) -> list[BoundingBox]:
    """
    Hậu xử lý OCR: Gom cụm các box thành từng dòng và sắp xếp trái->phải.
    
    Thuật toán:
    1. Sắp xếp tất cả box theo tọa độ Y trung tâm (y_center) từ trên xuống.
    2. Gom các box vào cùng 1 dòng nếu y_center của nó lệch so với trung bình
       của dòng hiện tại một khoảng nhỏ hơn 50% chiều cao của box.
    3. Trả về kết quả sau khi sắp xếp lại các box trong cùng một dòng theo tọa độ X.
    """
    if not boxes:
        return []
        
    # Tính y_center cho mỗi box
    boxes_with_y_center = [(b, (b.y1 + b.y2) / 2.0, b.y2 - b.y1) for b in boxes]
    
    # Bước 1: Sắp xếp theo y_center từ trên xuống dưới
    boxes_with_y_center.sort(key=lambda x: x[1])
    
    lines = []
    current_line = [boxes_with_y_center[0]]
    current_line_y_sum = boxes_with_y_center[0][1]
    
    for item in boxes_with_y_center[1:]:
        b, y_center, h = item
        # Y_center trung bình của dòng hiện tại
        avg_line_y = current_line_y_sum / len(current_line)
        
        # Nếu lệch Y quá ít (dưới 50% chiều cao của box), coi như cùng 1 dòng
        if abs(y_center - avg_line_y) < h * 0.5:
            current_line.append(item)
            current_line_y_sum += y_center
        else:
            lines.append(current_line)
            current_line = [item]
            current_line_y_sum = y_center
            
    if current_line:
        lines.append(current_line)
        
    # Bước 2 & 3: Sort mỗi dòng theo x1 (từ trái qua phải) và gom lại
    sorted_boxes = []
    for line in lines:
        line.sort(key=lambda x: x[0].x1)
        sorted_boxes.extend([x[0] for x in line])
        
    return sorted_boxes
