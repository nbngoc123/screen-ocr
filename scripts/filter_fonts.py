import os
from PIL import Image, ImageDraw, ImageFont

def get_char_image(font, char):
    img = Image.new('L', (40, 40), 0)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), char, font=font, fill=255)
    return img.tobytes()

def check_vietnamese_support(font_path):
    try:
        font = ImageFont.truetype(font_path, 20)
    except Exception:
        return False
        
    missing_mask = get_char_image(font, '\uffff')
    test_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
    
    for c in test_chars:
        # Nếu bất kỳ ký tự tiếng Việt nào in ra rỗng hoặc giống ký tự báo thiếu (tofu)
        char_mask = get_char_image(font, c)
        if char_mask == missing_mask:
            return False
            
    return True

if __name__ == "__main__":
    input_file = "data/font_list.txt"
    invalid_file = "data/font_list_unsupported.txt"
    
    if not os.path.exists(input_file):
        print(f"Không tìm thấy {input_file}")
        exit(1)
        
    with open(input_file, "r", encoding="utf-8") as f:
        fonts = [line.strip() for line in f if line.strip()]
        
    valid_fonts = []
    invalid_fonts = []
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    
    print(f"Đang kiểm tra {len(fonts)} fonts...")
    for font_path in fonts:
        if not os.path.exists(font_path):
            continue
            
        if check_vietnamese_support(font_path):
            valid_fonts.append(font_path)
        else:
            invalid_fonts.append(font_path)
            
    # Ghi đè file font_list.txt bằng danh sách hợp lệ
    with open(input_file, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_fonts) + "\n")
        
    # Ghi danh sách không hợp lệ ra file riêng
    with open(invalid_file, "w", encoding="utf-8") as f:
        f.write("\n".join(invalid_fonts) + "\n")
        
    print(f"Hoàn tất!")
    print(f" - Tổng số font ban đầu: {len(fonts)}")
    print(f" - Hỗ trợ Tiếng Việt (Giữ lại): {len(valid_fonts)}")
    print(f" - Lỗi Tiếng Việt (Đã bị loại): {len(invalid_fonts)}")
