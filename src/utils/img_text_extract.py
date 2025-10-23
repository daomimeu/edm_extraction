from PIL import Image
import pytesseract
import os


def text_extract(img, lang, link_to_tesseract, open_img=False):
    pytesseract.pytesseract.tesseract_cmd = link_to_tesseract
    if open_img:
        img = Image.open(img)
    text = pytesseract.image_to_string(img, lang=lang)
    
    return text


def create_text_file(text, text_folder_path, image_name):
    text_file_path = os.path.join(text_folder_path, f"{image_name}.txt")
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

