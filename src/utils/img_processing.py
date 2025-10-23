from PIL import Image
import os


def list_img(img_folder_path, open_img=False):
    img_file_name_list = os.listdir(img_folder_path)

    if open_img:
        return {file_name : Image.open(os.path.join(img_folder_path, file_name)) for file_name in img_file_name_list} #return a dict of file name and opened image
    
    return img_file_name_list


def crop(img, crop_pixel):
    img1 = img.crop(crop_pixel) 
    
    # Crop the width based on the background color
    pix = img1.load()
    width1, height1 = img1.size

    # Get the pixel color at the specified coordinates (x, y)
    x, y = 50, height1 // 2
    pixel_color = img1.getpixel((x, y))
    target_color = [pixel_color, (247, 247, 247, 250)]

    # Initialize cropping dimensions
    right = 0

    # Crop the left side and right side
    for i in range(width1):
        for j in range(height1 // 5, height1 // 1,5):
            pixel_color = pix[i, j]
            if pixel_color not in target_color:
                right = i
                break
    # cropped_width_img = img1.crop((width1 - right, 0, right, height1))

    #Crop the top and the bottom

    background_color = pix[3, 3]

    #crop the top 
    top = 0
    for i in range(height1):
        if any(pix[j, i] != background_color for j in range(width1)):
            top = i
            break

    bottom = height1
    for i in range(height1 - 1, -1, -1):
        if any(pix[j, i] != background_color for j in range(width1)):
            bottom = i + 1
            break
        
    cropped_img = img1.crop((width1 - right, top, right, bottom))

    return cropped_img


def recrop(cropped_img, img):
    cropped_width, _ = cropped_img.size
    _, height = img.size
    if cropped_width > 1650 or cropped_width < 600:
        recropped_img = img.crop((585, 360, 1335, height-68))
    else:
        recropped_img = cropped_img
    
    return recropped_img


def resize(recropped_img):
    width, height = recropped_img.size
    new_height = int(800 / width * height)   
    # Resize the image
    img_resized = recropped_img.resize((800, new_height), Image.Resampling.LANCZOS)
    return img_resized


def save_img(img_resized, output_folder, img_name):
    # Save the transformed image
    output_path = os.path.join(output_folder, f'{img_name}')
    img_resized.save(output_path)