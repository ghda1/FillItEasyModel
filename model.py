# Import necessary libraries
import sys
import os
import cv2
import io
import re
import fitz
import numpy as np
import pandas as pd
from PIL import Image
from paddleocr import PPStructure, save_structure_res
from img2table.document import Image as Img2TableImage
import easyocr
from easyocr import Reader


def extract_tables_from_image(image_path):
    img = Img2TableImage(src=image_path)
    tables = img.extract_tables()
    bbox_values = [(t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2) for t in tables]

    # Crop and save the table areas
    img = Image.open(image_path)
    cropped_image_dir = os.path.join(os.path.dirname(image_path), "cropped_image")
    os.makedirs(cropped_image_dir, exist_ok=True)
    for i, (x, y, w, h) in enumerate(bbox_values, start=1):
        cropped_img = img.crop((x, y, w, h))
        cropped_img.save(os.path.join(cropped_image_dir, f"cropped_image_{i}.png"))

    return None

def extract_images_from_pdf(pdf_path):
    pdf_file = fitz.open(pdf_path)
    extracted_images_dir = os.path.join(os.path.dirname(pdf_path), "extracted_images")
    os.makedirs(extracted_images_dir, exist_ok=True)
    min_width = 100
    min_height = 100
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            if image.width >= min_width and image.height >= min_height:
                output_path = os.path.join(extracted_images_dir, f"image{page_index + 1}_{image_index}.png")
                image.save(output_path, format="PNG")
    return extracted_images_dir

def extract_ocr_data(img_path):
    table_engine = PPStructure(layout=False, show_log=True, ocr_version='PP-OCRv3')
    cropped_image_path = os.path.join(os.path.dirname(img_path), "cropped_image", "cropped_image_1.png")
    img = cv2.imread(cropped_image_path)
    result2 = table_engine(img)
    save_structure_res(result2, os.path.join(os.path.dirname(img_path), "table_excel"), "cropped_image")
    df = pd.DataFrame({'CLO': [], 'Awarded marks':[],'Student No.': '', 'Course': '' })
    excels_dir = os.path.join(os.path.dirname(img_path), "table_excel", "cropped_image")
    for fn in os.listdir(excels_dir):
        if fn.endswith('.xlsx'):
            excel_path = os.path.join(excels_dir, fn)
            df = pd.read_excel(excel_path)
            df['CLO'] = (df['CLO'])
            df['Awarded marks'] = (df['Awarded marks'])
            os.remove(excel_path)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(img_path)
    id_pattern = r"\b\d{5,}\b"
    course_pattern = r"\b\d{3}\b"
    for (bbox, text, prob) in result:
        match1 = re.findall(id_pattern, text)
        match2 = re.findall(course_pattern, text)
        if match1:
            df['Student No.'] = (''.join(match1))
        if match2:
            df['Course'] = (''.join(match2))


    reshape = df.pivot(index=['Course', 'Student No.'], columns='CLO', values='Awarded marks')
    reshape = reshape.reset_index()
    reshape.columns = reshape.columns.tolist()
    return reshape

def save_excel(reshaped_data,output_table_path):
    excel_buffer = io.BytesIO()
    reshaped_data.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    # Save to a physical file
    with open(output_table_path, 'wb') as f:
        f.write(excel_buffer.read())

    return f"Output Excel file saved at: {output_table_path}"

def excel_file_from_pdf(image_path):
    for fn in os.listdir(image_path):
        if fn.endswith('.png'):
            img_path = os.path.join(image_path, fn)
            extract_tables_from_image(img_path)
            reshaped_data = extract_ocr_data(img_path)

            # Save the reshaped data to an Excel file
            output_table_path = os.path.join(os.path.dirname(image_path), "output_table.xlsx")
            save_excel(reshaped_data,output_table_path)
    return output_table_path

        
def excel_file_from_image(img_path):
    extract_tables_from_image(img_path)
    reshaped_data = extract_ocr_data(img_path)

    # Save the reshaped data to an Excel file
    output_table_path = os.path.join(os.path.dirname(img_path), "output_table.xlsx")
    save_excel(reshaped_data,output_table_path)
    return output_table_path

def predict(path):
    # for pdf
    if path.endswith('.pdf'):
        # Extract images from PDF
        extracted_image = extract_images_from_pdf(path)
        # Get the excel file
        return excel_file_from_pdf(extracted_image)
    # for png/jpg
    else:
        return excel_file_from_image(path)
        


if len(sys.argv) < 2:
    print("No file path provided")
    sys.exit(1)

file_path = sys.argv[1]

# Predict function to return the Excel file path
output_excel_path = predict(file_path)

# Print the result for the calling PHP script to process
print(f"Output Excel file saved at: {output_excel_path}")
