import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import pytesseract
import pandas as pd
import os
import requests
import uuid

# URL du fichier modèle sur GitHub
model_url = 'https://github.com/ibtiKH25/dataconvert/raw/main/TrainingModel.pt'
model_local_path = 'TrainingModel.pt'

@st.cache_data
def download_model(url, local_path):
    if not os.path.exists(local_path):
        st.write(f"Downloading model from: {url}")
        response = requests.get(url)
        with open(local_path, 'wb') as file:
            file.write(response.content)
    return local_path

download_model(model_url, model_local_path)
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

@st.cache_data
def load_model(model_path):
    try:
        st.write(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        st.write("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_local_path)

def clean_text(text):
    unwanted_chars = ['é', '°', 'è', 'à', 'ç', '<', '¢', '/', '\\' , '|' , '>']
    for char in unwanted_chars:
        text = text.replace(char, '')
    return text

def detect_objects(image, model):
    if model is None:
        st.error("Model is not loaded. Cannot perform detection.")
        return None
    try:
        results = model.predict(image)
        return results
    except Exception as e:
        st.error(f"Error during detection: {e}")
        return None

def extract_text_from_region(image, box):
    try:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_image = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cropped_image)
        text = clean_text(text)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from region: {e}")
        return ""

def determine_cable_type_from_table(image, box):
    try:
        x1, y1, x2, y2 = map(int, box[:4])
        table_region = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(table_region)
        text = clean_text(text)
        lines = text.strip().split('\n')
        num_lines = len(lines)
        if num_lines == 4 or num_lines == 5:
            cable_type = 'Ethernet'
        elif num_lines > 5:
            cable_type = 'Hsd'
        else:
            cable_type = 'Antenna'
        return cable_type
    except Exception as e:
        st.error(f"Error determining cable type from table: {e}")
        return "Unknown"

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    st.write(f"Checked/created directory: {directory}")

def main():
    st.title('Data Converter LEONI \n Convert Technical Drawings with Accuracy and Ease')

    # Print current working directory
    current_directory = os.getcwd()
    st.write(f"Current working directory: {current_directory}")

    # Ensure the 'Historique' directory exists
    create_directory('Historique')
    st.write("Created/Checked 'Historique' directory")

    uploaded_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "png", "jpeg", "pdf"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_np = np.array(image.convert('RGB'))
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        results_list = detect_objects(image_cv2, model)

        class_name_mapping = {
            "0- Side1": "Side1",
            "1- Side2": "Side2",
            "2- LEONIPartNumber": "LEONIPartNumber",
            "3- SupplierPartNumber": "SupplierPartNumber",
            "4- Wiretype": "Wiretype",
            "5- Length": "Length",
            "6- TypeOfCableAssembly": "TypeOfCableAssembly"
        }

        class_data = {new_name: [] for new_name in class_name_mapping.values()}

        if results_list:
            for results in results_list:
                if hasattr(results, 'boxes') and results.boxes is not None:
                    for i, box in enumerate(results.boxes.xyxy):
                        if len(box) >= 4:
                            class_id = int(results.boxes.cls[i]) if len(results.boxes.cls) > i else -1
                            label = results.names[class_id] if class_id in results.names else "Unknown"
                            new_label = class_name_mapping.get(label, label)
                            if label == '6- TypeOfCableAssembly':
                                cable_type = determine_cable_type_from_table(image_cv2, box)
                                text = cable_type
                            else:
                                text = extract_text_from_region(image_cv2, box)
                            if new_label in class_data:
                                class_data[new_label].append(text)
                            else:
                                st.warning(f"Detected label '{label}' is not in the specified columns.")
                            cv2.rectangle(image_cv2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            class_data['Pigtail'] = ['Non']
            class_data['HV'] = ['Non']

            annotated_image = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
            st.image(annotated_image, caption='Annotated Image', use_column_width=True)

            unique_dir = os.path.join('Historique', str(uuid.uuid4()))
            create_directory(unique_dir)
            st.write(f"Created/Checked unique directory: {unique_dir}")

            csv_file_path = os.path.join(unique_dir, 'extracted_data.csv')
            df = pd.DataFrame.from_dict(class_data, orient='index').transpose()
            column_order = ['Side1', 'Side2', 'LEONIPartNumber', 'SupplierPartNumber', 'Wiretype', 'Length', 'TypeOfCableAssembly', 'Pigtail', 'HV']
            df = df[column_order]
            df.to_csv(csv_file_path, index=False, sep=';', encoding='utf-8-sig')
            st.write(f"Saved CSV file to: {csv_file_path}")

            image_file_path = os.path.join(unique_dir, 'annotated_image.png')
            annotated_image.save(image_file_path)
            st.write(f"Saved annotated image to: {image_file_path}")

            # List contents of Historique directory
            all_items = os.listdir('Historique')
            st.write(f"All items in 'Historique' directory after saving: {all_items}")

            st.download_button(label="Download data as CSV",
                               data=df.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig'),
                               file_name='extracted_data.csv',
                               mime='text/csv')
            with open(image_file_path, 'rb') as f:
                st.download_button(label="Download Annotated Image",
                                   data=f,
                                   file_name='annotated_image.png',
                                   mime='image/png')
        else:
            st.write("No detections or incorrect result format.")

if __name__ == '__main__':
    main()
