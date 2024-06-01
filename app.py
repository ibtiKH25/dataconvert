import streamlit as st
from PIL import Image
import numpy as np
import cv2
import fitz  # PyMuPDF
from ultralytics import YOLO
import pytesseract
import pandas as pd
import os
import requests
import glob
import io

# Set the page configuration
st.set_page_config(page_title="IBSA Data Converter", layout="wide")

# URL du fichier modèle sur GitHub
model_url = 'https://github.com/ibtiKH25/dataconvert/raw/main/TrainingModel.pt'

# Chemin local où le fichier modèle sera sauvegardé
model_local_path = 'TrainingModel.pt'

# Téléchargement du modèle depuis GitHub
@st.cache_data
def download_model(url, local_path):
    if not os.path.exists(local_path):
        response = requests.get(url)
        with open(local_path, 'wb') as file:
            file.write(response.content)
    return local_path

# Télécharger le modèle
download_model(model_url, model_local_path)

# Configure the path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# Load the YOLO model
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_local_path)

# Function to clean text by removing unwanted characters
def clean_text(text):
    unwanted_chars = [
        '°', '<', '¢', '/', '\\', '|', '>', '--', '__', '<<', '>>', '@', '@@', '^', '^^', ',', '}', '{', '&', '&&', '//',
        ' Supplier P/N', 'Customer P/N', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï',
        'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Ø', 'Œ', 'Š', 'þ', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Ÿ', 'à', 'á', 'â', 'ã', 'ä', 
        'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', 'ø', 'œ', 'š', 'Þ', 
        'ù', 'ú', 'û', 'ü', 'ý', 'ÿ', '¢', 'ß', '¥', '£', '™', '©', 'ª', '×', '÷', '²', '³', '¼', '½', '¾', 'µ', '¿', 
        '¶', '·', '¸', 'º', '°', '¯', '§', '…', '¤', '¦', '≠', '¬', 'ˆ', '¨', '‰', "'", "'Supplier P/N", "'Customer P/N"
    ]
    for char in unwanted_chars:
        text = text.replace(char, '')
    return text

# Function to detect objects in the image using the YOLO model
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

# Function to extract text from a specific region in the image
def extract_text_from_region(image, box):
    try:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_image = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cropped_image)
        text = clean_text(text)  # Clean the extracted text
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from region: {e}")
        return ""

# Function to determine the cable type based on the text in the detected table
def determine_cable_type_from_table(image, box):
    try:
        x1, y1, x2, y2 = map(int, box[:4])
        table_region = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(table_region)
        text = clean_text(text)  # Clean the extracted text
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

# Add custom CSS for styling
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f0f4f7;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stTextInput>div>div>input {
        border: 2px solid #007BFF;
        padding: 5px;
        border-radius: 10px;
    }
    .stTextInput>div>label {
        font-weight: bold;
        color: #007BFF;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #007BFF;
    }
    .stDataFrame>div {
        border: 2px solid #007BFF;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    add_custom_css()
    
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox("Select a page", ["Convert Data", "Historique"])

    if page == "Convert Data":
        st.title('LEONI Data Converter \n Convert Technical Drawings with Accuracy and Ease')
        uploaded_files = st.file_uploader("Choose images to analyze...", type=["jpg", "png", "jpeg", "pdf"], accept_multiple_files=True)
        
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    # Convert PDF to image if uploaded file is PDF
                    doc = fitz.open(stream=uploaded_file.read())  # Open the PDF from stream
                    page = doc.load_page(0)  # Load the first page
                    zoom = 2.0  # Use a zoom factor to increase the resolution
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)  # Render page to an image with high resolution
                    img = Image.open(io.BytesIO(pix.tobytes('png')))  # Convert image to PIL format
                else:
                    # Directly load image if it is not a PDF
                    img = Image.open(uploaded_file)

                
                image_np = np.array(img.convert('RGB'))
                image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                results_list = detect_objects(image_cv2, model)

                # Mapping of old class names to new class names
                class_name_mapping = {
                    "0- Side1": "Side1",
                    "1- Side2": "Side2",
                    "2- LEONIPartNumber": "LEONIPartNumber",
                    "3- SupplierPartNumber": "SupplierPartNumber",
                    "4- Wiretype": "Wiretype",
                    "5- Length": "Length",
                    "6- TypeOfCableAssembly": "TypeOfCableAssembly"
                }

                # Dictionary to store the extracted data
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
                    st.image(annotated_image, caption=f'Annotated Image - {uploaded_file.name}', use_column_width=True)

                    # Create a DataFrame for the CSV export
                    df = pd.DataFrame.from_dict(class_data, orient='index').transpose()
                    column_order = ['Side1', 'Side2', 'LEONIPartNumber', 'SupplierPartNumber', 'Wiretype', 'Length', 'TypeOfCableAssembly', 'Pigtail', 'HV']
                    df = df[column_order]  # Reorder the columns

                    # Display data in a table
                    st.write(f"Extracted Data - {uploaded_file.name}:")
                    st.dataframe(df)

                    # Save the data and image
                    if st.button(f"Save Data and Technical Drawing - {uploaded_file.name}"):
                        output_dir = "saved_data"
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        # Afficher le chemin complet du répertoire de stockage
                        st.write(f"Répertoire de stockage : {os.path.abspath(output_dir)}")

                        # Save CSV
                        base_filename = os.path.splitext(uploaded_file.name)[0]
                        csv_path = os.path.join(output_dir, f"{base_filename}.csv")
                        df.to_csv(csv_path, index=False, sep=';', encoding='utf-8-sig')

                        # Save Image with the same extension as uploaded
                        image_extension = os.path.splitext(uploaded_file.name)[1]
                        image_path = os.path.join(output_dir, f"{base_filename}{image_extension}")
                        annotated_image.save(image_path)

                        st.success(f"Data and Technical Drawing saved successfully: {csv_path} and {image_path}")

                        # Provide a download button for the CSV file
                        csv = df.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
                        st.download_button(label=f"Download data as CSV - {uploaded_file.name}",
                                        data=csv,
                                        file_name=f'{base_filename}_extracted_data.csv',
                                        mime='text/csv')
                else:
                    st.write(f"No detections or incorrect result format for {uploaded_file.name}.")

    elif page == "Historique":
        st.title('Historique')

        # Add a search box
        search_query = st.text_input("Search by image name:")

        # Get all saved CSV files
        saved_files = glob.glob('saved_data/*.csv')
        if saved_files:
            for csv_file in saved_files:
                file_name = os.path.basename(csv_file)
                base_filename = os.path.splitext(file_name)[0]

                # Display only if the search query matches the base filename
                if search_query.lower() in base_filename.lower():
                    st.subheader(file_name)
                    df = pd.read_csv(csv_file, sep=';', encoding='utf-8-sig')
                    st.dataframe(df)

                    # Display corresponding image
                    image_files = glob.glob(f"saved_data/{base_filename}.*")
                    image_file = None
                    for ext in ['jpg', 'jpeg', 'png']:
                        potential_image_file = f"saved_data/{base_filename}.{ext}"
                        if potential_image_file in image_files:
                            image_file = potential_image_file
                            break

                    if image_file and os.path.exists(image_file):
                        try:
                            image = Image.open(image_file)
                            st.image(image, caption='Corresponding Image', use_column_width=True)
                        except Exception as e:
                            st.warning(f"Could not open image file: {image_file}. Error: {e}")
                    
                    # Add buttons for delete and modify actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Delete {file_name}"):
                            os.remove(csv_file)
                            if image_file:
                                os.remove(image_file)
                            st.success(f"Deleted {file_name} and its image.")
                            st.experimental_rerun()
                    with col2:
                        if st.button(f"Modify {file_name}"):
                            st.session_state['modify_file'] = csv_file
                            st.experimental_rerun()

                    # Add download button for CSV file
                    with open(csv_file, 'r', encoding='utf-8-sig') as f:
                        csv_data = f.read()
                    st.download_button(label=f"Download CSV - {file_name}",
                                       data=csv_data,
                                       file_name=file_name,
                                       mime='text/csv')

        else:
            st.write("No saved data found.")

        # Check if a file is selected for modification
        if 'modify_file' in st.session_state:
            modify_file = st.session_state['modify_file']
            if modify_file:
                st.write(f"Modifying: {modify_file}")
                df = pd.read_csv(modify_file, sep=';', encoding='utf-8-sig')
                new_data = st.experimental_data_editor(df)

                if st.button("Save Changes"):
                    new_data.to_csv(modify_file, index=False, sep=';', encoding='utf-8-sig')
                    st.success("Changes saved.")
                    del st.session_state['modify_file']
                    st.experimental_rerun()

if __name__ == '__main__':
    main()
