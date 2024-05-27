import streamlit as st
import os
import pandas as pd
from PIL import Image

def list_subdirectories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def display_csv_file(file_path):
    df = pd.read_csv(file_path, sep=';')
    st.write("CSV File Data:")
    st.dataframe(df)

def display_image_file(file_path):
    image = Image.open(file_path)
    st.image(image, caption=os.path.basename(file_path), use_column_width=True)

def main():
    st.title('View Saved Files')

    base_directory = 'Historique'
    subdirectories = list_subdirectories(base_directory)

    selected_subdirectory = st.selectbox('Select a directory', subdirectories)

    if selected_subdirectory:
        csv_file_path = os.path.join(base_directory, selected_subdirectory, 'extracted_data.csv')
        image_file_path = os.path.join(base_directory, selected_subdirectory, 'annotated_image.png')

        if os.path.exists(csv_file_path):
            display
