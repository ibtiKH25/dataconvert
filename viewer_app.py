import streamlit as st
import os
import pandas as pd
from PIL import Image

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to list available directories
def list_subdirectories(directory):
    create_directory(directory)  # Ensure the directory exists before listing
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

# Function to load and display a selected CSV file
def display_csv_file(file_path):
    df = pd.read_csv(file_path, sep=';')
    st.write("CSV File Data:")
    st.dataframe(df)

# Function to load and display a selected image file
def display_image_file(file_path):
    image = Image.open(file_path)
    st.image(image, caption=os.path.basename(file_path), use_column_width=True)

# Main function to run the Streamlit app
def main():
    st.title('View Saved Files')

    # Ensure the 'Historique' directory exists
    create_directory('Historique')
    st.write("Created/Checked 'Historique' directory")

    base_directory = 'Historique'
    subdirectories = list_subdirectories(base_directory)
    st.write(f"Subdirectories: {subdirectories}")

    if subdirectories:
        selected_subdirectory = st.selectbox('Select a directory', subdirectories)

        if selected_subdirectory:
            csv_file_path = os.path.join(base_directory, selected_subdirectory, 'extracted_data.csv')
            image_file_path = os.path.join(base_directory, selected_subdirectory, 'annotated_image.png')

            if os.path.exists(csv_file_path):
                display_csv_file(csv_file_path)
            else:
                st.write("CSV file not found.")

            if os.path.exists(image_file_path):
                display_image_file(image_file_path)
            else:
                st.write("Image file not found.")
    else:
        st.write("No directories found.")

if __name__ == '__main__':
    main()
