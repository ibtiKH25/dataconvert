# dataconverter
Data Converter is an advanced data extraction and conversion application designed to process technical drawings with precision and ease. This project leverages the powerful capabilities of YOLO (You Only Look Once) for object detection and Tesseract OCR for text recognition, ensuring accurate extraction of relevant information from images. The application is built using Streamlit, providing an intuitive and interactive web interface for users.
#Features

Object Detection with YOLO: Utilizes the YOLO v8 model to detect and identify specific regions of interest in technical drawings.
Text Recognition with Tesseract OCR: Extracts text from the identified regions, supporting various text extraction requirements.
Data Cleaning: Automatically cleans extracted text to remove unwanted characters, ensuring the extracted data is accurate and standardized.
CSV Export: Converts the extracted data into a structured CSV format, enabling easy data manipulation and integration with other systems.
Interactive Web Interface: Built with Streamlit, providing a user-friendly interface for uploading images, viewing detected objects, and downloading extracted data.
#Usage
Upload Image: Use the file uploader in the web interface to upload a technical drawing in JPG, PNG, JPEG, or PDF format.
View Annotations: The application processes the image, detects objects, and highlights the regions of interest.
Extracted Data: View the extracted text data in a table format.
Download CSV: Download the extracted data as a CSV file for further use.

#Project Structure
app.py: The main application script for Streamlit.
requirements.txt: Python dependencies required for the project.
packages.txt: System dependencies required for the project.
TrainingModel.pt: The pre-trained YOLO model file.
tesseract-ocr-setup-3.02.02.exe: Tesseract OCR executable installer (if applicable).

#Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the existing style and includes relevant tests.

#License
This project is licensed under the MIT License. See the LICENSE file for more details.

#Acknowledgements
YOLO (You Only Look Once): A state-of-the-art object detection system.
Tesseract OCR: An open-source OCR engine for text recognition.
Streamlit: An open-source app framework for Machine Learning and Data Science teams
