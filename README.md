# CIFAR-10 Image Classification with CNN

This project was developed as a task assigned by SkyHighes Technologies. The main objective was to create a Convolutional Neural Network (CNN) model to classify images from the CIFAR-10 dataset. In addition to the model, I have developed a Streamlit web application named ***CIFARInsight - Decode Your Images*** to provide an interactive platform for users to upload images and receive predictions based on the trained model. This web application was created to enhance my resume and demonstrate my skills in deploying machine learning models.

Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cifarinsight.streamlit.app/)

## Project Structure

The project consists of the following files:

- `Notebook.ipynb`: This Jupyter notebook serves as the main content for the project. It includes the data preprocessing steps, model training, and evaluation processes, providing a comprehensive overview of the workflow used to achieve image classification with the CNN.
- `model.py`: This file contains the code for downloading the CIFAR-10 dataset, defining the CNN model architecture, training the model, and saving it as `model.pth`.
- `app.py`: This file implements a Streamlit web application that allows users to upload images for classification. The application loads the trained model and displays the predicted class of the uploaded image.
- `model.pth`: The saved model file after training.

## Installation

To run the project, follow these steps:

1. Clone the repository:
   ```
   git clone "https://github.com/gandharvk422/Image-Classification-with-CNN"
   ```

2. Install the required libraries:
    ```
    pip install torch torchvision streamlit
    ```

3. **Train the Model:** Run the model.py file to train the model and create `model.pth`:
    ```
    python model.py
    ```

4. **Run the Streamlit App:** Start the Streamlit web application:
    ```
    streamlit run app.py
    ```

## Usage

* Upload an image of an object from the CIFAR-10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

* The model will predict the class of the uploaded image and display the result on the web page.

## Acknowledgments

* This project was developed as part of a task by SkyHighes Technologies.

* Special thanks to the contributors and open-source libraries that made this project possible.

Feel free to modify any sections as needed!
<hr>
