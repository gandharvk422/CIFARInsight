import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

st.set_page_config(page_title="CIFARInsight - Decode Your Images", page_icon="📸")

# Define the model architecture
class Cifar10Model(nn.Module):
    def __init__(self):
        super(Cifar10Model, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 classes for CIFAR-10
        )

    def forward(self, x):
        return self.network(x)

# Load the model
model = Cifar10Model()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

image_path = "0004.png"

def sample_image():
    # Open the sample image
    image = Image.open(image_path)
    st.session_state.sample_image = image  # Save the image to session state

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        st.session_state.predicted_class = class_names[predicted.item()]  # Save prediction to session state

# Streamlit web app layout
st.markdown("# CIFARInsight")
st.markdown("## Decode Your Images")

st.markdown("Want to try a sample image?")
if st.button("Sample Image"):
    sample_image()

# Display sample image and prediction if available
if 'sample_image' in st.session_state:
    st.image(st.session_state.sample_image, caption="Sample Image.", use_column_width=True)
    if 'predicted_class' in st.session_state:
        st.write(f"Predicted class: {st.session_state.predicted_class}")

st.markdown("")
st.write("Upload an image to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        st.write(f"Predicted class: {class_names[predicted.item()]}")

st.markdown("---")
st.markdown("##### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Created by [Gandharv Kulkarni](https://share.streamlit.io/user/gandharvk422)")

st.markdown("&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[![GitHub](https://img.shields.io/badge/GitHub-100000?style=the-badge&logo=github&logoColor=white&logoBackground=white)](https://github.com/gandharvk422) &emsp; [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/gandharvk422) &emsp; [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/gandharvk422)")
