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

# Streamlit web app layout
st.title("CIFARInsight - Decode Your Images")
st.write("Upload an image to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        st.write(f"Predicted class: {class_names[predicted.item()]}")

st.markdown("---")
st.markdown("##### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Created by [Gandharv Kulkarni](https://share.streamlit.io/user/gandharvk422)")

st.markdown("&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[![GitHub](https://img.shields.io/badge/GitHub-100000?style=the-badge&logo=github&logoColor=white&logoBackground=white)](https://github.com/gandharvk422) &emsp; [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/gandharvk422) &emsp; [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/gandharvk422)")
