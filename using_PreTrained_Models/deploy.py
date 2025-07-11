import streamlit as st
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load your model
PATH = 'D:\\Cellula Technologies\\Project_2\\Estimators\\'
# load the scripted model
preTrained_VGG_16_model = torch.jit.load(PATH + "VGG_16_model.pt")
#print(preTrained_VGG_16_model)


# Replace with the actual mean and std values
m = [0.73255072 ,0.50281198, 0.47917073]
s = [0.51907074 ,0.61126614 ,0.61097012]


basic_transform = transforms.Compose([
    #transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(m, s)
])

# def preprocess_input(image):
#   img = image.resize((224, 224))  # Adjust size as per your model input
#   img_array = np.array(img) / 255.0
  


teeth_diseases_classes = [ 
    'CaS',
    'CoS' ,     
    'Gum' ,   
    'MC'  ,   
    'OC'  ,   
    'OLP' ,    
    'OT'  ]    

st.markdown('<h1 style="color:black;">Vgg 19 Image classification model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The teeth disease diagnosis classifier can diagnose up to 7 teeth diseases. :</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> CaS ,CoS, Gum ,MC,  OC,  OLP , OT</h3>', unsafe_allow_html=True)


upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
    im= Image.open(upload)
    img= np.asarray(im)
    #image= cv2.resize(img,(224, 224))
    #img= preprocess_input(image)
    # Apply transform
    img_tensor = basic_transform(im).unsqueeze(0)  # Add batch dimension
    c1.header('Input Image')
    c1.image(im)
    c1.write(img.shape)

    # Predict
    preTrained_VGG_16_model.eval()
    with torch.no_grad():
        output = preTrained_VGG_16_model(img_tensor.to(device))
        predicted_class = output.argmax(1).item()
    #st.write( f'Actual Class: ', upload)
    probs = F.softmax(output, dim=1).cpu().numpy().flatten()
    print( probs )
    st.write(f"Predicted Class: {teeth_diseases_classes[predicted_class]}")
    st.write(f'with Trust: {probs[predicted_class] * 100:0.2f} %')