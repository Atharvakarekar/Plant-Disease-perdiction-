# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename
#
# # Load the trained model
# model = tf.keras.models.load_model('plant_disease_detection_model.h5')
#
# # Define the class labels
# class_labels = ['Not Affected', 'Affected']
#
# # Prompt the user to select an image file
# Tk().withdraw()  # Hide the Tkinter main window
# image_file_path = askopenfilename(title='Select Image File')
#
# # Load and preprocess the image
# img = image.load_img(image_file_path, target_size=(224, 224))
# img = image.img_to_array(img)
# img = img / 255.0
# img = tf.expand_dims(img, axis=0)
#
# # Get the prediction
# prediction = model.predict(img)
#
# # Print the predicted class and confidence
# predicted_class = tf.argmax(prediction, axis=1)[0]
# confidence = prediction[0][predicted_class]
# predicted_label = class_labels[predicted_class]
# print(f"Prediction: {predicted_label} (Confidence: {confidence})")


# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import numpy as np
#
# # Set page title and icon
# st.set_page_config(page_title='Plant Disease Detection', page_icon='🌱')
#
# # Load the trained model
# model = tf.keras.models.load_model('plant_disease_detection_model.h5')
# # Set the number of classes in your dataset
# num_classes = 37
# # Define the class labels
# class_labels = ['Not Affected', 'Affected']
# class_labels = ['class_{}'.format(i) for i in range(num_classes)]
# assert len(class_labels) == num_classes, "Number of class labels does not match the number of classes"
# # Set app title and description
# st.title('Plant Disease Detection')
# st.markdown('Upload an image to detect plant diseases.')
#
# # Create a file uploader widget
# uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
#
# if uploaded_file is not None:
#     # Load and preprocess the image
#     img = Image.open(uploaded_file)
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#
#     # Get the prediction
#     prediction = model.predict(img_array)
#
#     # Get the predicted class and confidence
#     predicted_class = tf.argmax(prediction, axis=1)[0]
#     confidence = prediction[0][predicted_class]
#     predicted_label = class_labels[predicted_class]
#
#     # Display the uploaded image and prediction result
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(img, caption='Uploaded Image', use_column_width=True)
#     with col2:
#         st.subheader('Prediction Result')
#         st.write(f"Class: {predicted_label}")
#         st.write(f"Confidence: {confidence:.2%}")


import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Set page title and icon
st.set_page_config(page_title='Plant Disease Detection', page_icon='🌱')

# Load the trained model
model = tf.keras.models.load_model('plant_disease_detection_model.h5')

# Set the number of classes in your dataset
num_classes = 37

# Define the class labels
class_labels = ['Not Affected', 'Affected']
class_labels += ['class_{}'.format(i) for i in range(num_classes - len(class_labels))]
assert len(class_labels) == num_classes, "Number of class labels does not match the number of classes"

# Set app title and description
st.title('Plant Disease Detection')
st.markdown('Upload an image to detect plant diseases.')

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Check if the uploaded file is an image
    try:
        img = Image.open(uploaded_file)
    except:
        st.error("Invalid image file. Please upload a valid image file.")
        st.stop()

    # Load and preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get the prediction
    prediction = model.predict(img_array)

    # Get the predicted class and confidence
    predicted_class = tf.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    predicted_label = class_labels[predicted_class]

    # Display the uploaded image and prediction result
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption='Uploaded Image', use_column_width=True)
    with col2:
        st.subheader('Prediction Result')
        st.write(f"Class: {predicted_label}")
        if predicted_class == 0:
            st.write("Plant Leaf: Not Affected")
        else:
            st.write("Plant Leaf: Affected")
        st.write(f"Confidence: {confidence:.2%}")