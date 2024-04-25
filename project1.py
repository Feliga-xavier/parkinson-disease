import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_path = "C:\\Users\\felic\\Desktop\\jupyter\\parkinson_model.h5"
loaded_model = load_model(model_path)  # Specify the correct path if needed

# Function to preprocess a single image for prediction
def preprocess_image(image_path, apply_blur=True):
    try:
        if image_array is not None:
            # Resize the image to the input shape of the model
            resized_image = cv2.resize(image_array, (224, 224))

            # Apply blur if specified
            if apply_blur:
                blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
            else:
                blurred_image = resized_image

            # Normalize the image
            normalized_image = blurred_image / 255.0

            # Reshape the image for the CNN input
            processed_image = normalized_image.reshape((224, 224, 1))
            processed_image = np.expand_dims(processed_image, axis=0)

            return processed_image, resized_image  # Return the resized image for display
        else:
            st.error("Error reading image. Please upload a valid image.")
            return None, None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None
# Check if label_encoder.joblib exists
label_encoder_path = "C:\\Users\\felic\\Desktop\\jupyter\\label_encoder.joblib"

try:
    # Attempt to load the label encoder
    label_encoder = joblib.load(label_encoder_path)
except FileNotFoundError:
    # If the file is not found, create a new label encoder
    label_encoder = LabelEncoder()
    # Fit and transform your labels (replace 'your_labels' with your actual labels)
    labels = ['class1', 'class2', 'class3']  # Modify with your actual classes
    label_encoder.fit(labels)
    # Save the label encoder
    joblib.dump(label_encoder, label_encoder_path)

# Streamlit app code
st.title("Parkinson's Disease Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Perform prediction
    image_array = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 0)

     # Display the original image
    st.image(image_array, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image_array, apply_blur=True)
        
    if processed_image is not None:
        # Unpack the tuple to get the processed image
        processed_image, original_image = preprocess_image(image_array, apply_blur=True)

        # Convert processed image to RGB format for display
        processed_image_rgb = cv2.cvtColor((processed_image.squeeze() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Convert the NumPy array to a format suitable for display
        image_for_display = (processed_image.squeeze() * 255).astype(np.uint8)

        # Display the image
        st.image([image_for_display], caption='Preprocessed Image', use_column_width=True)

        # Make predictions
        predictions = loaded_model.predict(processed_image)

        # Get the predicted class
        predicted_class = np.argmax(predictions)

        #    Map the predicted class to the corresponding label
        predicted_label = label_encoder.classes_[predicted_class]

        # Display the prediction result
        st.write(f"The predicted label for the given image is: {predicted_label}")
    else:
        st.write("Failed to preprocess the image.")
        