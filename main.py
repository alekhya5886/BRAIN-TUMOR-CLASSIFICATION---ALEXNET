import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="TumorScanAI.com",
    page_icon=":brain:",
    layout="centered",
)

# Load the Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key="#apikey")
chat_model = gen_ai.GenerativeModel('gemini-1.5-pro')

# Set up the brain tumor classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'alexnet_brain_tumor_classification.pth'

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

alexnet = models.alexnet(pretrained=False)
alexnet.classifier[6] = nn.Linear(4096, 4)  # 4 classes: glioma, meningioma, notumor, pituitary
alexnet.load_state_dict(torch.load(model_path, map_location=device))
alexnet = alexnet.to(device)
alexnet.eval()

class_names = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']

# Function to predict the class of an uploaded image
def predict_image(image, model):
    image = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    return predicted_class

# Function to assess emergency level and suggest treatment or vitamins
def assess_tumor_emergency(predicted_class):
    emergency_info = {
        "Glioma": {
            "emergency_level": "High",
            "action": "You should immediately see a doctor for this condition. Glioma can be aggressive and requires professional treatment.",
            "origin": "Arise from glial cells, which are supportive cells in the brain and spinal cord."
        },
        "Meningioma": {
            "emergency_level": "Moderate",
            "action": "Consult with a doctor soon. Meningioma is usually slow-growing, but medical supervision is needed.",
            "origin": "Arise from the meninges, the membranes that cover the brain and spinal cord"
        },
        "Notumor": {
            "emergency_level": "None",
            "action": "No tumor detected. You may not need any treatment, but continue a healthy lifestyle and monitor for symptoms.",
            "origin": "No tumor detected in the MRI scan."
        },
        "Pituitary": {
            "emergency_level": "Medium",
            "action": "Visit a doctor to discuss treatment options. Pituitary tumors can affect hormone levels and require medical attention.",
            "origin": "Arise from the pituitary gland, a small gland at the base of the brain responsible for hormone production."
        }
    }
    return emergency_info[predicted_class]

def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Main Streamlit App
def main():
    predicted_class = None
    # Add a heading for the website
    st.markdown("<h1 style='text-align: center;'>Brain Tumor Classification and Brainy bot</h1>", unsafe_allow_html=True)
    
    # Set the Streamlit title (it will appear below the custom heading)
    #st.title("Brain Tumor Classification & Chatbot")
    
    # Brain Tumor Classification Section
    st.header("Brain Tumor Classification")
    uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        resized_image = image.resize((300, 300))
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(resized_image, caption="Uploaded MRI Image", use_column_width=True)
        
        with col2:
            if st.button("Classify Tumor"):
                predicted_class = predict_image(image, alexnet)
                st.session_state.predicted_class = predicted_class
                classification_result = f"The predicted class is: {predicted_class}"
                
                # Update text area with the classification result
                st.markdown(f"<h2>{classification_result}</h2>", unsafe_allow_html=True)

                # Assess emergency level and suggest treatment/vitamins
                assessment = assess_tumor_emergency(predicted_class)
                st.markdown(f"**Emergency Level**: {assessment['emergency_level']}")
                st.markdown(f"**Recommended Action**: {assessment['action']}")
                st.markdown(f"**Origin**: {assessment['origin']}")

                # Store the classification result in session state
                st.session_state.classification_result = classification_result

                # Add the classification result to chatbot history
                if "chat_session" not in st.session_state:
                    st.session_state.chat_session = chat_model.start_chat(history=[])

                st.session_state.chat_session.send_message(classification_result)

    else:
        st.warning("Please upload an MRI image for classification.")

    # Chatbot Section
    st.header("Chat with Our Brainy Bot")

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = chat_model.start_chat(history=[])
    # Location Input Section
    st.subheader("Find Nearby Hospitals")

    # Input field for user's location
    user_location = st.text_input("Enter your area or city to find nearby hospitals:", key="location_input")
    if user_location:
        # Store the location in session state
        st.session_state.user_location = user_location

        # Add location to chat history
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = chat_model.start_chat(history=[])

        # Send location to chatbot
        st.session_state.chat_session.send_message(f"User is looking for hospitals in: {user_location} for treatment of {predicted_class}")

        # Display message indicating the location has been sent
        st.markdown(f"Location `{user_location}` has been sent to the chatbot for finding nearby hospitals.")


    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Type your message here...")
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

if __name__ == "__main__":
    main()

