import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cpu")  # Force CPU usage
    return pipe

pipe = load_model()

# Streamlit App Title
st.title("Text to Image Generator")

# Input text prompt from user
prompt = st.text_input("Enter a text prompt to generate an image:")

# Button to generate image
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            # Generate image
            image = pipe(prompt).images[0]
            
            # Display the image
            st.image(image, caption="Generated Image", use_column_width=True)
            
            # Save the image if needed
            image.save("generated_image.png")
            st.success("Image generated and saved!")
    else:
        st.warning("Please enter a prompt to generate an image.")
