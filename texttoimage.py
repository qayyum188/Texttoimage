
import streamlit as st
import torch
from diffusers import FluxPipeline, DiffusionPipeline
from huggingface_hub import login, HfFolder

# Log in to your Hugging Face account using your token
login(token="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")  # Replace with your actual token

# Title and subtitle
st.title("Flux Image Generator")
st.write("Created by Abdul Qayyum")

# Select pipeline type
pipeline_type = st.selectbox("Select Pipeline", ["FluxPipeline", "DiffusionPipeline"])

# Input prompt
prompt = st.text_input("Enter your prompt:")

# Generate image button
if st.button("Generate Image"):
    if pipeline_type == "FluxPipeline":
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()

        image = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=25,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)).images[0]
    else:  # DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        image = pipe(prompt).images[0]

    # Display the image
    st.image(image, caption="Generated Image", use_column_width=True)
