import os
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import streamlit as st

# Ensure necessary directories exist
os.makedirs('LR', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Model configuration
model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cpu')  # Change 'cpu' to 'cuda' for GPU

# Load the model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# Streamlit app
st.title("ESRGAN-Enhanced Generative Adversarial Network")
st.write("Upload a low-resolution image to enhance it using the ESRGAN model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Load in BGR format

    # Save input image to LR folder (BGR format)
    input_image_path = os.path.join('LR', uploaded_file.name)
    cv2.imwrite(input_image_path, img)  # Save input image as-is (BGR)

    # Convert BGR to RGB for correct display in Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the input image
    # st.write("Input Image:")
    # st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for the model
    img = img * 1.0 / 255  # Normalize
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()  # BGR to RGB, HWC to CHW
    img_LR = img.unsqueeze(0).to(device)

    # Run the model
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # Post-process output
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW to HWC, RGB to BGR
    output = (output * 255.0).round().astype(np.uint8)

    # Save enhanced image to results folder (BGR format)
    output_image_path = os.path.join('results', f"{os.path.splitext(uploaded_file.name)[0]}_rlt.png")
    cv2.imwrite(output_image_path, output)  # Save in BGR format

    # Convert output to RGB for display
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

       # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # st.write("Input Image:")
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    with col2:
        # st.write("Enhanced Image:")
        st.image(output_rgb, caption="Enhanced Image", use_column_width=True)


    # Provide download link for the enhanced image
    with open(output_image_path, "rb") as file:
        st.download_button(
            label="Download Enhanced Image",
            data=file,
            file_name=os.path.basename(output_image_path),
            mime="image/png"
        )
