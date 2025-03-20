import os
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

# Ensure the necessary directories exist
os.makedirs('LR', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load the ESRGAN model
model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cpu')  # Use 'cuda' for GPU if available

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", error="No file selected")

        # Save the file to the LR directory
        input_image_path = os.path.join("LR", file.filename)
        file.save(input_image_path)

        # Load and preprocess the input image
        img = cv2.imread(input_image_path)
        img = img * 1.0 / 255  # Normalize
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0).to(device)

        # Run the ESRGAN model
        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        # Post-process the output
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # Convert to BGR
        output = (output * 255.0).round().astype(np.uint8)

        # Save the enhanced image
        output_image_path = os.path.join("results", f"{os.path.splitext(file.filename)[0]}_rlt.png")
        cv2.imwrite(output_image_path, output)

        return render_template(
            "index.html",
            input_image=file.filename,
            output_image=os.path.basename(output_image_path)
        )

    return render_template("index.html")

@app.route("/LR/<filename>")
def serve_LR_file(filename):
    return send_from_directory("LR", filename)

@app.route("/results/<filename>")
def download_file(filename):
    return send_from_directory("results", filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
    
