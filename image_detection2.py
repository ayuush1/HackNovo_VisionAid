from flask import Flask, jsonify, request, render_template
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

app = Flask(__name__)

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model and processor on the chosen device
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)

# Directory to store uploaded images
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_form():
    # Render the HTML form
    return render_template('upload.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    # Check if an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    # Ensure the uploaded file is valid
    if image_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    # Open the image
    img = Image.open(image_path)

    # Resize the image for faster processing
    img = img.resize((224, 224))

    # Example labels for detection
    labels = ["apple", "orange", "banana", "milk", "carrot","human"]

    # Process image through CLIP (move inputs to GPU)
    inputs = processor(text=labels, images=img, return_tensors="pt", padding=True).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the label with the highest similarity score
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    best_match_idx = torch.argmax(probs).item()
    description = labels[best_match_idx]

    # Return the description as a JSON response
    return jsonify({"description": description})


if __name__ == '__main__':
    app.run(debug=True)
