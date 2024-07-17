from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import requests
import torch

app = Flask(__name__)
CORS(app)

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)

@app.route('/predict', methods=['POST'])
def predict():
    # ... (previous code remains the same)
    data = request.json
    image_url = data['image_url']
    question = data['question']

    # Prepare image and question
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Process inputs
    encoding = processor(image, question, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]

    return jsonify({"answer": answer})
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)