# VLM Question Answering System Deployment Guide

This guide will walk you through the process of setting up and deploying the VLM Question Answering system.

## Prerequisites

- Python 3.9 or later
- pip (Python package manager)
- Git (optional, for version control)

## Step 1: Set Up the Environment

1. Clone the repository (if using version control):
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install flask transformers torch pillow requests gunicorn
   ```

## Step 2: Prepare the Application Files

1. Create a file named `app.py` with the following content:

   ```python
   from flask import Flask, request, jsonify, render_template
   from transformers import ViltProcessor, ViltForQuestionAnswering
   from PIL import Image
   import requests
   import torch

   app = Flask(__name__)

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
   model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)

   @app.route('/', methods=['GET'])
   def home():
       return render_template('index.html')

   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.json
       image_url = data['image_url']
       question = data['question']

       image = Image.open(requests.get(image_url, stream=True).raw)
       encoding = processor(image, question, return_tensors="pt").to(device)

       with torch.no_grad():
           outputs = model(**encoding)
       logits = outputs.logits
       idx = logits.argmax(-1).item()
       answer = model.config.id2label[idx]

       return jsonify({"answer": answer})

   if __name__ == '__main__':
       app.run(debug=True)
   ```

2. Create a `templates` folder and add the `index.html` file inside it with the HTML content provided earlier.

## Step 3: Local Testing

1. Run the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000` to test the application locally.

## Step 4: Deployment

### Option 1: Deploy on a Linux Server

1. Transfer your application files to the server.

2. Install required system dependencies:
   ```
   sudo apt-get update
   sudo apt-get install python3-pip python3-venv
   ```

3. Follow steps 1-4 from "Set Up the Environment" section on the server.

4. Install Gunicorn:
   ```
   pip install gunicorn
   ```

5. Create a `wsgi.py` file with the following content:
   ```python
   from app import app

   if __name__ == "__main__":
       app.run()
   ```

6. Start the application with Gunicorn:
   ```
   gunicorn --bind 0.0.0.0:8000 wsgi:app
   ```

7. Set up a reverse proxy (e.g., Nginx) to forward requests to Gunicorn.

### Option 2: Deploy on Heroku

1. Install the Heroku CLI and login.

2. Create a `Procfile` with the following content:
   ```
   web: gunicorn wsgi:app
   ```

3. Create a `runtime.txt` file specifying the Python version:
   ```
   python-3.9.7
   ```

4. Create a `requirements.txt` file:
   ```
   pip freeze > requirements.txt
   ```

5. Initialize a Git repository (if not already done):
   ```
   git init
   git add .
   git commit -m "Initial commit"
   ```

6. Create a Heroku app and deploy:
   ```
   heroku create your-app-name
   git push heroku main
   ```

7. Open the app in a browser:
   ```
   heroku open
   ```

## Step 5: SSL/TLS Configuration (for production)

For a production environment, it's crucial to set up SSL/TLS:

1. Obtain an SSL/TLS certificate (e.g., using Let's Encrypt).
2. Configure your web server (e.g., Nginx) to use the certificate.
3. Ensure all resources are loaded over HTTPS.

## Troubleshooting

- If you encounter CUDA/GPU issues, ensure you have the correct CUDA version installed for your PyTorch version.
- For memory issues, you may need to adjust the model size or use a CPU-only version.
- Check firewall settings if you can't access the deployed application.

## Maintenance

- Regularly update dependencies to patch security vulnerabilities.
- Monitor server resources (CPU, memory, disk space) to ensure optimal performance.
- Implement logging for easier debugging and monitoring.

Remember to never expose sensitive information (e.g., API keys) in your code or version control. Use environment variables for sensitive data.
