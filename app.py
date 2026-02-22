from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os

from model import FruitCNN

app = Flask(__name__, template_folder='templates')
app.config["UPLOAD_FOLDER"] = "static/uploads"


class_names = ['apples', 'avocados', 'bananas', 'grapes', 'guava', 'limes', 'mangos', 'oranges', 'pineapples', 'watermelons']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FruitCNN(num_classes=len(class_names))
model.load_state_dict(torch.load("fruit_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img = Image.open(image_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img)
                _, pred = torch.max(outputs, 1)
                prediction = class_names[pred.item()]

    return render_template("index.html",
                           prediction=prediction,
                           image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)