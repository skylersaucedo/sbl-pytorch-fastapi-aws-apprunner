"""
Build a PyTorch model that can be used for prediction served out via FastAPI
"""

import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

#import fastapi
#from fastapi import File, UploadFile
#import uvicorn


# updating with Flask


from flask import Flask, render_template, request, redirect
from flask_swagger_ui import get_swaggerui_blueprint

# dont need inference
#from inference import get_prediction
#from commons import format_class_name

app = Flask(__name__)

SWAGGER_URL = '/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL ="http://localhost:8080" # Our API url (can of course be a local resource)

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
    # oauth_config={  # OAuth config. See https://github.com/swagger-api/swagger-ui#oauth2-configuration .
    #    'clientId': "your-client-id",
    #    'clientSecret': "your-client-secret-if-required",
    #    'realm': "your-realms",
    #    'appName': "your-app-name",
    #    'scopeSeparator': " ",
    #    'additionalQueryStringParams': {'test': "hello"}
    # }
)

app.register_blueprint(swaggerui_blueprint)


#app = fastapi.FastAPI()

model_10 = torch.load("model_10_class_not_jit.pt", map_location=torch.device('cpu'))
model_3 = torch.load("model_3_class_not_jit.pt", map_location=torch.device('cpu'))

model_10.eval()
model_3.eval()

#imagenet_class_index = json.load(open("imagenet_class_index.json", encoding="utf-8"))
model10_class_index = json.load(open("10class_index.json", encoding="utf-8"))
model3_class_index = json.load(open("3class_index.json", encoding="utf-8"))


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)

    outputs_10 = model_10(tensor)
    outputs_3 = model_3(tensor)


    _10, y_hat10 = torch.max(outputs_10.data,1)
    _3, y_hat3 = torch.max(outputs_3.data,1)

    # print('look at your prediction: ', y_hat)
    # print('pred index: ', str(y_hat.item()))

    model_10_pred_idx = str(y_hat10.item())
    model_3_pred_idx = str(y_hat3.item())

    label_pred_10 = model10_class_index[model_10_pred_idx]
    label_pred_3 = model3_class_index[model_3_pred_idx]



    return model_3_pred_idx, label_pred_3, model_10_pred_idx, label_pred_10


# @app.get("/")
# def index():
#     return {"message": "Hello Hair Disease July 16 2023"}


# @app.post("/files/")
# async def create_file(file: bytes = File()):
#     return {"file_size": len(file)}


# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):
#     return {"filename": file.filename}


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     print(len(image_bytes))
#     model_3_pred_idx, label_pred_3, model_10_pred_idx, label_pred_10 = get_prediction(image_bytes=image_bytes)
#     return {"earlyorlateID": model_3_pred_idx, "class_name_3": label_pred_3, "diseaseID": model_10_pred_idx, "class_name_10":label_pred_10}

@app.route('/')
def index():
    return {"message": "Hello Hair Disease July 16 2023"}

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()

        #class_id, class_name = get_prediction(image_bytes=img_bytes)
        #class_name = format_class_name(class_name)
        #return render_template('result.html', class_id=class_id, class_name=class_name)
        # return render_template('index.html')

        model_3_pred_idx, label_pred_3, model_10_pred_idx, label_pred_10 = get_prediction(image_bytes=img_bytes)


        
        return {"earlyorlateID": model_3_pred_idx, "class_name_3": label_pred_3, "diseaseID": model_10_pred_idx, "class_name_10":label_pred_10}

if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8080)
    app.run(debug=False, host='0.0.0.0', port=8080)