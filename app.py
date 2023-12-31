"""
Build a PyTorch model that can be used for prediction served out via FastAPI
"""

# import io
# import json
# import torch
# from torchvision import models
# import torchvision.transforms as transforms
# from PIL import Image

# #import fastapi
# #from fastapi import File, UploadFile
# #import uvicorn


# # updating with Flask
# from crypt import methods
# from flask import Flask,jsonify,request,redirect

# #from flask import Flask, render_template, request, redirect
# from flask_swagger_ui import get_swaggerui_blueprint

# # dont need inference
# #from inference import get_prediction
# #from commons import format_class_name

# app = Flask(__name__)

# SWAGGER_URL="/swagger"
# API_URL="swagger.json"

# # look for cuda device, not on image...

# print('is_available: ', torch.cuda.is_available())

# swagger_ui_blueprint = get_swaggerui_blueprint(
#     SWAGGER_URL,
#     API_URL,
#     config={
#         'app_name': 'Access API'
#     }
# )

# app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

# #app = fastapi.FastAPI()

import io
import json
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import fastapi
from fastapi import File, UploadFile, Request
import uvicorn
import torch
from fastapi.responses import RedirectResponse, HTMLResponse
import base64

#from flask import Flask, request, redirect

app = fastapi.FastAPI()

model_10 = torch.load("model_10_class_not_jit.pt", map_location=torch.device('cpu'))
model_3 = torch.load("model_3_class_not_jit.pt", map_location=torch.device('cpu'))

model_10.eval()
model_3.eval()

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


@app.get("/")
def index():
    return {"message": "Hello Hair Disease July 19 2023"}

@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

@app.post("/predictOLD")
async def predictFIle(file: UploadFile = File(...)):
    image_bytes = await file.read()
    print('length of inc image', len(image_bytes))
    model_3_pred_idx, label_pred_3, model_10_pred_idx, label_pred_10 = get_prediction(image_bytes=image_bytes)
    return {"earlyorlateID": model_3_pred_idx, "class_name_3": label_pred_3, "diseaseID": model_10_pred_idx, "class_name_10":label_pred_10}

# adding Matthias's route, using requests
# https://fastapi.tiangolo.com/advanced/using-request-directly/
@app.post("/predict")
async def predictRequest(request: Request):
    
  if request.method == 'POST':
    content_type = request.headers.get('Content-type')
        
    if (content_type == 'application/json'):
      
      data = await request.json()
      if not data:
        return
      
      img_string = data.get('file')
      #Clean string
      img_string = img_string[img_string.find(",")+1:]
      img_bytes = base64.b64decode(img_string)      
    elif (content_type == 'multipart/form-data'):
      print('you have multiformish dater!')
      if 'file' not in request.files:
        return {"oops":"no data in form"}
      file = request.files.get('file')
      if not file:
        return
      img_bytes = file.read()
    else: 
      return "Content type is not supported."
  
    if len(img_bytes) > 0:   # not sure if that works like that in Python...
      model_3_pred_idx, label_pred_3, model_10_pred_idx, label_pred_10 = get_prediction(image_bytes=img_bytes)        
      return {"earlyorlateID": model_3_pred_idx, "class_name_3": label_pred_3, "diseaseID": model_10_pred_idx, "class_name_10":label_pred_10}
    else: 
      return "Cannot extract image data from request"

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#   if request.method == 'POST':
#     content_type = request.headers.get('Content-Type')
#     if (content_type == 'application/json'):
#       data = request.json
#       if not data:
#          return
#       img_bytes = data.get('file')
#     elif (content_type == 'multipart/form-data'):
#       if 'file' not in request.files:
#         return redirect(request.url)
#       file = request.files.get('file')
#       if not file:
#         return
#       img_bytes = file.read()
#     else: 
#       return "Content type is not supported."
   
#     if img_bytes:   # not sure if that works like that in Python...
#       model_3_pred_idx, label_pred_3, model_10_pred_idx, label_pred_10 = get_prediction(image_bytes=img_bytes)        
#       return {"earlyorlateID": model_3_pred_idx, "class_name_3": label_pred_3, "diseaseID": model_10_pred_idx, "class_name_10":label_pred_10}
#     else: 
#       return "Cannot extract image data from request"


# # Flask stuff below

# @app.route('/')
# def index():
#     return {"message": "FLASK -- Hello Hair Disease July 16 2023"}

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files.get('file')
#         if not file:
#             return
#         img_bytes = file.read()

#         #class_id, class_name = get_prediction(image_bytes=img_bytes)
#         #class_name = format_class_name(class_name)
#         #return render_template('result.html', class_id=class_id, class_name=class_name)
#         # return render_template('index.html')

#         model_3_pred_idx, label_pred_3, model_10_pred_idx, label_pred_10 = get_prediction(image_bytes=img_bytes)
        
#         return {"earlyorlateID": model_3_pred_idx, "class_name_3": label_pred_3, "diseaseID": model_10_pred_idx, "class_name_10":label_pred_10}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
    #app.run(debug=True, host='0.0.0.0', port=8080)
