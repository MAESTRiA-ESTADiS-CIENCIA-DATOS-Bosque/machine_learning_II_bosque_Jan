from fastapi import FastAPI
from pydantic import BaseModel
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *
import uvicorn
from fastapi.responses import FileResponse
import numpy as np

# #@app.get("/")
# #async def root():
# #    return {"mensaje": "Holaa api!!"}

# class Item(BaseModel):
#     nombre: str
#     edad: int
#     es_hombre: bool

# @app.post('/item/')
# def create_item(item:Item):
#     return item
#dir='/home/javier/yolo3_tf_new/TensorFlow-2.x-YOLOv3/'
yolo = Load_Yolo_model()


app =FastAPI()
@app.get('/predict')
def predict(url_image:str):
    path_url= tf.keras.utils.get_file("", url_image)
    detect_image(yolo, path_url, "./IMAGES/get1.png",\
         input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    return FileResponse("./IMAGES/get1.png")
