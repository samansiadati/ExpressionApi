from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from random import randint
from ExpressionApi.express import Xpress
import matplotlib.pyplot as plt
import uuid
import logging
import uvicorn
import os
logger = logging.getLogger(__name__)
router = APIRouter()

IMAGEDIR = "/home/sam/projects/images/"

app =FastAPI(title="Expression Analysis",
    description=(
        "This is a Rest Api generated with FastAPI"
    ),
    root_path="/",
)

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename": file.filename}


@app.get('/expression/')
async def express():

    files = os.listdir(IMAGEDIR)
    random_index = randint(0, len(files) - 1)
    path = f"{IMAGEDIR}{files[random_index]}"

    test_image_one = plt.imread(path)
    emo_detector = Xpress(mtcnn=True)

    captured_emotions = emo_detector.detect_emotions(test_image_one)
    emotions_dict = [d['emotions'] for d in captured_emotions][0]
    #print(emotions_dict)

    FileResponse(path)
    #plt.imshow(test_image_one)
    return emotions_dict


@app.get("/images/")
async def read_uploaded_file():
    files = os.listdir(IMAGEDIR)
    random_index = randint(0, len(files) - 1)

    path = f"{IMAGEDIR}{files[random_index]}"

    return FileResponse(path)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)