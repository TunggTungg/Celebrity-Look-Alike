from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from utils.triton_serving import triton_client
from utils.database import DataBase
from utils.utils import Camera, cv2, plot
import imutils
import asyncio
import time

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Init 
cap = Camera(0)
client = triton_client()
db = DataBase("http://10.5.0.4:9200")

def gen_frames():
    while cap.isOpened():
        success, frame = cap.read()
        frame = imutils.resize(frame, width=720)
        if success:
            start = time.time()
            # Run YOLOv8 inference on the frame
            embedding_vectors, coordinates = client.query(frame)
        
            if embedding_vectors.shape[0] != 0:
                paths = db.search(embedding_vectors)
                # Visualize the results on the frame
                frame = plot(coordinates, paths, frame)
            elapsed_time = time.time() - start

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(max(0, 0.03 - elapsed_time))

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
