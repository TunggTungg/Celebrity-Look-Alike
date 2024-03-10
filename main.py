from utils.utils import Camera, cv2, plot
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from utils.detector import YoloDetector
from utils.feature_extractor import Feature_Extractor
from utils.database import DataBase
import imutils

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Init 
cap = Camera(0)
detector = YoloDetector()
fe = Feature_Extractor()
db = DataBase("http://10.5.0.4:9200")

def gen_frames():
    while cap.isOpened():
        success, frame = cap.read()
        frame = imutils.resize(frame, width=720)
        if success:
            # Run YOLOv8 inference on the frame
            aligned_faces, coors = detector.detect(frame)
            
            if len(aligned_faces) != 0:
                embeddings = fe.pre_processing(aligned_faces)
                paths = db.search(embeddings)

                # Visualize the results on the frame
                frame = plot(coors, paths, frame)
            _, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
