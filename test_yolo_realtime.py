import cv2
from ultralytics import YOLO
from Yolo import YoloDetector
from feature_extractor import Feature_Extractor
from elasticsearch import Elasticsearch
import numpy as np
from database import DataBase
import imutils

cap = cv2.VideoCapture("/dev/video0")
# cap = cv2.VideoCapture("http://192.168.1.38:4747/video")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

detector = YoloDetector()
fe = Feature_Extractor()
es = Elasticsearch("http://localhost:9200")
db = DataBase()
pivot_img_size = 120
resolution_x = 720
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = imutils.resize(frame, width=resolution_x)
    if success:
        # Run YOLOv8 inference on the frame
        results = detector.detect(frame)
        
        aligned_faces, coor = detector.pre_processing_faces(frame, results)
        
        if len(aligned_faces) != 0:
            embeddings = fe.pre_processing(aligned_faces)
            paths = []
            for e in embeddings:
                res = db.search(e)
                paths.append(res)

        # Visualize the results on the frame
        annotated_frame = results.plot(kpt_radius=0) # kpt_radius=0 for no plotting poses 
        base_img = annotated_frame.copy()

        for i in range(len(coor)):
            label = paths[i].split("/")[0]
            print(label)
            display_img = cv2.imread("./celeb_data/"+paths[i])
            display_img = cv2.resize(display_img, (pivot_img_size,pivot_img_size))
            facial_area = coor[i]
            x, y, w, h = facial_area.x, facial_area.y, facial_area.w, facial_area.h

            if (y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x):
                # top right
                base_img[
                    y - pivot_img_size : y,
                    x + w : x + w + pivot_img_size,
                :] = display_img
                overlay = base_img.copy()
                opacity = 0.4
                cv2.rectangle(
                            base_img,
                            (x + w, y),
                            (x + w + pivot_img_size, y + 20),
                            (46, 200, 255),
                            cv2.FILLED,
                        )
                cv2.addWeighted(
                                overlay,
                                opacity,
                                base_img,
                                1 - opacity,
                                0,
                                base_img,
                            )
                cv2.putText(
                            base_img,
                            label,
                            (x + w, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255,255,255),
                            1,
                        )
                # connect face and text
                cv2.line(
                    base_img,
                    (x + int(w / 2), y),
                    (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                    (0, 0, 255),
                    2,
                )
                cv2.line(
                    base_img,
                    (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                    (x + w, y - int(pivot_img_size / 2)),
                    (0, 0, 255),
                    2,
                )

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", base_img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
cv2.destroyAllWindows()
