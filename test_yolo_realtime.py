import cv2
from utils.detector import YoloDetector
from utils.feature_extractor import Feature_Extractor
from utils.database import DataBase
from utils.utils import Camera, Plotter
import imutils

cap = Camera(0)
plotter = Plotter()

detector = YoloDetector()
fe = Feature_Extractor()
db = DataBase()
resolution = 720
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = imutils.resize(frame, width=resolution)
    if success:
        # Run YOLOv8 inference on the frame
        aligned_faces, coors = detector.detect(frame)

        if len(aligned_faces) != 0:
            embeddings = fe.pre_processing(aligned_faces)
            paths = db.search(embeddings)

        # Visualize the results on the frame
        base_img = frame.copy()

        base_img = plotter.plot(coors, paths, base_img)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", base_img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
cv2.destroyAllWindows()
