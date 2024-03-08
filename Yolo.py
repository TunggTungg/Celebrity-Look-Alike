
from typing import Any, List, Tuple, Union, Optional
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from abc import ABC, abstractmethod


class FacialAreaRegion:
    x: int
    y: int
    w: int
    h: int
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    confidence: float

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        left_eye: Optional[Tuple[int, int]] = None,
        right_eye: Optional[Tuple[int, int]] = None,
        confidence: Optional[float] = None,
    ):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.confidence = confidence

class YoloDetector():
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build a yolo detector model
        Returns:
            model (Any)
        """

        # Import the Ultralytics YOLO model
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as e:
            raise ImportError(
                "Yolo is an optional detector, ensure the library is installed. \
                Please install using 'pip install ultralytics' "
            ) from e
        
        # Model's weights paths
        weight_path = f"yolov8n-face.pt"
        model = YOLO(weight_path, verbose=False)    
        return model
    
    def detect(self, img: np.ndarray) -> List:
        """Return results from detector(YOLOv8)
        Args:
            img (np.ndarray): frame from camera
        Returns:
            results (List): detected faces
        """
        results = self.model(img, agnostic_nms=True, conf=0.5, verbose = False)[0]
        return results
    
    def align_face(self, img: np.ndarray, left_eye: Union[list, tuple], right_eye: Union[list, tuple]) -> Tuple[np.ndarray, float]:
        """Align a given image horizantally with respect to their left and right eye locations
        Args:
            img (np.ndarray): pre-loaded image with detected face
            left_eye (list or tuple): coordinates of left eye with respect to the you
            right_eye(list or tuple): coordinates of right eye with respect to the you
        Returns:
            img (np.ndarray): aligned facial image
        """
        # if eye could not be detected for the given image, return image itself
        if left_eye is None or right_eye is None:
            return img

        # sometimes unexpectedly detected images come with nil dimensions
        if img.shape[0] == 0 or img.shape[1] == 0:
            return img
    
        # compute the angle between the eye centroids
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        desiredLeftEye = (0.3,0.3)
        desiredRightEyeX = 1.0 - desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredFaceWidth, desiredFaceHeight = 160, 160
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2)
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        aligned_face = cv2.warpAffine(img, M, (w, h),
            flags=cv2.INTER_CUBIC)
        aligned_face = aligned_face.astype(float)
        # return the aligned face
        return aligned_face
    
    def pre_processing_faces(self, img: np.ndarray,  results: List) -> List:
        """
        Detect and align face with yolo

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            aligned_faces (List[]): A list of aligned_faces objects
        """
        #For each face, extract the bounding box, the landmarks and confidence
        aligned_faces = []
        coor = []
        for result in results:
            # Extract the bounding box and the confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]
            
            # left_eye_conf = result.keypoints.conf[0][0]
            # right_eye_conf = result.keypoints.conf[0][1]
            left_eye = result.keypoints.xy[0][0].tolist()
            right_eye = result.keypoints.xy[0][1].tolist()

            # eyes are list of float, need to cast them tuple of int
            left_eye = tuple(int(i) for i in left_eye)
            right_eye = tuple(int(i) for i in right_eye)
                # left_eye, right_eye = None, None
            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
            )
            coor.append(facial_area)
            aligned = self.align_face(img, left_eye=facial_area.left_eye, right_eye=facial_area.right_eye)
            aligned_faces.append(aligned)

        return aligned_faces, coor
        

if __name__ == "__main__":
    import time
    
    detector = YoloDetector()
    image_path = 'test2.jpg'
    
    image = cv2.imread(image_path)
    
    image = cv2.resize(image, (640, 640))
    start = time.time()
    out = detector.detect(image)
    a = detector.pre_processing_faces
    

    print(time.time() - start)
    