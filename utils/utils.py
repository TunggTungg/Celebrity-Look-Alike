import cv2
from typing import List
class Camera():
    def __init__(self, id: str = "0"):
        self.camera = cv2.VideoCapture(id)
        # self.camera = cv2.VideoCapture("http://192.168.1.38:4747/video")
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    def read(self):
        return self.camera.read()
    def isOpened(self):
        return self.camera.isOpened()
    def release(self):
        self.camera.release()


def plot(coors, paths, base_img):
    pivot_img_size = 120
    resolution = 720
    for i in range(len(coors)):
        label = paths[i].split("/")[0]
        display_img = cv2.imread("./celeb_data/" + paths[i])
        display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))
        facial_area = coors[i]
        x, y, w, h = facial_area.x, facial_area.y, facial_area.w, facial_area.h

        # Plot Bounding Box
        base_img = cv2.rectangle(base_img, (x,y), (x+w, y+h), (0,255,0), 1) 

        if (y - pivot_img_size > 0 and x + w + pivot_img_size < resolution):
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
    return base_img