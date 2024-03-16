import cv2

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
    resolution_x = 720
    resolution_y = 540
    for i in range(len(paths)):
        label = paths[i].split("/")[0]
        display_img = cv2.imread("./celeb_data/" + paths[i])
        display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))
        x, y, w, h = coors[i]

        # Plot Bounding Box
        base_img = cv2.rectangle(base_img, (x,y), (x+w, y+h), (0,255,0), 1) 

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

        elif (  y + h + pivot_img_size < resolution_y
                and x - pivot_img_size > 0
            ):
            # bottom left
            base_img[
                y + h : y + h + pivot_img_size,
                x - pivot_img_size : x,
            ] = display_img

            overlay = base_img.copy()
            opacity = 0.4
            cv2.rectangle(
                base_img,
                (x - pivot_img_size, y + h - 20),
                (x, y + h),
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
                (x - pivot_img_size, y + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1,
            )

            # connect face and text
            cv2.line(
                base_img,
                (x + int(w / 2), y + h),
                (
                    x + int(w / 2) - int(w / 4),
                    y + h + int(pivot_img_size / 2),
                ),
                (67, 67, 67),
                1,
            )
            cv2.line(
                base_img,
                (
                    x + int(w / 2) - int(w / 4),
                    y + h + int(pivot_img_size / 2),
                ),
                (x, y + h + int(pivot_img_size / 2)),
                (67, 67, 67),
                1,
            )

        elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
            # top left
            base_img[
                y - pivot_img_size : y, x - pivot_img_size : x
            ] = display_img

            overlay = base_img.copy()
            opacity = 0.4
            cv2.rectangle(
                base_img,
                (x - pivot_img_size, y),
                (x, y + 20),
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
                (x - pivot_img_size, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1,
            )

            # connect face and text
            cv2.line(
                base_img,
                (x + int(w / 2), y),
                (
                    x + int(w / 2) - int(w / 4),
                    y - int(pivot_img_size / 2),
                ),
                (67, 67, 67),
                1,
            )
            cv2.line(
                base_img,
                (
                    x + int(w / 2) - int(w / 4),
                    y - int(pivot_img_size / 2),
                ),
                (x, y - int(pivot_img_size / 2)),
                (67, 67, 67),
                1,
            )

        elif (
                x + w + pivot_img_size < resolution_x
                and y + h + pivot_img_size < resolution_y
            ):
            # bottom righ
            base_img[
                y + h : y + h + pivot_img_size,
                x + w : x + w + pivot_img_size,
            ] = display_img

            overlay = base_img.copy()
            opacity = 0.4
            cv2.rectangle(
                base_img,
                (x + w, y + h - 20),
                (x + w + pivot_img_size, y + h),
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
                (x + w, y + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1,
            )

            # connect face and text
            cv2.line(
                base_img,
                (x + int(w / 2), y + h),
                (
                    x + int(w / 2) + int(w / 4),
                    y + h + int(pivot_img_size / 2),
                ),
                (67, 67, 67),
                1,
            )
            cv2.line(
                base_img,
                (
                    x + int(w / 2) + int(w / 4),
                    y + h + int(pivot_img_size / 2),
                ),
                (x + w, y + h + int(pivot_img_size / 2)),
                (67, 67, 67),
                1,
            )


    return base_img