import triton_python_backend_utils as pb_utils
from typing import Union
import numpy as np
import cv2, torch
import torchvision
import json

class input_object():
    def __init__(self, ori_image):
        self.ori_image = ori_image
        self.ori_shape = ori_image.shape[:2]
        self.processed_image, self.processed_shape = self.pre_precessing(self.ori_image)

    def pre_precessing(self, ori_image):
        shape = self.ori_shape 
        new_shape = (480, 480)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        img = cv2.resize(ori_image, new_unpad, interpolation=cv2.INTER_LINEAR) if shape[::-1] != new_unpad else ori_image.copy()
        center = True
        top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        processed_frame = np.ascontiguousarray(img) 
        processed_frame = processed_frame.astype(np.float32) / 255.0
        # Channel first
        processed_frame = processed_frame.transpose(2, 0, 1)
        # Expand dimensions
        processed_frame = np.expand_dims(processed_frame, 0)

        return processed_frame, processed_frame.shape[2:]
class Ops():
    def xywh2xyxy(self,x):
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = x[..., 0] + dw  # bottom right x
        y[..., 3] = x[..., 1] + dh  # bottom right y
        return y
    
    def xyxy2xywh(self,x):
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
        y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
        return y

    def non_max_suppression(self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        nc=0,  # number of classes (optional)
        max_det=300,
        max_nms=30000,
        max_wh=7680,
    ):
        # Checks
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output
        
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # xywh to xyxy

        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = x[i]
        return output
    
    def clip_boxes(self, boxes, shape):
        if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
            boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
            boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
            boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
            boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes

    def scale_boxes(self, img1_shape, boxes, img0_shape, padding=True, xywh=False):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding

        if padding:
            boxes[..., 0] -= pad[0]  # x padding
            boxes[..., 1] -= pad[1]  # y padding
            if not xywh:
                boxes[..., 2] -= pad[0]  # x padding
                boxes[..., 3] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        return self.clip_boxes(boxes, img0_shape)

    def clip_coords(self, coords, shape):
        if isinstance(coords, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
            coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
            coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
        else:  # np.array (faster grouped)
            coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
            coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
        return coords

    def scale_coords(self, img1_shape, coords, img0_shape, padding=True):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        
        if padding:
            coords[..., 0] -= pad[0]  # x padding
            coords[..., 1] -= pad[1]  # y padding
        coords[..., 0] /= gain
        coords[..., 1] /= gain
        return self.clip_coords(coords, img0_shape)
    
class Faces():
    def __init__(self):
        self.cropped_faces = []
        self.coordinates = []
    
    def processing(self, boxes, poses, origin_image):
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            left_eye = poses[i][0].tolist()
            right_eye = poses[i][1].tolist()
            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)

            self.coordinates.append([x, y, w, h])
            aligned_face = self.align_face(origin_image, left_eye=left_eye, right_eye=right_eye)
            self.cropped_faces.append(aligned_face)

    def align_face(self, img: np.ndarray, left_eye: Union[list, tuple], right_eye: Union[list, tuple]) -> np.ndarray:
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
        scale = desiredDist / (dist+1e-7)

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
        aligned_face = aligned_face.astype(np.float16) / 255.0
        # return the aligned face
        return aligned_face

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(
                        json.loads(args["model_config"]), "embedding_output"
                    )["data_type"]
                )
        
    def execute(self, requests):
        responses = []
        for request in requests:
            # Get INPUTs
            input_image = pb_utils.get_input_tensor_by_name(request, "input_image").as_numpy()
            
            # Preprocessing
            input_obj = input_object(input_image)

            # YOLO detecting
            yolo_input = pb_utils.Tensor(
                "images", input_obj.processed_image
            )
            detecting_request = pb_utils.InferenceRequest(
                model_name="face_extractor",
                requested_output_names=["output0"],
                inputs=[yolo_input],
                preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU) 
            )

            response = detecting_request.exec()
            if response.has_error():
                raise pb_utils.TritonModelException(response.error().message())
            else:
                yolo_outputs = pb_utils.get_output_tensor_by_name(
                    response, "output0"
                ).as_numpy()

            # YOLO postprocess: boxes, eyes
            ops = Ops()
            preds = ops.non_max_suppression(
                torch.tensor(yolo_outputs),
                conf_thres=0.25,
                agnostic=True,
                classes=[0],
                nc=1
            )
            if preds[0].size()[0] != 0:
                boxes, keypoints = [], []
                for _, pred in enumerate(preds):
                    pred[:, :4] = ops.scale_boxes(input_obj.processed_shape, pred[:, :4], input_obj.ori_shape).round()
                    boxes.extend(ops.xyxy2xywh(pred[:, :4]))
                    pred_kpts = pred[:, 6:].view(len(pred), 5,3) if len(pred) else pred[:, 6:]
                    pred_kpts = ops.scale_coords(input_obj.processed_shape, pred_kpts, input_obj.ori_shape)
                    keypoints.extend(pred_kpts)

                # Cropped Faces
                face = Faces()
                face.processing(boxes, keypoints, input_obj.ori_image)

                feature_extractor_input = np.array(face.cropped_faces)
                feature_extractor_input = pb_utils.Tensor(
                    "input_image", feature_extractor_input.astype(np.float32)
                )

                extracting_request = pb_utils.InferenceRequest(
                    model_name="feature_extractor",
                    requested_output_names=["embedding_output"],
                    inputs=[feature_extractor_input],
                )
                response = extracting_request.exec()
                if response.has_error():
                    raise pb_utils.TritonModelException(response.error().message())
                else:
                    feature_outputs = pb_utils.get_output_tensor_by_name(
                        response, "embedding_output"
                    )
                    coordinates= np.array(face.coordinates)
                    coordinates = pb_utils.Tensor(
                        "coordinates", coordinates.astype(np.uint16))
                    cropped_faces = np.array(face.cropped_faces)
                    cropped_faces = pb_utils.Tensor(
                        "cropped_faces", cropped_faces.astype(np.float32))
            else: 
                feature_outputs = pb_utils.Tensor(
                        "embedding_output",
                        np.empty((0, 512), dtype=np.float32)
                )
                coordinates = pb_utils.Tensor(
                        "coordinates", np.empty((0, 4), dtype=np.uint16)
                )
                cropped_faces = pb_utils.Tensor(
                        "cropped_faces", np.empty((0, 160,160,3), dtype=np.float32)
                )
                                                  
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    cropped_faces,
                    coordinates,
                    feature_outputs])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass
