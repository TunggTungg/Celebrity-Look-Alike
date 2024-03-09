import onnxruntime as ort
import cv2
import numpy as np
"""INFO - Model inputs: ['input_image']
INFO - Model outputs: ['embedding_output']"""

# Load the ONNX model
onnx_model_path = 'models/model.onnx'
ort_session = ort.InferenceSession(onnx_model_path)




# Prepare input data
image_path = 'test_image.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
image = cv2.resize(image, (160, 160))  # Resize to match model input size
image = image.astype(np.float32)
mean, std = image.mean(), image.std()
image = (image - mean) / std
# Prepare input data
input_data = np.expand_dims(image, axis=0) 

# Run inference
output = ort_session.run(None, {'input_image': input_data})[0]
print(output)
# 'output' will contain the model's output(s), which you can use as needed
