from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
from typing import Any, List
import numpy as np

class Feature_Extractor():
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build a FaceNet extractor model
        Returns:
            model (Any)
        """
        model = tf.saved_model.load(export_dir="models/saved_model", tags=[tag_constants.SERVING])
        model = model.signatures["serving_default"]

        # Model Warm-up
        print("<-----Feature Extractor Warm-Up------>")
        zeros_array = np.zeros((8, 160, 160, 3))
        zeros_array = tf.constant(zeros_array, dtype=tf.float32)
        tmp = model(zeros_array)['embedding_output'].numpy()
        return model
    
    def pre_processing(self, aligned_faces: List):
        faces = np.array(aligned_faces) / 255.0
        # Calculate mean and standard deviation for each image separately
        faces = tf.constant(faces, dtype=tf.float32)
        embeddings = self.model(faces)['embedding_output'].numpy()
        
        return embeddings

if __name__ == "__main__":
    fe = Feature_Extractor()
    zeros_array = np.zeros((1, 160, 160, 3))
    zeros_array = tf.constant(zeros_array, dtype=tf.float32)
    tmp = fe(zeros_array)['embedding_output'].numpy()
