import tritonclient.grpc as grpc_client
import contextlib
import numpy as np
import cv2
import time

class triton_client():
    def __init__(self, host = "10.5.0.5", port ="8001"):
        self.host = host
        self.port = port
        self.triton_client = grpc_client.InferenceServerClient(url=":".join([self.host, self.port]))
        self.connected = False

    def connect(self, img:np.array): 
        while not self.connected:
            with contextlib.suppress(Exception):
                assert self.triton_client.is_model_ready('pipeline')
                self.connected = True
                print(f'Client connected to {self.host}:{self.port} succesfully')
            time.sleep(1)

    def query(self, frame: np.array) -> np.array:  
        input = grpc_client.InferInput('input_image', frame.shape, datatype='UINT8')
        input.set_data_from_numpy(frame)
        embeddings = grpc_client.InferRequestedOutput('embedding_output')
        coordinates = grpc_client.InferRequestedOutput('coordinates')
        result = self.triton_client.infer(model_name='pipeline', inputs=[input], outputs=[embeddings, coordinates])
        embeddings = result.as_numpy('embedding_output')
        coordinates = result.as_numpy('coordinates')
        return embeddings, coordinates

if __name__ == "__main__":
    fe = triton_client()
    zeros_array = np.zeros((1, 160, 160, 3))
    tmp = fe(zeros_array)['embedding_output'].numpy()


