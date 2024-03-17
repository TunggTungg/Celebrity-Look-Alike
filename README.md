
# **Celebrity Look-alike Detector**

## Demo
<p align="center">
  <img src="demo/video.gif"><br/>
  <i>Project deployment.</i>
</p>

## Overview
  The Celebrity Look-alike Detector is an innovative project designed to provide users with an entertaining and engaging experience by comparing their facial features to those of celebrities. Leveraging state-of-the-art deep learning models and efficient deployment techniques, the system swiftly analyzes user-provided images to find the closest match among a database of celebrities.

## Technologies Used
  * [YOLOv8-face](https://github.com/derronqi/yolov8-face) for face detection.
  * [FaceNet-512](https://github.com/timesler/facenet-pytorch) for facial feature extraction.
  * [Elasticsearch](https://www.elastic.co/) for storing and querying celebrity facial features.
  * [Triton serving](https://github.com/triton-inference-server/server) for efficient model serving.
  * [FastAPI](https://fastapi.tiangolo.com/) for web deployment.


## Run Locally
Clone the project

```bash
  git clone https://github.com/TunggTungg/Celebrity-Look-Alike.git
```

Go to the project directory

```bash
  cd Celebrity-Look-Alike
  docker-compose up --build
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)