import urllib.request

# Define the URLs
cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
weights_url = "https://pjreddie.com/media/files/yolov4.weights"
names_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

# Define the local paths
cfg_path = "models/dnn_model/yolov4.cfg"
weights_path = "models/dnn_model/yolov4.weights"
names_path = "models/dnn_model/coco.names"

# Download the files
urllib.request.urlretrieve(cfg_url, cfg_path)
urllib.request.urlretrieve(weights_url, weights_path)
urllib.request.urlretrieve(names_url, names_path)

print("Downloaded YOLOv4 files successfully.")