import cv2
from inference import VitInference

# Image to run inference RGB format
img = cv2.imread('img_2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# set is_video=True to enable tracking in video inference
model_path = 'vitpose-25-l.pth'
yolo_path = 'yolov5s.pt'
model = VitInference(model_path, yolo_path, model_name='l', yolo_size=320, is_video=False)

# Infer keypoints, output is a dict where keys are person ids and values are keypoints (np.ndarray (25, 3): (y, x,
# score))
keypoints = model.inference(img)

img = model.draw(show_yolo=True)  # Returns RGB image with drawings
cv2.imshow('image', img[..., ::-1])
cv2.waitKey()
