from ultralytics import YOLO
import cv2

model_path = "runs/pose/train/weights/last.pt"
image_path = "test/000000568213.jpg"
img = cv2.imread(image_path)

model = YOLO(model_path)

results = model(image_path)[0]

for result in results:
    for keypoint_indx, keypoint in enumerate(result.keypoints.numpy()):
        cv2.putText(
            img,
            str(keypoint_indx),
            (int(keypoint[0]), int(keypoint[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

cv2.imwrite("annotated_image.jpg", img)
