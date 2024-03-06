from ultralytics import YOLO
import cv2

model_path = "runs/pose/train/weights/last.pt"
image_path = "test/000000568213.jpg"
img = cv2.imread(image_path)

# Load the model
model = YOLO(model_path)

# Perform inference
results = model(image_path)[0]

# Iterate through results and keypoints
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

# Since we're in a script, cv2.imshow might not work as expected. Consider saving the image or using an appropriate display method for your environment.
cv2.imwrite("annotated_image.jpg", img)
