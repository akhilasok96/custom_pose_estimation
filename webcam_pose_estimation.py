import cv2
from ultralytics import YOLO

# Load the YOLO model
model_path = "runs/pose/train/weights/last.pt"
model = YOLO(model_path)

# Define connections between keypoints based on the provided indices
SKELETON_CONNECTIONS = [
    (0, 1),  # Nose to Left Eye
    (0, 2),  # Nose to Right Eye
    (1, 3),  # Left Eye to Left Ear
    (2, 4),  # Right Eye to Right Ear
    (5, 6),  # Left Shoulder to Right Shoulder
    (5, 7),  # Left Shoulder to Left Elbow
    (7, 9),  # Left Elbow to Left Wrist
    (6, 8),  # Right Shoulder to Right Elbow
    (8, 10),  # Right Elbow to Right Wrist
    (5, 11),  # Left Shoulder to Left Hip
    (6, 12),  # Right Shoulder to Right Hip
    (11, 12),  # Left Hip to Right Hip
    (11, 13),  # Left Hip to Left Knee
    (13, 15),  # Left Knee to Left Ankle
    (12, 14),  # Right Hip to Right Knee
    (14, 16),  # Right Knee to Right Ankle
]


def draw_skeleton(frame, results):
    for result in results:
        # Access the .data attribute to get the keypoints array
        keypoints = result.keypoints.data

        # Draw lines for the skeleton connections
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            start_point = keypoints[0][start_idx][:2]  # Get x, y for the start point
            end_point = keypoints[0][end_idx][:2]  # Get x, y for the end point
            # Only draw the line if both keypoints were detected (confidence > 0)
            if keypoints[0][start_idx][2] > 0.5 and keypoints[0][end_idx][2] > 0.5:
                cv2.line(
                    frame,
                    (int(start_point[0]), int(start_point[1])),
                    (int(end_point[0]), int(end_point[1])),
                    (255, 0, 0),
                    2,
                )

        # Draw keypoints
        for keypoint in keypoints[0]:
            x, y, conf = keypoint
            if conf > 0.5:  # Use a confidence threshold to filter keypoints
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)


# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    results = model(frame)[
        0
    ]  # Adjust this line if needed based on how your model returns results

    # Draw the skeleton on the frame
    draw_skeleton(frame, results)

    # Display the resulting frame
    cv2.imshow("Pose Estimation", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
