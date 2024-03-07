import cv2
from ultralytics import YOLO

model_path = "runs/pose/train/weights/last.pt"
model = YOLO(model_path)

SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def draw_skeleton(frame, results):
    for result in results:
        keypoints = result.keypoints.data

        for start_idx, end_idx in SKELETON_CONNECTIONS:
            start_point = keypoints[0][start_idx][:2]
            end_point = keypoints[0][end_idx][:2]

            if keypoints[0][start_idx][2] > 0.5 and keypoints[0][end_idx][2] > 0.5:
                cv2.line(
                    frame,
                    (int(start_point[0]), int(start_point[1])),
                    (int(end_point[0]), int(end_point[1])),
                    (255, 0, 0),
                    2,
                )

        for keypoint in keypoints[0]:
            x, y, conf = keypoint
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    draw_skeleton(frame, results)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
