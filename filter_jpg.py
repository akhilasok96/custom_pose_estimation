import os

labels_dir = "labels/val"
images_dir = "images/val"

label_files = {
    file.split(".")[0] for file in os.listdir(labels_dir) if file.endswith(".txt")
}

for image_file in os.listdir(images_dir):
    if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_name = image_file.rsplit(".", 1)[0]

        if image_name not in label_files:
            image_path = os.path.join(images_dir, image_file)
            try:
                os.remove(image_path)
                print(f"Deleted: {image_path}")
            except Exception as e:
                print(f"Error deleting {image_path}: {e}")

print("Cleanup completed.")
