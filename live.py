import cv2
import os
from datetime import datetime

def create_photo_directory():
    photo_dir = "captured_photos"
    if not os.path.exists(photo_dir):
        os.makedirs(photo_dir)
    return photo_dir

def capture_photo(frame, photo_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"photo_{timestamp}.jpg"
    filepath = os.path.join(photo_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"Photo saved: {filepath}")

def main():
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    photo_dir = create_photo_directory()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the frame
        cv2.imshow('Camera', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        # Press 'c' to capture photo
        elif key == ord('c'):
            capture_photo(frame, photo_dir)

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()