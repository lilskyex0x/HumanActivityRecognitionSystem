import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model we just trained
try:
    model = load_model('human_activity_model.h5')
except:
    print("Error: Could not find 'human_activity_model.h5'. Run train_model.py first!")
    exit()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --- CHANGED LINE BELOW ---
# Replace 'your_video_file.mp4' with the actual name of your video
cap = cv2.VideoCapture('running.mp4')

if not cap.isOpened():
    print("Error: Could not open the video file. Check the filename and path!")
    exit()

prev_gray = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. Pre-processing (Requirement: Gaussian Filtering)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 2. Human Detection (Requirement: HOG)
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))

    # 3. Motion Analysis (Requirement: Optical Flow)
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    else:
        mag = np.zeros(gray.shape)

    prev_gray = gray

    for (x, y, w, h) in boxes:
        # Calculate motion inside the human bounding box
        person_motion = np.mean(mag[y:y + h, x:x + w])

        # --- MOTION FLOW LABEL LOGIC ---
        if person_motion > 2.5:  # Adjust this number if it's too sensitive
            motion_label = "RUNNING"
            color = (0, 0, 255)  # Red
        elif person_motion > 0.5:
            motion_label = "WALKING"
            color = (0, 255, 255)  # Yellow
        else:
            motion_label = "STANDING"
            color = (0, 255, 0)  # Green

        # 4. Joint Detection (Requirement: Harris Corners)
        roi_gray = gray[y:y + h, x:x + w]
        dst = cv2.cornerHarris(np.float32(roi_gray), 2, 3, 0.04)
        frame[y:y + h, x:x + w][dst > 0.01 * dst.max()] = [0, 0, 255]

        # Draw the Result Label on the screen
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"MOTION: {motion_label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the Flow Result Window
    flow_vis = cv2.applyColorMap(np.uint8(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)), cv2.COLORMAP_JET)
    cv2.imshow('Motion Flow Visualizer', flow_vis)
    cv2.imshow('Final System Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break