from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
import time

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize SORT tracker
tracker = Sort()

# Load video
cap = cv2.VideoCapture(r"C:\Users\prajw\Documents\ResoluteAi_Internship\test1.mp4")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Define class names of interest
class_names_of_interest = ['car', 'truck']

# Initialize counts for cars and trucks
class_counts = {name: 0 for name in class_names_of_interest}

# Initialize counts for up and down directions
direction_counts = {'up': 0, 'down': 0}

# Initialize frame vehicle counts and peak time variables
frame_vehicle_counts = []
peak_time = 0
peak_count = 0

# Track objects to avoid double counting and to calculate speed
object_id_to_class = {}
object_id_to_positions = {}  # Stores positions of objects
object_id_to_times = {}  # Stores the time of each position
object_id_to_direction = {}  # Stores the direction of each object

# Real-world conversion factors (example values, adjust as per your calibration)
pixels_per_meter = 10  # Example: 10 pixels represent 1 meter in the real world

# Define the line for up and down direction detection
line_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current time
    current_time = time.time()

    # Perform detection
    results = model(frame)

    # Collect detections
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name in class_names_of_interest:
                # Append detection to list
                detections.append([x1, y1, x2, y2, conf])

    # Convert detections to NumPy array
    detections = np.array(detections)

    # Update tracker with new detections
    tracked_objects = tracker.update(detections)

    # Count the number of vehicles in the current frame
    frame_vehicle_count = 0

    # Draw tracking results on frame
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj[:5])

        # Calculate the centroid of the bounding box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Calculate speed if we have previous position and time
        if track_id in object_id_to_positions:
            old_cx, old_cy = object_id_to_positions[track_id]
            old_time = object_id_to_times[track_id]

            # Calculate distance traveled in pixels
            distance_px = np.sqrt((cx - old_cx) ** 2 + (cy - old_cy) ** 2)

            # Convert to meters
            distance_m = distance_px / pixels_per_meter

            # Calculate time difference
            time_diff = current_time - old_time

            # Calculate speed in m/s and convert to km/h
            speed_mps = distance_m / time_diff
            speed_kmph = speed_mps * 3.6

            # Draw speed on frame
            label = f'{model.names[cls]}: {speed_kmph:.2f} km/h'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Determine the direction of movement (up or down)
            if track_id not in object_id_to_direction:
                if old_cy < cy:
                    direction_counts['down'] += 1
                    object_id_to_direction[track_id] = 'down'
                else:
                    direction_counts['up'] += 1
                    object_id_to_direction[track_id] = 'up'

        # Update position and time for this track ID
        object_id_to_positions[track_id] = (cx, cy)
        object_id_to_times[track_id] = current_time

        # Count the vehicle if it's newly tracked
        if track_id not in object_id_to_class:
            object_id_to_class[track_id] = class_name
            class_counts[class_name] += 1

        # Increment the vehicle count for the frame
        frame_vehicle_count += 1

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Store the vehicle count for the frame
    frame_vehicle_counts.append(frame_vehicle_count)

    # Check if this frame has the peak vehicle count
    if frame_vehicle_count > peak_count:
        peak_count = frame_vehicle_count
        peak_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Time in seconds

    # Create a GUI box at the top of the frame
    gui_box_height = 70
    cv2.rectangle(frame, (0, 0), (frame.shape[1], gui_box_height), (0, 0, 0), -1)

    # Display class counts and direction counts on the GUI box
    text = (f"Cars: {class_counts['car']} | Trucks: {class_counts['truck']} | "
            f"Up: {direction_counts['up']} | Down: {direction_counts['down']} | "
            f"Peak Traffic Time: {peak_time:.2f}s")
    cv2.putText(frame, text, (10, gui_box_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Write the frame into the output video file
    out.write(frame)

    # Display the resulting frame (optional, can comment out if not needed)
    # cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Save the video file
cv2.destroyAllWindows()
