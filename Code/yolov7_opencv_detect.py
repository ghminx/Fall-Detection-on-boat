#%%
import cv2
import numpy as np
import torch

# Load YOLOv7 model
model = torch.hub.load(r'C:/Users/admin/Desktop/project/yolo/yolov7-segmentation', 'custom', 'C:/Users/admin/Desktop/project/yolo/yolov7-segmentation/best.pt', source='local')

# Define the list of classes or labels
classes = ['Fall', 'Stand']

# OpenCV webcam capture
cap = cv2.VideoCapture(0)  # Use the appropriate index for your webcam

while True:
    ret, frame = cap.read()  # Capture a frame from the webcam

    # Perform object detection
    with torch.no_grad():
        # Convert the frame to a torch tensor
        img = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        img = img.unsqueeze(0)

        # Pass the frame through the YOLOv7 model
        pred = model(img)[0]

        # Extract the class indices, scores, and bounding boxes from the prediction
        class_indices = pred[..., 5:].argmax(1).flatten()
        scores = pred[..., 4].flatten()
        bboxes = pred[..., :4].reshape(-1, 4)

        # Filter the detections based on confidence threshold
        conf_thresh = 0.5  # Set your desired confidence threshold here
        mask = scores > conf_thresh
        detections = bboxes[mask]
        detection_classes = [class_indices[i] for i, m in enumerate(mask) if m]

        # Draw bounding boxes on the frame
        for bbox, cls_idx in zip(detections, detection_classes):
            x, y, w, h = bbox.tolist()
            cls_name = classes[cls_idx]
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, cls_name, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Webcam Object Detection', frame)  # Show the frame with detections

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

# %%
