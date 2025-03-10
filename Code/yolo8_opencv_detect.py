#%%

# 1. 기본 웹캠 실행
import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('C:/Users/rmsgh/Desktop/project/yolo/yolo_result/yolov8s-seg_123/weights/best.pt')

# FPS 계산 변수
prev_frame_time = 0
new_frame_time = 0

# fall detection 변수
fall_count = 0
threshold = 3 * 20  # 초 * FPS

# Open the video file
cap = cv2.VideoCapture(0)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = "C:/Users/rmsgh/Desktop/project/result_video/yolo8-seg-123-recode_cam.avi"  # Set the desired output folder and file name
out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the videoq
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        annotated_frame = results[0].plot()

        # FPS 화면 출력
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(annotated_frame, fps_text, (annotated_frame.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        
        # Write the frame to the output file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("Fall-Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, video writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()


#%%

# 2. Fall Detection of Video
import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('C:/Users/rmsgh/Desktop/project/yolo/yolo_result/yolov8s-seg_123/weights/best.pt')

video_path = "C:/Users/rmsgh/Desktop/project/test/test2.mp4"

# FPS 계산 변수
prev_frame_time = 0
new_frame_time = 0
# Open the video file
cap = cv2.VideoCapture(video_path)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_path = "C:/Users/rmsgh/Desktop/project/result_video/yolo8-seg-123-recode.avi"  # Set the desired output folder and file name


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))


fall_count = 0
threshold = 3 * 20  # Assuming 20 FPS, so 3 seconds would be 100 frames

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the videoq
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Check if a fall is detected in the current frame
        fall_detected = any(box.cls.item() == 0 for box in results[0].boxes)


        if fall_detected:
            fall_count += 1
        else:
            fall_count = 0

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Fall Detection 메세지 출력
        if fall_count > threshold:
            cv2.putText(annotated_frame, "Fall Detection", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # 캡쳐
            cv2.imwrite("C:/Users/rmsgh/Desktop/project/fall_detection_capture.jpg", annotated_frame)
        
        # FPS 화면 출력
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(annotated_frame, fps_text, (annotated_frame.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        
        # Write the frame to the output file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("Fall-Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, video writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

# %%
# 3. Fall Detection of Webcam
import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('C:/Users/rmsgh/Desktop/project/yolo/yolo_result/yolov8s-seg_123/weights/best.pt')

# FPS 계산 변수
prev_frame_time = 0
new_frame_time = 0
# Open the video file
cap = cv2.VideoCapture(0)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_path = "C:/Users/rmsgh/Desktop/project/result_video/yolo8-seg-123-recode-ccc.avi"  # Set the desired output folder and file name


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))


fall_count = 0
threshold = 3 * 20  # Assuming 20 FPS, so 3 seconds would be 100 frames

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the videoq
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # 현재 프레임에서 Fall이 감지되는지 확인
        fall_detected = any(box.cls.item() == 0 for box in results[0].boxes)


        if fall_detected:
            fall_count += 1
        else:
            fall_count = 0

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # fall_count가 임계값을 초과하면 "Fall Detection" 메시지 표시
        if fall_count > threshold:
            cv2.putText(annotated_frame, "Fall Detection", (150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # FPS 화면 출력
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(annotated_frame, fps_text, (annotated_frame.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        
        # Write the frame to the output file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("Fall-Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, video writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()


# %%
# 4. bbox Fall Detection of Webcam
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('C:/Users/admin/Desktop/project/yolo/yolo_result/yolov8_bbox/weights/best.pt')

# FPS 계산 변수
prev_frame_time = 0
new_frame_time = 0
# Open the video file
cap = cv2.VideoCapture(0)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_path = "C:/Users/admin/Desktop/project/yolo/result_video/Yolo_Fall_Detect_WebCam.avi"  # Set the desired output folder and file name


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))


fall_count = 0
threshold = 3 * 20  # Assuming 20 FPS, so 3 seconds would be 100 frames

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the videoq
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # 현재 프레임에서 Fall이 감지되는지 확인
        fall_detected = any(box.cls.item() == 0 for box in results[0].boxes)


        if fall_detected:
            fall_count += 1
        else:
            fall_count = 0

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # fall_count가 임계값을 초과하면 "Fall Detection" 메시지 표시
        if fall_count > threshold:
            cv2.putText(annotated_frame, "Fall Detection", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite("C:/Users/admin/Desktop/project/yolo/result_video/bbox_Yolo_Fall_Detect_WebCam.jpg", annotated_frame)
        
        # FPS 화면 출력
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(annotated_frame, fps_text, (annotated_frame.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        

        
        # Write the frame to the output file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("Fall-Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, video writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()





#%%

# 5. bbox Fall Detection of Video

import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('C:/Users/admin/Desktop/project/yolo/yolo_result/yolov8_bbox/weights/best.pt')

video_path = "C:/Users/admin/Desktop/project/yolo/test/test1.mp4"

# FPS 계산 변수
prev_frame_time = 0
new_frame_time = 0
# Open the video file
cap = cv2.VideoCapture(video_path)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_path = "C:/Users/admin/Desktop/project/yolo/result_video/bbox_Yolo_Fall_Detect_Video.avi"  # Set the desired output folder and file name


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))


fall_count = 0
threshold = 3 * 20  # Assuming 20 FPS, so 3 seconds would be 100 frames

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the videoq
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Check if a fall is detected in the current frame
        fall_detected = any(box.cls.item() == 0 for box in results[0].boxes)


        if fall_detected:
            fall_count += 1
        else:
            fall_count = 0

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # If fall_count exceeds the threshold, display the "Fall Detection" message
        if fall_count > threshold:
            cv2.putText(annotated_frame, "Fall Detection", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite("C:/Users/admin/Desktop/project/yolo/result_video/bbox_Yolo_Fall_Detect_Video.jpg", annotated_frame)
        
        # FPS 화면 출력
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(annotated_frame, fps_text, (annotated_frame.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        
        # Write the frame to the output file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("Fall-Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, video writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()