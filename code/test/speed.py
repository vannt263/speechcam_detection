
import numpy as np
from ultralytics import YOLO
import math
import cv2
import cvzone
from sort import *
import time

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
cap = cv2.VideoCapture("D:/study/CV/CK/Object-Detection-101/Videos/cars.mp4")  # For Video

mask = cv2.imread("D:/study/CV/CK/Object-Detection-101/Project 1 - Car Counter/mask1.png")

# Khởi tạo đối tượng tracker với các tham số
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Định nghĩa giới hạn theo tọa độ [x1, y1, x2, y2]
first_limit = [300, 340, 700, 340]
last_limit = [0, 597, 673, 597]

# Danh sách lưu trữ các đối tượng đã đếm
total_count = []
start = {}

while True:
    # Đọc frame từ camera
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask) # assuming you have 'mask' defined somewhere

    # Gọi mô hình để nhận diện đối tượng trong vùng quan tâm
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Lấy thông tin về hộp giới hạn (bounding box)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Lọc ra các đối tượng là xe và có độ tin cậy lớn hơn 0.3
            if currentClass == 'car' and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Cập nhật thông tin tracking của đối tượng
    resultsTracker = tracker.update(detections)

    # Vẽ đường giới hạn trên ảnh
    cv2.line(img, (first_limit[0], first_limit[1]), (first_limit[2], first_limit[3]), (0, 0, 255), 5)
    cv2.line(img, (last_limit[0], last_limit[1]), (last_limit[2], last_limit[3]), (0, 0, 255), 5)

    # Hiển thị thông tin vận tốc và tính toán vận tốc trung bình
    for result in resultsTracker:
        x1, y1, x2, y2, object_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1+y2) // 2

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        if first_limit[1] < cy + 15 and cy - 15 < first_limit[1] :
            start[object_id] = time.time()
        if object_id in start:
            if last_limit[1] < cy + 15 and last_limit[1] > cy -15:
                    elapsed_time = time.time() - start[object_id]
                    if total_count.count(object_id)==0:
                        total_count.append(object_id) 
                        # Lấy thông tin vận tốc
                        speed = 20/ elapsed_time *3.6
                        # Hiển thị thông tin vận tốc
                        speed_text = f'{speed:.2f} km/h'
                        cv2.circle(img,(cx,cy),4,(0,0,255),-1)
                        cv2.putText(img,str(object_id),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                        cv2.putText(img,speed_text,(x2, y2),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    # Hiển thị ảnh và chờ nhấn phím
    cv2.imshow("Image", img)
    cv2.waitKey(1)
