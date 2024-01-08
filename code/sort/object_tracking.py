import numpy as np
from ultralytics import YOLO
import cv2
import time
import cvzone
import math
from object_detection import ObjectDetection
from sort import *

od = ObjectDetection()
class_names = od.load_class_names()

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# Tracking vehicles roundabout
vehicles_entering = {}
vehicles_elapsed_time = {}

cap = cv2.VideoCapture("../../data/cars.mp4")  # For Video
area = [(400, 300), (690, 300), (710, 425), (150, 425)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = od.detect(frame)
    boxes = result[0].boxes

    detections = np.empty((0, 5))
    for box in boxes:
        (x1, y1, x2, y2) = np.array(box.xyxy[0], dtype=int)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        conf = np.array(boxes.conf[0], dtype=float)
        cls = np.array(boxes.cls[0], dtype=int)
        currentClass = class_names[cls]

        if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)

    # Ước lượng vận tốc
    

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1

        cx, cy  = x1 + w//2, y1 + h//2

        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)

        if result >= 0:
            vehicles_entering[id] = time.time()

        if id in vehicles_entering:
            # result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
            # if result < 0:
            #     elapsed_time = time.time() - vehicles_entering[id]
            #     if id not in vehicles_elapsed_time:
            #         vehicles_elapsed_time[id] = elapsed_time

            #     if id in vehicles_elapsed_time:
            #         elapsed_time = vehicles_elapsed_time[id] + 10**(-5)
            #     # Calc average speed
            #     distance = 5
            #     a_speed_ms = distance/elapsed_time
            #     a_speed_kh = a_speed_ms* 3.6

            #     vehicles_elapsed_time[id] = elapsed_time
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 170, 66), 2)
                # cv2.rectangle(frame, (x1, y1), (x1+100, y1-20), (245, 170, 66), -1)
                # cv2.putText(frame, str(round(a_speed_kh, 2)) + "km/h", (x1, y1-5), 0, 0.5, (255, 255, 255), 2)
                # cv2.circle(frame, (cx, cy), 5, (245, 170, 66), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 170, 66), 2)
            cv2.rectangle(frame, (x1, y1), (x1+100, y1-20), (245, 170, 66), -1)
            cv2.putText(frame, str(id), (x1, y1-5), 0, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (245, 170, 66), -1)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (15, 220, 10), 6)

    # Đếm xe
    # cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    # for result in resultsTracker:
    #     x1, y1, x2, y2, id = result
    #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #     print(result)
    #     w, h = x2 - x1, y2 - y1
    #     cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
    #     cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
    #                        scale=2, thickness=3, offset=10)

    #     cx, cy = x1 + w // 2, y1 + h // 2
    #     cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    #     if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
    #         if totalCount.count(id) == 0:
    #             totalCount.append(id)
    #             cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cv2.putText(frame,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
