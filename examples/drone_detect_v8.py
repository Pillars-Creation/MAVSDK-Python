import cv2
import time
import numpy as np
# from object_detection import ObjectDetection
from ultralytics import YOLO


# od = ObjectDetection()
model = YOLO("yolov8n.pt")
# cap = cv2.VideoCapture("../wrj.mp4")
# 使用摄像头，索引为0（通常是笔记本内置摄像头）
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # (class_ids, scores, boxes) = od.detect(frame)
    start_time = time.time()
    results = model(source=frame, save=False)
    end_time = time.time()
    # 计算推理耗时
    inference_time = end_time - start_time

    # 获取 class_ids, scores, boxes
    class_ids = results[0].boxes.cls.cpu().numpy()  # 类别 ID
    scores = results[0].boxes.conf.cpu().numpy()  # 置信度分数
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 边界框坐标 (x_min, y_min, x_max, y_max)

    # 打印结果
    print(f'Class IDs: {class_ids}')
    print(f'Scores: {scores}')
    print(f'Boxes: {boxes}')
    print(f'Inference Time: {inference_time:.4f} seconds')

    # 如果有检测结果
    if len(class_ids) > 0:
        for i, class_id in enumerate(class_ids):
            # 将 (x_min, y_min, x_max, y_max) 转换为 (x, y, w, h)
            x_min, y_min, x_max, y_max = boxes[i]
            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)

            # 如果检测到的类别是 'airplane' (假设类别 ID 为 0)
            if class_id == 4:
                color = (0, 0, 255)  # 红色
            else:
                color = (0, 255, 0)  # 绿色

            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("window", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()