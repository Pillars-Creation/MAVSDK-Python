import cv2
import sys
# from object_detection import ObjectDetection
from ultralytics import YOLO
import time

# 初始化变量
tracker = None
drone_detected = False
nmsThreshold = 0.4
confThreshold = 0.5
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# 使用摄像头，索引为0（通常是笔记本内置摄像头）
cap = cv2.VideoCapture(0)

# 检查是否成功打开摄像头
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # 从摄像头读取帧
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    if not drone_detected:
        # 检测无人机
        # 记录开始时间
        start_time = time.time()
        results = model(source=frame, save=False)
        # 记录结束时间
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

        # (class_ids, scores, boxes) = od.detect(frame)

        for i, class_id in enumerate(class_ids):
            if class_id == 0:  # 假设类别ID 4 表示无人机
                drone_detected = True
                bbox = boxes[i]
                # 转换边界框格式为 (x, y, width, height)
                x_min, y_min, x_max, y_max = map(int, bbox)  # 转换为整数
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                tracker = cv2.TrackerMIL_create()

                # 初始化追踪器
                ok = tracker.init(frame, bbox)
                if ok:
                    print("无人机检测到并开始跟踪！")
                else:
                    print("无法初始化跟踪器")
                break

    else:
        # 使用追踪器跟踪无人机
        success, bbox = tracker.update(frame)
        saved_bbox = bbox  # 保存边界框

    # success = True
        if success:
            # 跟踪成功，绘制边界框
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            # 跟踪失败
            cv2.putText(frame, "Lost", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # 显示帧
    cv2.imshow("Camera Feed", frame)
    # 检查退出条件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
