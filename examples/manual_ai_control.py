import asyncio
import cv2
import time
from ultralytics import YOLO
from mavsdk import System
from mavsdk.offboard import PositionNedYaw
from mavsdk.offboard import (Attitude, VelocityNedYaw)

# 初始化YOLO模型
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# 初始化无人机状态
drone_detected = False


async def takeoff(drone):
    print("-- Arming")
    await drone.action.arm()
    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(5)  # 等待几秒钟以确保起飞完成

async def get_current_yaw(drone):
    async for heading in drone.telemetry.heading():
        return heading.heading_deg  # 提取航向角度

async def turn_to_north(drone):
    print("-- Go North 2 m/s, turn to face East")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
    await drone.offboard.start()  # 启动 Offboard 模式
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -5.0, 0.0))
    await asyncio.sleep(2)  # 等待几秒钟以确保起飞完成
async def control_drone(drone, bbox):
    """控制无人机进行追踪"""
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    img_center_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2
    img_center_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2

    # 计算偏移
    offset_x = (center_x - img_center_x) / img_center_x
    offset_y = (center_y - img_center_y) / img_center_y
    gain = 2.0  # 根据需要调整
    roll = max(-1.0, min(1.0, float(offset_x) * gain))
    pitch = max(-1.0, min(1.0, float(-offset_y) * gain))
    throttle = float(1)
    yaw = float(0)

    try:
        await drone.manual_control.set_manual_control_input(pitch, roll, throttle, yaw)
    except asyncio.CancelledError:
        print("-- asyncio.CancelledError!")
        pass


    # 等待 0.1 秒
    await asyncio.sleep(0.1)


async def main():
    drone = System(mavsdk_server_address='localhost', port=50051)
    await drone.connect(system_address="udp://192.168.255.10:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break


    await turn_to_north(drone)
    await takeoff(drone)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        start_time = time.time()
        results = model(source=frame, save=False)
        end_time = time.time()

        # 获取 class_ids, scores, boxes
        class_ids = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # 打印结果
        print(f'Class IDs: {class_ids}')
        print(f'Scores: {scores}')
        print(f'Boxes: {boxes}')
        print(f'Inference Time: {end_time - start_time:.4f} seconds')

        # 检查无人机是否被检测到
        for i, class_id in enumerate(class_ids):
            bbox = boxes[i]
            bbox = tuple(map(int, bbox))  # 转换为整数
            x_min, y_min, x_max, y_max = bbox

            # 如果检测到无人机
            if class_id == 0 and scores[i] > 0.5:  # 假设类别 ID 0 表示无人机
                drone_detected = True
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # 红色
                await control_drone(drone, bbox)
            else:
                # 绘制其他物体的边界框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # 绿色

        cv2.imshow("window", frame)
        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
