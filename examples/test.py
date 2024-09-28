#!/usr/bin/env python3

import asyncio
import keyboard
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw
from mavsdk.offboard import PositionNed

async def run():
    # Create MAVSDK server connection instance
    drone = System(mavsdk_server_address='localhost', port=50051)

    # Connect to the drone via the simulator
    await drone.connect(system_address="udp://192.168.255.10:14540")  # Set to simulator's UDP address

    status_text_task = asyncio.ensure_future(print_status_text(drone))

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()

    # Start the keyboard control task
    control_task = asyncio.ensure_future(control_drone(drone))

    print("-- Drone is now in the air. Use keyboard controls to navigate. Press 'q' to land.")

    try:
        await control_task
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Landing drone.")
    finally:
        print("-- Landing")
        await drone.action.land()
        status_text_task.cancel()
        control_task.cancel()
        try:
            await status_text_task
        except asyncio.CancelledError:
            pass
        try:
            await control_task
        except asyncio.CancelledError:
            pass


async def print_status_text(drone):
    try:
        async for status_text in drone.telemetry.status_text():
            print(f"Status: {status_text.type}: {status_text.text}")
    except asyncio.CancelledError:
        return


async def control_drone(drone):
    # Default yaw angle set to 0 degrees (facing north)
    yaw_angle = 0.0
    # 初始化 Offboard 模式
    # 创建位置对象
    position = PositionNed(0, 0, -10)  # NED 坐标

    # 设置无人机目标位置
    await drone.offboard.set_position_ned(position)

    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(north_m_s=0.0, east_m_s=0.0, down_m_s=-5.0, yaw_deg=yaw_angle)
    )
    # 设置目标位置
    await drone.offboard.start()

    while True:
        try:

            await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(north_m_s=1.0, east_m_s=0.0, down_m_s=0.0, yaw_deg=yaw_angle)
            )
            print("Moving forward")

            await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting
        except asyncio.CancelledError:
            return


if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())
