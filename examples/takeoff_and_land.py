#!/usr/bin/env python3

import asyncio
from mavsdk import System


async def run():

    # drone = System()
    # await drone.connect(system_address="udp://:14540")
    # # 创建 MAVSDK 服务器连接实例
    drone = System(mavsdk_server_address='localhost', port=50051)
    #
    # # 连接到仿真器上的飞控系统
    await drone.connect(system_address="udp://192.168.255.10:14540")  # 设置为仿真器的 UDP 地址

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

    await asyncio.sleep(2)

    print("-- Landing")
    await drone.action.land()

    status_text_task.cancel()


async def print_status_text(drone):
    try:
        async for status_text in drone.telemetry.status_text():
            print(f"Status: {status_text.type}: {status_text.text}")
    except asyncio.CancelledError:
        return


if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())
