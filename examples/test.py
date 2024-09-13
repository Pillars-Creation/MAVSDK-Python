#!/usr/bin/env python3

import asyncio
import keyboard
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

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


async def print_status_text(drone):
    try:
        async for status_text in drone.telemetry.status_text():
            print(f"Status: {status_text.type}: {status_text.text}")
    except asyncio.CancelledError:
        return


async def control_drone(drone):
    # Default yaw angle set to 0 degrees (facing north)
    yaw_angle = 0.0

    while True:
        try:
            if keyboard.is_pressed('w'):  # Move forward
                await drone.offboard.set_velocity_ned(
                    velocity_ned=[1.0, 0.0, 0.0]  # Adjust speed as needed
                )
                print("KeyboardInterrupt up")
            elif keyboard.is_pressed('s'):  # Move backward
                await drone.offboard.set_velocity_ned(
                    velocity_ned=[-1.0, 0.0, 0.0]
                )
                print("KeyboardInterrupt down")
            elif keyboard.is_pressed('a'):  # Move left
                await drone.offboard.set_velocity_ned(
                    velocity_ned=[0.0, 1.0, 0.0]
                )
                print("KeyboardInterrupt left")
            elif keyboard.is_pressed('right'):  # Move right
                await drone.offboard.set_velocity_ned(
                    velocity_ned=[0.0, -1.0, 0.0]
                )
                print("KeyboardInterrupt right")
            elif keyboard.is_pressed('up'):  # Move up
                await drone.offboard.set_velocity_ned(
                    velocity_ned=[0.0, 0.0, 1.0]
                )
                print("KeyboardInterrupt up")
            elif keyboard.is_pressed('down'):  # Move down
                await drone.offboard.set_velocity_ned(
                    velocity_ned=[0.0, 0.0, -1.0]
                )
                print("KeyboardInterrupt down")
            else:
                # Stop the drone if no key is pressed
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(north_m_s=0.0, east_m_s=0.0, down_m_s=0.0, yaw_deg=yaw_angle)
                )
                print("no key is pressed")
            await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting
        except asyncio.CancelledError:
            return


if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())
