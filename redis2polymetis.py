import redis
import time
import torch
from polymetis import RobotInterface


def main(
    z_height: float = 0.3,
    redis_host: str = "localhost",
    polymetis_ip: str = "10.10.10.210",
):
    r = redis.Redis(redis_host, decode_responses=True)

    stream_name = "cart_cmd"

    last_id = "$"

    robot = RobotInterface(ip_address=polymetis_ip)

    robot.go_home()

    while not r.xread({stream_name: last_id}, block=1000, count=1):
        print("Waiting for control signals")

    first_cmd = True
    try:
        while True:
            messages = r.xread({stream_name: last_id}, block=1000, count=1)[0][1]

            if messages:
                message_id, payload = messages[-1]
                last_id = message_id  # Update last_id for the next iteration

                # Calculate latency
                message_time = float(message_id.split("-")[0])
                current_time = float(time.time() * 1000)
                latency = current_time - message_time
                x = float(payload["x"])
                y = float(payload["y"])

                goal_pos = torch.Tensor([x, y, z_height])
                goal_quat = None  # torch.Tensor([0, 0, 1, 0])
                if first_cmd:
                    robot.move_to_ee_pose(
                        position=goal_pos, orientation=goal_quat, time_to_go=2.0
                    )
                    robot.start_cartesian_impedance()
                else:
                    robot.update_desired_ee_pose(goal_pos, goal_quat)

                print(f"x: {x:.2f}, y: {y:.2f}, Latency: {latency:.2f} ms")

            else:
                print("No message received for 1000ms. Exiting.")
                break
    finally:
        robot.go_home()


if __name__ == "__main__":
    import typer

    typer.run(main)
