import redis
import time
import torch
from polymetis import RobotInterface
import json
import numpy as np


def main(
    z_height: float = 0.3,
    redis_host: str = "localhost",
    polymetis_ip: str = "10.10.10.210",
    rate_limit_hz: int = 50,
):
    r = redis.Redis(redis_host, decode_responses=True)

    cmd_stream = "cart_cmd"
    obs_stream = "real_robot_obs"

    last_id = "$"

    robot = RobotInterface(ip_address=polymetis_ip)

    robot.go_home()
    print("Ready.")

    # while not r.xread({cmd_stream: last_id}, block=1000, count=1):
    #    print("Waiting for control signals")
    x = None
    y = None

    goal_quat = torch.Tensor([0.92387975, 0.3826829, 0.0, 0.0])
    first_cmd = True
    try:
        while True:
            start = time.perf_counter()
            if rate_limit_hz:
                time.sleep(1 / rate_limit_hz)
            messages = r.xread({cmd_stream: last_id}, block=50, count=1)
            redis_time = start - time.perf_counter()
            if messages:
                message_id, payload = messages[0][1][-1]
                # for fast sources, simply waiting for a new message is better
                last_id = message_id  # Update last_id for the next iteration
                # robot.start_joint_impedance()
                if payload["cmd"] == "GOTO":
                    # Calculate latency
                    message_time = float(message_id.split("-")[0])
                    current_time = float(time.time() * 1000)
                    latency = current_time - message_time
                    x = float(payload["x"])
                    y = float(payload["y"])

                    goal_pos = torch.Tensor([x, y, z_height])
                    if first_cmd:
                        robot.move_to_ee_pose(
                            position=goal_pos, orientation=goal_quat, time_to_go=2.0
                        )
                        robot.start_joint_impedance()
                        first_cmd = False
                    else:
                        start = time.perf_counter()
                        robot.update_desired_ee_pose(goal_pos, goal_quat)
                        update_time = time.perf_counter() - start
                        # print(f"x: {x:.2f}, y: {y:.2f}, Latency: {latency:.2f} ms")
                        print(
                            f"x: {x:.2f}, y: {y:.2f}, Inverse update time: {(1/update_time):.2f}, Inverse Redis Time: {(1/redis_time):.2f}"
                        )
                else:
                    assert (
                        payload["cmd"] == "RESET"
                    ), f"Unknown Command: {payload['cmd']}"
                    robot.start_joint_impedance()
                    ee_pos, _ = robot.get_ee_pose()
                    ee_pos[2] = 0.35
                    robot.move_to_ee_pose(
                        position=ee_pos, orientation=goal_quat, time_to_go=2.0
                    )
                    robot.go_home()
                    print("RESET: Please move box and press enter to confirm.")
                    input()
                    res = r.xrevrange("box_tracking", "+", "-", count=1)
                    assert res
                    _, payload = res[0]  # type: ignore
                    transform = json.loads(payload["transform"])
                    transform = np.array(transform).reshape(4, 4)
                    x, y = transform[:2, 3].flatten()
                    goal_pos = torch.Tensor([x, y, 0.35])
                    robot.move_to_ee_pose(
                        position=goal_pos, orientation=goal_quat, time_to_go=2.0
                    )
                    print("Press Enter to confirm position")
                    input()
                    print(f"x: {x:.2f}, y: {y:.2f}, RESET DONE")
                    goal_pos = torch.Tensor([x, y, z_height])
                    robot.move_to_ee_pose(
                        position=goal_pos, orientation=goal_quat, time_to_go=2.0
                    )
                    robot.start_joint_impedance()
                    r.xadd(obs_stream, {"reset": 1, "x": x, "y": y})

            else:
                if x is None or y is None:
                    continue
                goal_pos = torch.Tensor([x, y, z_height])
                robot.update_desired_ee_pose(goal_pos, goal_quat)
                print("No message received for 50ms. Exiting.")
    finally:
        robot.go_home()


if __name__ == "__main__":
    import typer

    typer.run(main)
