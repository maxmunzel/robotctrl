import redis
import time
import torch
from polymetis import RobotInterface
import json
import numpy as np
from typing import Tuple


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
    last_cmd = "GOTO"

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
            if rate_limit_hz:
                time.sleep(1 / rate_limit_hz)
            messages = r.xread({cmd_stream: last_id}, block=50, count=1)
            if messages:
                message_id, payload = messages[0][1][-1]
                # for fast sources, simply waiting for a new message is better
                last_id = message_id  # Update last_id for the next iteration
                cmd = payload["cmd"]
                if cmd == "GOTO":
                    if x is None:
                        x = float(payload["x"])
                        y = float(payload["y"])
                    else:
                        x_ = float(payload["x"])
                        y_ = float(payload["y"])
                        dist = ((x - x_) ** 2 + (y - y_) ** 2) ** 0.5
                        limit = 0.05
                        if dist > limit:
                            print("Ignoring GOTO, as it exceeds max speed")
                            continue
                        x, y = x_, y_

                    goal_pos = torch.Tensor([x, y, z_height])
                    if first_cmd:
                        robot.move_to_ee_pose(
                            position=goal_pos, orientation=goal_quat, time_to_go=2.0
                        )
                        robot.start_joint_impedance()
                        first_cmd = False
                    else:
                        robot.update_desired_ee_pose(goal_pos, goal_quat)
                        ee_pos, _ = robot.get_ee_pose()
                        r.xadd("ack", {"x": ee_pos[0], "y": ee_pos[1], "z": ee_pos[2]})
                        print(f"x: {x:.2f}, y: {y:.2f}")
                else:
                    assert cmd == "RESET", f"Unknown Command: {payload['cmd']}"
                    first_cmd = False

                    if last_cmd == "RESET":
                        print("Skipping Double Reset CMD")
                        # Sleep for 3 seconds or the next command
                        r.xread({cmd_stream: last_id}, block=3000, count=1)
                        r.xadd(obs_stream, {"reset": 1, "x": x, "y": y})
                        continue

                    print("RESET: Please move box and press enter to confirm.")
                    # Move robot just outside the box in a vertical movement
                    ee_pos, _ = robot.get_ee_pose()
                    ee_pos[2] = 0.35
                    robot.move_to_ee_pose(
                        position=ee_pos, orientation=goal_quat, time_to_go=2.0
                    )

                    # Go home and wait for box movement
                    robot.go_home()
                    input()

                    # Go above box and wait for confirmation
                    x, y = get_most_recent_box_xy(r)
                    goal_pos = torch.Tensor([x, y, 0.35])
                    robot.move_to_ee_pose(
                        position=goal_pos, orientation=goal_quat, time_to_go=2.0
                    )
                    print("Press Enter to confirm position")
                    input()

                    # Go inside box, start the controller and confirm the reset
                    goal_pos = torch.Tensor([x, y, z_height])
                    robot.move_to_ee_pose(
                        position=goal_pos, orientation=goal_quat, time_to_go=2.0
                    )
                    robot.start_joint_impedance()
                    robot.update_desired_ee_pose(goal_pos, goal_quat)
                    r.xadd(obs_stream, {"reset": 1, "x": x, "y": y})
                    print(f"x: {x:.2f}, y: {y:.2f}, RESET DONE")
                last_cmd = cmd

    finally:
        robot.go_home()


def get_most_recent_box_xy(r: redis.Redis) -> Tuple[float, float]:
    res = r.xrevrange("box_tracking", "+", "-", count=1)
    assert res
    _, payload = res[0]  # type: ignore
    transform = json.loads(payload["transform"])
    transform = np.array(transform).reshape(4, 4)
    x, y = transform[:2, 3].flatten()
    return x, y


if __name__ == "__main__":
    import typer

    typer.run(main)
