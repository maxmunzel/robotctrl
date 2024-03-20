from typing import List, Union
import json
import time
import numpy as np
from pydantic import BaseModel
import redis
import typer
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import (
    affine_matrix_to_xpos_and_xquat,
)


class Config(BaseModel):
    experiments: List[str]


class EpisodeFeedback(BaseModel):
    box_goal_pos_dist: float
    box_goal_rot_dist: float
    episode_energy: float
    is_success: int
    num_steps: int
    reward: float


class Mocap(BaseModel):
    time_redis: float
    pos: List[float]
    quat: List[float]


class Run(BaseModel):
    feedback: EpisodeFeedback
    start_time: float
    end_time: float
    start_pos: Mocap
    end_pos: Mocap
    end_pos_delayed: Mocap
    experiment: str
    rep: int


class Result(BaseModel):
    runs: List[Run]
    agent: str


class RealRobotObs(BaseModel):
    reset: int
    x: float
    y: float


# Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0)


def read_episode_feedback() -> EpisodeFeedback:
    stream = r.xread({"episode_feedback": "$"}, count=1, block=0)

    message = stream[0]
    stream_id, messages = message
    msg_id, msg_data = messages[0]

    event = EpisodeFeedback.validate(
        {k.decode(): float(v.decode()) for k, v in msg_data.items()}
    )
    return event


def check_mocap() -> Mocap:
    res = r.xrevrange("box_tracking", "+", "-", count=1)
    assert res
    msg_id, payload = res[0]  # type: ignore
    transform = json.loads(payload["transform"])
    transform = np.array(transform).reshape(4, 4)
    time_redis = float(msg_id.split("-")[0]) / 1000
    box_pos, box_quat = affine_matrix_to_xpos_and_xquat(transform)

    return Mocap.validate(
        {
            "pos": box_pos.tolist(),
            "quat": box_quat.tolist(),
            "time_redis": time_redis,
        }
    )


def get_obs_or_feedback() -> Union[EpisodeFeedback, RealRobotObs]:
    stream = r.xread({"episode_feedback": "$", "real_robot_obs": "$"}, count=1, block=0)

    message = stream[0]
    stream_id, messages = message
    msg_id, msg_data = messages[0]

    if stream_id == "episode_feedback":
        event = EpisodeFeedback.validate(
            {k.decode(): float(v.decode()) for k, v in msg_data.items()}
        )
        return event

    else:
        assert stream_id == "real_robot_obs"
        x = float(msg_data["x"])
        y = float(msg_data["y"])
        reset = int(msg_data["reset"])
        event = RealRobotObs.validate({"x": x, "y": y, "reset": reset})
        return event


def collect_run(exp: str, rep: int, reps_per_exp: int) -> Run:
    print(f"Please Prepare Experiment {exp}. [Iteration {rep+1}/{reps_per_exp}]")
    print("Waiting for reset")
    while True:
        event = get_obs_or_feedback()
        start_time = None
        start_pos = None
        if isinstance(event, RealRobotObs):
            print("Recieved Reset.")
            start_time = time.time()
            start_pos = check_mocap()
        else:
            assert isinstance(event, EpisodeFeedback)
            feedback = event
            break
    end_time = time.time()
    end_pos = check_mocap()
    time.sleep(1)
    end_pos_delayed = check_mocap()
    return Run.validate(
        {
            "feedback": feedback,
            "start_time": start_time,
            "end_time": end_time,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "end_pos_delayed": end_pos_delayed,
            "experiment": exp,
            "rep": rep,
        }
    )


def main(n_reps: int, agent: str, config: str):
    output = f"results-{agent}.json".replace("/", "_")
    with open(config) as f:
        config = Config.validate(json.load(f))

    # todo: start env

    runs = list()

    try:
        for exp in config.experiments:
            for i in range(n_reps):
                run = collect_run(exp, i, n_reps)
                runs.append(run)
                print(run.model_dump_json(indent=2))
    finally:
        result = Result.validate({"runs": runs, "agent": agent})
        with open(output, "w") as f:
            f.write(result.model_dump_json(indent=2))


# Example usage
if __name__ == "__main__":
    typer.run(main)
