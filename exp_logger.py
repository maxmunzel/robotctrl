from typing import List, Dict
import json
from pydantic import BaseModel
import redis
import typer


class Config(BaseModel):
    experiments: List[str]


class EpisodeFeedback(BaseModel):
    box_goal_pos_dist: float
    box_goal_rot_dist: float
    episode_energy: float
    is_success: int
    num_steps: int
    reward: float


# Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0)

# Function to read events from Redis and parse them
def read_episode_feedback() -> EpisodeFeedback:
    stream = r.xread({"episode_feedback": "$"}, count=1, block=0)

    message = stream[0]
    stream_id, messages = message
    msg_id, msg_data = messages[0]

    event = EpisodeFeedback.validate(
        {k.decode(): float(v.decode()) for k, v in msg_data.items()}
    )
    return event


class Run(BaseModel):
    feedback: EpisodeFeedback
    experiment: str


class Result(BaseModel):
    runs: List[Run]
    agent: str


def main(n_reps: int, agent: str, config: str):
    output = f"results-{agent}.json"
    with open(config) as f:
        config = Config.validate(json.load(f))

    # todo: start env

    runs = list()

    try:
        for exp in config.experiments:
            for i in range(n_reps):
                print(f"Please Prepare Experiment {exp}. [Iteration {i+1}/{n_reps}]")
                feedback = read_episode_feedback()
                runs.append(Run.validate({"feedback": feedback, "experiment": exp}))
                print(feedback.model_dump_json(indent=2))
                print()
    finally:
        result = Result.validate({"runs": runs, "agent": agent})
        with open(output, "w") as f:
            f.write(result.model_dump_json(indent=2))


# Example usage
if __name__ == "__main__":
    typer.run(main)
