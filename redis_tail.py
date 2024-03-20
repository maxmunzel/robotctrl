import redis
import json
import numpy as np
from typing import Tuple


def main(
    redis_host: str = "localhost",
):
    r = redis.Redis(redis_host, decode_responses=True)

    cmd_stream = "cart_cmd"
    obs_stream = "real_robot_obs"

    last_ids = {
        cmd_stream: "$",
        obs_stream: "$",
        "ack": "$",
    }

    while True:
        messages = r.xread(last_ids, block=0, count=1)
        if messages:
            topic, body = messages[0]
            message_id, payload = body[0]
            last_ids[topic] = message_id
            timestamp = int(message_id.split("-")[0]) % 100_000
            timestamp /= 1000
            if topic == "cart_cmd" and payload["cmd"] == "GOTO":
                payload = f'x={float(payload["x"]):2.2f} y={float(payload["y"]):2.2f} {payload.get("src" "?")}'
            print(topic, "\t", timestamp, payload)


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
