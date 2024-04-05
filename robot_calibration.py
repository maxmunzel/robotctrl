import marimo

__generated_with = "0.3.4"
app = marimo.App(width="medium")


@app.cell
def __():
    from exp_logger import Result, Mocap, Run
    from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import (
        rotation_distance,
        affine_matrix_to_xpos_and_xquat,
    )
    from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import BoxPushingDense
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    import numpy as np
    from typing import Tuple, List
    from redis import Redis
    import matplotlib.gridspec as gridspec
    import sys
    import io

    return (
        BoxPushingDense,
        List,
        Mocap,
        Redis,
        Result,
        Rotation,
        Run,
        Tuple,
        affine_matrix_to_xpos_and_xquat,
        gridspec,
        io,
        np,
        plt,
        rotation_distance,
        sys,
    )


@app.cell
def __(Result):
    def load_exp(path: str) -> Result:
        with open(path) as f:
            return Result.model_validate_json(f.read())

    return (load_exp,)


@app.cell
def __(Mocap, Rotation, Tuple, np, plt):
    def mocap_to_44marix(m: Mocap) -> np.ndarray:
        M = np.zeros((4, 4))
        quat = np.array(m.quat)
        M[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        M[3, 3] = 1
        M[:3, 3] = np.array(m.pos).flatten()
        return M.copy()

    def draw_pos(
        m: Mocap,
        ax: plt.Axes,
        color_body: str = "orange",
        color_face: str = "blue",
        alpha=1,
    ):
        M = mocap_to_44marix(m)

        def proj2d(x: float, y: float) -> Tuple[float, float]:
            vec = M @ np.array([x, y, 0, 1])
            vec = vec.flatten().tolist()
            # swap x and y so the plot gets the perspective of the operator
            # standing in front of the table
            return vec[1], vec[0]

        def draw_line(p1: Tuple[float, float], p2: Tuple[float, float], color: str):
            x1, y1 = proj2d(*p1)
            x2, y2 = proj2d(*p2)
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha)

        box_w = 0.1
        d = box_w / 2
        #    (d, d)  │######│  (d, -d)      ▲ x
        #            │      │               │
        #            │      │               │     -y
        #   (-d, d)  └──────┘  (-d, -d)     └─────►
        # points going clockwise starting from the top right
        points = [(d, -d), (-d, -d), (-d, d), (d, d)]
        # draw body
        for p1, p2 in zip(points, points[1:]):
            draw_line(p1, p2, color=color_body)
        # draw face
        draw_line(points[0], points[-1], color=color_face)

    from heapq import heapify, heappop
    from typing import NamedTuple

    class Event(NamedTuple):
        timestamp_str: str
        kind: str
        pos: np.ndarray

        @property
        def timestamp_ms(self) -> float:
            return float(self.timestamp_str.split("-")[0])

        def dist(self, other: "Event") -> float:
            return np.linalg.norm(self.pos - other.pos)

    return Event, NamedTuple, draw_pos, heapify, heappop, mocap_to_44marix


@app.cell
def __(load_exp):
    mocap = load_exp("results-sweep[44]--base=4-seed=1800.json").runs[0].start_pos
    return (mocap,)


@app.cell
def __(
    BoxPushingDense,
    Event,
    List,
    Redis,
    Run,
    heapify,
    heappop,
    io,
    load_exp,
    np,
    plt,
    sys,
):
    def plotcard(
        filename: str, run_index: int = 0, damping=20, mass=1, maxforce=20, kp=20
    ) -> plt.Figure:
        res = load_exp(filename)
        run: Run = res.runs[run_index]
        r = Redis(decode_responses=True)

        # build event heap
        events = []
        events: List[Event]
        for timestamp, msg in r.xrange(
            "cart_cmd",
            min=f"{int(run.start_pos.time_redis*1000)}-0",
            max=f"{int(run.end_pos.time_redis*1000)}-0",
            count=1000,
        ):
            x = float(msg["x"])
            y = float(msg["y"])
            events.append(Event(timestamp, "cmd", np.array([x, y])))

        for timestamp, msg in r.xrange(
            "ack",
            min=f"{int(run.start_pos.time_redis*1000)}-0",
            max=f"{int(run.end_pos.time_redis*1000)}-0",
            count=1000,
        ):
            x = float(msg["x"])
            y = float(msg["y"])
            events.append(Event(timestamp, "ack", np.array([x, y])))

        heapify(events)
        last_cmd: Event = None
        last_ack: Event = None

        while events[0].kind != "cmd":
            heappop(events)

        last_cmd = heappop(events)

        while events[0].kind != "ack":
            heappop(events)

        last_ack = heappop(events)

        fig, ax = plt.subplots()

        pos_ax = ax
        cmds = list()
        acks = list()

        while events:
            e = heappop(events)
            if e.kind == "cmd":
                cmds.append(e.pos)
                last_cmd = e
            else:
                acks.append(e.pos)
                last_ack = e

        l = min(len(cmds), len(acks))
        cmds = np.array(cmds[:l])
        acks = np.array(acks[:l])
        pos_ax.plot(cmds[:, 0], cmds[:, 1], color="green", label="cmd")
        pos_ax.plot(acks[:, 0], acks[:, 1], color="red", label="robot")

        finger_pos = []
        env = BoxPushingDense()
        env.reset()

        maxforce = 47.75

        env.replacements_by_file = {
            "finger.xml": [
                ('damping="15"', f'damping="{damping}"'),
                ('mass="1.0"', f'mass="{mass}"'),
                ('forcerange="-12 12"', f'forcerange="-{maxforce} {maxforce}"'),
                ('kp="20"', f'kp="{kp}"'),
            ]
        }

        env.randomize()

        env.data.joint("finger_x_joint").qpos = cmds[0][0]
        env.data.joint("finger_y_joint").qpos = cmds[0][1]
        try:
            stdout = sys.stdout
            sys.stdout = io.StringIO()
            for x, y in cmds:
                env.step([x, y])
                finger_pos.append(env.data.body("finger").xpos.copy()[:2])
        finally:
            sys.stdout = stdout

        finger_pos = np.array(finger_pos)
        pos_ax.plot(finger_pos[:, 0], finger_pos[:, 1], color="orange", label="sim")
        fig.legend()

        error = []
        for r, f in zip(acks, finger_pos):
            error.append(np.linalg.norm(r - f))

        print(f"Mean Error: {np.mean(error):.3f}")

        return fig

    return (plotcard,)


@app.cell
def __(plotcard):
    import glob

    for filename in [
        # "results-sweep[44]--base=4-seed=1800.json",
        # "results-sweep37-seed1700.json",
        # "results-sweep31.json",
        # "results-sweep29-i=27-seed=140.json",
        # "results-sweep37-seed1700-realtime-3.json",
        # "results-sweep37-seed1700-realtime-3-busy-wait.json",
        "results-sweep45-1700-2024-03-28.json",
    ]:
        fig = plotcard(filename, 0)
    fig
    return fig, filename, glob


@app.cell
def __(
    BoxPushingDense,
    Event,
    List,
    Redis,
    Run,
    heapify,
    heappop,
    io,
    load_exp,
    np,
    sys,
):
    res = load_exp("results-sweep45-1700-2024-03-28.json")
    run: Run = res.runs[-6]
    r = Redis(decode_responses=True)

    # build event heap
    events = []
    events: List[Event]
    for timestamp, msg in r.xrange(
        "cart_cmd",
        min=f"{int(run.start_pos.time_redis*1000)}-0",
        max=f"{int(run.end_pos.time_redis*1000)}-0",
        count=1000,
    ):
        x = float(msg["x"])
        y = float(msg["y"])
        events.append(Event(timestamp, "cmd", np.array([x, y])))

    for timestamp, msg in r.xrange(
        "ack",
        min=f"{int(run.start_pos.time_redis*1000)}-0",
        max=f"{int(run.end_pos.time_redis*1000)}-0",
        count=1000,
    ):
        x = float(msg["x"])
        y = float(msg["y"])
        events.append(Event(timestamp, "ack", np.array([x, y])))

    heapify(events)
    last_cmd: Event = None
    last_ack: Event = None

    while events[0].kind != "cmd":
        heappop(events)

    last_cmd = heappop(events)

    while events[0].kind != "ack":
        heappop(events)

    last_ack = heappop(events)

    cmds = list()
    acks = list()

    while events:
        e = heappop(events)
        if e.kind == "cmd":
            cmds.append(e.pos)
            last_cmd = e
        else:
            acks.append(e.pos)
            last_ack = e

    l = min(len(cmds), len(acks))
    cmds = np.array(cmds[:l])
    acks = np.array(acks[:l])

    def eval_params(maxforce, mass, damping, kp):
        if min(maxforce, mass, kp) <= 0 or damping < 0:
            return 1
        finger_pos = []

        try:
            stdout = sys.stdout
            sys.stdout = io.StringIO()
            env = BoxPushingDense()
            env.reset()

            env.replacements_by_file = {
                "finger.xml": [
                    ('damping="15"', f'damping="{damping}"'),
                    ('mass="1.0"', f'mass="{mass}"'),
                    ('forcerange="-12 12"', f'forcerange="-{maxforce} {maxforce}"'),
                    ('kp="20"', f'kp="{kp}"'),
                ]
            }

            env.randomize()

            env.data.joint("finger_x_joint").qpos = cmds[0][0]
            env.data.joint("finger_y_joint").qpos = cmds[0][1]

            for x, y in cmds:
                env.step([x, y])
                finger_pos.append(env.data.body("finger").xpos.copy()[:2])
        finally:
            sys.stdout = stdout

        finger_pos = np.array(finger_pos)

        error = []
        for r, f in zip(acks, finger_pos):
            error.append(np.linalg.norm(r - f) ** 2)

        return -np.sqrt(np.mean(error))

    return (
        acks,
        cmds,
        e,
        eval_params,
        events,
        l,
        last_ack,
        last_cmd,
        msg,
        r,
        res,
        run,
        timestamp,
        x,
        y,
    )


@app.cell
def __(eval_params):
    eval_params(20, 1, 0, 20)
    return


@app.cell
def __():
    from scipy.optimize import minimize

    # minimize(eval_params, [17,1,3])
    return (minimize,)


@app.cell
def __(eval_params):
    from bayes_opt import BayesianOptimization

    optimizer = BayesianOptimization(
        f=eval_params,
        pbounds={
            "mass": (1, 10),
            "maxforce": (1, 100),
            "damping": (1, 50),
            "kp": (1, 200),
        },
    )
    return BayesianOptimization, optimizer


@app.cell
def __(optimizer):
    optimizer.probe(
        {
            "damping": 14.180179681179924,
            "kp": 168.49563075463564,
            "mass": 1.0,
            "maxforce": 47.74829824923389,
        }
    )
    optimizer.probe(
        {
            "damping": 3,
            "kp": 20,
            "mass": 1.0,
            "maxforce": 17,
        }
    )
    optimizer.probe(
        {
            "damping": 11.782275037281728,
            "kp": 140.5756458718141,
            "mass": 1.0,
            "maxforce": 34.494222061526806,
        }
    )
    return


@app.cell
def __(optimizer):
    optimizer.maximize(init_points=100, n_iter=400)
    return


@app.cell
def __(optimizer):
    optimizer.max
    return


@app.cell
def __(filename, optimizer, plotcard):
    plotcard(filename, -1, **optimizer.max["params"])
    return


@app.cell
def __(optimizer):
    str(optimizer.max["params"])
    return


@app.cell
def __(filename, plotcard):
    plotcard(filename, -1, {"damping": 200, "kp": 20, "mass": 1.0, "maxforce": 20})
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
