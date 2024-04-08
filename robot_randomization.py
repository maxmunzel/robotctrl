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
    from functools import lru_cache

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
        lru_cache,
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
def __(np):
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

    return Event, NamedTuple, heapify, heappop


@app.cell
def __(load_exp):
    mocap = load_exp("results-sweep45-1700-2024-03-28.json").runs[0].start_pos
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
    lru_cache,
    np,
    sys,
):
    res = load_exp("results-sweep46-1700-2024-04-04.json")
    run: Run = res.runs[-1]
    r = Redis(decode_responses=True)

    # build event heap
    events = []
    events: List[Event]
    for timestamp, msg in r.xrange(
        "cart_cmd",
        min=f"{int(run.start_pos.time_redis*1000)}-0",
        max=f"{int(run.end_pos.time_redis*1000)}-0",
        count=2000,
    ):
        x = float(msg["x"])
        y = float(msg["y"])
        events.append(Event(timestamp, "cmd", np.array([x, y])))

    for timestamp, msg in r.xrange(
        "ack",
        min=f"{int(run.start_pos.time_redis*1000)}-0",
        max=f"{int(run.end_pos.time_redis*1000)}-0",
        count=2000,
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

    @lru_cache()
    def simulate_robot(mass, damping, kp):
        if min(mass, kp) <= 0 or damping < 0:
            return 1
        finger_pos = []

        try:
            stdout = sys.stdout
            sys.stdout = io.StringIO()
            env = BoxPushingDense()
            env.reset()

            env.replacements_by_file = {
                "finger.xml": [
                    ('damping="10"', f'damping="{damping}"'),
                    ('mass="1.0"', f'mass="{mass}"'),
                    # ('forcerange="-12 12"', ""),
                    # ('forcelimited="true"', ""),
                    ('kp="200"', f'kp="{kp}"'),
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

        return finger_pos

    return (
        acks,
        cmds,
        e,
        events,
        l,
        last_ack,
        last_cmd,
        msg,
        r,
        res,
        run,
        simulate_robot,
        timestamp,
        x,
        y,
    )


@app.cell
def __(acks, cmds, np, simulate_robot):
    real_error = []
    for c, a in zip(cmds, acks):
        real_error.append(np.linalg.norm(c - a))

    def constraint(*args):
        """calculates the step-wise difference between the simulated ctrl error and the real-robot ctrl error"""
        finger = simulate_robot(*args)
        sim_error = []
        for c, f in zip(cmds, finger):
            sim_error.append(np.linalg.norm(c - f))
        return np.array(sim_error)

    return a, c, constraint, real_error


@app.cell
def __(constraint, plt, real_error):
    high = (1, 10, 200)
    low = (1, 10, 70)
    low = (5, 14, 443)
    # low = (5, 14, 200)

    index = 90

    # mdk=(5,40,700)
    # mdk = (5, 40, 400)
    fig_err, ax_err = plt.subplots()
    ax_err.plot(real_error, label="real", color="violet")

    for mdk, label, color in [(low, "low", "blue"), (high, "high", "red")]:
        ax_err.plot(constraint(*mdk), label=label, color=color)
        ax_err.axvline(index)

    fig_err.legend()
    fig_err
    return ax_err, color, fig_err, high, index, label, low, mdk


@app.cell
def __(acks, cmds, high, index, low, plt, simulate_robot):
    fig, ax = plt.subplots()
    data_sources = [
        (cmds, "cmds", "green"),
        (acks, "acks", "violet"),
        (simulate_robot(*low), "low", "blue"),
        (simulate_robot(*high), "high", "red"),
    ]
    for data, label2, color2 in data_sources:
        ax.plot(data[:, 0], data[:, 1], label=label2, color=color2)
        ax.scatter(data[index, 0], data[index, 1], color=color2)

    fig.legend()
    fig
    return ax, color2, data, data_sources, fig, label2


@app.cell
def __():
    return


@app.cell
def __(Redis, np, plt, res):
    def plot_speeds():
        speeds = []
        for run in res.runs:
            r = Redis(decode_responses=True)
            cmds = []
            for timestamp, msg in r.xrange(
                "cart_cmd",
                min=f"{int(run.start_pos.time_redis*1000)}-0",
                max=f"{int(run.end_pos.time_redis*1000)}-0",
                count=2000,
            ):
                x = float(msg["x"])
                y = float(msg["y"])
                cmds.append(np.array([x, y]))

            dt = 0.02
            for p1, p2 in zip(cmds, cmds[1:]):
                speeds.append(np.linalg.norm(p1 - p2) / dt)

        fig_speeds, ax_speeds = plt.subplots()
        ax_speeds.plot(speeds)
        # ax_speeds.set_xlim(min(speeds), max(speeds))
        print(f"Max Speed: {max(speeds):.2f} m/s")
        return fig_speeds

    plot_speeds()
    return (plot_speeds,)


@app.cell
def __():
    return


@app.cell
def __(Redis, plt, res):
    def plot_trajs():
        colors = ["red", "green", "blue"]
        fig_traj, ax_traj = plt.subplots()
        for i, run in enumerate(res.runs):
            r = Redis(decode_responses=True)
            xs = []
            ys = []
            for timestamp, msg in r.xrange(
                "cart_cmd",
                min=f"{int(run.start_pos.time_redis*1000)}-0",
                max=f"{int(run.end_pos.time_redis*1000)}-0",
                count=2000,
            ):
                x = float(msg["x"])
                y = float(msg["y"])
                xs.append(x)
                ys.append(y)
            ax_traj.plot(xs, ys, alpha=0.2, color=colors[i // 12])

        return fig_traj

    plot_trajs()
    return (plot_trajs,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
