import marimo

__generated_with = "0.6.0"
app = marimo.App()


@app.cell
def __():
    from exp_logger import Result, Mocap, Run
    from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import (
        rotation_distance,
        affine_matrix_to_xpos_and_xquat,
    )
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    import numpy as np
    from typing import Tuple, List
    from redis import Redis
    import matplotlib.gridspec as gridspec

    return (
        List,
        Mocap,
        Redis,
        Result,
        Rotation,
        Run,
        Tuple,
        affine_matrix_to_xpos_and_xquat,
        gridspec,
        np,
        plt,
        rotation_distance,
    )


@app.cell
def __(Rotation, np, rotation_distance):
    import json

    def project_quaternion_to_z_rotation(quaternion):
        """
        Projects a quaternion to the space of rotations around the Z-axis (yaw only).

        Parameters:
        quaternion (array-like): The input quaternion in the form [x, y, z, w].

        Returns:
        numpy.ndarray: The output quaternion representing a rotation around the Z-axis.
        """
        # Convert the input quaternion to Euler angles with 'ZYX' convention
        quaternion = np.array(quaternion)
        euler_angles = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_euler("ZYX")

        # Zero out the roll and pitch components (X and Y axes rotations)
        euler_angles[1] = 0  # Roll
        euler_angles[2] = 0  # Pitch

        # Convert the modified Euler angles back to a quaternion
        return Rotation.from_euler("ZYX", euler_angles).as_quat()[[3, 0, 1, 2]]

    def yank_deg(quat) -> float:
        # how many degrees are between the given quat and its flat projection?
        quat = np.array(quat)
        proj = project_quaternion_to_z_rotation(quat)
        return rotation_distance(proj, quat) / np.pi * 180

    return json, project_quaternion_to_z_rotation, yank_deg


@app.cell
def __(Result):
    def load_exp(path: str) -> Result:
        with open(path) as f:
            return Result.model_validate_json(f.read())

    return (load_exp,)


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    # Sweep 44 / 37 / 29
    # (Pdb++) pp target_pos
    # array([0.51505285, 0.        , 0.        ])
    # (Pdb++) pp target_quat
    # array([1., 0., 0., 0.])

    # Sweep 17 / 31
    # (Pdb++) pp target_pos
    # array([0.51505285, 0.        , 0.        ])
    # (Pdb++) pp target_quat
    # array([0.51505285, 0.        , 0.        , 0.85715842])
    return


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
def __(
    Event,
    List,
    Mocap,
    Redis,
    draw_pos,
    heapify,
    heappop,
    load_exp,
    np,
    plt,
    project_quaternion_to_z_rotation,
    rotation_distance,
):
    def plotcard(filename: str, delayed: bool) -> plt.Figure:
        res = load_exp(filename)
        for run_ in res.runs:
            run_.end_pos.quat = project_quaternion_to_z_rotation(run_.end_pos.quat)
            run_.start_pos.quat = project_quaternion_to_z_rotation(run_.start_pos.quat)

        target_pos = np.array([0.51505285, 0.0, 0.0])
        target_quat = (
            np.array([0.51505285, 0.0, 0.0, 0.85715842])
            if "sweep31" in res.agent
            else np.array([1.0, 0.0, 0.0, 0.0])
        )

        def pos_err(pos) -> float:
            return np.linalg.norm(pos - target_pos)

        # fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(6, 10))
        fig, ax = plt.subplot_mosaic(
            [
                ["final_pos", "pos_err", "rot_err"],
                ["score", "ctrl_dt", "ctrl_err"],
                ["start", "start", "start"],
                ["traj", "end_yank", "ack_latency"],
            ],
            figsize=(7.5, 12),
            tight_layout=True,
        )

        fig.subplots_adjust(
            left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4
        )

        fig.suptitle(f"{res.agent}{' (delayed)' if delayed else ''}\n\n", fontsize=16)

        # Final Position

        x = target_pos[0]
        y = target_pos[1]
        d = 0.4
        final_pos_ax = ax["final_pos"]
        final_pos_ax.set_ylabel("x")
        final_pos_ax.set_xlabel("y")
        final_pos_ax.set_ylim(x + d, x - d)
        final_pos_ax.set_xlim(y - d, y + d)
        final_pos_ax.set_aspect("equal")
        final_pos_ax.set_title(f"Final Position")

        for run in res.runs:
            if delayed:
                end_pos = run.end_pos_delayed
            else:
                end_pos = run.end_pos
            draw_pos(
                end_pos,
                final_pos_ax,
                alpha=0.1,
                color_body="orange",
                color_face="blue",
            )

        draw_pos(
            Mocap.validate(
                {
                    "time_redis": 42,
                    "pos": target_pos.tolist(),
                    "quat": target_quat.tolist(),
                }
            ),
            final_pos_ax,
            alpha=1,
            color_body="green",
            color_face="red",
        )

        # Pos Error Plot

        pos_err_ax = ax["pos_err"]
        pos_err_ax: plt.Axes
        pos_tolerance = 0.05
        if delayed:
            pos_err_data = [pos_err(r.end_pos_delayed.pos) for r in res.runs]
        else:
            pos_err_data = [pos_err(r.end_pos.pos) for r in res.runs]

        pos_err_data = np.array(pos_err_data)
        pos_err_ax.set_title(
            f"Pos Error Dist\n"
            f"µ={np.mean(pos_err_data):.2f} "
            f"σ={np.var(pos_err_data)**.5:.2f} "
            f"suc={np.mean(pos_err_data<pos_tolerance):.2f}"
        )
        pos_err_ax.hist(
            pos_err_data,
            alpha=1,
            # bins=np.arange(0, 1, 0.01),
        )
        pos_err_ax.set_xlim(0, 0.5)
        pos_err_ax.axvline(pos_tolerance, color="green", linestyle="--")

        # Rot Error Plot

        rot_err = ax["rot_err"]
        rot_tolerance = 0.5 / np.pi * 180  # what the env considers a success
        rot_err_data = np.array(
            [
                rotation_distance(
                    np.array(r.end_pos_delayed.quat if delayed else r.end_pos.quat),
                    target_quat,
                )
                / np.pi
                * 180
                for r in res.runs
                if None not in r.end_pos.quat
            ]
        )
        _ = rot_err.hist(rot_err_data)
        _ = rot_err.set_xticks([0, 45, 90, 135, 180])
        rot_err.set_title(
            f"Rot Error /deg\nµ={np.mean(rot_err_data):.2f} suc={np.mean(rot_err_data<rot_tolerance):.2f}"
        )
        _ = rot_err.axvline(x=rot_tolerance, color="green", linestyle="--")

        # Start Positions
        start_ax = ax["start"]
        start_ax: plt.Axes
        start_ax.remove()
        for rep in range(4):
            pos_ax = fig.add_subplot(4, 4, 9 + rep)
            x = 0.4
            y = 0.0
            d = 0.4

            pos_ax.set_ylabel("x")
            pos_ax.set_xlabel("y")
            # pos_ax.set_ylim(x + d, x - d)
            # pos_ax.set_xlim(y + d, y - d)
            pos_ax.set_aspect("equal")
            pos_ax.set_title(f"Start Exp X-{rep}")
            for run in res.runs:
                if run.experiment.endswith(str(rep + 1)):
                    if None in run.start_pos.quat:
                        print(
                            f"Skipping start_pos of run {filename}-{run.experiment} It. {run.rep+1}"
                        )
                        continue

                    # run.start_pos.quat = project_quaternion_to_z_rotation(run.start_pos.quat)
                    draw_pos(
                        run.start_pos,
                        pos_ax,
                        alpha=0.25,
                        color_body="orange",
                        color_face="blue",
                    )

        # Score
        score_ax = ax["score"]
        scores = []
        successes = []
        for run in res.runs:
            success = 1
            if None in run.end_pos.quat:
                continue
            if delayed:
                box_goal_dist = np.linalg.norm(run.end_pos_delayed.pos - target_pos)
            else:
                box_goal_dist = np.linalg.norm(run.end_pos.pos - target_pos)

            if box_goal_dist > 0.05:
                success = 0

            box_goal_pos_dist_reward = -3.5 * box_goal_dist * 100
            box_goal_rot_dist_reward = (
                -rotation_distance(
                    np.array(run.end_pos_delayed.quat if delayed else run.end_pos.quat),
                    target_quat,
                )
                / np.pi
                * 100
            )
            if box_goal_rot_dist_reward <= -17.5:
                success = 0

            reward = box_goal_pos_dist_reward + box_goal_rot_dist_reward
            if not np.isnan(reward):
                scores.append(reward)
            successes.append(success)

        score_tolerance = -67.5
        score_ax.set_title(
            f"Score µ={np.mean(scores):.2f} suc={np.mean(successes):.2f}"
        )
        score_ax.hist(scores)
        _ = score_ax.axvline(x=score_tolerance, color="green", linestyle="--")
        score_ax.set_xlim(-200, 0)

        # Latency
        latency_ax = ax["ctrl_dt"]
        r = Redis(decode_responses=True)
        dt = []
        for run in res.runs:
            r = Redis(decode_responses=True)
            times = []
            for timestamp, msg in r.xrange(
                "cart_cmd",
                min=f"{int(run.start_pos.time_redis*1000)}-0",
                max=f"{int(run.end_pos.time_redis*1000)}-0",
                count=1000,
            ):
                times.append(float(timestamp.split("-")[0]))

            dt += np.diff(times).tolist()
            warn_limit = 500
            if len(np.diff(times)) and max(np.diff(times)) > warn_limit:
                print(
                    f"{res.agent} exp {run.experiment} it {run.rep} had ctrl latency over {warn_limit}ms"
                )
        dt = np.array(dt)
        latency_ax.plot(dt[dt < 1000000], alpha=0.3)
        latency_ax.axhline(20, color="green")
        ctrl_limit = 100
        latency_ax.set_title(
            f"Ctrl dt med={np.median(dt):.1f}\n"
            f"{np.sum(dt>ctrl_limit)} times >{ctrl_limit}ms"
        )
        latency_ax.set_ylim(0, 100)

        # Ctrl Error
        ack_latencies = []
        ctrl_err_ax = ax["ctrl_err"]
        ack_latency_ax = ax["ack_latency"]

        for run in res.runs:
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

            while events and events[0].kind != "cmd":
                heappop(events)

            ctrl_errors = []
            while events:
                e = heappop(events)
                if e.kind == "cmd":
                    last_cmd = e
                else:
                    last_ack = e
                    ctrl_errors.append(e.dist(last_cmd))
                    ack_latencies.append(e.timestamp_ms - last_cmd.timestamp_ms)

            ctrl_err_ax.plot(ctrl_errors, alpha=0.3)
            # ctrl_err_ax.axhline(50)
            ctrl_err_ax.set_title("Positional Ctrl Error")
            ctrl_err_ax.set_ylim(0, 0.3)
        ack_latency_ax.set_title("Ack Latency /ms")
        ack_latency_ax.plot(ack_latencies, alpha=0.3)
        ack_latency_ax.set_ylim(0, 250)

        colors = ["red", "green", "blue"]
        ax_traj = ax["traj"]
        ax_traj.set_title("Trajectories")
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
            # swap x/y and mirror x to resemble operator view
            ax_traj.set_ylim(1, 0)
            ax_traj.set_xlim(-0.5, 0.5)
            ax_traj.plot(ys, xs, alpha=0.2, color=colors[i // 12])

        return fig

    return (plotcard,)


@app.cell
def __(plotcard):
    import glob

    for filename in glob.glob("*06-07*.json"):
        for delayed in [False, True]:
            plotcard(filename, delayed=delayed).savefig(
                f"{filename}_card{'_delayed' if delayed else ''}.pdf"
            )
    return delayed, filename, glob


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
