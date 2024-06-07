import marimo

__generated_with = "0.6.0"
app = marimo.App(width="medium")


@app.cell
def __():
    from exp_logger import Result, Mocap, Run
    from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import (
        # rotation_distance,
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
    import json
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
        json,
        lru_cache,
        np,
        plt,
        sys,
    )


@app.cell
def __(Rotation, np):
    def rotation_distance(matrix1, matrix2):
        """
        Compute the rotation distance in radians between two affine transformation matrices.

        Parameters:
        - matrix1: 4x4 numpy array representing the first affine transformation.
        - matrix2: 4x4 numpy array representing the second affine transformation.

        Returns:
        - distance: The rotation distance in radians.
        """
        # Extract the rotational parts of the matrices (upper 3x3 submatrices)
        rot1 = matrix1[:3, :3]
        rot2 = matrix2[:3, :3]

        # Convert the rotation matrices to rotation objects
        r1 = Rotation.from_matrix(rot1)
        r2 = Rotation.from_matrix(rot2)

        # Compute the relative rotation
        relative_rotation = r1.inv() * r2

        # Get the rotation vector and its magnitude (the angle)
        rotation_vector = relative_rotation.as_rotvec()
        distance = np.linalg.norm(rotation_vector)

        return distance


    def quat_distance(q1, q2):
        # Convert the rotation matrices to rotation objects
        r1 = Rotation.from_quat(q1)
        r2 = Rotation.from_quat(q2)

        # Compute the relative rotation
        relative_rotation = r1.inv() * r2

        # Get the rotation vector and its magnitude (the angle)
        rotation_vector = relative_rotation.as_rotvec()
        distance = np.linalg.norm(rotation_vector)

        return distance
    return quat_distance, rotation_distance


@app.cell
def __(Redis, json, np, rotation_distance):
    r = Redis(decode_responses=True)

    # build event heap
    events = []

    last_time = None
    last_trans = None

    times = list()
    rot_speeds = list()
    for timestamp, msg in r.xrange(
        "box_tracking",
        min=f"-",
        max=f"+",
        count=20000,
    ):
        timestamp = float(timestamp.split("-")[0]) / 1000
        trans = np.array(json.loads(msg["transform"])).reshape(4, 4)

        if last_time is None:
            last_time = timestamp
            last_trans = trans
            continue

        times.append((timestamp + last_time) / 2)
        dt = timestamp - last_time
        rot_speeds.append(rotation_distance(last_trans, trans) / dt)

        last_time = timestamp
        last_trans = trans
    return (
        dt,
        events,
        last_time,
        last_trans,
        msg,
        r,
        rot_speeds,
        times,
        timestamp,
        trans,
    )


@app.cell
def __(np, plt, times):
    fig_fps, ax_fps = plt.subplots()
    _ = ax_fps.hist(1/np.diff(times))
    fig_fps
    return ax_fps, fig_fps


@app.cell
def __(plt, rot_speeds, times):
    plt.plot(times, rot_speeds)
    return


@app.cell
def __(plt, rot_speeds, times):
    fig_runs, ax_runs = plt.subplots()

    spin_limit = 20
    no_spin = 1

    def plot_real_spins(ax: plt.Axes):

        
        DISCARD = 0  # discard data until speeds exceed spin_limit
        WRITE = 1  # write data into new run until speed falls below no_spin
        STATE = DISCARD
        
        runs = []
        current_run = []
        
        for t, s in zip(times, rot_speeds):
            if STATE == DISCARD:
                if s > spin_limit:
                    STATE = WRITE
                    current_run.append((t, s))
        
            elif STATE == WRITE:
                current_run.append((t, s))
                if s < no_spin:
                    runs.append(current_run)
                    current_run = []
                    state = DISCARD
            
        trimmed = []
        for run in runs:
            max_speed, i = max([(r[1], i) for i, r in enumerate(run)])
            trimmed.append(run[i:])
        
        
        for run in trimmed:
            if len(run) < 5:
                continue
            ax.plot(
                [r[0] - run[-1][0] for r in run], [r[1] for r in run], "-o", alpha=0.4, 
            )

    plot_real_spins(ax_runs)
    fig_runs
    return ax_runs, fig_runs, no_spin, plot_real_spins, spin_limit


@app.cell
def __(
    BoxPushingDense,
    io,
    no_spin,
    np,
    plot_real_spins,
    plt,
    quat_distance,
    sys,
):
    def simulate_robot(lat_fric, rot_fric, yank_damp, yank_gain, damping, ax):
        mass = 1
        kp = 100
        box_quats = []
        try:
            stdout = sys.stdout
            sys.stdout = io.StringIO()
            env = BoxPushingDense(frame_skip=1)
            env.reset()

            env.replacements_by_file = {
                "push_box.xml": [
                    #("5308", "1"),
                ('friction="0.4296 0.001 0.0001"', f'friction="{lat_fric} {rot_fric} 0.0001"'),
                ('name="box_rot_joint"', f'name="box_rot_joint" damping="{damping}"'),
                ('limited="true"', f'limited="true" damping="{yank_damp}"')
                ],
            }

            env.randomize()

            env.data.joint("finger_x_joint").qpos = -0.25
            env.data.joint("finger_y_joint").qpos = -0.5
            env.data.joint("box_rot_joint").qvel = 20
            env.data.joint("box_z_joint").qpos = 0.001
            #env.data.joint("box_roll_joint").qvel = 1
            #env.data.joint("box_pitch_joint").qvel = 2


            sim_roll_speeds = []

            for _ in range(500):
                env.step([-1, -1])
                box_quats.append(
                    env.data.body("box_0").xquat[[3, 0, 1, 2]].copy()
                )
                # sim_roll_speeds.append(abs(float(env.data.joint("box_roll_joint").qvel.copy()+env.data.joint("box_pitch_joint").qvel.copy()))*1000)
                sim_roll_speeds.append(env.data.body("box_0").xpos[2]*10000)
                env.data.joint("box_roll_joint").qvel *= yank_gain
                env.data.joint("box_pitch_joint").qvel *= yank_gain
            t = 0
            sim_times = []
            sim_rot_speeds = []
            for q1, q2 in zip(box_quats, box_quats[1:]):
                sim_times.append(t)
                t += env.dt
                rot_speed = quat_distance(q1, q2) / env.dt
                sim_rot_speeds.append(rot_speed)
                if rot_speed < no_spin:
                    break
        finally:
            sys.stdout = stdout

        sim_times = np.array(sim_times) - max(sim_times) 

        ax.plot(sim_times, sim_rot_speeds, "o-")
        #ax.plot(sim_times, sim_roll_speeds[:len(sim_times)], "o-", label="Height in mm")


    fig_comb, ax_comb = plt.subplots()

    simulate_robot(lat_fric=0.4296, rot_fric=0, damping=-0, yank_damp=0.00, yank_gain=0.1, ax=ax_comb)
    simulate_robot(lat_fric=0.3, rot_fric=0, damping=-0, yank_damp=0.00, yank_gain=0, ax=ax_comb)
    simulate_robot(lat_fric=0.2, rot_fric=0, damping=-0, yank_damp=0.00, yank_gain=0, ax=ax_comb)
    simulate_robot(lat_fric=0.1, rot_fric=0, damping=-0, yank_damp=0.00, yank_gain=0, ax=ax_comb)

    plot_real_spins(ax_comb)
    ax_comb.legend()


    return ax_comb, fig_comb, simulate_robot


@app.cell
def __(fig_comb):
    fig_comb
    return


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


    high = (1, 12, 100)
    low = (1, 12, 50)
    # low = (5, 14, 443)
    # low = (5, 14, 200)
    return a, c, constraint, high, low, real_error


@app.cell
def __(constraint, high, low, plt, real_error):
    index = 380

    # mdk=(5,40,700)
    # mdk = (5, 40, 400)
    fig_err, ax_err = plt.subplots()
    ax_err.plot(real_error, label="real", color="violet")

    for mdk, label, color in [(low, "low", "blue"), (high, "high", "red")]:
        ax_err.plot(constraint(*mdk), label=label, color=color)
        ax_err.axvline(index)

    fig_err.legend()
    fig_err
    return ax_err, color, fig_err, index, label, mdk


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

    ax.axis("equal")
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
    return plot_speeds,


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
    return plot_trajs,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
