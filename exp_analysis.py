import marimo

__generated_with = "0.3.4"
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
    return (
        List,
        Mocap,
        Result,
        Rotation,
        Run,
        Tuple,
        affine_matrix_to_xpos_and_xquat,
        np,
        plt,
        rotation_distance,
    )


@app.cell
def __(List, Rotation, np, rotation_distance):
    import json


    def proj_as_2d_rot(quat) -> List[float]:
        return quat
        r = Rotation.from_quat(quat)
        z_rot = r.as_euler("zyx")[1]
        return Rotation.from_euler("y", z_rot).as_quat()


    def project_quaternion_to_z_rotation(quaternion):  # chatgpt
        """
        Projects a quaternion to the space of rotations around the Z-axis (yaw only).

        Parameters:
        quaternion (array-like): The input quaternion in the form [x, y, z, w].

        Returns:
        numpy.ndarray: The output quaternion representing a rotation around the Z-axis.
        """
        # Convert the input quaternion to Euler angles with 'ZYX' convention
        euler_angles = Rotation.from_quat(quaternion).as_euler("ZYX")

        # Zero out the roll and pitch components (X and Y axes rotations)
        euler_angles[0] = 0  # Roll
        euler_angles[1] = 0  # Pitch

        # Convert the modified Euler angles back to a quaternion
        return Rotation.from_euler("ZYX", euler_angles).as_quat()


    def yank_deg(quat) -> float:
        # how many degrees are between the given quat and its flat projection?
        quat = np.array(quat)
        proj = project_quaternion_to_z_rotation(quat)
        return rotation_distance(proj, quat) / np.pi * 180
    return json, proj_as_2d_rot, project_quaternion_to_z_rotation, yank_deg


@app.cell
def __(Result):
    def load_exp(path: str) -> Result:
        with open(path) as f:
            return Result.model_validate_json(f.read())
    return load_exp,


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

        def draw_line(
            p1: Tuple[float, float], p2: Tuple[float, float], color: str
        ):
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
    return draw_pos, mocap_to_44marix


@app.cell
def __(affine_matrix_to_xpos_and_xquat, load_exp, mocap_to_44marix):
    mocap = load_exp("results-sweep[44]--base=4-seed=1800.json").runs[0].start_pos
    M0 = mocap_to_44marix(mocap)
    xpos0, xquat0 = affine_matrix_to_xpos_and_xquat(M0)
    print(xquat0)
    print(mocap.quat)
    return M0, mocap, xpos0, xquat0


@app.cell
def __(
    Mocap,
    draw_pos,
    load_exp,
    np,
    plt,
    proj_as_2d_rot,
    rotation_distance,
    yank_deg,
):
    def plotcard(filename: str) -> plt.Figure:
        res = load_exp(filename)
        for run_ in res.runs:
            run_.end_pos.quat = proj_as_2d_rot(run_.end_pos.quat)
            run_.start_pos.quat = proj_as_2d_rot(run_.start_pos.quat)

        target_pos = np.array([0.51505285, 0.0, 0.0])
        target_quat = (
            np.array([0.51505285, 0.0, 0.0, 0.85715842])
            if "sweep31" in res.agent
            else np.array([1.0, 0.0, 0.0, 0.0])
        )

        def pos_err(pos) -> float:
            return np.linalg.norm(pos - target_pos)

        fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(6, 10))
        fig.suptitle(f"{res.agent}\n\n", fontsize=16)

        # Final Position

        x = target_pos[0]
        y = target_pos[1]
        d = 0.4

        ax[0][0].set_ylabel("x")
        ax[0][0].set_xlabel("y")
        ax[0][0].set_ylim(x + d, x - d)
        ax[0][0].set_xlim(y - d, y + d)
        ax[0][0].set_aspect("equal")
        ax[0][0].set_title("Final Position")
        for run in res.runs:
            draw_pos(
                run.end_pos,
                ax[0][0],
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
            ax[0][0],
            alpha=1,
            color_body="green",
            color_face="red",
        )

        # Pos Error Plot

        pos_err_ax = ax[0][1]
        pos_err_ax: plt.Axes
        pos_tolerance = 0.05
        pos_err_data = np.array([pos_err(run.end_pos.pos) for r in res.runs])
        pos_err_ax.set_title(
            f"Pos Error Dist\n"
            f"µ={np.mean(pos_err_data):.2f} "
            f"σ={np.var(pos_err_data)**.5:.2f} "
            f"suc={np.mean(pos_err_data<pos_tolerance):.2f}"
        )
        pos_err_ax.hist(pos_err_data, alpha=1, bins=np.arange(0, 1, 0.01))
        pos_err_ax.set_xlim(0, 0.3)
        pos_err_ax.axvline(pos_tolerance, color="green", linestyle="--")
        # Rot Error Plot

        rot_err = ax[0, 2]
        rot_tolerance = 0.5 / np.pi * 180  # what the env considers a success
        rot_err_data = np.array(
            [
                rotation_distance(np.array(r.end_pos.quat), target_quat)
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

        for rep, pos_ax in enumerate([ax[2][0], ax[2][1], ax[2][2], ax[3][0]]):
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

        # Start Yank
        start_yank_ax = ax[3][1]

        start_yanks = []
        for run in res.runs:
            if None in run.start_pos.quat:
                continue
            yank = yank_deg(run.start_pos.quat)
            if not np.isnan(yank):
                start_yanks.append(yank)
        color = None if np.max(start_yanks) < 5 else "red"
        start_yank_ax.hist(start_yanks, color=color)
        start_yank_ax.set_xlim(0, 5)
        start_yank_ax.set_title(
            "Start Yank /deg\n"
            f"µ={np.mean(start_yanks):.2f} "
            f"max={np.max(start_yanks):.2f}"
        )

        # End Yank
        end_yank_ax = ax[3][2]

        end_yanks = []
        for run in res.runs:
            if None in run.end_pos.quat:
                continue
            yank = yank_deg(run.end_pos.quat)
            if not np.isnan(yank):
                end_yanks.append(yank)
        color = None if np.max(end_yanks) < 5 else "red"
        end_yank_ax.hist(end_yanks, color=color)
        end_yank_ax.set_xlim(0, 5)
        end_yank_ax.set_title(
            "End Yank /deg\n"
            f"µ={np.mean(end_yanks):.2f} "
            f"max={np.max(end_yanks):.2f}"
        )

        fig.tight_layout()

        # Score
        score_ax = ax[1, 0]
        scores = []
        for run in res.runs:
            if None in run.end_pos.quat:
                continue
            box_goal_dist = np.linalg.norm(run.end_pos.pos - target_pos)

            box_goal_pos_dist_reward = -3.5 * box_goal_dist * 100
            box_goal_rot_dist_reward = (
                -rotation_distance(np.array(run.end_pos.quat), target_quat)
                / np.pi
                * 100
            )
            reward = box_goal_pos_dist_reward + box_goal_rot_dist_reward
            if not np.isnan(reward):
                scores.append(reward)
        score_ax.set_title(f"Score µ={np.mean(scores):.2f}")
        score_ax.hist(scores)
        score_ax.set_xlim(-200, 0)

        return fig
    return plotcard,


@app.cell
def __(plotcard):
    import glob

    for filename in [
        "results-sweep[44]--base=4-seed=1800.json",
        "results-sweep37-seed1700.json",
        "results-sweep31.json",
        "results-sweep29-i=27-seed=140.json",
    ]:
        plotcard(filename).savefig(f"{filename}_card.pdf")
    return filename, glob


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()