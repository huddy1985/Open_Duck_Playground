import argparse
import pickle
import time

import mujoco
import mujoco.viewer
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase


class ObsReplayer(MJInferBase):
    def __init__(self, model_path: str, obs_path: str) -> None:
        super().__init__(model_path)
        with open(obs_path, "rb") as f:
            self.obs_data = pickle.load(f)

    def run(self) -> None:
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            for obs in self.obs_data:
                idx = 0
                idx += 3  # gyro
                idx += 3  # accelerometer
                idx += 7  # commands

                # joint positions relative to home posture
                pos_rel = obs[idx : idx + self.num_dofs]
                idx += self.num_dofs
                # joint velocities scaled by 0.05 in saved obs
                vel_scaled = obs[idx : idx + self.num_dofs]
                idx += self.num_dofs

                # skip action history
                idx += self.num_dofs * 3

                motor_targets = obs[idx : idx + self.num_dofs]

                # reconstruct positions/velocities
                joint_pos = pos_rel + self.default_actuator
                joint_vel = vel_scaled / 0.05

                self.set_actuator_joints_qpos(joint_pos, self.data.qpos)
                self.set_actuator_joints_qvel(joint_vel, self.data.qvel)
                self.data.ctrl[:] = motor_targets

                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(self.model.opt.timestep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obs",
        type=str,
        required=True,
        help="Path to pickled observation list",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    args = parser.parse_args()

    player = ObsReplayer(args.model_path, args.obs)
    player.run()
