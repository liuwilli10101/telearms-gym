"""Safety-limited bridge from bimanual YAM policy actions to I2RT hardware."""
from __future__ import annotations

from typing import Any

import numpy as np

from roboeval.robots.configs.yam import (
    YAM_COULOMB_FRICTION_NM,
    YAM_GRAV_COMP_KD,
    YAM_GRAVITY_COMP_FACTOR,
    YAM_GRIPPER_FORCE_LIMIT_N,
    YAM_GRIPPER_KD,
    YAM_GRIPPER_KP,
    YAM_GRIPPER_RANGE,
    YAM_HARDWARE_POSITION_LIMITS_RAD,
    YAM_KD,
    YAM_KP,
    YAM_POLICY_CONTROL_FREQUENCY_HZ,
    YAM_TORQUE_LIMITS_NM,
    YAM_USE_COULOMB_FRICTION,
    YAM_VELOCITY_LIMITS_RAD_S,
)


class YAMLimitViolation(RuntimeError):
    """Raised when a YAM command or observation violates a hardware limit."""


class BimanualYAMCommandLimiter:
    """Limit 14-D policy actions before sending them to two real YAMs."""

    ACTION_DIM = 14

    def __init__(
        self, control_frequency_hz: float = YAM_POLICY_CONTROL_FREQUENCY_HZ
    ) -> None:
        if control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")
        self.control_frequency_hz = float(control_frequency_hz)
        self._position_limits = np.asarray(
            YAM_HARDWARE_POSITION_LIMITS_RAD, dtype=np.float64
        )
        self._velocity_limits = np.asarray(
            YAM_VELOCITY_LIMITS_RAD_S, dtype=np.float64
        )
        self._max_target_delta = self._velocity_limits / self.control_frequency_hz
        self._previous_targets: tuple[np.ndarray, np.ndarray] | None = None

    def reset(
        self, left_joint_position: np.ndarray, right_joint_position: np.ndarray
    ) -> None:
        left = self._validated_arm_vector(left_joint_position, "left position")
        right = self._validated_arm_vector(right_joint_position, "right position")
        self._check_position(left, "left")
        self._check_position(right, "right")
        self._previous_targets = (left.copy(), right.copy())

    def limit_policy_action(
        self, policy_action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._previous_targets is None:
            raise RuntimeError("reset the limiter from current hardware state first")
        action = np.asarray(policy_action, dtype=np.float64)
        if action.shape != (self.ACTION_DIM,):
            raise ValueError(
                f"expected a {self.ACTION_DIM}-D YAM policy action, got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise YAMLimitViolation("policy action contains non-finite values")

        left = self._limit_arm_target(action[:6], self._previous_targets[0])
        right = self._limit_arm_target(action[7:13], self._previous_targets[1])
        left_gripper = float(np.clip(action[6], *YAM_GRIPPER_RANGE))
        right_gripper = float(np.clip(action[13], *YAM_GRIPPER_RANGE))
        self._previous_targets = (left.copy(), right.copy())
        return (
            np.concatenate((left, [left_gripper])),
            np.concatenate((right, [right_gripper])),
        )

    def validate_feedback(
        self,
        side: str,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        joint_torque: np.ndarray,
        *,
        position_tolerance_rad: float = 0.01,
        velocity_tolerance_rad_s: float = 0.5,
        torque_tolerance_nm: float = 1.0,
    ) -> None:
        position = self._validated_arm_vector(joint_position, f"{side} position")
        velocity = self._validated_arm_vector(joint_velocity, f"{side} velocity")
        torque = self._validated_arm_vector(joint_torque, f"{side} torque")
        self._check_position(position, side, position_tolerance_rad)

        velocity_excess = np.abs(velocity) - (
            self._velocity_limits + velocity_tolerance_rad_s
        )
        if np.any(velocity_excess > 0):
            joint = int(np.argmax(velocity_excess))
            raise YAMLimitViolation(
                f"{side} J{joint + 1} velocity {velocity[joint]:.3f} rad/s "
                f"exceeds {self._velocity_limits[joint]:.3f} rad/s"
            )

        torque_limits = np.asarray(YAM_TORQUE_LIMITS_NM)
        torque_excess = np.abs(torque) - (torque_limits + torque_tolerance_nm)
        if np.any(torque_excess > 0):
            joint = int(np.argmax(torque_excess))
            raise YAMLimitViolation(
                f"{side} J{joint + 1} torque {torque[joint]:.3f} N m "
                f"exceeds {torque_limits[joint]:.3f} N m"
            )

    def _limit_arm_target(
        self, requested: np.ndarray, previous: np.ndarray
    ) -> np.ndarray:
        mechanical = np.clip(
            requested, self._position_limits[:, 0], self._position_limits[:, 1]
        )
        limited = previous + np.clip(
            mechanical - previous, -self._max_target_delta, self._max_target_delta
        )
        return np.clip(
            limited, self._position_limits[:, 0], self._position_limits[:, 1]
        )

    @staticmethod
    def _validated_arm_vector(values: np.ndarray, label: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (6,):
            raise ValueError(f"{label} must have shape (6,), got {vector.shape}")
        if not np.isfinite(vector).all():
            raise YAMLimitViolation(f"{label} contains non-finite values")
        return vector

    def _check_position(
        self, position: np.ndarray, side: str, tolerance: float = 0.0
    ) -> None:
        below = self._position_limits[:, 0] - position - tolerance
        above = position - self._position_limits[:, 1] - tolerance
        violation = np.maximum(below, above)
        if np.any(violation > 0):
            joint = int(np.argmax(violation))
            lower, upper = self._position_limits[joint]
            raise YAMLimitViolation(
                f"{side} J{joint + 1} position {position[joint]:.5f} rad is "
                f"outside [{lower:.5f}, {upper:.5f}] rad"
            )


class BimanualYAMHardware:
    """Two real YAM followers guarded by the shared simulation contract."""

    def __init__(
        self,
        left_robot: Any,
        right_robot: Any,
        *,
        control_frequency_hz: float = YAM_POLICY_CONTROL_FREQUENCY_HZ,
    ) -> None:
        self.left_robot = left_robot
        self.right_robot = right_robot
        self.limiter = BimanualYAMCommandLimiter(control_frequency_hz)
        self._validate_runtime_config()
        observation = self.get_policy_observation(validate=False)
        self.limiter.reset(observation[:6], observation[7:13])

    @classmethod
    def connect(
        cls,
        *,
        left_channel: str = "can_follower_l",
        right_channel: str = "can_follower_r",
        control_frequency_hz: float = YAM_POLICY_CONTROL_FREQUENCY_HZ,
    ) -> "BimanualYAMHardware":
        """Connect to two standard YAMs using I2RT's official SDK."""
        try:
            from i2rt.robots.get_robot import get_yam_robot
            from i2rt.robots.utils import ArmType, GripperType
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "the official i2rt package is required for real YAM deployment"
            ) from error

        left = get_yam_robot(
            channel=left_channel,
            arm_type=ArmType.YAM,
            gripper_type=GripperType.LINEAR_4310,
            zero_gravity_mode=False,
        )
        right = None
        try:
            right = get_yam_robot(
                channel=right_channel,
                arm_type=ArmType.YAM,
                gripper_type=GripperType.LINEAR_4310,
                zero_gravity_mode=False,
            )
            return cls(
                left,
                right,
                control_frequency_hz=control_frequency_hz,
            )
        except Exception:
            if right is not None:
                right.enter_gravity_comp_idle()
                right.close()
            left.enter_gravity_comp_idle()
            left.close()
            raise

    def command_policy_action(self, policy_action: np.ndarray) -> None:
        """Validate feedback, limit one policy action, and command both arms."""
        self.get_policy_observation(validate=True)
        left_command, right_command = self.limiter.limit_policy_action(policy_action)
        try:
            self.left_robot.command_joint_pos(left_command)
            self.right_robot.command_joint_pos(right_command)
        except Exception:
            self.enter_safe_idle()
            raise

    def get_policy_observation(self, *, validate: bool = True) -> np.ndarray:
        """Return ``left 6 + gripper + right 6 + gripper`` hardware state."""
        left = self.left_robot.get_observations()
        right = self.right_robot.get_observations()
        if validate:
            try:
                self.limiter.validate_feedback(
                    "left", left["joint_pos"], left["joint_vel"], left["joint_eff"]
                )
                self.limiter.validate_feedback(
                    "right",
                    right["joint_pos"],
                    right["joint_vel"],
                    right["joint_eff"],
                )
            except YAMLimitViolation:
                self.enter_safe_idle()
                raise
        return np.concatenate(
            (
                left["joint_pos"],
                np.asarray(left["gripper_pos"]).reshape(1),
                right["joint_pos"],
                np.asarray(right["gripper_pos"]).reshape(1),
            )
        ).astype(np.float64)

    def enter_safe_idle(self) -> None:
        """Return both arms to the SDK's gravity-compensated idle mode."""
        for robot in (self.left_robot, self.right_robot):
            try:
                robot.enter_gravity_comp_idle()
            except Exception:
                pass

    def close(self) -> None:
        self.enter_safe_idle()
        self.left_robot.close()
        self.right_robot.close()

    def _validate_runtime_config(self) -> None:
        expected_kp = np.asarray(YAM_KP)
        expected_kd = np.asarray(YAM_KD)
        for side, robot in (("left", self.left_robot), ("right", self.right_robot)):
            info = robot.get_robot_info()
            actual_kp = np.asarray(info["kp"], dtype=np.float64)
            actual_kd = np.asarray(info["kd"], dtype=np.float64)
            if not np.allclose(actual_kp[:6], expected_kp) or not np.allclose(
                actual_kd[:6], expected_kd
            ):
                raise RuntimeError(
                    f"{side} YAM SDK gains do not match the deployment contract: "
                    f"kp={actual_kp.tolist()} kd={actual_kd.tolist()}"
                )
            if actual_kp.shape[0] < 7 or actual_kd.shape[0] < 7 or not np.isclose(
                actual_kp[6], YAM_GRIPPER_KP
            ) or not np.isclose(actual_kd[6], YAM_GRIPPER_KD):
                raise RuntimeError(
                    f"{side} YAM gripper gains do not match I2RT linear_4310: "
                    f"kp={actual_kp.tolist()} kd={actual_kd.tolist()}"
                )

            expected_runtime_arrays = {
                "joint_limits": np.asarray(YAM_HARDWARE_POSITION_LIMITS_RAD),
                "gravity_comp_factor": np.asarray([*YAM_GRAVITY_COMP_FACTOR, 1.0]),
                "grav_comp_kd": np.asarray([*YAM_GRAV_COMP_KD, 0.0]),
                "coulomb_friction": np.asarray([*YAM_COULOMB_FRICTION_NM, 0.0]),
            }
            for key, expected in expected_runtime_arrays.items():
                actual = np.asarray(info[key], dtype=np.float64)
                if not np.allclose(actual, expected):
                    raise RuntimeError(
                        f"{side} YAM SDK {key} does not match the official config: "
                        f"{actual.tolist()} != {expected.tolist()}"
                    )
            if bool(info.get("use_coulomb_friction", False)) is not YAM_USE_COULOMB_FRICTION:
                raise RuntimeError(
                    f"{side} YAM use_coulomb_friction does not match the official default"
                )
            gripper_limit = info.get("limit_gripper_effort")
            if gripper_limit is not None and not np.isclose(
                float(gripper_limit), YAM_GRIPPER_FORCE_LIMIT_N
            ):
                raise RuntimeError(
                    f"{side} gripper force limit {gripper_limit} N does not match "
                    f"{YAM_GRIPPER_FORCE_LIMIT_N} N"
                )
