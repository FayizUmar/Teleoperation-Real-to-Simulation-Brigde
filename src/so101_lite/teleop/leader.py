"""Physical leader-arm factory and serial-port diagnostics for teleop.

All ``lerobot`` imports are deferred into :func:`get_leader` so this module can
be imported without the optional ``teleop`` extra installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from so101_lite.config import SO101_JOINT_NAMES

DEFAULT_WRIST_ROLL_OFFSET_DEG = -90.0

ROBOT_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "so100": SO101_JOINT_NAMES,
    "so101": SO101_JOINT_NAMES,
}


class LeaderProtocol(Protocol):
    """Minimal interface the teleop loop requires from a leader-arm driver."""

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_action(self) -> dict: ...


@dataclass(frozen=True)
class LeaderPortDiagnostic:
    """Structured diagnosis for a requested leader-arm serial path."""

    kind: str
    message: str
    recovery_hint: str = ""


def apply_wrist_roll_offset_deg(action: dict[str, float], offset_deg: float) -> dict[str, float]:
    """Return a copy of *action* with ``wrist_roll.pos`` shifted by *offset_deg*."""
    result = dict(action)
    if "wrist_roll.pos" in result:
        result["wrist_roll.pos"] = float(result["wrist_roll.pos"]) + offset_deg
    return result


def _permission_recovery_hint(port: str) -> str:
    return (
        "The device exists but the current user cannot open it.\n"
        "Fix the port permissions in another terminal, then retry here:\n"
        f"  sudo chmod 666 {port}\n"
        "Run 'lerobot-find-port' if you want to confirm the current device path."
    )


def diagnose_leader_port(port: str) -> LeaderPortDiagnostic:
    """Inspect *port* and return a structured diagnosis."""
    if not os.path.exists(port):
        return LeaderPortDiagnostic(
            kind="not_found",
            message=f"Serial device '{port}' was not found.",
            recovery_hint="Check the USB connection and run 'lerobot-find-port' to locate the arm.",
        )
    if not os.access(port, os.R_OK | os.W_OK):
        return LeaderPortDiagnostic(
            kind="permission_denied",
            message=f"Serial device '{port}' exists but is not readable/writable by this user.",
            recovery_hint=_permission_recovery_hint(port),
        )
    return LeaderPortDiagnostic(kind="ok", message=f"Serial device '{port}' looks accessible.")


def format_leader_connection_error(port: str, exc: Exception) -> str:
    """Return a user-facing connection error with recovery guidance."""
    details = str(exc).strip() or type(exc).__name__
    lower_details = details.lower()
    if "permission denied" in lower_details:
        diag = LeaderPortDiagnostic(
            kind="permission_denied",
            message=f"Serial device '{port}' rejected the connection with a permission error.",
            recovery_hint=_permission_recovery_hint(port),
        )
    elif "no such file" in lower_details or "could not connect on port" in lower_details:
        diag = LeaderPortDiagnostic(
            kind="not_found",
            message=f"Serial device '{port}' could not be opened.",
            recovery_hint=(
                "Check the USB connection and run 'lerobot-find-port' to locate the arm."
            ),
        )
    else:
        diag = diagnose_leader_port(port)

    message = f"Failed to connect on {port}: {details}"
    if diag.kind != "ok":
        message += f"\n{diag.message}"
    if diag.recovery_hint:
        message += f"\n{diag.recovery_hint}"
    return message


def get_leader(robot_type: str, port: str, leader_id: str) -> LeaderProtocol:
    """Create and return the appropriate ``SOLeader`` for *robot_type*."""
    if robot_type == "so100":
        from lerobot.teleoperators.so_leader.config_so_leader import SO100LeaderConfig
        from lerobot.teleoperators.so_leader.so_leader import SO100Leader

        return SO100Leader(SO100LeaderConfig(port=port, use_degrees=True, id=leader_id))

    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    return SO101Leader(SO101LeaderConfig(port=port, use_degrees=True, id=leader_id))


class DummyLeader:
    """Satisfies LeaderProtocol with no hardware — arm holds its rest pose."""

    _REST_DEG: tuple[float, ...] = (0.0, -90.0, 90.0, 37.8, 0.0, 50.0)

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_action(self) -> dict:
        names = (*SO101_JOINT_NAMES[:-1], "gripper")
        return {f"{name}.pos": v for name, v in zip(names, self._REST_DEG)}


def import_backend_for_env_id(env_id: str) -> None:
    """Ensure the so101-lite MuJoCo envs are registered for *env_id*."""
    import so101_lite.envs  # noqa: F401 — registers all MuJoCo env ids on import

    if env_id.startswith("MuJoCo"):
        return

    import gymnasium as gym

    try:
        gym.spec(env_id)
    except gym.error.Error as exc:
        raise ValueError(f"Unknown env_id {env_id!r}.") from exc
