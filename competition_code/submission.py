print("submission.py loaded")

from typing import List
import roar_py_interface
import numpy as np


def normalize_rad(rad: float):
    """Normalize angle to [-pi, pi)."""
    return (rad + np.pi) % (2 * np.pi) - np.pi


def filter_waypoints(location: np.ndarray, current_idx: int, waypoints: List[roar_py_interface.RoarPyWaypoint]) -> int:
    loc2d = location[:2]
    for i in range(current_idx, current_idx + len(waypoints)):
        wp = waypoints[i % len(waypoints)]
        if np.linalg.norm(loc2d - wp.location[:2]) < 3:
            return i % len(waypoints)
    return current_idx


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

        self._prev_steer = 0.0
        self._steer_alpha = 0.3
        self._base_speed = 20.0

    async def initialize(self) -> None:
        self._prev_steer = 0.0
        self._last_yaw = None
        self._last_dt = 0.05

        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        heading = np.array([np.cos(vehicle_rotation[2]), np.sin(vehicle_rotation[2])])

        # Pick closest waypoint in front
        best_idx, min_cost = 0, np.inf
        for i, wp in enumerate(self.maneuverable_waypoints):
            vec = wp.location[:2] - vehicle_location[:2]
            if np.dot(vec, heading) > 0:  # in front
                dist = np.linalg.norm(vec)
                if dist < min_cost:
                    min_cost, best_idx = dist, i
        self.current_waypoint_idx = best_idx

    async def step(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        v = float(np.linalg.norm(self.velocity_sensor.get_last_gym_observation()))
        yaw = float(vehicle_rotation[2])
        pos_xy = vehicle_location[:2]

        if self._last_yaw is None:
            self._last_yaw = yaw

        # Update to nearest waypoint
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        # Collect waypoints ahead
        num_fit_points, lookahead_distance = 10, 6.0
        wp_xy = np.array([
            self.maneuverable_waypoints[(self.current_waypoint_idx + i) % len(self.maneuverable_waypoints)].location[:2]
            for i in range(num_fit_points)
        ])
        print(f"Current idx: {self.current_waypoint_idx}, Waypoint: {wp_xy[0]}")

        # Transform into vehicle local frame
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        local_wp = (wp_xy - pos_xy) @ rot.T

        # Prevent driving backwards
        if sum(pt[0] > 0.5 for pt in local_wp) < len(local_wp) // 2:
            print("Too many waypoints behind car — skipping step")
            return

        # Quadratic fit
        x, y = local_wp[:, 0], local_wp[:, 1]
        coeffs = np.polyfit(x, y, 2)

        # Steering
        y_target = np.polyval(coeffs, lookahead_distance)
        steer_angle = np.arctan2(2.0 * y_target, lookahead_distance)
        raw_steer = float(np.clip(steer_angle / 0.7, -1.0, 1.0))
        self._prev_steer = (1 - self._steer_alpha) * self._prev_steer + self._steer_alpha * raw_steer

        # Speed control
        curvature = abs(coeffs[0])
        target_speed = max(6.0, self._base_speed * (1.0 - 10.0 * curvature))
        accel = 0.10 * (target_speed - v)
        throttle = np.clip(accel, 0.0, 1.0)
        brake = np.clip(-accel, 0.0, 1.0)

        control = {
            "throttle": throttle,
            "steer": self._prev_steer,
            "brake": brake,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0,
        }
        await self.vehicle.apply_action(control)
        return control
