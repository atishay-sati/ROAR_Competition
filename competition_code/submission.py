"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""
print("submission.py loaded")

from typing import List
import roar_py_interface
import numpy as np


def normalize_rad(rad: float):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def filter_waypoints(location: np.ndarray, current_idx: int, waypoints: List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint: roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(location[:2] - waypoint.location[:2])
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i % len(waypoints)]) < 3:
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
        self._steer_alpha = 0.5   # stronger smoothing
        self._base_speed = 20.0   # normal cruising speed

    async def initialize(self) -> None:
        self._prev_steer = 0.0
        self._last_yaw = None
        self._last_dt = 0.05

        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_heading = np.array([np.cos(vehicle_rotation[2]), np.sin(vehicle_rotation[2])])

        # Pick a waypoint ahead of the car
        min_cost = np.inf
        best_idx = 0
        for i, wp in enumerate(self.maneuverable_waypoints):
            vec_to_wp = wp.location[:2] - vehicle_location[:2]
            if np.dot(vec_to_wp, vehicle_heading) > 0:  # must be in front
                cost = np.linalg.norm(vec_to_wp)
                if cost < min_cost:
                    min_cost = cost
                    best_idx = i

        # 🔹 Skip ahead to avoid clipping first corner
        self.current_waypoint_idx = (best_idx + 10) % len(self.maneuverable_waypoints)
        self._startup_counter = 50  # ~50 steps (~2–3 seconds) of cautious driving

    async def step(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        v = float(np.linalg.norm(vehicle_velocity))
        yaw = float(vehicle_rotation[2])
        pos_xy = vehicle_location[:2]

        if self._last_yaw is None:
            self._last_yaw = yaw

        # Update waypoint index
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        # --- PARAMETERS FOR SAFE DRIVING ---
        num_fit_points = 15
        lookahead_distance = 12.0
        max_speed = 20.0
        min_speed = 5.0

        # Gather waypoints ahead
        wp_xy = np.array([
            self.maneuverable_waypoints[(self.current_waypoint_idx + i) % len(self.maneuverable_waypoints)].location[:2]
            for i in range(num_fit_points)
        ])

        rot = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
        local_wp = (wp_xy - pos_xy) @ rot.T

        # Ensure most waypoints are in front
        forward_points = [pt for pt in local_wp if pt[0] > 1.0]
        if len(forward_points) < len(local_wp) // 2:
            print("Too many waypoints behind car — braking to re-center")
            await self.vehicle.apply_action({
                "throttle": 0.0, "steer": 0.0, "brake": 1.0,
                "hand_brake": 0.0, "reverse": 0, "target_gear": 0
            })
            return

        # Fit curve
        x = local_wp[:, 0]
        y = local_wp[:, 1]
        coeffs = np.polyfit(x, y, 2)

        # Target farther point
        x_target = lookahead_distance
        y_target = np.polyval(coeffs, x_target)

        # Steering
        steer_angle = np.arctan2(2.0 * y_target, lookahead_distance)
        raw_steer = float(np.clip(steer_angle / 0.7, -1.0, 1.0))
        steer_control = (1 - self._steer_alpha) * self._prev_steer + self._steer_alpha * raw_steer
        self._prev_steer = steer_control

        # --- SPEED CONTROL ---
        curvature = abs(coeffs[0])
        target_speed = max(min_speed, max_speed * (1.0 - 12.0 * curvature))
        target_speed = np.clip(target_speed, min_speed, max_speed)

        # During startup, keep speed capped
        if self._startup_counter > 0:
            self._startup_counter -= 1
            target_speed = min(target_speed, 8.0)

        speed_error = target_speed - v
        accel = 0.1 * speed_error
        throttle = np.clip(accel, 0.0, 1.0)
        brake = np.clip(-accel, 0.0, 1.0)

        # --- EMERGENCY COLLISION STOP ---
        if self.collision_sensor and self.collision_sensor.get_last_gym_observation():
            print("Collision detected — emergency stop")
            throttle, brake = 0.0, 1.0

        control = {
            "throttle": throttle,
            "steer": steer_control,
            "brake": brake,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control
