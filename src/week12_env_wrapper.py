import os
import gymnasium as gym
import highway_env
import pandas as pd


class PhantomJamRewardWrapper(gym.Wrapper):
    """
    Week12 튜닝용 Reward Wrapper

    기존 reward 구성:
    - 속도 유지 보상
    - 충돌 페널티
    - 차선 변경 페널티
    - 후방 차량 평균 속도 보상

    Week12 추가:
    - 급감속 페널티
    """

    def __init__(
        self,
        env,
        log_filename="logs/week12_training_log.csv",
        w_speed=1.5,
        w_col=7.0,
        w_lane=0.3,
        w_global=2.0,
        w_hard_brake=0.7,
        target_speed=25.0,
        hard_brake_decel_threshold=3.0,
        enable_logging=False,
    ):
        super().__init__(env)

        self.log_filename = log_filename
        self.enable_logging = enable_logging

        self.w_speed = w_speed
        self.w_col = w_col
        self.w_lane = w_lane
        self.w_global = w_global
        self.w_hard_brake = w_hard_brake

        self.target_speed = target_speed
        self.hard_brake_decel_threshold = hard_brake_decel_threshold

        self.log_data = []
        self.step_count = 0
        self.prev_ego_speed = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.step_count = 0
        self.log_data = []

        ego_vehicle = self.env.unwrapped.controlled_vehicles[0]
        self.prev_ego_speed = float(ego_vehicle.speed)

        return obs, info

    def step(self, action):
        obs, default_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        action_int = int(action)

        reward, reward_info = self._calculate_custom_reward(action_int)

        if self.enable_logging:
            self._record_state(action_int, reward, reward_info)

        info.update(reward_info)

        if terminated or truncated:
            if self.enable_logging:
                self._save_log()

        return obs, reward, terminated, truncated, info

    def _calculate_custom_reward(self, action):
        ego_vehicle = self.env.unwrapped.controlled_vehicles[0]
        vehicles = self.env.unwrapped.road.vehicles

        crashed = bool(ego_vehicle.crashed)

        # 1. 충돌 페널티
        r_collision = -1.0 if crashed else 0.0

        # 2. 속도 유지 보상
        if crashed:
            r_speed = 0.0
        else:
            r_speed = min(ego_vehicle.speed / self.target_speed, 1.0)

        # 3. 차선 변경 페널티
        is_lane_change = action in [0, 2]
        r_lane_change = -1.0 if is_lane_change else 0.0

        # 4. 후방 차량 흐름 보상
        rear_vehicles_speed = []

        for v in vehicles:
            if v is ego_vehicle:
                continue

            distance = ego_vehicle.position[0] - v.position[0]

            if 0 < distance < 100:
                rear_vehicles_speed.append(v.speed)

        if rear_vehicles_speed:
            avg_rear_speed = sum(rear_vehicles_speed) / len(rear_vehicles_speed)
            r_global = min(avg_rear_speed / self.target_speed, 1.0)
        else:
            avg_rear_speed = 0.0
            r_global = 0.0

        # 5. 급감속 페널티
        current_speed = float(ego_vehicle.speed)

        if self.prev_ego_speed is None:
            decel = 0.0
        else:
            # policy_frequency=5 기준, step 간 시간은 0.2초
            dt = 1.0 / 5.0
            decel = (self.prev_ego_speed - current_speed) / dt

        is_hard_brake = decel >= self.hard_brake_decel_threshold
        r_hard_brake = -1.0 if is_hard_brake else 0.0

        self.prev_ego_speed = current_speed

        total_reward = (
            self.w_speed * r_speed
            + self.w_col * r_collision
            + self.w_lane * r_lane_change
            + self.w_global * r_global
            + self.w_hard_brake * r_hard_brake
        )

        reward_info = {
            "custom/speed_reward": r_speed,
            "custom/collision_penalty": r_collision,
            "custom/lane_change_penalty": r_lane_change,
            "custom/global_reward": r_global,
            "custom/hard_brake_penalty": r_hard_brake,
            "custom/ego_speed": current_speed,
            "custom/avg_rear_speed": avg_rear_speed,
            "custom/deceleration": decel,
            "custom/crashed": crashed,
            "custom/is_lane_change": is_lane_change,
            "custom/is_hard_brake": is_hard_brake,
            "custom/action": action,
            "custom/total_reward": total_reward,
        }

        return total_reward, reward_info

    def _record_state(self, action, reward, reward_info):
        ego_vehicle = self.env.unwrapped.controlled_vehicles[0]

        for vehicle in self.env.unwrapped.road.vehicles:
            self.log_data.append(
                {
                    "step": self.step_count,
                    "vehicle_id": id(vehicle),
                    "x_position": vehicle.position[0],
                    "y_position": vehicle.position[1],
                    "speed": vehicle.speed,
                    "lane_index": vehicle.lane_index[2],
                    "is_ego": vehicle is ego_vehicle,
                    "action": action if vehicle is ego_vehicle else None,
                    "reward": reward if vehicle is ego_vehicle else None,
                    "crashed": reward_info["custom/crashed"] if vehicle is ego_vehicle else None,
                    "is_hard_brake": reward_info["custom/is_hard_brake"] if vehicle is ego_vehicle else None,
                }
            )

    def _save_log(self):
        if not self.log_data:
            return

        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        df = pd.DataFrame(self.log_data)
        df.to_csv(self.log_filename, index=False)

        print(f"[week12] log saved: {self.log_filename}")


def create_phantom_jam_env(
    render_mode=None,
    enable_logging=False,
    log_filename="logs/week12_training_log.csv",
    reward_config=None,
):
    env = gym.make("highway-v0", render_mode=render_mode)

    env.unwrapped.configure(
        {
            "lanes_count": 2,
            "vehicles_count": 60,
            "vehicles_density": 2.0,
            "controlled_vehicles": 1,
            "duration": 50,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "vehicles_count": 15,
                "normalize": True,
                "absolute": False,
                "order": "sorted",
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
        }
    )

    env.reset()

    if reward_config is None:
        reward_config = {}

    env = PhantomJamRewardWrapper(
        env,
        log_filename=log_filename,
        enable_logging=enable_logging,
        **reward_config,
    )

    return env