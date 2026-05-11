import os
import gymnasium as gym
import highway_env
import pandas as pd


class PhantomJamRewardWrapper(gym.Wrapper):
    """
    10주차 본 학습용 Reward Wrapper

    기존 week5_env_wrapper.py의 다중 목표 보상 구조를 유지하면서,
    TensorBoard 모니터링을 위해 info에 추가 지표를 기록한다.
    """

    def __init__(
        self,
        env,
        log_filename="logs/week10_training_log.csv",
        w_speed=1.0,
        w_col=5.0,
        w_lane=0.2,
        w_global=1.5,
        target_speed=25.0,
        enable_logging=False,
    ):
        super().__init__(env)

        self.log_filename = log_filename
        self.enable_logging = enable_logging

        self.w_speed = w_speed
        self.w_col = w_col
        self.w_lane = w_lane
        self.w_global = w_global
        self.target_speed = target_speed

        self.log_data = []
        self.step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.step_count = 0
        self.log_data = []

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

        total_reward = (
            self.w_speed * r_speed
            + self.w_col * r_collision
            + self.w_lane * r_lane_change
            + self.w_global * r_global
        )

        reward_info = {
            "custom/speed_reward": r_speed,
            "custom/collision_penalty": r_collision,
            "custom/lane_change_penalty": r_lane_change,
            "custom/global_reward": r_global,
            "custom/ego_speed": ego_vehicle.speed,
            "custom/avg_rear_speed": avg_rear_speed,
            "custom/crashed": crashed,
            "custom/is_lane_change": is_lane_change,
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
                }
            )

    def _save_log(self):
        if not self.log_data:
            return

        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        df = pd.DataFrame(self.log_data)
        df.to_csv(self.log_filename, index=False)

        print(f"[week10] log saved: {self.log_filename}")


def create_phantom_jam_env(
    render_mode=None,
    enable_logging=False,
    log_filename="logs/week10_training_log.csv",
    reward_config=None,
):
    """
    10주차 본 학습용 환경 생성 함수

    render_mode=None:
        본 학습용. 화면 렌더링 없이 빠르게 학습.

    render_mode="human":
        학습된 모델 테스트용. 차량 움직임 확인.
    """

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

    # configure 이후 observation space 동기화
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