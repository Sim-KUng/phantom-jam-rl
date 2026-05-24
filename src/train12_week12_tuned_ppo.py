import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from week12_env_wrapper import create_phantom_jam_env


class Week12StableMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.lane_change_count = 0
        self.hard_brake_count = 0
        self.collision_count = 0
        self.action_count = 0

        self.ego_speed_sum = 0.0
        self.rear_speed_sum = 0.0
        self.speed_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            self.action_count += 1

            if info.get("custom/is_lane_change", False):
                self.lane_change_count += 1

            if info.get("custom/is_hard_brake", False):
                self.hard_brake_count += 1

            if info.get("custom/crashed", False):
                self.collision_count += 1

            self.ego_speed_sum += info.get("custom/ego_speed", 0.0)
            self.rear_speed_sum += info.get("custom/avg_rear_speed", 0.0)
            self.speed_count += 1

        return True

    def _on_rollout_end(self) -> None:
        lane_change_ratio = self.lane_change_count / max(self.action_count, 1)
        hard_brake_ratio = self.hard_brake_count / max(self.action_count, 1)
        collision_rate = self.collision_count / max(self.action_count, 1)
        avg_ego_speed = self.ego_speed_sum / max(self.speed_count, 1)
        avg_rear_speed = self.rear_speed_sum / max(self.speed_count, 1)

        self.logger.record("custom/lane_change_ratio", lane_change_ratio)
        self.logger.record("custom/hard_brake_ratio", hard_brake_ratio)
        self.logger.record("custom/collision_rate", collision_rate)
        self.logger.record("custom/avg_ego_speed", avg_ego_speed)
        self.logger.record("custom/avg_rear_speed", avg_rear_speed)

        self.lane_change_count = 0
        self.hard_brake_count = 0
        self.collision_count = 0
        self.action_count = 0

        self.ego_speed_sum = 0.0
        self.rear_speed_sum = 0.0
        self.speed_count = 0


def main():
    print("[week12] 안정성 우선 재튜닝 PPO 추가 학습을 시작합니다.")

    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    reward_config = {
        "w_speed": 1.2,
        "w_col": 10.0,
        "w_lane": 0.8,
        "w_global": 1.2,
        "w_hard_brake": 2.0,
        "target_speed": 23.0,
        "hard_brake_decel_threshold": 2.5,
    }

    env = create_phantom_jam_env(
        render_mode=None,
        enable_logging=False,
        reward_config=reward_config,
    )

    env = Monitor(env)

    # 직전 모델에서 이어서 학습
    # 너무 공격적으로 변한 week12_tuned_final에서 이어서 하되,
    # 이번에는 안정성 보상으로 방향을 되돌림
    model_path = "models/ppo_2lane_week12_tuned_final"

    model = PPO.load(
        model_path,
        env=env,
        tensorboard_log="./logs/tensorboard/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix="ppo_2lane_week12_stable",
    )

    metrics_callback = Week12StableMetricsCallback()

    model.learn(
        total_timesteps=100_000,
        tb_log_name="ppo_2lane_week12_stable",
        callback=[checkpoint_callback, metrics_callback],
        reset_num_timesteps=False,
        progress_bar=True,
    )

    final_model_path = "models/ppo_2lane_week12_stable_final"
    model.save(final_model_path)

    env.close()

    print(f"[week12] 안정성 우선 재튜닝 완료: {final_model_path}.zip")


if __name__ == "__main__":
    main()