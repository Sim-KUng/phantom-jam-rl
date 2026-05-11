import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from week10_env_wrapper import create_phantom_jam_env


class DrivingMetricsCallback(BaseCallback):
    """
    TensorBoard에 10주차 핵심 지표를 기록하는 Callback

    기록 지표:
    - custom/collision_rate
    - custom/lane_change_ratio
    - custom/avg_ego_speed
    - custom/avg_rear_speed
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.collision_count = 0
        self.lane_change_count = 0
        self.action_count = 0

        self.ego_speed_sum = 0.0
        self.rear_speed_sum = 0.0
        self.speed_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            self.action_count += 1

            if info.get("custom/crashed", False):
                self.collision_count += 1

            if info.get("custom/is_lane_change", False):
                self.lane_change_count += 1

            self.ego_speed_sum += info.get("custom/ego_speed", 0.0)
            self.rear_speed_sum += info.get("custom/avg_rear_speed", 0.0)
            self.speed_count += 1

        return True

    def _on_rollout_end(self) -> None:
        """
        PPO는 n_steps만큼 rollout을 모은 뒤 학습 업데이트를 수행한다.
        이 시점에 custom 지표를 기록하면 TensorBoard에 안정적으로 표시된다.
        """

        collision_rate = self.collision_count / max(self.action_count, 1)
        lane_change_ratio = self.lane_change_count / max(self.action_count, 1)
        avg_ego_speed = self.ego_speed_sum / max(self.speed_count, 1)
        avg_rear_speed = self.rear_speed_sum / max(self.speed_count, 1)

        self.logger.record("custom/collision_rate", collision_rate)
        self.logger.record("custom/lane_change_ratio", lane_change_ratio)
        self.logger.record("custom/avg_ego_speed", avg_ego_speed)
        self.logger.record("custom/avg_rear_speed", avg_rear_speed)

        # 다음 rollout 구간을 위해 초기화
        self.collision_count = 0
        self.lane_change_count = 0
        self.action_count = 0

        self.ego_speed_sum = 0.0
        self.rear_speed_sum = 0.0
        self.speed_count = 0


def main():
    print("[week10] 2차선 PPO 본 학습을 시작합니다.")

    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    reward_config = {
        "w_speed": 1.0,
        "w_col": 5.0,
        "w_lane": 0.2,
        "w_global": 1.5,
        "target_speed": 25.0,
    }

    # 본 학습에서는 render_mode=None으로 해야 빠르게 학습됨
    env = create_phantom_jam_env(
        render_mode=None,
        enable_logging=False,
        reward_config=reward_config,
    )

    env = Monitor(env)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/tensorboard/",
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix="ppo_2lane_week10",
    )

    metrics_callback = DrivingMetricsCallback()
    
    model.learn(
        total_timesteps=100_000,
        tb_log_name="ppo_2lane_week10_main",
        callback=[checkpoint_callback, metrics_callback],
        progress_bar=True,
    )

    final_model_path = "models/ppo_2lane_week10_final"
    model.save(final_model_path)

    env.close()

    print(f"[week10] 본 학습 완료: {final_model_path}.zip")


if __name__ == "__main__":
    main()