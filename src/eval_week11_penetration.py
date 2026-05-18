import os
import random
from dataclasses import dataclass

import gymnasium as gym
import highway_env
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


@dataclass
class EvaluationConfig:
    vehicles_count: int = 60
    lanes_count: int = 2
    duration: int = 50
    simulation_frequency: int = 15
    policy_frequency: int = 5
    episodes_per_scenario: int = 10
    throughput_x: float = 500.0
    model_path: str = "models/ppo_2lane_week11_tuned_final"
    output_csv: str = "results/week11_penetration_eval.csv"


def make_eval_env(penetration_rate: int, config: EvaluationConfig):
    """
    침투율별 평가 환경 생성.

    penetration_rate = 0:
        Baseline 조건. controlled vehicle은 1대로 두고 IDLE action만 사용한다.

    penetration_rate = 5 or 10:
        전체 차량 중 일부를 controlled vehicle로 설정하고,
        MultiAgentObservation / MultiAgentAction 구조로 PPO action을 적용한다.
    """

    controlled_count = int(config.vehicles_count * penetration_rate / 100)

    if controlled_count < 1:
        controlled_count = 1

    env = gym.make("highway-v0", render_mode=None)

    # 0% baseline은 단일 observation/action 구조 유지
    if penetration_rate == 0:
        observation_config = {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "vehicles_count": 15,
            "normalize": True,
            "absolute": False,
            "order": "sorted",
        }

        action_config = {
            "type": "DiscreteMetaAction",
        }

    # 5%, 10%는 다중 PPO 에이전트 평가 구조
    else:
        observation_config = {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "vehicles_count": 15,
                "normalize": True,
                "absolute": False,
                "order": "sorted",
            },
        }

        action_config = {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
            },
        }

    env.unwrapped.configure(
        {
            "lanes_count": config.lanes_count,
            "vehicles_count": config.vehicles_count,
            "vehicles_density": 2.0,
            "controlled_vehicles": controlled_count,
            "duration": config.duration,
            "simulation_frequency": config.simulation_frequency,
            "policy_frequency": config.policy_frequency,
            "observation": observation_config,
            "action": action_config,
        }
    )

    env.reset()
    return env, controlled_count


def predict_actions(model, obs, controlled_count, penetration_rate):
    """
    PPO action 생성.

    0% Baseline:
        PPO 모델을 사용하지 않고 IDLE action=1 사용.

    5%, 10%:
        MultiAgentObservation으로 들어온 각 agent observation에 대해
        PPO 모델이 action을 하나씩 예측한다.
    """

    # 0% baseline은 PPO를 사용하지 않음
    if penetration_rate == 0:
        return 1

    actions = []

    # MultiAgentObservation은 보통 obs가 list/tuple 형태로 들어온다.
    if isinstance(obs, (list, tuple)):
        for single_obs in obs:
            action, _ = model.predict(single_obs, deterministic=True)
            actions.append(int(action))
        return tuple(actions)

    # numpy array로 들어오는 경우 처리
    obs_array = np.asarray(obs)

    # 정상적인 다중 에이전트 observation: (N, 15, 5)
    if obs_array.ndim == 3:
        for i in range(obs_array.shape[0]):
            single_obs = obs_array[i]
            action, _ = model.predict(single_obs, deterministic=True)
            actions.append(int(action))
        return tuple(actions)

    # 혹시 단일 observation으로 들어온 경우: (15, 5)
    if obs_array.ndim == 2:
        action, _ = model.predict(obs_array, deterministic=True)
        return int(action)

    raise ValueError(f"Unexpected observation shape: {obs_array.shape}")


def collect_step_metrics(env, previous_positions, throughput_x):
    """
    현재 step에서 전체 차량 평균 속도와 throughput을 계산한다.

    throughput:
        차량이 throughput_x 지점을 처음 통과하면 1대로 카운트.
    """

    vehicles = env.unwrapped.road.vehicles

    speeds = []
    passed_count = 0

    for vehicle in vehicles:
        vehicle_id = id(vehicle)
        x_position = float(vehicle.position[0])
        speed = float(vehicle.speed)

        speeds.append(speed)

        prev_x = previous_positions.get(vehicle_id)

        if prev_x is not None:
            if prev_x < throughput_x <= x_position:
                passed_count += 1

        previous_positions[vehicle_id] = x_position

    avg_speed = float(np.mean(speeds)) if speeds else 0.0

    return avg_speed, passed_count, previous_positions


def run_one_episode(model, penetration_rate, episode_idx, config):
    env, controlled_count = make_eval_env(penetration_rate, config)

    obs, info = env.reset()

    previous_positions = {}
    step_avg_speeds = []
    total_throughput = 0
    crashed_count = 0
    step_count = 0
    crashed_vehicle_ids = set()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = predict_actions(
            model=model,
            obs=obs,
            controlled_count=controlled_count,
            penetration_rate=penetration_rate,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        avg_speed, passed_count, previous_positions = collect_step_metrics(
            env=env,
            previous_positions=previous_positions,
            throughput_x=config.throughput_x,
        )

        step_avg_speeds.append(avg_speed)
        total_throughput += passed_count

        for vehicle in env.unwrapped.road.vehicles:
            if getattr(vehicle, "crashed", False):
                crashed_vehicle_ids.add(id(vehicle))

        step_count += 1

    env.close()

    episode_avg_speed = float(np.mean(step_avg_speeds)) if step_avg_speeds else 0.0

    return {
        "penetration_rate": penetration_rate,
        "episode": episode_idx,
        "controlled_count": controlled_count if penetration_rate > 0 else 0,
        "vehicles_count": config.vehicles_count,
        "avg_speed": episode_avg_speed,
        "throughput": total_throughput,
        "crashed_count": crashed_count,
        "steps": step_count,
    }


def main():
    config = EvaluationConfig()

    os.makedirs("results", exist_ok=True)

    print("[week11] 침투율별 성능 평가를 시작합니다.")
    print(f"[week11] PPO model path: {config.model_path}.zip")

    model = PPO.load(config.model_path)

    penetration_rates = [0, 5, 10]
    all_results = []

    for rate in penetration_rates:
        print(f"\n[week11] penetration_rate={rate}% 평가 시작")

        for episode in range(1, config.episodes_per_scenario + 1):
            result = run_one_episode(
                model=model,
                penetration_rate=rate,
                episode_idx=episode,
                config=config,
            )

            all_results.append(result)

            print(
                f"rate={rate:>2}% | "
                f"episode={episode:>2} | "
                f"avg_speed={result['avg_speed']:.2f} | "
                f"throughput={result['throughput']} | "
                f"crashed={result['crashed_count']}"
            )

    df = pd.DataFrame(all_results)
    df.to_csv(config.output_csv, index=False)

    print("\n[week11] 평가 완료")
    print(f"[week11] 결과 CSV 저장: {config.output_csv}")


if __name__ == "__main__":
    main()