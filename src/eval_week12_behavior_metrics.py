import os
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

    # 통행량 측정 기준 지점
    throughput_x: float = 500.0

    # 정체 시간 측정 구간
    congestion_x_min: float = 200.0
    congestion_x_max: float = 500.0
    congestion_speed_threshold: float = 10.0  # m/s, 약 36km/h 이하를 정체로 판단
    min_vehicles_in_segment: int = 3

    # 급감속 기준
    hard_brake_decel_threshold: float = 3.0  # m/s^2 이상 감속이면 급감속으로 판단

    # 평가 모델
    model_path: str = "models/ppo_2lane_week12_stable_final"

    # 결과 저장
    output_csv: str = "results/week12_behavior_eval.csv"


def make_eval_env(penetration_rate: int, config: EvaluationConfig):
    """
    침투율별 평가 환경 생성.

    0%:
        PPO 차량 없음. Baseline으로 사용한다.

    5%, 10%:
        MultiAgentObservation / MultiAgentAction으로 여러 PPO 차량을 제어한다.
    """

    controlled_count = int(config.vehicles_count * penetration_rate / 100)

    if controlled_count < 1:
        controlled_count = 1

    env = gym.make("highway-v0", render_mode=None)

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


def predict_actions(model, obs, penetration_rate: int):
    """
    PPO action 생성.

    0% Baseline:
        PPO 모델을 쓰지 않고 IDLE action=1 사용.

    5%, 10%:
        각 controlled vehicle observation에 대해 PPO action을 예측한다.
    """

    if penetration_rate == 0:
        return 1

    actions = []

    if isinstance(obs, (list, tuple)):
        for single_obs in obs:
            action, _ = model.predict(single_obs, deterministic=True)
            actions.append(int(action))
        return tuple(actions)

    obs_array = np.asarray(obs)

    if obs_array.ndim == 3:
        for i in range(obs_array.shape[0]):
            single_obs = obs_array[i]
            action, _ = model.predict(single_obs, deterministic=True)
            actions.append(int(action))
        return tuple(actions)

    if obs_array.ndim == 2:
        action, _ = model.predict(obs_array, deterministic=True)
        return int(action)

    raise ValueError(f"Unexpected observation shape: {obs_array.shape}")


def initialize_vehicle_state(env):
    """
    차선 변경과 급감속 계산을 위해 차량별 이전 상태를 초기화한다.
    """

    previous_state = {}

    for vehicle in env.unwrapped.road.vehicles:
        previous_state[id(vehicle)] = {
            "lane_index": vehicle.lane_index[2],
            "speed": float(vehicle.speed),
            "x_position": float(vehicle.position[0]),
        }

    return previous_state


def collect_step_metrics(env, previous_state, config: EvaluationConfig):
    """
    한 step에서 차선 변경, 급감속, throughput, 정체 여부를 계산한다.
    """

    vehicles = env.unwrapped.road.vehicles

    lane_change_count = 0
    hard_brake_count = 0
    throughput_count = 0
    crashed_vehicle_ids = set()

    all_speeds = []
    segment_speeds = []

    dt = 1.0 / config.policy_frequency

    for vehicle in vehicles:
        vehicle_id = id(vehicle)

        current_lane = vehicle.lane_index[2]
        current_speed = float(vehicle.speed)
        current_x = float(vehicle.position[0])

        all_speeds.append(current_speed)

        if getattr(vehicle, "crashed", False):
            crashed_vehicle_ids.add(vehicle_id)

        if config.congestion_x_min <= current_x <= config.congestion_x_max:
            segment_speeds.append(current_speed)

        prev = previous_state.get(vehicle_id)

        if prev is not None:
            prev_lane = prev["lane_index"]
            prev_speed = prev["speed"]
            prev_x = prev["x_position"]

            # 1. 차선 변경 횟수
            if current_lane != prev_lane:
                lane_change_count += 1

            # 2. 급감속 횟수
            decel = (prev_speed - current_speed) / dt
            if decel >= config.hard_brake_decel_threshold:
                hard_brake_count += 1

            # 3. throughput
            if prev_x < config.throughput_x <= current_x:
                throughput_count += 1

        previous_state[vehicle_id] = {
            "lane_index": current_lane,
            "speed": current_speed,
            "x_position": current_x,
        }

    avg_speed = float(np.mean(all_speeds)) if all_speeds else 0.0

    if len(segment_speeds) >= config.min_vehicles_in_segment:
        segment_avg_speed = float(np.mean(segment_speeds))
        is_congested = segment_avg_speed < config.congestion_speed_threshold
    else:
        segment_avg_speed = 0.0
        is_congested = False

    return {
        "avg_speed": avg_speed,
        "segment_avg_speed": segment_avg_speed,
        "lane_change_count": lane_change_count,
        "hard_brake_count": hard_brake_count,
        "throughput_count": throughput_count,
        "is_congested": is_congested,
        "crashed_vehicle_ids": crashed_vehicle_ids,
    }, previous_state


def run_one_episode(model, penetration_rate: int, episode_idx: int, config: EvaluationConfig):
    env, controlled_count = make_eval_env(penetration_rate, config)

    obs, info = env.reset()

    previous_state = initialize_vehicle_state(env)

    step_avg_speeds = []
    step_segment_speeds = []

    total_lane_changes = 0
    total_hard_brakes = 0
    total_throughput = 0
    congestion_steps = 0
    crashed_vehicle_ids = set()

    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated):
        action = predict_actions(
            model=model,
            obs=obs,
            penetration_rate=penetration_rate,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        metrics, previous_state = collect_step_metrics(
            env=env,
            previous_state=previous_state,
            config=config,
        )

        step_avg_speeds.append(metrics["avg_speed"])
        step_segment_speeds.append(metrics["segment_avg_speed"])

        total_lane_changes += metrics["lane_change_count"]
        total_hard_brakes += metrics["hard_brake_count"]
        total_throughput += metrics["throughput_count"]

        if metrics["is_congested"]:
            congestion_steps += 1

        crashed_vehicle_ids.update(metrics["crashed_vehicle_ids"])

        step_count += 1

    env.close()

    episode_avg_speed = float(np.mean(step_avg_speeds)) if step_avg_speeds else 0.0
    episode_segment_avg_speed = (
        float(np.mean(step_segment_speeds)) if step_segment_speeds else 0.0
    )

    congestion_time_sec = congestion_steps * (1.0 / config.policy_frequency)

    return {
        "penetration_rate": penetration_rate,
        "episode": episode_idx,
        "controlled_count": controlled_count if penetration_rate > 0 else 0,
        "vehicles_count": config.vehicles_count,
        "avg_speed": episode_avg_speed,
        "segment_avg_speed": episode_segment_avg_speed,
        "throughput": total_throughput,
        "lane_change_count": total_lane_changes,
        "hard_brake_count": total_hard_brakes,
        "congestion_steps": congestion_steps,
        "congestion_time_sec": congestion_time_sec,
        "crashed_count": len(crashed_vehicle_ids),
        "steps": step_count,
    }


def main():
    config = EvaluationConfig()

    os.makedirs("results", exist_ok=True)

    print("[week12] 차선 변경, 급감속, 정체 시간 평가를 시작합니다.")
    print(f"[week12] PPO model path: {config.model_path}.zip")

    model = PPO.load(config.model_path)

    penetration_rates = [0, 5, 10]
    all_results = []

    for rate in penetration_rates:
        print(f"\n[week12] penetration_rate={rate}% 평가 시작")

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
                f"lane_changes={result['lane_change_count']} | "
                f"hard_brakes={result['hard_brake_count']} | "
                f"congestion_time={result['congestion_time_sec']:.2f}s | "
                f"crashed={result['crashed_count']}"
            )

    df = pd.DataFrame(all_results)
    df.to_csv(config.output_csv, index=False)

    print("\n[week12] 평가 완료")
    print(f"[week12] 결과 CSV 저장: {config.output_csv}")


if __name__ == "__main__":
    main()