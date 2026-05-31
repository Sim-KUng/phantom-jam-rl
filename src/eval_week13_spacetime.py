"""
Week 13: 시공간 다이어그램용 차량 궤적 데이터 수집

수정 사항 (v2):
- NoCrashTerminationWrapper 추가: 충돌 발생 시에도 에피소드가 종료되지 않고
  전체 duration 동안 계속 실행된다. 충돌 차량은 정지 상태로 시뮬레이션에 남는다.
- DURATION = 40초 (= 200 policy steps @ 5 Hz)

실행 방법:
    cd phantom-jam-rl/src
    python eval_week13_spacetime.py
"""

import os

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


VEHICLES_COUNT = 60
LANES_COUNT = 2
DURATION = 40           # 초 단위 (highway-env 기본값). 40초 = 200 policy steps @ 5Hz
SIMULATION_FREQUENCY = 15
POLICY_FREQUENCY = 5

MODEL_PATH = "models/ppo_2lane_week12_stable_final"
LOG_DIR = "logs"


# ──────────────────────────────────────────────
# 충돌 후에도 에피소드를 종료시키지 않는 래퍼
# ──────────────────────────────────────────────

class NoCrashTerminationWrapper(gym.Wrapper):
    """
    highway-env는 controlled vehicle이 충돌하면 terminated=True를 반환한다.
    이 래퍼는 terminated를 항상 False로 덮어써서, 에피소드가 전체 duration 동안
    계속 실행되도록 한다. 충돌 차량은 속도 0으로 정지하지만 시뮬레이션은 계속된다.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, False, truncated, info


# ──────────────────────────────────────────────
# 환경 생성
# ──────────────────────────────────────────────

def make_trajectory_env(penetration_rate: int, vehicles_count: int = VEHICLES_COUNT):
    controlled_count = max(1, int(vehicles_count * penetration_rate / 100))

    if penetration_rate == 0:
        observation_config = {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "vehicles_count": 15,
            "normalize": True,
            "absolute": False,
            "order": "sorted",
        }
        action_config = {"type": "DiscreteMetaAction"}
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
            "action_config": {"type": "DiscreteMetaAction"},
        }

    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure(
        {
            "lanes_count": LANES_COUNT,
            "vehicles_count": vehicles_count,
            "vehicles_density": 2.0,
            "controlled_vehicles": controlled_count,
            "duration": DURATION,
            "simulation_frequency": SIMULATION_FREQUENCY,
            "policy_frequency": POLICY_FREQUENCY,
            "observation": observation_config,
            "action": action_config,
        }
    )
    env.reset()

    # 충돌 후에도 에피소드가 계속 실행되도록 래퍼 적용
    env = NoCrashTerminationWrapper(env)

    return env, controlled_count


# ──────────────────────────────────────────────
# 액션 생성
# ──────────────────────────────────────────────

def get_action(model, obs, penetration_rate: int):
    if penetration_rate == 0:
        return 1  # IDLE

    obs_arr = np.asarray(obs)

    if isinstance(obs, (list, tuple)):
        return tuple(
            int(model.predict(np.asarray(o), deterministic=True)[0]) for o in obs
        )

    if obs_arr.ndim == 3:
        return tuple(
            int(model.predict(obs_arr[i], deterministic=True)[0])
            for i in range(obs_arr.shape[0])
        )

    return int(model.predict(obs_arr, deterministic=True)[0])


# ──────────────────────────────────────────────
# 궤적 수집
# ──────────────────────────────────────────────

def collect_episode_trajectories(
    penetration_rate: int, model, vehicles_count: int = VEHICLES_COUNT
) -> pd.DataFrame:
    env, controlled_count = make_trajectory_env(penetration_rate, vehicles_count)
    obs, _ = env.reset()

    # NoCrashTerminationWrapper로 감싼 경우 unwrapped를 두 단계 벗겨야 한다
    base_env = env.env.unwrapped
    controlled_ids = {id(v) for v in base_env.controlled_vehicles}

    records = []
    step = 0
    terminated = truncated = False

    while not (terminated or truncated):
        for vehicle in base_env.road.vehicles:
            records.append(
                {
                    "step": step,
                    "time_sec": round(step / POLICY_FREQUENCY, 2),
                    "vehicle_id": id(vehicle),
                    "x_position": float(vehicle.position[0]),
                    "speed": float(vehicle.speed),
                    "lane_index": int(vehicle.lane_index[2]),
                    "is_controlled": id(vehicle) in controlled_ids,
                    "crashed": bool(getattr(vehicle, "crashed", False)),
                }
            )

        action = get_action(model, obs, penetration_rate)
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1

    env.close()

    df = pd.DataFrame(records)
    print(f"  → {len(df):,} records | steps={step} | time={step/POLICY_FREQUENCY:.1f}s")
    return df


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    print("[week13] PPO 모델 로드 중...")
    model = PPO.load(MODEL_PATH)

    configs = [
        (0, "week13_baseline_traj.csv", "Baseline (0% PPO)"),
        (5, "week13_rl5pct_traj.csv",   "RL Agent (5% PPO)"),
    ]

    for rate, filename, label in configs:
        print(f"\n[week13] {label} 시뮬레이션 실행 중 (duration={DURATION}s)...")
        df = collect_episode_trajectories(rate, model)

        n_vehicles = df["vehicle_id"].nunique()
        avg_speed = df["speed"].mean()
        crash_count = df.groupby("vehicle_id")["crashed"].any().sum()

        output_path = os.path.join(LOG_DIR, filename)
        df.to_csv(output_path, index=False)

        print(f"     unique_vehicles={n_vehicles}, avg_speed={avg_speed:.2f} m/s, crashed_vehicles={crash_count}")
        print(f"     저장: {output_path}")

    print("\n[week13] 궤적 데이터 수집 완료.")
    print("다음 단계: python visualize_week13_spacetime.py")


if __name__ == "__main__":
    main()
