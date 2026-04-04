import gymnasium as gym
import highway_env
from highway_env.vehicle.behavior import IDMVehicle
import pandas as pd
import pprint


def run_baseline_simulation():
    print("베이스라인 시뮬레이션(에이전트 없음)을 시작합니다...")

    # 1. 2차선 커스텀 환경 생성 (화면 렌더링 켜기)
    env = gym.make("highway-v0", render_mode="human")

    # highway-env 커스텀 파라미터 덮어쓰기
    env.unwrapped.configure(
        {
            "lanes_count": 2,  # 2차선 고속도로
            "vehicles_count": 60,  # 차량을 빽빽하게 60대 배치
            "vehicles_density": 2.2,  # 차량 밀도 (높을수록 차간 거리가 좁아짐)
            "controlled_vehicles": 1,  # 핵심: RL 에이전트 1대 (모두 IDM+MOBIL 인간 운전자)
            "duration": 150,  # 시뮬레이션 지속 시간 (프레임)
            "simulation_frequency": 15,  # 물리 엔진 초당 연산 횟수
            "policy_frequency": 1,  # 의사결정 주기
            "show_trajectories": True,  # 화면에 차량 궤적 표시 (정체 확인용)
        }
    )

    # 초기화
    env.reset()

    # ---------------------------------------------------------
    # 😈 정체 유발 트리거: 앞쪽 차량 하나를 '느린 트럭(빌런)'으로 만들기
    # 다른 차량들의 목표 속도는 보통 25~30m/s 이지만, 이 차량만 15m/s (약 54km/h)로 고정합니다.
    villain = env.unwrapped.road.vehicles[5]  # 배열의 앞쪽에 있는 차량 하나 선택
    villain.target_speed = 15.0
    villain.color = (200, 0, 0)  # 렌더링 화면에서 눈에 띄게 빨간색으로 칠하기 (옵션)
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 🎩 마법의 코드: 멍청한 RL 에이전트를 똑똑한 사람(NPC)으로 바꿔치기
    ego = env.unwrapped.controlled_vehicles[0]
    npc_ego = IDMVehicle.create_from(ego)  # 에이전트와 완벽히 똑같은 NPC 복제

    # 도로에서 기존 에이전트 삭제하고 복제한 NPC 투입
    env.unwrapped.road.vehicles.remove(ego)
    env.unwrapped.road.vehicles.append(npc_ego)

    # 시뮬레이터가 에러를 뿜지 않게 통제권(조이스틱)을 NPC에게 쥐어줌
    env.unwrapped.controlled_vehicles[0] = npc_ego
    # ---------------------------------------------------------

    # 차량 속도 및 위치 기록용 리스트
    log_data = []

    done = truncated = False
    step_count = 0

    # 2. 에이전트 없는 시뮬레이션 루프 실행
    while not (done or truncated):
        # controlled_vehicles가 0이므로, action은 None으로 넘김
        obs, reward, done, truncated, info = env.step(1)

        # 3. 로깅 데이터 추출: 현재 프레임의 모든 차량 상태 기록
        for vehicle in env.unwrapped.road.vehicles:
            log_data.append(
                {
                    "step": step_count,
                    "vehicle_id": id(vehicle),  # 차량 고유 ID
                    "x_position": vehicle.position[0],
                    "y_position": vehicle.position[1],
                    "speed": vehicle.speed,  # 현재 속도 (이후 평균 속도 비교에 사용)
                    "lane_index": vehicle.lane_index[2],  # 현재 차선
                }
            )

        step_count += 1

    env.close()

    # 4. 수집된 데이터를 CSV 파일로 저장
    df = pd.DataFrame(log_data)
    df.to_csv("baseline_log.csv", index=False)

    print("시뮬레이션 완료!")
    print(f"총 {step_count} 스텝의 로그가 'baseline_log.csv' 파일로 저장되었습니다.")
    print(f"평균 속도: {df['speed'].mean():.2f} m/s")


if __name__ == "__main__":
    run_baseline_simulation()
