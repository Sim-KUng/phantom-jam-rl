import pandas as pd
import matplotlib.pyplot as plt


def plot_speed_degradation(
    csv_file="baseline_log.csv", output_img="speed_degradation.png"
):
    print(f"[{csv_file}] 데이터를 분석합니다...")

    # 1. 데이터 로드
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(
            f"오류: '{csv_file}' 파일이 없습니다. custom_highway.py를 먼저 실행해 주세요."
        )
        return

    # 2. 스텝(시간)별 전체 차량의 평균 속도 계산
    avg_speed_per_step = df.groupby("step")["speed"].mean()

    # 3. 그래프 스타일링 및 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(
        avg_speed_per_step.index,
        avg_speed_per_step.values,
        color="#FF5733",
        linewidth=2.5,
        label="Avg Speed (Baseline)",
    )

    # 4. 라벨 및 타이틀 설정
    plt.title(
        "Phantom Jam: Average Speed Degradation Over Time",
        fontsize=15,
        fontweight="bold",
    )
    plt.xlabel("Simulation Step (Time)", fontsize=12)
    plt.ylabel("Average Speed (m/s)", fontsize=12)

    # y축 범위를 0부터 시작하게 하여 속도 저하를 더 명확하게 보여줌
    plt.ylim(0, max(avg_speed_per_step.values) * 1.1)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # 5. 고화질(dpi=300) 이미지로 저장
    plt.savefig(output_img, dpi=300)
    print(f"✅ 그래프가 성공적으로 저장되었습니다: {output_img}")

    # 로컬에서 바로 확인하기 위해 창 띄우기
    plt.show()


if __name__ == "__main__":
    plot_speed_degradation()
