import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    input_csv = "results/week11_penetration_eval.csv"
    output_dir = "results/week11_figures"

    if not os.path.exists(input_csv):
        print(f"오류: {input_csv} 파일이 없습니다.")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # 1. 침투율별 평균 속도 요약
    speed_summary = (
        df.groupby("penetration_rate")["avg_speed"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        speed_summary["penetration_rate"].astype(str),
        speed_summary["mean"],
        yerr=speed_summary["std"],
        capsize=5,
    )
    plt.xlabel("Penetration Rate (%)")
    plt.ylabel("Average Speed")
    plt.title("Average Speed by PPO Agent Penetration Rate")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/week11_avg_speed_by_penetration.png", dpi=300)
    plt.close()

    # 2. 침투율별 throughput 요약
    throughput_summary = (
        df.groupby("penetration_rate")["throughput"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        throughput_summary["penetration_rate"].astype(str),
        throughput_summary["mean"],
        yerr=throughput_summary["std"],
        capsize=5,
    )
    plt.xlabel("Penetration Rate (%)")
    plt.ylabel("Throughput")
    plt.title("Throughput by PPO Agent Penetration Rate")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/week11_throughput_by_penetration.png", dpi=300)
    plt.close()

    # 3. episode별 평균 속도 변화
    plt.figure(figsize=(9, 5))
    for rate in sorted(df["penetration_rate"].unique()):
        subset = df[df["penetration_rate"] == rate]
        plt.plot(
            subset["episode"],
            subset["avg_speed"],
            marker="o",
            label=f"{rate}%"
        )

    plt.xlabel("Episode")
    plt.ylabel("Average Speed")
    plt.title("Episode-wise Average Speed")
    plt.legend(title="Penetration")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/week11_episode_avg_speed.png", dpi=300)
    plt.close()

    # 4. episode별 throughput 변화
    plt.figure(figsize=(9, 5))
    for rate in sorted(df["penetration_rate"].unique()):
        subset = df[df["penetration_rate"] == rate]
        plt.plot(
            subset["episode"],
            subset["throughput"],
            marker="o",
            label=f"{rate}%"
        )

    plt.xlabel("Episode")
    plt.ylabel("Throughput")
    plt.title("Episode-wise Throughput")
    plt.legend(title="Penetration")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/week11_episode_throughput.png", dpi=300)
    plt.close()

    print("[week11] 시각화 완료")
    print(f"저장 위치: {output_dir}")


if __name__ == "__main__":
    main()