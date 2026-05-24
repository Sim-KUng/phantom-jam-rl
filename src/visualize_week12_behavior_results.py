import os
import pandas as pd
import matplotlib.pyplot as plt


def save_bar_chart(summary, metric_col, std_col, ylabel, title, output_path):
    plt.figure(figsize=(8, 5))

    x = summary["penetration_rate"].astype(str)
    y = summary[metric_col]
    yerr = summary[std_col]

    plt.bar(x, y, yerr=yerr, capsize=5)
    plt.xlabel("Penetration Rate (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_line_chart(df, metric_col, ylabel, title, output_path):
    plt.figure(figsize=(9, 5))

    for rate in sorted(df["penetration_rate"].unique()):
        subset = df[df["penetration_rate"] == rate]
        plt.plot(
            subset["episode"],
            subset[metric_col],
            marker="o",
            label=f"{rate}%",
        )

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Penetration Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    eval_csv = "results/week12_behavior_eval.csv"
    summary_csv = "results/week12_behavior_summary.csv"
    output_dir = "results/week12_figures"

    if not os.path.exists(eval_csv):
        print(f"오류: {eval_csv} 파일이 없습니다.")
        return

    if not os.path.exists(summary_csv):
        print(f"오류: {summary_csv} 파일이 없습니다.")
        print("먼저 analyze_week12_behavior_results.py를 실행하세요.")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(eval_csv)
    summary = pd.read_csv(summary_csv)

    save_bar_chart(
        summary=summary,
        metric_col="lane_change_mean",
        std_col="lane_change_std",
        ylabel="Lane Change Count",
        title="Lane Change Count by Penetration Rate",
        output_path=f"{output_dir}/week12_lane_change_by_penetration.png",
    )

    save_bar_chart(
        summary=summary,
        metric_col="hard_brake_mean",
        std_col="hard_brake_std",
        ylabel="Hard Brake Count",
        title="Hard Brake Count by Penetration Rate",
        output_path=f"{output_dir}/week12_hard_brake_by_penetration.png",
    )

    save_bar_chart(
        summary=summary,
        metric_col="congestion_time_mean",
        std_col="congestion_time_std",
        ylabel="Congestion Time (sec)",
        title="Congestion Time by Penetration Rate",
        output_path=f"{output_dir}/week12_congestion_time_by_penetration.png",
    )

    save_line_chart(
        df=df,
        metric_col="lane_change_count",
        ylabel="Lane Change Count",
        title="Episode-wise Lane Change Count",
        output_path=f"{output_dir}/week12_episode_lane_change.png",
    )

    save_line_chart(
        df=df,
        metric_col="hard_brake_count",
        ylabel="Hard Brake Count",
        title="Episode-wise Hard Brake Count",
        output_path=f"{output_dir}/week12_episode_hard_brake.png",
    )

    save_line_chart(
        df=df,
        metric_col="congestion_time_sec",
        ylabel="Congestion Time (sec)",
        title="Episode-wise Congestion Time",
        output_path=f"{output_dir}/week12_episode_congestion_time.png",
    )

    print("[week12] 시각화 완료")
    print(f"[week12] 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()