import os
import pandas as pd


def reduction_rate(baseline_value, target_value):
    """
    Baseline 대비 감소율 계산.
    값이 작아질수록 좋은 지표에 사용한다.
    """

    if baseline_value == 0:
        return None

    return (baseline_value - target_value) / baseline_value * 100.0


def improvement_rate(baseline_value, target_value):
    """
    Baseline 대비 증가율 계산.
    값이 커질수록 좋은 지표에 사용한다.
    """

    if baseline_value == 0:
        return None

    return (target_value - baseline_value) / baseline_value * 100.0


def main():
    input_csv = "results/week12_behavior_eval.csv"
    summary_csv = "results/week12_behavior_summary.csv"
    hypothesis_csv = "results/week12_hypothesis_summary.csv"

    if not os.path.exists(input_csv):
        print(f"오류: {input_csv} 파일이 없습니다.")
        print("먼저 eval_week12_behavior_metrics.py를 실행하세요.")
        return

    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(input_csv)

    summary = (
        df.groupby("penetration_rate")
        .agg(
            episodes=("episode", "count"),
            avg_speed_mean=("avg_speed", "mean"),
            avg_speed_std=("avg_speed", "std"),
            throughput_mean=("throughput", "mean"),
            throughput_std=("throughput", "std"),
            lane_change_mean=("lane_change_count", "mean"),
            lane_change_std=("lane_change_count", "std"),
            hard_brake_mean=("hard_brake_count", "mean"),
            hard_brake_std=("hard_brake_count", "std"),
            congestion_time_mean=("congestion_time_sec", "mean"),
            congestion_time_std=("congestion_time_sec", "std"),
            crashed_count_mean=("crashed_count", "mean"),
        )
        .reset_index()
    )

    baseline = summary[summary["penetration_rate"] == 0].iloc[0]

    hypothesis_rows = []

    for _, row in summary.iterrows():
        rate = int(row["penetration_rate"])

        lane_change_reduction = reduction_rate(
            baseline["lane_change_mean"], row["lane_change_mean"]
        )

        hard_brake_reduction = reduction_rate(
            baseline["hard_brake_mean"], row["hard_brake_mean"]
        )

        congestion_time_reduction = reduction_rate(
            baseline["congestion_time_mean"], row["congestion_time_mean"]
        )

        throughput_improvement = improvement_rate(
            baseline["throughput_mean"], row["throughput_mean"]
        )

        avg_speed_improvement = improvement_rate(
            baseline["avg_speed_mean"], row["avg_speed_mean"]
        )

        if rate == 0:
            hypothesis_result = "baseline"
        else:
            conditions = [
                lane_change_reduction is not None and lane_change_reduction > 0,
                hard_brake_reduction is not None and hard_brake_reduction > 0,
                congestion_time_reduction is not None and congestion_time_reduction > 0,
                row["crashed_count_mean"] == 0,
            ]

            if all(conditions):
                hypothesis_result = "supported"
            elif any(conditions):
                hypothesis_result = "partially_supported"
            else:
                hypothesis_result = "not_supported"

        hypothesis_rows.append(
            {
                "penetration_rate": rate,
                "lane_change_reduction_pct": lane_change_reduction,
                "hard_brake_reduction_pct": hard_brake_reduction,
                "congestion_time_reduction_pct": congestion_time_reduction,
                "throughput_improvement_pct": throughput_improvement,
                "avg_speed_improvement_pct": avg_speed_improvement,
                "crashed_count_mean": row["crashed_count_mean"],
                "hypothesis_result": hypothesis_result,
            }
        )

    hypothesis_df = pd.DataFrame(hypothesis_rows)

    summary.to_csv(summary_csv, index=False)
    hypothesis_df.to_csv(hypothesis_csv, index=False)

    print("[week12] 행동 지표 평가 요약")
    print(summary)

    print("\n[week12] 가설 검증 결과")
    print(hypothesis_df)

    print(f"\n[week12] 요약 CSV 저장: {summary_csv}")
    print(f"[week12] 가설 검증 CSV 저장: {hypothesis_csv}")


if __name__ == "__main__":
    main()