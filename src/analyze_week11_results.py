import os
import pandas as pd


def main():
    input_csv = "results/week11_penetration_eval.csv"
    output_csv = "results/week11_penetration_summary.csv"

    if not os.path.exists(input_csv):
        print(f"오류: {input_csv} 파일이 없습니다.")
        print("먼저 evaluate_week11_penetration.py를 실행하세요.")
        return

    df = pd.read_csv(input_csv)

    summary = (
        df.groupby("penetration_rate")
        .agg(
            episodes=("episode", "count"),
            controlled_count_mean=("controlled_count", "mean"),
            avg_speed_mean=("avg_speed", "mean"),
            avg_speed_std=("avg_speed", "std"),
            throughput_mean=("throughput", "mean"),
            throughput_std=("throughput", "std"),
            crashed_count_mean=("crashed_count", "mean"),
        )
        .reset_index()
    )

    os.makedirs("results", exist_ok=True)
    summary.to_csv(output_csv, index=False)

    print("[week11] 침투율별 평가 요약")
    print(summary)
    print(f"\n[week11] 요약 CSV 저장: {output_csv}")


if __name__ == "__main__":
    main()