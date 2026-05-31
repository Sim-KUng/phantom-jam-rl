"""
Week 13: 2차선 시공간 다이어그램 시각화 (v2 - scatter plot 방식)

히트맵 격자 대신 산점도(scatter plot) 방식을 사용한다.
산점도는 데이터가 드문 경우에도 각 차량 궤적이 정확하게 표시된다.

실행 방법:
    cd phantom-jam-rl/src
    python visualize_week13_spacetime.py
"""

import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# Windows 한글 폰트 (맑은 고딕). 없으면 NanumGothic → DejaVu 순으로 fallback
import matplotlib.font_manager as fm
_korean_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
_available = {f.name for f in fm.fontManager.ttflist}
_chosen = next((f for f in _korean_candidates if f in _available), None)
if _chosen:
    plt.rcParams["font.family"] = _chosen
plt.rcParams["axes.unicode_minus"] = False


RESULTS_DIR = "results/week13_figures"
BASELINE_CSV = "logs/week13_baseline_traj.csv"
RL_CSV = "logs/week13_rl5pct_traj.csv"

BLOG_STATIC_IMG = os.path.join(
    os.path.dirname(__file__), "..", "..", "phantom-jam-blog", "static", "img"
)

SPEED_VMIN = 0.0
SPEED_VMAX = 30.0
CMAP = "RdYlGn"    # 빨강=저속(정체), 초록=고속(원활)
DOT_SIZE = 3       # 산점도 점 크기 (pt^2)
DOT_ALPHA = 0.7    # 투명도

LANE_LABEL = {0: "1차선 (Lane 1)", 1: "2차선 (Lane 2)"}


# ──────────────────────────────────────────────
# 범위 계산
# ──────────────────────────────────────────────

def compute_common_ranges(dfs):
    """여러 DataFrame에서 공통 x/t 범위를 계산해 동일한 스케일로 비교한다."""
    x_min = min(df["x_position"].min() for df in dfs if len(df) > 0)
    x_max = max(df["x_position"].max() for df in dfs if len(df) > 0)
    t_min = min(df["time_sec"].min() for df in dfs if len(df) > 0)
    t_max = max(df["time_sec"].max() for df in dfs if len(df) > 0)
    return (x_min, x_max), (t_min, t_max)


# ──────────────────────────────────────────────
# 단일 패널 그리기 (scatter plot)
# ──────────────────────────────────────────────

def draw_scatter(ax, df_lane, title, x_range, t_range):
    """
    시공간 다이어그램을 산점도로 그린다.
    - X축: 도로 위치 (m)
    - Y축: 시간 (s)
    - 색상: 속도 (m/s)  빨강=저속, 초록=고속
    """
    if len(df_lane) == 0:
        ax.set_title(title + "\n(데이터 없음)", fontsize=12)
        return None

    sc = ax.scatter(
        df_lane["x_position"],
        df_lane["time_sec"],
        c=df_lane["speed"],
        cmap=CMAP,
        vmin=SPEED_VMIN,
        vmax=SPEED_VMAX,
        s=DOT_SIZE,
        alpha=DOT_ALPHA,
        linewidths=0,
        rasterized=True,
    )

    ax.set_xlim(x_range)
    ax.set_ylim(t_range)
    ax.set_xlabel("도로 위치 (m)", fontsize=11)
    ax.set_ylabel("시간 (s)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    return sc


# ──────────────────────────────────────────────
# 출력 함수들
# ──────────────────────────────────────────────

def save_lane_comparison(baseline_df, rl_df, lane, output_path):
    """단일 차선 Baseline vs RL 나란히 비교."""
    df_base = baseline_df[baseline_df["lane_index"] == lane]
    df_rl = rl_df[rl_df["lane_index"] == lane]

    x_range, t_range = compute_common_ranges([df_base, df_rl])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=120)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.88, bottom=0.10, wspace=0.25)
    fig.suptitle(
        f"시공간 다이어그램 비교 — {LANE_LABEL[lane]}\n"
        "색상: 빨강=저속(정체) / 초록=고속(원활)  |  Y축: 시간(s)  X축: 위치(m)",
        fontsize=13, fontweight="bold",
    )

    sc = draw_scatter(axes[0], df_base, f"Baseline (0% PPO)\n{LANE_LABEL[lane]}", x_range, t_range)
    sc = draw_scatter(axes[1], df_rl,   f"RL Agent (5% PPO)\n{LANE_LABEL[lane]}", x_range, t_range)

    if sc is not None:
        cax = fig.add_axes([0.90, 0.15, 0.018, 0.65])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("속도 (m/s)", fontsize=11)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {output_path}")


def save_overview_2x2(baseline_df, rl_df, output_path):
    """(2행×2열) 전체 개요: 행=조건, 열=차선."""
    all_sub = [
        baseline_df[baseline_df["lane_index"] == 0],
        baseline_df[baseline_df["lane_index"] == 1],
        rl_df[rl_df["lane_index"] == 0],
        rl_df[rl_df["lane_index"] == 1],
    ]
    x_range, t_range = compute_common_ranges(all_sub)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=120)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)
    fig.suptitle(
        "2차선 시공간 다이어그램 전체 비교\n"
        "Baseline (0% PPO) vs RL Agent (5% PPO)  |  빨강=저속 / 초록=고속",
        fontsize=14, fontweight="bold",
    )

    panel_configs = [
        (0, 0, baseline_df, "Baseline (0% PPO)", 0),
        (0, 1, baseline_df, "Baseline (0% PPO)", 1),
        (1, 0, rl_df,       "RL Agent (5% PPO)", 0),
        (1, 1, rl_df,       "RL Agent (5% PPO)", 1),
    ]

    sc = None
    for row, col, df, label, lane in panel_configs:
        ax = axes[row][col]
        result = draw_scatter(
            ax,
            df[df["lane_index"] == lane],
            f"{label}\n{LANE_LABEL[lane]}",
            x_range,
            t_range,
        )
        if result is not None:
            sc = result

    if sc is not None:
        cax = fig.add_axes([0.90, 0.12, 0.018, 0.75])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("속도 (m/s)", fontsize=11)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {output_path}")


def save_single_2panel(df, label, output_path):
    """한 가지 조건에 대해 1차선/2차선을 나란히."""
    df0 = df[df["lane_index"] == 0]
    df1 = df[df["lane_index"] == 1]
    x_range, t_range = compute_common_ranges([df0, df1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=120)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.88, bottom=0.10, wspace=0.25)
    fig.suptitle(
        f"{label} — 1차선 · 2차선 시공간 다이어그램\n"
        "색상: 빨강=저속(정체) / 초록=고속(원활)",
        fontsize=13, fontweight="bold",
    )

    sc = draw_scatter(axes[0], df0, f"{label}\n{LANE_LABEL[0]}", x_range, t_range)
    sc = draw_scatter(axes[1], df1, f"{label}\n{LANE_LABEL[1]}", x_range, t_range)

    if sc is not None:
        cax = fig.add_axes([0.90, 0.15, 0.018, 0.65])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("속도 (m/s)", fontsize=11)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {output_path}")


# ──────────────────────────────────────────────
# 블로그 복사
# ──────────────────────────────────────────────

def copy_to_blog(src_dir, blog_img_dir):
    blog_img_dir = os.path.normpath(blog_img_dir)
    if not os.path.isdir(blog_img_dir):
        print(f"\n[복사 건너뜀] 블로그 이미지 폴더 없음: {blog_img_dir}")
        return

    copied = 0
    for fname in os.listdir(src_dir):
        if fname.endswith(".png"):
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(blog_img_dir, fname))
            copied += 1

    print(f"\n[블로그] {copied}개 이미지 복사 완료 → {blog_img_dir}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[week13] 궤적 CSV 로드 중...")
    baseline_df = pd.read_csv(BASELINE_CSV)
    rl_df = pd.read_csv(RL_CSV)

    for name, df in [("Baseline", baseline_df), ("RL 5%", rl_df)]:
        steps = df["step"].max() + 1
        t_max = df["time_sec"].max()
        n_v = df["vehicle_id"].nunique()
        print(f"  {name}: {len(df):,} rows | steps={steps} | time={t_max:.1f}s | vehicles={n_v}")

    print("\n[week13] 시공간 다이어그램 생성 중...\n")

    # 1차선 비교
    save_lane_comparison(
        baseline_df, rl_df, lane=0,
        output_path=os.path.join(RESULTS_DIR, "week13_spacetime_lane1_comparison.png"),
    )

    # 2차선 비교
    save_lane_comparison(
        baseline_df, rl_df, lane=1,
        output_path=os.path.join(RESULTS_DIR, "week13_spacetime_lane2_comparison.png"),
    )

    # 2×2 전체 개요
    save_overview_2x2(
        baseline_df, rl_df,
        output_path=os.path.join(RESULTS_DIR, "week13_spacetime_2lane_overview.png"),
    )

    # Baseline 단독
    save_single_2panel(
        baseline_df, "Baseline (0% PPO)",
        output_path=os.path.join(RESULTS_DIR, "week13_spacetime_baseline_only.png"),
    )

    # RL 단독
    save_single_2panel(
        rl_df, "RL Agent (5% PPO)",
        output_path=os.path.join(RESULTS_DIR, "week13_spacetime_rl_only.png"),
    )

    print(f"\n[week13] 모든 그래프 생성 완료 → {RESULTS_DIR}/")
    copy_to_blog(RESULTS_DIR, BLOG_STATIC_IMG)


if __name__ == "__main__":
    main()
