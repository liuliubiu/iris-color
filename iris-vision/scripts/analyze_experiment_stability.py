"""实验记录稳定性分析：巩膜参考调色前后同人 L* 离散度对比。

数据源：MySQL/SQLite experiment_records（实验记录界面写入）。
分析维度见 app.services.experiment_stability。

用法（在 iris-vision 目录）：
    python scripts/analyze_experiment_stability.py [--json out.json] [--min-n 3]
"""

import json
import sys
from pathlib import Path

import pymysql
import yaml
from pymysql.cursors import DictCursor

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"

sys.path.insert(0, str(ROOT))
from app.services.experiment_stability import analyze_stability  # noqa: E402


def _load_records() -> tuple[list, list]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mysql_cfg = (cfg.get("experiments") or {}).get("mysql") or {}
    conn = pymysql.connect(
        host=mysql_cfg.get("host", "127.0.0.1"),
        port=int(mysql_cfg.get("port", 3306)),
        user=mysql_cfg.get("user", "root"),
        password=str(mysql_cfg.get("password", "")),
        database=mysql_cfg.get("database", "iris_experiment"),
        charset=mysql_cfg.get("charset", "utf8mb4"),
        cursorclass=DictCursor,
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, group_name, subgroup_name, operator, camera_device, "
            "light_device, illuminance, color, lstar_before, lstar_after, "
            "grade_before, grade_after, include_in_stats FROM experiment_records "
            "WHERE lstar_before IS NOT NULL AND lstar_after IS NOT NULL"
        )
        rows = cur.fetchall()
    conn.close()
    boundaries = (cfg.get("grade") or {}).get("boundaries", [55, 45, 29, 19])
    return rows, boundaries


def _print_report(report: dict) -> None:
    boundaries = report["boundaries"]
    print(f"记录总数（含调色前后 L*）：{report['total']}    grade 边界：{boundaries}")

    ws = report["within_subgroup"]
    print("\n" + "=" * 110)
    print(f"一、小组内重复性（n>={report['filters']['min_subgroup_n']} 的小组）")
    print("=" * 110)
    header = (
        f"{'人':<3} {'组':<14} {'设备':<15} {'照度':>4} {'n':>3} "
        f"| {'调色前 σ':>8} {'极差':>6} {'grade':<14} "
        f"| {'调色后 σ':>8} {'极差':>6} {'grade':<14} | 结论"
    )
    print(header)
    print("-" * 110)
    verdict_map = {"better": "更稳", "worse": "更散", "same": "持平"}
    for g in ws["groups"]:
        label = f"{g['group']}-{g.get('subgroup') or '-'}"
        print(
            f"{g['operator']:<3} {label:<14} {g['camera']:<15} {g['lux']!s:>4} {g['n']:>3} "
            f"| {g['std_before']:>8.3f} {g['range_before']:>6.2f} {g['grades_before']:<14} "
            f"| {g['std_after']:>8.3f} {g['range_after']:>6.2f} {g['grades_after']:<14} "
            f"| {verdict_map.get(g['verdict'], g['verdict'])}"
        )
    print("-" * 110)
    if ws["median_std_before"] is not None:
        print(
            f"小组数={len(ws['groups'])}  调色后更稳={ws['wins']}  更散={ws['losses']}  持平={ws['ties']}\n"
            f"σ 中位数：调色前 {ws['median_std_before']:.2f} → 调色后 {ws['median_std_after']:.2f}"
            f"（差值中位数 {ws['median_std_diff']:+.2f}）\n"
            f"符号检验 p={ws['p_sign']:.4f}   Wilcoxon 符号秩 p={ws['p_wilcoxon']}"
        )

    print("\n" + "=" * 110)
    print("二、同人跨条件一致性")
    print("=" * 110)
    for p in report["per_person"]:
        print(
            f"\n{p['operator']}（n={p['n']}，条件数={p['conditions']}）\n"
            f"  调色前：总 σ={p['std_before']:.2f} 极差={p['range_before']:.2f}  "
            f"条件间 σ={p['between_cond_std_before']:.2f}  条件内 σ={p['within_cond_std_before']:.2f}  "
            f"grade: {p['grades_before']}（众数占比 {p['majority_before']:.0%}）\n"
            f"  调色后：总 σ={p['std_after']:.2f} 极差={p['range_after']:.2f}  "
            f"条件间 σ={p['between_cond_std_after']:.2f}  条件内 σ={p['within_cond_std_after']:.2f}  "
            f"grade: {p['grades_after']}（众数占比 {p['majority_after']:.0%}）"
        )

    print("\n  跨条件组均值差 |Δμ|（同人两两条件配对）：")
    for c in report["cross_condition"]:
        print(
            f"  {c['operator']}: 条件对数={c['pairs']}  |Δμ| 中位数 调色前 {c['median_gap_before']:.2f} "
            f"→ 调色后 {c['median_gap_after']:.2f}  "
            f"（缩小 {c['improved']} 对 / 扩大 {c['worsened']} 对，符号检验 p={c['p_sign']:.4f}）"
        )

    be = report["boundary_effect"]
    print("\n" + "=" * 110)
    print("三、grade 分散的边界效应分析")
    print("=" * 110)
    print(
        f"L* 到最近 grade 边界的距离中位数：调色前 {be['median_dist_before']:.2f} → 调色后 {be['median_dist_after']:.2f}\n"
        f"距边界 <2 的记录：调色前 {be['near_boundary_before']}/{be['total_records']}  "
        f"调色后 {be['near_boundary_after']}/{be['total_records']}\n"
        f"调色导致 grade 改变的记录：{be['grade_changed']}/{be['total_records']}，"
        f"其中 |ΔL*|<=3 的『边界翻转』占 {be['grade_changed_small_move']}/{be['grade_changed']}"
    )
    if report["grade_changes"]:
        print("\n  grade 改变明细（id: L*前→后, G前→后, 距边界）：")
        for r in report["grade_changes"]:
            print(
                f"  #{r['id']:>4} {r['operator']} {r['camera']:<15} lux={r['lux']!s:>3}  "
                f"L* {r['lstar_before']:6.2f}→{r['lstar_after']:6.2f} (Δ{r['delta_lstar']:+5.2f})  "
                f"G{r['grade_before']}→G{r['grade_after']}  "
                f"距边界 前{r['dist_before']:4.1f} 后{r['dist_after']:4.1f}"
            )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    out_json = None
    min_n = 3
    args = list(sys.argv[1:])
    if "--json" in args:
        i = args.index("--json")
        out_json = Path(args[i + 1])
    if "--min-n" in args:
        i = args.index("--min-n")
        min_n = max(2, int(args[i + 1]))

    rows, boundaries = _load_records()
    report = analyze_stability(rows, boundaries, min_subgroup_n=min_n)
    _print_report(report)

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已导出：{out_json}")


if __name__ == "__main__":
    main()
