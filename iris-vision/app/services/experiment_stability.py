"""实验记录 L* 离散度（稳定性）分析。

对比巩膜参考调色前后，同人/同条件下的 L* 标准差、极差与 grade 一致性。
供 CLI 脚本与实验记录管理 API 共用。
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Any, Iterable, Optional

import numpy as np


def grade_of(l_star: float, boundaries: list[float]) -> int:
    for i, b in enumerate(boundaries):
        if l_star > b:
            return i + 1
    return len(boundaries) + 1


def sign_test_p(wins: int, losses: int) -> float:
    """双侧符号检验精确 p 值（二项分布，p=0.5，忽略平局）。"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def wilcoxon_signed_rank_p(diffs: np.ndarray) -> Optional[float]:
    """双侧 Wilcoxon 符号秩检验（正态近似，含平局校正）。"""
    d = diffs[diffs != 0]
    n = len(d)
    if n < 6:
        return None
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    abs_d = np.abs(d)
    for v in np.unique(abs_d):
        sel = abs_d == v
        if sel.sum() > 1:
            ranks[sel] = ranks[sel].mean()
    w_pos = float(ranks[d > 0].sum())
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return None
    z = (w_pos - mean_w) / math.sqrt(var_w)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))
    return p


def lstar_stats(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "range": float(arr.max() - arr.min()) if arr.size else 0.0,
    }


def majority_ratio(grades: list[int]) -> float:
    if not grades:
        return 0.0
    counts = Counter(grades)
    return max(counts.values()) / len(grades)


def grade_summary(grades: list[int]) -> str:
    counts = Counter(grades)
    return "/".join(f"G{g}x{c}" for g, c in sorted(counts.items()))


def dist_to_boundary(l_star: float, boundaries: list[float]) -> float:
    return min(abs(l_star - b) for b in boundaries)


def _is_included_in_stats(row: dict) -> bool:
    v = row.get("include_in_stats")
    if v is None:
        return True
    return bool(v)


def _prepare_rows(rows: Iterable[dict], boundaries: list[float]) -> list[dict]:
    out = []
    for r in rows:
        if not _is_included_in_stats(r):
            continue
        if r.get("lstar_before") is None or r.get("lstar_after") is None:
            continue
        rec = dict(r)
        rec["g_before"] = grade_of(float(rec["lstar_before"]), boundaries)
        rec["g_after"] = grade_of(float(rec["lstar_after"]), boundaries)
        out.append(rec)
    return out


def _decompose_variance(rs: list[dict], field: str) -> tuple[float, float, int]:
    conds: dict[tuple, list[float]] = {}
    for r in rs:
        ck = (r.get("camera_device"), r.get("light_device"), r.get("illuminance"))
        conds.setdefault(ck, []).append(float(r[field]))
    all_vals = [v for vs in conds.values() for v in vs]
    if not all_vals:
        return 0.0, 0.0, len(conds)
    grand = float(np.mean(all_vals))
    n_total = len(all_vals)
    between = sum(len(vs) * (float(np.mean(vs)) - grand) ** 2 for vs in conds.values()) / n_total
    within = sum(sum((v - float(np.mean(vs))) ** 2 for v in vs) for vs in conds.values()) / n_total
    return math.sqrt(between), math.sqrt(within), len(conds)


def analyze_stability(
    rows: Iterable[dict],
    boundaries: list[float],
    *,
    min_subgroup_n: int = 3,
    operators: Optional[list[str]] = None,
    group_names: Optional[list[str]] = None,
    subgroup_names: Optional[list[str]] = None,
    record_ids: Optional[list[int]] = None,
) -> dict[str, Any]:
    """对实验记录做 L* 离散度分析。

    可选 filters：operators、group_names、subgroup_names、record_ids（均为白名单，空则不过滤）。
    min_subgroup_n：小组内重复性分析要求的最少重复拍摄次数。
    """
    prepared = _prepare_rows(rows, boundaries)

    if record_ids is not None:
        id_set = {int(i) for i in record_ids}
        prepared = [r for r in prepared if int(r.get("id", 0)) in id_set]
    if operators:
        op_set = set(operators)
        prepared = [r for r in prepared if r.get("operator") in op_set]
    if group_names:
        grp_set = set(group_names)
        prepared = [r for r in prepared if r.get("group_name") in grp_set]
    if subgroup_names:
        sub_set = set(subgroup_names)
        prepared = [r for r in prepared if r.get("subgroup_name") in sub_set]

    report: dict[str, Any] = {
        "boundaries": boundaries,
        "total": len(prepared),
        "filters": {
            "min_subgroup_n": min_subgroup_n,
            "operators": operators or [],
            "group_names": group_names or [],
            "subgroup_names": subgroup_names or [],
            "record_ids": record_ids or [],
        },
    }
    if not prepared:
        report["within_subgroup"] = {
            "groups": [],
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "median_std_before": None,
            "median_std_after": None,
            "p_sign": None,
            "p_wilcoxon": None,
        }
        report["per_person"] = []
        report["cross_condition"] = []
        report["boundary_effect"] = {
            "median_dist_before": None,
            "median_dist_after": None,
            "near_boundary_before": 0,
            "near_boundary_after": 0,
            "grade_changed": 0,
            "grade_changed_small_move": 0,
        }
        report["grade_changes"] = []
        return report

    # ---------- 1. 小组内重复性 ----------
    subgroups: dict[tuple, list[dict]] = {}
    for r in prepared:
        key = (
            r["operator"],
            r["group_name"],
            r.get("subgroup_name"),
            r.get("camera_device"),
            r.get("light_device"),
            r.get("illuminance"),
        )
        subgroups.setdefault(key, []).append(r)

    paired: list[tuple[float, float, float, float]] = []
    sub_report: list[dict] = []
    for key in sorted(subgroups, key=lambda k: (k[0], k[1], k[2] or "")):
        rs = subgroups[key]
        if len(rs) < min_subgroup_n:
            continue
        sb = lstar_stats([float(r["lstar_before"]) for r in rs])
        sa = lstar_stats([float(r["lstar_after"]) for r in rs])
        gb = [r["g_before"] for r in rs]
        ga = [r["g_after"] for r in rs]
        paired.append((sb["std"], sa["std"], sb["range"], sa["range"]))
        op, grp, sub, cam, light, lux = key
        sub_report.append({
            "operator": op,
            "group": grp,
            "subgroup": sub,
            "camera": cam,
            "light": light,
            "lux": lux,
            "n": sb["n"],
            "std_before": round(sb["std"], 3),
            "std_after": round(sa["std"], 3),
            "range_before": round(sb["range"], 3),
            "range_after": round(sa["range"], 3),
            "grades_before": grade_summary(gb),
            "grades_after": grade_summary(ga),
            "majority_before": round(majority_ratio(gb), 3),
            "majority_after": round(majority_ratio(ga), 3),
            "verdict": (
                "better" if sa["std"] < sb["std"] - 1e-9
                else ("worse" if sa["std"] > sb["std"] + 1e-9 else "same")
            ),
        })

    if paired:
        arr = np.asarray(paired)
        wins = int(np.sum(arr[:, 1] < arr[:, 0] - 1e-9))
        losses = int(np.sum(arr[:, 1] > arr[:, 0] + 1e-9))
        ties = len(arr) - wins - losses
        diffs = arr[:, 0] - arr[:, 1]
        p_sign = sign_test_p(wins, losses)
        p_wilcoxon = wilcoxon_signed_rank_p(diffs)
        report["within_subgroup"] = {
            "groups": sub_report,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "median_std_before": round(float(np.median(arr[:, 0])), 3),
            "median_std_after": round(float(np.median(arr[:, 1])), 3),
            "median_std_diff": round(float(np.median(diffs)), 3),
            "p_sign": round(p_sign, 5),
            "p_wilcoxon": round(p_wilcoxon, 5) if p_wilcoxon is not None else None,
        }
    else:
        report["within_subgroup"] = {
            "groups": sub_report,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "median_std_before": None,
            "median_std_after": None,
            "median_std_diff": None,
            "p_sign": None,
            "p_wilcoxon": None,
        }

    # ---------- 2. 同人跨条件 ----------
    persons: dict[str, list[dict]] = {}
    for r in prepared:
        persons.setdefault(r["operator"], []).append(r)

    person_report: list[dict] = []
    for op in sorted(persons):
        rs = persons[op]
        sb = lstar_stats([float(r["lstar_before"]) for r in rs])
        sa = lstar_stats([float(r["lstar_after"]) for r in rs])
        gb = [r["g_before"] for r in rs]
        ga = [r["g_after"] for r in rs]
        bwt_b, wth_b, n_cond = _decompose_variance(rs, "lstar_before")
        bwt_a, wth_a, _ = _decompose_variance(rs, "lstar_after")
        person_report.append({
            "operator": op,
            "n": sb["n"],
            "conditions": n_cond,
            "std_before": round(sb["std"], 3),
            "std_after": round(sa["std"], 3),
            "range_before": round(sb["range"], 3),
            "range_after": round(sa["range"], 3),
            "between_cond_std_before": round(bwt_b, 3),
            "between_cond_std_after": round(bwt_a, 3),
            "within_cond_std_before": round(wth_b, 3),
            "within_cond_std_after": round(wth_a, 3),
            "grades_before": grade_summary(gb),
            "grades_after": grade_summary(ga),
            "majority_before": round(majority_ratio(gb), 3),
            "majority_after": round(majority_ratio(ga), 3),
        })
    report["per_person"] = person_report

    cross_report: list[dict] = []
    for op in sorted(persons):
        rs = persons[op]
        conds: dict[tuple, dict[str, list[float]]] = {}
        for r in rs:
            ck = (r.get("camera_device"), r.get("light_device"), r.get("illuminance"))
            conds.setdefault(ck, {"b": [], "a": []})
            conds[ck]["b"].append(float(r["lstar_before"]))
            conds[ck]["a"].append(float(r["lstar_after"]))
        pairs_b, pairs_a = [], []
        for c1, c2 in combinations(conds.keys(), 2):
            pairs_b.append(abs(float(np.mean(conds[c1]["b"])) - float(np.mean(conds[c2]["b"]))))
            pairs_a.append(abs(float(np.mean(conds[c1]["a"])) - float(np.mean(conds[c2]["a"]))))
        if not pairs_b:
            continue
        med_b, med_a = float(np.median(pairs_b)), float(np.median(pairs_a))
        better = sum(1 for b, a in zip(pairs_b, pairs_a) if a < b - 1e-9)
        worse = sum(1 for b, a in zip(pairs_b, pairs_a) if a > b + 1e-9)
        cross_report.append({
            "operator": op,
            "pairs": len(pairs_b),
            "median_gap_before": round(med_b, 3),
            "median_gap_after": round(med_a, 3),
            "improved": better,
            "worsened": worse,
            "p_sign": round(sign_test_p(better, worse), 5),
        })
    report["cross_condition"] = cross_report

    # ---------- 3. 边界效应 ----------
    db = [dist_to_boundary(float(r["lstar_before"]), boundaries) for r in prepared]
    da = [dist_to_boundary(float(r["lstar_after"]), boundaries) for r in prepared]
    changed = [r for r in prepared if r["g_before"] != r["g_after"]]
    small_move = [
        r for r in changed
        if abs(float(r["lstar_after"]) - float(r["lstar_before"])) <= 3.0
    ]
    report["boundary_effect"] = {
        "median_dist_before": round(float(np.median(db)), 3),
        "median_dist_after": round(float(np.median(da)), 3),
        "near_boundary_before": sum(1 for d in db if d < 2),
        "near_boundary_after": sum(1 for d in da if d < 2),
        "grade_changed": len(changed),
        "grade_changed_small_move": len(small_move),
        "total_records": len(prepared),
    }
    report["grade_changes"] = [
        {
            "id": r["id"],
            "operator": r.get("operator"),
            "camera": r.get("camera_device"),
            "lux": r.get("illuminance"),
            "lstar_before": round(float(r["lstar_before"]), 2),
            "lstar_after": round(float(r["lstar_after"]), 2),
            "delta_lstar": round(float(r["lstar_after"]) - float(r["lstar_before"]), 2),
            "grade_before": r["g_before"],
            "grade_after": r["g_after"],
            "dist_before": round(dist_to_boundary(float(r["lstar_before"]), boundaries), 1),
            "dist_after": round(dist_to_boundary(float(r["lstar_after"]), boundaries), 1),
        }
        for r in sorted(changed, key=lambda x: x["id"])
    ]
    return report
