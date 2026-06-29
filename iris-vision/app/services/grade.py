"""L* 映射 Pan 2017 风格 Grade 1–5。"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class GradeResult:
    """分档结果。"""

    grade: int
    confidence: float


def load_config(config_path: Path) -> dict:
    """从 YAML 加载配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def map_l_star_to_grade(l_star: float, boundaries: List[float]) -> GradeResult:
    """
    将 L* 映射到 Grade 1–5。

    boundaries 为 4 个递减上界：[b1, b2, b3, b4]
    - Grade 1: L* > b1
    - Grade 2: b2 < L* <= b1
    - Grade 3: b3 < L* <= b2
    - Grade 4: b4 < L* <= b3
    - Grade 5: L* <= b4

    Pan 2017 规则：恰在边界 → 取较深档（本实现中边界归属较深档）。
    """
    b1, b2, b3, b4 = boundaries

    if l_star > b1:
        grade = 1
        nearest_boundary = b1
        distance = l_star - b1
    elif l_star > b2:
        grade = 2
        nearest_boundary = min(abs(l_star - b1), abs(l_star - b2))
        distance = nearest_boundary
    elif l_star > b3:
        grade = 3
        nearest_boundary = min(abs(l_star - b2), abs(l_star - b3))
        distance = nearest_boundary
    elif l_star > b4:
        grade = 4
        nearest_boundary = min(abs(l_star - b3), abs(l_star - b4))
        distance = nearest_boundary
    else:
        grade = 5
        distance = b4 - l_star

    # 置信度：距最近边界越远越高，最大边界间距约 6 L* 单位
    max_span = max(boundaries[0] - boundaries[-1], 1.0)
    confidence = min(1.0, max(0.3, distance / (max_span / 2)))

    return GradeResult(grade=grade, confidence=round(confidence, 2))


def get_grade_boundaries(config: dict) -> List[float]:
    """从配置读取 Pan Grade 边界（兼容旧版顶层 boundaries）。"""
    grade_cfg = config.get("grade", {})
    if "boundaries" in grade_cfg:
        return grade_cfg["boundaries"]
    return config["boundaries"]


def grade_from_l_star(l_star: float, config_path: Path) -> GradeResult:
    """加载配置并分档（供 routes 调用）。"""
    config = load_config(config_path)
    boundaries = get_grade_boundaries(config)
    return map_l_star_to_grade(l_star, boundaries)
