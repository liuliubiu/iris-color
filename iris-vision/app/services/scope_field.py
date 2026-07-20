"""镜筒特写预处理：检测亮圆视场、裁剪黑边、统一工作分辨率。

实拍图（镜筒特写）画面约一半是纯黑边框，中间是亮圆视场。黑边会干扰
Otsu/百分位阈值、CLAHE 与 Hough；先裁掉黑边再统一降到工作分辨率，
既提升定位稳定性，又把后续全流程的像素量降一个数量级。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ScopeField:
    """亮圆视场（镜筒视野）。坐标为所属图像的像素坐标。"""

    center_x: float
    center_y: float
    radius: float
    bright_ratio: float


@dataclass
class ImageTransform:
    """工作图与原图的坐标换算：work = (orig - offset) * scale。"""

    offset_x: int = 0
    offset_y: int = 0
    scale: float = 1.0

    @property
    def identity(self) -> bool:
        return self.offset_x == 0 and self.offset_y == 0 and abs(self.scale - 1.0) < 1e-9

    def to_original_xy(self, x: float, y: float) -> Tuple[float, float]:
        return self.offset_x + x / self.scale, self.offset_y + y / self.scale

    def to_original_len(self, value: float) -> float:
        return value / self.scale

    def to_working_xy(self, x: float, y: float) -> Tuple[float, float]:
        return (x - self.offset_x) * self.scale, (y - self.offset_y) * self.scale

    def to_working_len(self, value: float) -> float:
        return value * self.scale


@dataclass
class PreprocessResult:
    """预处理输出：工作图 + 坐标换算 + 视场圆（工作图坐标，非镜筒图为 None）。"""

    image: np.ndarray
    transform: ImageTransform
    scope: Optional[ScopeField]


_SCOPE_DETECT_DIM = 256
# 手机紧裁镜筒图常把暗边框裁掉，固定低阈值会把周围皮肤并进亮区导致外接圆越界失败。
# 在多个亮度阈值上选「高填充 + 低越界 + 亮区占比适中」的最佳圆。
_SCOPE_V_THRESHOLDS = tuple(range(8, 96, 4))


def _scope_candidate_at_threshold(
    gray: np.ndarray,
    v_threshold: int,
    skip_if_bright_ratio: float,
    min_fill_ratio: float,
    min_radius_ratio: float,
    max_overflow_ratio: float = 0.12,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """单阈值下的视场圆候选。返回 (cx, cy, r, bright_ratio, fill, score)；失败 None。"""
    bright = (gray > v_threshold).astype(np.uint8) * 255
    bright_ratio = float(np.mean(bright > 0))
    if bright_ratio >= skip_if_bright_ratio or bright_ratio < 0.08:
        return None

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if num <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == idx).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(contour)

    sh, sw = gray.shape[:2]
    if r < min(sh, sw) * min_radius_ratio:
        return None
    # 视场圆须基本落在画面内；普通特写亮区顶到边时外接圆会大幅越界
    overflow = max(r - cx, cx + r - sw, r - cy, cy + r - sh, 0.0)
    if overflow > r * max_overflow_ratio:
        return None
    fill = float(stats[idx, cv2.CC_STAT_AREA]) / max(np.pi * r * r, 1.0)
    if fill < min_fill_ratio:
        return None

    # 镜筒图亮区占比多在 0.25~0.55；偏离越大分越低
    br_term = 1.0 - abs(bright_ratio - 0.40) / 0.50
    score = fill * (1.0 - overflow / max(r, 1.0)) * max(br_term, 0.1)
    return float(cx), float(cy), float(r), bright_ratio, fill, float(score)


def detect_scope_field(
    image_bgr: np.ndarray,
    v_threshold: int = 12,
    skip_if_bright_ratio: float = 0.90,
    min_fill_ratio: float = 0.55,
    min_radius_ratio: float = 0.25,
    allow_synthetic_fallback: bool = True,
) -> Optional[ScopeField]:
    """
    检测镜筒亮圆视场。返回原图坐标下的视场圆；非镜筒图返回 None。

    在 ~256px 小图上多阈值扫描亮度 → 最大连通域 → 最小外接圆，选取综合分最高者。
    固定单阈值在「紧裁手机图」上常把皮肤并进亮区而失败；自适应阈值可找回真视场圆。
    若仍失败但画面像紧裁眼部特写，则用画面中心 + 0.40×最短边作为弱先验（合成视场），
    供后续角膜缘搜索使用——半径尺度与实测 iris/min_dim≈0.20 对齐后落入 0.5×scope 一带。
    """
    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)
    scale = _SCOPE_DETECT_DIM / min_dim if min_dim > _SCOPE_DETECT_DIM else 1.0
    small = (
        cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image_bgr
    )

    # V 通道（BGR 最大值）判亮暗，纯黑镜筒边框远低于阈值
    gray = small.max(axis=2)
    # 优先尝试配置阈值，再扫其余阈值（去重保序）
    thresholds: list[int] = []
    for vt in (int(v_threshold), *_SCOPE_V_THRESHOLDS):
        if vt not in thresholds:
            thresholds.append(vt)

    best: Optional[Tuple[float, float, float, float, float, float]] = None
    for vt in thresholds:
        cand = _scope_candidate_at_threshold(
            gray,
            vt,
            skip_if_bright_ratio=skip_if_bright_ratio,
            min_fill_ratio=min_fill_ratio,
            min_radius_ratio=min_radius_ratio,
        )
        if cand is None:
            continue
        if best is None or cand[5] > best[5]:
            best = cand

    inv = 1.0 / scale
    if best is not None:
        cx, cy, r, bright_ratio, _fill, _score = best
        return ScopeField(
            center_x=float(cx) * inv,
            center_y=float(cy) * inv,
            radius=float(r) * inv,
            bright_ratio=bright_ratio,
        )

    if not allow_synthetic_fallback:
        return None

    # 紧裁特写兜底：无很低阈值看是否仍有大块非黑区（排除普通白底照片）
    loose = (gray > max(int(v_threshold), 8)).astype(np.uint8)
    loose_ratio = float(np.mean(loose > 0))
    if loose_ratio < 0.35 or loose_ratio >= skip_if_bright_ratio:
        return None
    # 合成视场：中心取画面中心，半径取 0.40×最短边
    return ScopeField(
        center_x=float(w) * 0.5,
        center_y=float(h) * 0.5,
        radius=float(min_dim) * 0.40,
        bright_ratio=loose_ratio,
    )


def preprocess_capture(image_bgr: np.ndarray, eye_cfg: dict) -> PreprocessResult:
    """
    分析入口预处理：视场裁剪 + 统一工作分辨率。

    - 检出视场圆（镜筒特写）则裁剪到其外接框（留 margin），黑边不再进入后续
      流程，并把最短边压到 process_max_dim 的工作分辨率；
    - 非镜筒图原样返回（identity transform），保证既有图片行为完全不变。
    """
    scope_cfg = eye_cfg.get("scope_field", {})
    process_max_dim = int(eye_cfg.get("process_max_dim", 1600))

    scope_full: Optional[ScopeField] = None
    if scope_cfg.get("enabled", True):
        scope_full = detect_scope_field(
            image_bgr,
            v_threshold=int(scope_cfg.get("v_threshold", 12)),
            skip_if_bright_ratio=float(scope_cfg.get("skip_if_bright_ratio", 0.90)),
            min_fill_ratio=float(scope_cfg.get("min_fill_ratio", 0.55)),
            min_radius_ratio=float(scope_cfg.get("min_radius_ratio", 0.25)),
        )

    if scope_full is None:
        return PreprocessResult(image=image_bgr, transform=ImageTransform(), scope=None)

    h, w = image_bgr.shape[:2]
    margin = float(scope_cfg.get("margin_ratio", 0.02)) * scope_full.radius
    x1 = int(max(scope_full.center_x - scope_full.radius - margin, 0))
    y1 = int(max(scope_full.center_y - scope_full.radius - margin, 0))
    x2 = int(min(scope_full.center_x + scope_full.radius + margin, w))
    y2 = int(min(scope_full.center_y + scope_full.radius + margin, h))
    work = image_bgr
    off_x = off_y = 0
    if x2 - x1 >= 32 and y2 - y1 >= 32:
        work = image_bgr[y1:y2, x1:x2]
        off_x, off_y = x1, y1

    min_dim = min(work.shape[:2])
    scale = 1.0
    if process_max_dim > 0 and min_dim > process_max_dim:
        scale = float(process_max_dim) / float(min_dim)
        work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = np.ascontiguousarray(work)

    transform = ImageTransform(offset_x=off_x, offset_y=off_y, scale=scale)
    wx, wy = transform.to_working_xy(scope_full.center_x, scope_full.center_y)
    scope_work = ScopeField(
        center_x=wx,
        center_y=wy,
        radius=transform.to_working_len(scope_full.radius),
        bright_ratio=scope_full.bright_ratio,
    )

    return PreprocessResult(image=work, transform=transform, scope=scope_work)
