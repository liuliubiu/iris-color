# Pan 2017 虹膜颜色分级参考

## 文献

Pan C-W, Qiu Q-X, Qian D-J, Hu D-N, Li J, Saw S-M, Zhong H. **Iris colour in relation to myopia among Chinese school-aged children.** Ophthalmic Physiol Opt. 2018;38:48–55.

DOI: https://doi.org/10.1111/opo.12427

## 研究人群

- 中国云南墨江 13–14 岁中学生，n=2346
- 虹膜颜色变化范围：**浅棕～深棕**（中国人典型范围）

## 分级协议

1. **拍照**：暗室 + Topcon DC-3 裂隙灯数码相机，JPEG 3120×4160
2. **查看**：Photoshop CS
3. **评分**：两名评分者盲评，与 Figure 1 参考图板做**整体虹膜颜色**对比
4. **档位**：Grade 1（最浅）～ Grade 5（最深）
5. **边界规则**：介于两档之间 → **取较高档（更深）**
6. **不一致处理**：第三人裁定
7. **信度**：评分者间 Kappa=0.74，评分者内 Kappa=0.88

## 人群 Grade 分布（Figure 2）

| Grade | 人数 | 占比 |
|-------|------|------|
| 1 | 23 | ~1% |
| 2 | 276 | ~12% |
| 3 | 1662 | **~71%** |
| 4 | 314 | ~13% |
| 5 | 71 | ~3% |

## 与屈光/眼轴的关系（仅供参考，非本系统目标）

- Grade 1 平均 S.E. ≈ +0.40 D，Grade 5 ≈ -0.89 D
- Grade 1 平均 AL ≈ 23.2 mm，Grade 5 ≈ 23.9 mm

## 论文局限（对本项目的影响）

- 分级是**主观的**，**未提供** Lab/RGB 阈值或量化公式
- 作者建议未来采用更客观的量化方法
- 本 MVP 用手机/摄像头 + CIELAB 自动近似，**不可直接声称与论文分级等效**

## MVP 标定策略

1. 初期：用 L* 占位阈值（见 `iris-vision/config/grade_thresholds.yaml`）
2. 收集 ~100 张样张 + 人工 Pan 式分级
3. 调整阈值，使分布接近 G3≈71%，或简化为 3 档（浅/中/深）
