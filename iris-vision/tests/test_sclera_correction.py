import unittest

import numpy as np

from app.services.color import extract_iris_lab_median
from app.services.sclera import (
    ScleraReference,
    _reference_quality,
    compute_channel_gains,
    compute_profile_lstar_delta,
)


def _reference(*, quality: float = 1.0, rgb=(0.72, 0.70, 0.68)) -> ScleraReference:
    return ScleraReference(
        ok=True,
        reason="ok",
        pixel_count=2000,
        rgb_linear=np.asarray(rgb, dtype=np.float64),
        quality_score=quality,
    )


class ScleraCorrectionTests(unittest.TestCase):
    def test_unstable_reference_scores_lower(self):
        stable = _reference_quality(
            pixel_count=3000,
            min_pixels=500,
            clipped_ratio=0.01,
            side_luminance_gap=0.01,
            luminance_mad_ratio=0.01,
            cfg={},
        )
        unstable = _reference_quality(
            pixel_count=600,
            min_pixels=500,
            clipped_ratio=0.25,
            side_luminance_gap=0.18,
            luminance_mad_ratio=0.10,
            cfg={},
        )
        self.assertGreater(stable, unstable)
        self.assertGreaterEqual(stable, 0.0)
        self.assertLessEqual(stable, 1.0)

    def test_low_quality_reference_falls_back(self):
        ref = _reference(quality=0.1)
        gains, status = compute_channel_gains(
            ref,
            {
                "normalize_luminance": True,
                "normalize_chroma": True,
                "adaptive_strength": True,
                "quality_min_apply": 0.2,
            },
        )
        self.assertIsNone(gains)
        self.assertEqual(status, "sclera_low_quality")

    def test_quality_shrinks_luminance_and_chroma_strengths(self):
        ref = _reference(quality=0.5)
        gains, status = compute_channel_gains(
            ref,
            {
                "target_l": 88.0,
                "normalize_luminance": True,
                "normalize_chroma": True,
                "luminance_strength": 0.6,
                "chroma_strength": 0.4,
                "adaptive_strength": True,
                "quality_min_apply": 0.1,
                "quality_strength_power": 1.0,
                "quality_strength_floor": 0.0,
                "gain_min": 0.25,
                "gain_max": 4.0,
            },
        )
        self.assertEqual(status, "ok")
        self.assertIsNotNone(gains)
        self.assertAlmostEqual(ref.effective_luminance_strength, 0.3)
        self.assertAlmostEqual(ref.effective_chroma_strength, 0.2)

    def test_log_gain_limit_is_symmetric(self):
        ref = _reference(quality=1.0, rgb=(0.12, 0.10, 0.08))
        gains, status = compute_channel_gains(
            ref,
            {
                "target_l": 95.0,
                "normalize_luminance": True,
                "normalize_chroma": True,
                "luminance_strength": 1.0,
                "chroma_strength": 1.0,
                "adaptive_strength": False,
                "max_gain_ratio": 1.15,
                "gain_min": 0.25,
                "gain_max": 4.0,
            },
        )
        self.assertEqual(status, "ok")
        self.assertTrue(ref.gains_limited)
        self.assertTrue(np.all(gains <= 1.15 + 1e-12))
        self.assertTrue(np.all(gains >= 1.0 / 1.15 - 1e-12))

    def test_luminance_deadband_suppresses_small_exposure_change(self):
        ref = _reference(quality=1.0, rgb=(0.82, 0.82, 0.82))
        gains, status = compute_channel_gains(
            ref,
            {
                "target_l": 92.0,
                "normalize_luminance": True,
                "normalize_chroma": False,
                "luminance_strength": 0.5,
                "adaptive_strength": False,
                "luminance_log_deadband": 0.2,
            },
        )
        self.assertEqual(status, "ok")
        np.testing.assert_allclose(gains, np.ones(3), rtol=0.0, atol=1e-12)

    def test_chroma_deadband_suppresses_small_channel_change(self):
        ref = _reference(quality=1.0, rgb=(0.72, 0.70, 0.68))
        gains, status = compute_channel_gains(
            ref,
            {
                "normalize_luminance": False,
                "normalize_chroma": True,
                "chroma_strength": 0.4,
                "adaptive_strength": False,
                "chroma_log_deadband": 0.2,
            },
        )
        self.assertEqual(status, "ok")
        np.testing.assert_allclose(gains, np.ones(3), rtol=0.0, atol=1e-12)

    def test_deterministic_for_same_reference(self):
        cfg = {
            "target_l": 88.0,
            "normalize_luminance": True,
            "normalize_chroma": True,
            "luminance_strength": 0.4,
            "chroma_strength": 0.3,
            "adaptive_strength": True,
            "quality_min_apply": 0.1,
            "quality_strength_floor": 0.1,
            "max_gain_ratio": 1.2,
        }
        gains_a, status_a = compute_channel_gains(_reference(quality=0.7), cfg)
        gains_b, status_b = compute_channel_gains(_reference(quality=0.7), cfg)
        self.assertEqual(status_a, status_b)
        np.testing.assert_allclose(gains_a, gains_b, rtol=0.0, atol=0.0)

    def test_chroma_correction_can_preserve_iris_luminance(self):
        image = np.full((24, 24, 3), (80, 120, 160), dtype=np.uint8)
        mask = np.full((24, 24), 255, dtype=np.uint8)
        baseline = extract_iris_lab_median(image, mask)
        corrected = extract_iris_lab_median(
            image,
            mask,
            channel_gains=np.asarray((1.10, 0.95, 0.85)),
            preserve_luminance=True,
        )
        self.assertAlmostEqual(baseline.L, corrected.L, delta=1e-4)
        self.assertGreater(abs(baseline.a - corrected.a) + abs(baseline.b - corrected.b), 1.0)

    def test_device_profile_uses_sclera_lstar_and_quality(self):
        ref = _reference(quality=0.8)
        ref.lab = (95.0, 0.0, 0.0)
        delta, profile = compute_profile_lstar_delta(
            ref,
            (1920, 1440, 3),
            {
                "device_luminance_profiles": {
                    "enabled": True,
                    "quality_min_apply": 0.1,
                    "quality_power": 1.0,
                    "delta_lstar_max": 1.0,
                    "profiles": {
                        "honor70": {
                            "dimensions": [[1440, 1920]],
                            "sclera_target_l": 93.5,
                            "iris_l_per_sclera_l": 0.35,
                        }
                    },
                }
            },
        )
        self.assertEqual(profile, "honor70")
        self.assertAlmostEqual(delta, -0.42, places=6)


if __name__ == "__main__":
    unittest.main()
