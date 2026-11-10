import unittest
import numpy as np
import pandas as pd
from stats.adversarial_analysis import digit_configuration_comparisons


class ConfigurationTests(unittest.TestCase):
    def test_pairing_adjustment_and_zero_endpoint(self):
        rows = []
        for model in ["A", "B", "C"]:
            for scenario in range(5):
                rows.append(
                    dict(
                        model=model,
                        setting=model,
                        scenario=scenario,
                        adversarial=int(model == "A" and scenario > 0),
                        redacted=0,
                        errored=0,
                        usable=True,
                    )
                )
        result = digit_configuration_comparisons(pd.DataFrame(rows))
        self.assertEqual(len(result), 6)
        self.assertTrue(result.family_size.eq(6).all())
        self.assertTrue(result.p_t_Holm_all.ge(result.p_t).all())
        self.assertTrue(result.p_sign_Holm_all.ge(result.p_sign).all())
        zero = result[result.metric.eq("redaction_rate")]
        self.assertTrue(zero.p_t_Holm_all.eq(1).all())
        self.assertTrue(zero.CI_95_low.isna().all())
        ab = result[
            (result.target == "A")
            & (result.reference == "B")
            & result.metric.eq("adversarial_rate")
        ].iloc[0]
        self.assertAlmostEqual(ab.mean_difference, 0.8)
        self.assertAlmostEqual(ab.p_sign, 0.125)


if __name__ == "__main__":
    unittest.main()
