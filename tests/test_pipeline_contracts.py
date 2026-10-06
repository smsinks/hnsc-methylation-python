"""Small independent checks of numerical and portability contracts."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy.stats import false_discovery_control
from hnsc_methylation.core import bh_fdr, TrainingVarianceFilter
from hnsc_methylation.robustness import TrainingCompletenessFilter
from hnsc_methylation.pipeline import PipelineConfig


class PipelineContracts(unittest.TestCase):
    def test_bh_matches_scipy_with_nonfinite_tests(self):
        p = np.array([.8, .001, np.nan, .03, .02])
        actual = bh_fdr(p)
        finite = np.isfinite(p)
        np.testing.assert_allclose(actual[finite], false_discovery_control(p[finite]))
        self.assertTrue(np.isnan(actual[2]))

    def test_training_filters_ignore_held_out_observations(self):
        train = np.array([[0., 1., np.nan], [0., 2., np.nan], [0., 3., 4.], [0., 4., 5.]])
        variance = TrainingVarianceFilter(k=1).fit(train)
        np.testing.assert_array_equal(variance.indices_, [1])
        held_out = np.array([[10000., 7., 9.]])
        np.testing.assert_array_equal(variance.transform(held_out), [[7.]])
        complete = TrainingCompletenessFilter().fit(train, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(complete.keep_, [True, True, False])
        np.testing.assert_array_equal(complete.transform(held_out), [[10000., 7.]])

    def test_outputs_resolve_from_repository_and_protect_inputs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            config = PipelineConfig(root, Path("run-output"), permutations=1)
            self.assertEqual(config.output_dir, root / "run-output")
            with self.assertRaises(ValueError):
                PipelineConfig(root, root / "data" / "inputs")
            with self.assertRaises(ValueError):
                PipelineConfig(root, permutations=0)


if __name__ == "__main__":
    unittest.main()
