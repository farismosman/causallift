from pathlib import Path
import sys, unittest, mock
from easydict import EasyDict
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from causallift.dataobjects.loggers import Loggers
from causallift.nodes import model_for_each


class ModelForEchTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_should_initialize_object_with_treatment_argument(self):
        model = model_for_each.ModelForTreatedOrUntreated(treatment_val=0)

        self.assertEqual(0, model.treatment_val)
        self.assertEqual('untreated', model.treatment_label)

    def test_should_not_initialize_object_with_treatment_argument_is_invalid(self):
        with self.assertRaises(AssertionError):
            model_for_each.ModelForTreatedOrUntreated(treatment_val=3)

    def test_should_fit_model_fail_when_object_passed_is_not_dataframe(self):
        model = model_for_each.ModelForTreatedOrUntreated(treatment_val=0)
        with self.assertRaises(AssertionError):
            model.fit(args={}, df_='DataFrame')

    def test_should_set_logger_level(self):
        Loggers().setup(verbose=2)
        self.assertEqual(20, model_for_each.log.level)

        Loggers().setup(verbose=3)
        self.assertEqual(10, model_for_each.log.level)

    @mock.patch('causallift.nodes.model_for_each.log', autospec=True)
    def test_should_log_only_info_level(self, mock_logger):
        Loggers().setup(verbose=2)
        model = model_for_each.ModelForTreatedOrUntreated(treatment_val=0)

        model._display_model_info()

        mock_logger.info.assert_called_with("## Model for Treatment = 0")




if __name__ == "__main__":
    unittest.main()