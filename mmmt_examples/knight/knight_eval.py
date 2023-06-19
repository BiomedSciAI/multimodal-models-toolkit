import os

from fuse.eval.evaluator import EvaluatorDefault
from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAUCROC,
    MetricROCCurve,
)
from fuse.eval.metrics.metrics_common import CI


def knight_eval(args_dict: dict) -> dict:
    """
    Evaluation of the binary task in knight
    Expect as input either a dataframe or a path to a dataframe that includes 3 columns:
    1. "id" - unique identifier per sample - can be a running index
    2. "pred" - prediction scores per sample
    3. "target" - the ground truth label
    """

    evaluation_directory = os.path.join(
        args_dict["root_dir"], args_dict["step_args"]["evaluation_directory"]
    )
    test_results_filename = os.path.join(
        evaluation_directory, args_dict["step_args"]["test_results_filename"]
    )

    metrics = {
        "auc": CI(
            MetricAUCROC(
                pred="pred",
                target="target",
            ),
            stratum="target",
        ),
        "roc_curve": MetricROCCurve(
            pred="pred",
            target="target",
            output_filename=os.path.join(evaluation_directory, "roc.png"),
        ),
    }

    evaluator = EvaluatorDefault()
    return evaluator.eval(
        ids=None,
        data=test_results_filename,
        metrics=metrics,
        output_dir=evaluation_directory,
    )
