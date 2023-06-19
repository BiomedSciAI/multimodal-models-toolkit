import logging
from mmmt.pipeline.pipeline import MMMTPipeline

import fuseimg.datasets.knight
import knight_eval
import get_splits

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

logging.basicConfig(level=logging.INFO)

# pre-requisite for KNIGHT data
# git clone https://github.com/neheller/KNIGHT.git
# python KNIGHT/knight/scripts/get_imaging.py
# mv KNIGHT/knight/data downloads/knight_data



if __name__ == "__main__":

    parser = ArgumentParser(
        description="Multimodal fusion Experiments",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--path_to_config",
        dest="path_to_config",
        type=str,
        default="mmmt_examples/knight/mmmt_pipeline_config_demonstration.yaml",
        help="Path to pipeline configuration",
    )

    # 2 yaml files are provided in this example:
    # - mmmt_pipeline_config.yaml for training a model using all the data and GPUs
    # - mmmt_pipeline_config_demonstration.yaml for training a model with a small subset of the data and CPUs

    parser_args = parser.parse_args()

    mmmt_pipeline_config_path = parser_args.path_to_config

    # Specify specific objects needed for this particular example
    specific_objects = {
        "KNIGHT.static_pipeline": {
            "object": fuseimg.datasets.knight.KNIGHT.static_pipeline,
        },
        "Eval": {
            "object": knight_eval.knight_eval,
            "need_cache": True,
        },
        "get_splits_str_ids": {
            "object": get_splits.get_splits_str_ids,
        },
    }

    # Initialize the pipeline
    MMMTP = MMMTPipeline(
        mmmt_pipeline_config_path, specific_objects, defaults=mmmt_pipeline_config_path
    )

    # Run the pipeline - option debugging=True will only use the first 3 samples for each dataset
    MMMTP.run_pipeline(debugging=True)

    logging.info("MMMT pipeline completed")
