import fuse.data.ops.ops_cast
import fuse.data.ops.ops_common
import fuse.eval.metrics.classification.metrics_classification_common

import mmmt.models.multimodal_graph_model
import mmmt.models.multimodal_mlp
import mmmt.data.operators.op_forwardpass
import mmmt.data.representation.encoded_unimodal_to_concept
import mmmt.data.graph.concept_to_graph
import mmmt.data.representation.modality_encoding
import mmmt.data.representation.encoded_unimodal_to_concept
import mmmt.data.graph.visualization


class ObjectRegistry:
    """
    Registry of objects commonly used in an MMMT pipeline.
    """

    def __init__(self, specific_objects=None):
        """

        Args:
            specific_objects (dict, optional): Specific objects needed by the pipeline and not contaiined in the default objects. Defaults to None.
        """

        self.object_dict = {
            "OpToTensor": {
                "object": fuse.data.ops.ops_cast.OpToTensor,
                "need_cache": True,
            },
            "OpLambda": {
                "object": fuse.data.ops.ops_common.OpLambda,
                "need_cache": True,
            },
            "MetricAUCROC": {
                "object": fuse.eval.metrics.classification.metrics_classification_common.MetricAUCROC,
                "need_cache": True,
            },
            "MetricROCCurve": {
                "object": fuse.eval.metrics.classification.metrics_classification_common.MetricROCCurve,
                "need_cache": True,
            },
            "MetricAccuracy": {
                "object": fuse.eval.metrics.classification.metrics_classification_common.MetricAccuracy,
                "need_cache": True,
            },
            "ForwardPass": {
                "object": mmmt.data.operators.op_forwardpass.OpForwardPass,
                "need_cache": True,
            },
            "MultimodalGraphModel": {
                "object": mmmt.models.multimodal_graph_model.MultimodalGraphModel,
                "need_cache": True,
                "need_pipeline": True,
                "need_object_registry": True,
                "need_call_method": True,
            },
            "MultimodalMLP": {
                "object": mmmt.models.multimodal_mlp.MultimodalMLP,
                "need_cache": True,
                "need_pipeline": True,
                "need_object_registry": True,
                "need_call_method": True,
            },
            "ConceptToGraph": {
                "object": mmmt.data.graph.concept_to_graph.ConceptToGraph,
                "need_cache": True,
                "need_pipeline": True,
                "need_call_method": True,
            },
            "ModalityEncoding": {
                "object": mmmt.data.representation.modality_encoding.ModalityEncoding,
                "need_cache": True,
                "need_pipeline": True,
                "need_call_method": True,
                "need_object_registry": True,
            },
            "EncodedUnimodalToConcept": {
                "object": mmmt.data.representation.encoded_unimodal_to_concept.EncodedUnimodalToConcept,
                "need_cache": True,
                "need_pipeline": True,
                "need_call_method": True,
            },
            "GraphVisualization": {
                "object": mmmt.data.graph.visualization.GraphVisualization,
                "need_cache": True,
                "need_pipeline": True,
                "need_call_method": True,
            },
        }

        if specific_objects:
            self.object_dict.update(specific_objects)

    def instance_object(self, op_key, op_arguments):
        """
        Instanciate the selected object.

        Args:
            op_key (str): object identifier
            op_arguments (dict): arguments needed by the selected object

        Returns:
            instanciated object
        """

        if op_arguments is None:
            return self.object_dict[op_key]["object"]()
        elif any(isinstance(i, dict) for i in op_arguments.values()):
            return self.object_dict[op_key]["object"](op_arguments)
        else:
            return self.object_dict[op_key]["object"](**op_arguments)
