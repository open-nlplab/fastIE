from tests.utils import UnifiedTaskTest
from tests.dummy import dummy_ner_dataset

from fastie.tasks.ner import BertNER

class TestBertNER(UnifiedTaskTest):
    def setup_class(self):
        super().setup_class(self, task_cls=BertNER, data_bundle=dummy_ner_dataset(), extra_parameters={
            "pretrained_model_name_or_path": "prajjwal1/bert-tiny"
        })
