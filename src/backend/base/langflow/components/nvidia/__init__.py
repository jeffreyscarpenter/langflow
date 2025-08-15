import sys

from .nemo_guardrails import NVIDIANeMoGuardrailsComponent
from .nemo_guardrails_validator import NVIDIANeMoGuardrailsValidator
from .nvidia import NVIDIAModelComponent
from .nvidia_customizer import NvidiaCustomizerComponent
from .nvidia_data_preparation import NeMoDataPreparationComponent
from .nvidia_embedding import NVIDIAEmbeddingsComponent
from .nvidia_evaluator import NvidiaEvaluatorComponent
from .nvidia_ingest import NvidiaIngestComponent
from .nvidia_rerank import NvidiaRerankComponent

if sys.platform == "win32":
    from .system_assist import NvidiaSystemAssistComponent

    __all__ = [
        "NVIDIAEmbeddingsComponent",
        "NVIDIAModelComponent",
        "NVIDIANeMoGuardrailsComponent",
        "NVIDIANeMoGuardrailsValidator",
        "NeMoDataPreparationComponent",
        "NvidiaCustomizerComponent",
        "NvidiaEvaluatorComponent",
        "NvidiaIngestComponent",
        "NvidiaRerankComponent",
        "NvidiaSystemAssistComponent",
    ]
else:
    __all__ = [
        "NVIDIAEmbeddingsComponent",
        "NVIDIAModelComponent",
        "NVIDIANeMoGuardrailsComponent",
        "NVIDIANeMoGuardrailsValidator",
        "NeMoDataPreparationComponent",
        "NvidiaCustomizerComponent",
        "NvidiaEvaluatorComponent",
        "NvidiaIngestComponent",
        "NvidiaRerankComponent",
    ]
