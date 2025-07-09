import sys

from .nvidia import NVIDIAModelComponent
from .nvidia_customizer import NvidiaCustomizerComponent
from .nvidia_embedding import NVIDIAEmbeddingsComponent
from .nvidia_evaluator import NvidiaEvaluatorComponent
from .nvidia_ingest import NvidiaIngestComponent
from .nvidia_rerank import NvidiaRerankComponent

if sys.platform == "win32":
    from .system_assist import NvidiaSystemAssistComponent

    __all__ = [
        "NVIDIAEmbeddingsComponent",
        "NVIDIAModelComponent",
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
        "NvidiaCustomizerComponent",
        "NvidiaEvaluatorComponent",
        "NvidiaIngestComponent",
        "NvidiaRerankComponent",
    ]
