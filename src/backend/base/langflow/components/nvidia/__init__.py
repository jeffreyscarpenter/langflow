import sys

from .nvidia_ingest import NvidiaIngestComponent
from .nvidia_rerank import NvidiaRerankComponent
from .nvidia_customizer import NvidiaCustomizerComponent
from .nvidia_evaluator import NvidiaEvaluatorComponent

if sys.platform == "win32":
    from .system_assist import NvidiaSystemAssistComponent
    __all__ = [
        "NvidiaCustomizerComponent",
        "NvidiaEvaluatorComponent",
        "NvidiaIngestComponent",
        "NvidiaRerankComponent",
        "NvidiaSystemAssistComponent",
    ]
else:
    __all__ = [
        "NvidiaCustomizerComponent",
        "NvidiaEvaluatorComponent",
        "NvidiaIngestComponent",
        "NvidiaRerankComponent",
    ]
