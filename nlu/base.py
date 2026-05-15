from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedIntent:
    intent: str  # QUERY | SUBMIT_PRICE | GREETING | UNKNOWN
    product: Optional[str]
    unit: Optional[str]
    location: Optional[str]
    price: Optional[float]
    quantity: Optional[float]
    confidence: float  # 0.0 – 1.0

    # Minimum confidence to proceed to Price Intelligence Engine
    CONFIDENCE_GATE = 0.65

    def above_gate(self) -> bool:
        return self.confidence >= self.CONFIDENCE_GATE


class BaseNLUParser(ABC):
    @abstractmethod
    def parse(self, message: str) -> ParsedIntent:
        """
        Takes a preprocessed message string.
        Returns a ParsedIntent.
        Must never raise — catch internally and return UNKNOWN with low confidence.
        """
        pass
