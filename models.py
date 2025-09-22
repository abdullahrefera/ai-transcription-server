from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# ===== REQUEST MODELS =====

class AITailoringRequest(BaseModel):
    originalTranscript: str = Field(..., description="Required transcript text")
    productDescription: str = Field(..., description="Required product description")

class ProductScraperRequest(BaseModel):
    url: str = Field(..., description="Product URL to scrape")

# ===== RESPONSE MODELS =====

class SectionBreakdown(BaseModel):
    sectionName: str
    triggerEmotionalState: str
    originalQuote: str
    rewrittenVersion: str
    sceneDescription: str
    psychologicalPrinciples: List[str]
    timestamp: str

class SutherlandAlchemy(BaseModel):
    explanation: str
    valueReframing: List[Dict[str, Any]]
    identityShifts: List[str]

class HormoziValueStack(BaseModel):
    coreOffer: str
    valueElements: List[Dict[str, Any]]
    totalStack: Dict[str, Any]
    grandSlamElements: List[str]

class AITailoringData(BaseModel):
    tailoredScript: str
    confidence: float
    processingTime: float
    wordCount: int
    estimatedReadTime: str
    sectionBreakdown: List[SectionBreakdown]
    sutherlandAlchemy: SutherlandAlchemy
    hormoziValueStack: HormoziValueStack

class AITailoringResponse(BaseModel):
    success: bool
    data: AITailoringData
    metadata: Dict[str, Any]

class ProductScraperData(BaseModel):
    description: str
    title: Optional[str] = None

class ProductScraperResponse(BaseModel):
    success: bool
    data: ProductScraperData
    metadata: Dict[str, Any]

# ===== ERROR MODELS =====

class ErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
