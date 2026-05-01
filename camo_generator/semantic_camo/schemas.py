from typing import Dict, List

from pydantic import BaseModel, Field

class NaturalScoreOutput(BaseModel):

    scores: List[float]

class SemanticScoreOutput(BaseModel):

    scores: List[float]

class AbstractionScoreOutput(BaseModel):

    scores: List[float]

class ExposureScoreOutput(BaseModel):

    scores: List[float]

class CandidateItem(BaseModel):

    text: str

    mode: str = Field(description="functional_substitution / scene_substitution / metaphorical_expression")

    source_type: str = Field(description="feature_abstract")

    explanation: str

class SemanticGenOutput(BaseModel):

    items: List[CandidateItem]

    evidence: List[str]

class SemanticFeatureOutput(BaseModel):

    features: Dict[str, List[str]] = Field(description="语义特征槽位字典")

    evidence: List[str] = Field(description="证据列表（原样返回）")

class JudgeItem(BaseModel):

    ok: bool

    reasons: List[str] = Field(default_factory=list)

    violations: List[str] = Field(default_factory=list)

    suggested_fix: str = ""

class JudgeOutput(BaseModel):

    items: List[JudgeItem]
