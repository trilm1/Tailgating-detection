from dataclasses import dataclass


@dataclass(frozen=True)
class Grant:
    grant_id: str
    t_unlock_s: float
    t_relock_s: float


@dataclass(frozen=True)
class Observation:
    """One confirmed tracker observation; bbox uses image pixels (x1,y1,x2,y2)."""

    track_id: int
    t_s: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Crossing:
    track_id: int
    t_cross_s: float
    direction: str


@dataclass(frozen=True)
class GroundTruth:
    person_id: str
    t_cross_s: float
    direction: str
    grant_id: str | None
