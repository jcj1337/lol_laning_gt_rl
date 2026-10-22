"""The demo API's typed contract (M6b): pydantic models for every request and response of `lanegame.demo.api`.

They describe the shapes the session core already produces (`session.view`, `frames.position`, `frames.turn`,
`actions.menu`, `session.recording`, `session.bundle`), so that FastAPI publishes them as an OpenAPI schema and the
frontend's TypeScript types are generated from it (`web/openapi.json`, `web/src/api/schema.d.ts`). Every model forbids
unknown fields, and a test validates the core's real output against them, so a field added on one side fails on the
other.

The models change no value: the API serializes the core's dicts through them. Field names follow the core, including
`from`, `with` and `schema`, which Python spells with an alias.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt

Seat = Literal["A", "B"]
ModelId = Literal["sp:0", "om:0"]
BeliefMode = Literal["hidden", "shown"]
Kind = Literal["research_record", "new_simulation", "human_game"]
SessionState = Literal["awaiting_action", "stopped"]
StopReason = Literal["visitor", "turn_cap", "expired"]
Packet = Tuple[int, int, bool]                 # [cell, HP, is the leading wave]
Siege = Tuple[int, int]                        # [cell, HP] of the crashed mass


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------------------------- models and status
class ModelCard(Strict):
    id: str
    arm: Literal["sp", "om"]
    seed: int
    label: str
    description: str
    config: str
    inputs: int
    file_sha256: str
    weights_digest: str
    dependencies: List[str]
    training: str
    evidence: List[str]


class Status(Strict):
    ready: bool
    models: List[ModelCard]
    error: Optional[str] = None
    turn_cap: Optional[int] = None
    lane_exit_position: Optional[int] = None
    read_position: Optional[int] = None
    simulations: Literal["ready", "starting", "in_process", "unavailable"] = "unavailable"
    replays: int = 0
    sessions: int = 0


# ---------------------------------------------------------------------------------------------- frames
class LaneView(Strict):
    packets: Dict[Seat, List[Packet]]
    siege: Dict[Seat, Siege]
    contact: Optional[float]               # the cell where the two fronts fight, half-integers between cells


class ChampionStatus(Strict):
    level: int
    hp: int
    hp_max: int
    xp: int
    xp_level: int
    xp_next: Optional[int]
    gold_earned: float
    gold_unspent: float
    invested: float
    attack_power: float
    status: Literal["present", "recall", "dead"]
    returns_in: int
    cell: Optional[int]
    at_ward: bool
    vision: int
    capacity: Optional[int]
    wave_budgets: Dict[str, int]


class Tower(Strict):
    damage: int
    damage_max: int
    plates_lost: int
    plates_total: int


class Position(Strict):
    p: int
    k: int
    lane: LaneView
    champions: Dict[Seat, ChampionStatus]
    towers: Dict[Seat, Tower]
    lead_A: float
    wave_status: str


class MinionFire(Strict):
    cause: str
    units: int


class Tick(Strict):
    tick: int
    packets: Dict[Seat, List[Packet]]
    siege: Dict[Seat, Siege]
    contact: Optional[float]               # the cell where the two fronts fight, half-integers between cells
    cells: Dict[Seat, Optional[int]]
    at_ward: Dict[Seat, bool]
    wave_removed: Dict[Seat, int]
    crashed: Dict[Seat, int]
    minion_fire: Dict[Seat, MinionFire]


class Outcome(Strict):
    kind: str
    seat: Optional[Seat]
    major: bool
    text: str
    data: Optional[Dict[str, Union[int, float, str]]] = None


class CodeParts(Strict):
    kind: Literal["lane", "recall", "ward", "absent"]
    wave: Optional[int] = None
    behaviour: Optional[int] = None
    objective: Optional[int] = None


class Delta(Strict):
    gold: float
    farm: float
    xp: int
    hp: int
    hp_taken: int


class Turn(Strict):
    t: int
    k: int
    codes: Tuple[int, int]
    names: Tuple[str, str]
    parts: Tuple[CodeParts, CodeParts]
    modes: List[str]
    ticks: List[Tick]
    outcomes: List[Outcome]
    delta: Dict[Seat, Delta]
    reward_A: float


# ---------------------------------------------------------------------------------------------- the menu
class MenuChoice(Strict):
    id: int
    name: str
    legal: bool
    help: Optional[str] = None
    reason: Optional[str] = None
    with_: Optional[List[str]] = Field(None, alias="with")


class MenuSingle(Strict):
    code: int
    name: str
    legal: bool
    help: str
    reason: Optional[str] = None


class Away(Strict):
    status: Literal["recall", "dead"]
    returns_in: int


class Menu(Strict):
    legal: List[int]
    forced: Optional[int]
    away: Optional[Away]
    waves: List[MenuChoice]
    behaviours: List[MenuChoice]
    objectives: List[MenuChoice]
    singles: List[MenuSingle]
    lane_codes: Dict[str, int]


# ---------------------------------------------------------------------------------------------- the session
class SeatPublic(Strict):
    seat: Seat
    kind: Literal["human", "model", "scripted"]
    id: Optional[str] = None
    label: Optional[str] = None
    arm: Optional[Literal["sp", "om"]] = None
    seed: Optional[int] = None
    file_sha256: Optional[str] = None
    weights_digest: Optional[str] = None
    sampling: Optional[str] = None
    rule: Optional[str] = None


class Seeds(Strict):
    chance: Optional[int]
    chance_chosen: bool
    policy: Optional[int]
    policy_rule: str


class BeliefNow(Strict):
    types: List[str]
    posterior: List[float]


class BeliefHistory(Strict):
    types: List[str]
    posteriors: List[List[float]]


class LaneExitMark(Strict):
    position: int
    panel: Dict[str, float]


class LeadMark(Strict):
    position: int
    lead_A: float
    note: str


class Marks(Strict):
    lane_exit: Optional[LaneExitMark] = None
    lead_114: Optional[LeadMark] = None


class View(Strict):
    session: str
    kind: Kind
    state: SessionState
    stop_reason: Optional[StopReason]
    position: int
    next_turn: Optional[int]
    turn_cap: int
    lane_exit_position: int
    read_position: int
    seats: Tuple[SeatPublic, SeatPublic]
    belief_mode: BeliefMode
    seeds: Seeds
    current: Position
    marks: Marks
    human_seat: Optional[Seat] = None
    opponent_ready: Optional[bool] = None
    menu: Optional[Menu] = None
    belief: Optional[Dict[Seat, BeliefNow]] = None


class History(Strict):
    session: str
    from_: int = Field(alias="from")
    to: int
    positions: List[Position]
    turns: List[Turn]
    beliefs: Optional[Dict[Seat, BeliefHistory]] = None


class CreateRequest(Strict):
    opponent: ModelId = "sp:0"
    belief: BeliefMode = "hidden"
    seed: Optional[StrictInt] = Field(None, ge=0, lt=2**53, description="a chosen chance seed (recorded as chosen)")


class CreateResponse(Strict):
    session: str
    view: View
    history: History


class SubmitRequest(Strict):
    turn: StrictInt
    code: StrictInt


class TurnResult(Strict):
    turn: Turn
    position: Position
    belief: Optional[Dict[Seat, BeliefNow]] = None
    view: View


class Summary(Strict):
    session: str
    state: SessionState
    stop_reason: Optional[StopReason]
    turns: int
    position: Position
    panel: Dict[str, float]
    marks: Marks
    note: str


class Provenance(Strict):
    kind: Kind
    note: Optional[str] = None
    source: Optional[Dict[str, object]] = None


class RecordStatus(Strict):
    state: SessionState
    stop_reason: Optional[StopReason]
    turns: int


class RecordingBeliefs(Strict):
    types: List[str]
    posteriors: Dict[Seat, List[List[float]]]


class Recording(Strict):
    schema_: str = Field(alias="schema")
    version: int
    provenance: Provenance
    config: Dict[str, str]
    start: str
    seeds: Seeds
    policy_rule: str
    seats: Tuple[SeatPublic, SeatPublic]
    belief_mode: BeliefMode
    turn_cap: int
    status: RecordStatus
    turns: List[Tuple[int, int, str]]
    lead_A: float
    marks: Marks
    beliefs: Optional[RecordingBeliefs] = None


class StopResponse(Strict):
    summary: Summary
    recording: Recording
    history: History
    view: View


# ---------------------------------------------------------------------------------------------- Watch
class SimulateRequest(Strict):
    a: ModelId = "om:0"
    b: ModelId = "sp:0"
    turns: int = Field(150, ge=1, le=150)


class BundleMeta(Strict):
    provenance: Provenance
    config: Dict[str, str]
    seeds: Seeds
    policy_rule: str
    seats: Tuple[SeatPublic, SeatPublic]
    belief_mode: BeliefMode
    turn_cap: int
    status: RecordStatus
    lead_A: float
    marks: Marks


class GameBundle(Strict):
    """A whole bot game for Watch: a new simulation, or a verified research recording. Positions 0..n, turns 1..n,
    the recording (replayable) and the beliefs the game may show."""
    name: str
    label: str
    meta: BundleMeta
    positions: List[Position]
    turns: List[Turn]
    beliefs: Optional[RecordingBeliefs]
    recording: Recording
    lane_exit_position: int
    read_position: int
    verification: Optional[Dict[str, object]] = None


class ReplayEntry(Strict):
    name: str
    label: str
    kind: Kind
    turns: int
    seats: Tuple[SeatPublic, SeatPublic]


class ErrorBody(Strict):
    error: str
    message: str


__all__ = [n for n in dir() if n[:1].isupper()]
