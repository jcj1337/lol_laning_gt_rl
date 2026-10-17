"""Bayesian opponent modelling (OM2, docs/opponent_modelling/om2_design.md): opponent types that play and report
action likelihoods (`types`), and the belief over them (`belief`). Opt-in research code: nothing in the game, the
environment's defaults or the trainer's defaults depends on it."""

from .belief import Belief, loglik, prior_features
from .types import (LibraryMixtureType, MixtureAgent, NetAgent, NetType, ScriptedAgent, ScriptedType, SwitchAgent,
                    UniformType)

__all__ = ["Belief", "loglik", "prior_features", "ScriptedType", "NetType", "UniformType", "LibraryMixtureType",
           "ScriptedAgent", "NetAgent", "MixtureAgent", "SwitchAgent"]
