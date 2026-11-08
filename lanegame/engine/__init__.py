"""The game engine: every ruleset's state, parameters and turn resolution.

    config        GameConfig (a frozen dataclass) and its presets, from v1() to the research preset m45_calibrated()
    state         the Markov state of the lane and its canonical encoding
    mechanics     the deterministic wave step (`step_turn`): movement, merging, fights, crashes, tower fire, spawns
    game          the exact game's layer on top: legality, farm, plates, expected gank cost, reachable states
    champions     the M2 champion layer (`GameConfig.champions()`) and the chance vector shared by later resolvers
    champions_m4  the resolver of every `ruleset="m4"` preset (`resolve_turn_m4`; its docstring orders a turn)
    progression   the integer level and item tables
    positional    the M4.5 movement graph: lane cells plus the shared Vision node WARD
    lane_enum     exhaustive enumeration of the reachable lane
    render        plain-text lane traces

The only randomness is ganks and trade or tower rolls, drawn as a fixed number of uniforms per turn
(docs/model.md describes the research preset).
"""
