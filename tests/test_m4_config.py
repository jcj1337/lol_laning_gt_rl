"""WP0: the ruleset contract (plan WP0, tests C1-C3)."""

import json

import pytest

from lanegame import GameConfig
from lanegame.config import M4_ECONOMY_FIELDS, _FIELD_DEFAULTS


def test_c1_presets_round_trip_through_json():
    for cfg in (GameConfig.m4(), GameConfig.champions_continuing(), GameConfig.m4(skill_a=0.7, gamma=0.98)):
        cfg.validate()
        back = GameConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        assert back == cfg


def test_c2_every_contract_row_is_accepted():
    for cfg in (GameConfig(), GameConfig.v1(), GameConfig.v2(), GameConfig.champions()):
        cfg.validate()
        assert cfg.ruleset == "legacy" and cfg.time_mode == "finite" and cfg.time_features and cfg.gamma == 1.0
    c = GameConfig.champions_continuing()
    c.validate()
    assert (c.ruleset, c.time_mode, c.time_features, c.gamma, c.champion_mode) == ("m2c", "continuing", False, 0.99, True)
    m = GameConfig.m4()
    m.validate()
    assert (m.ruleset, m.time_mode, m.time_features, m.gamma) == ("m4", "continuing", False, 0.99)
    assert m.plate_mode == "durability" and m.freeze_maintenance and m.chase_rule == "contact_cell"
    # ablations stay valid M4 configurations
    GameConfig.m4(freeze_maintenance=False).validate()
    GameConfig.m4(tower_adjacent_p=0.0, tower_adjacent_p_busy=0.0).validate()
    GameConfig.m4(chase_rule="m2").validate()


@pytest.mark.parametrize("kw", [dict(time_mode="continuing"), dict(time_features=False), dict(gamma=0.99)])
def test_c2_legacy_rejects_non_legacy_time_fields(kw):
    for preset in (GameConfig, GameConfig.v1, GameConfig.v2, GameConfig.champions):
        with pytest.raises(ValueError, match=list(kw)[0]):
            preset(**kw).validate()


@pytest.mark.parametrize("field", M4_ECONOMY_FIELDS)
def test_c2_legacy_and_m2c_reject_any_m4_economy_field(field):
    on = GameConfig.m4()
    value = getattr(on, field)
    if value == _FIELD_DEFAULTS[field]:          # fields m4() leaves at the default: switch them some other way
        value = {"xp_range": 2, "xp_base": 9, "xp_step": 4, "level_cap": 10, "hp_base": 700, "hp_growth": 1.06,
                 "ap_level_growth": 1.06, "item_mult": 1.3, "item_prices": (3.0, 5.0),
                 "plate_thresholds": (1000, 2000, 3000), "plate_gold": 1.5, "pressure_tower_damage": 300,
                 "plate_gold_requires_presence": True, "gold_unit": 1e-3, "rear_count_max": 5,
                 "item_model": "continuous", "invest_alpha": 0.06, "invest_p": 0.8, "invest_obs_gold_ref": 64,
                 "invest_obs_ref": 512, "invest_curve": "saturating", "invest_gain": 2.0, "invest_scale": 8.0,
                 "invest_obs_encoding": "compressed", "farm_spot_rule": "pre_crash_front"}[field]
    for preset in (GameConfig.champions, GameConfig.champions_continuing, GameConfig):
        with pytest.raises(ValueError, match=field):
            preset(**{field: value}).validate()
    with pytest.raises(ValueError, match="plate_mode"):
        GameConfig.champions(plate_mode="durability").validate()


def test_c2_m4_and_m2c_reject_finite_time_and_m4_rejects_reduced_modes():
    with pytest.raises(ValueError, match="time_mode"):
        GameConfig.m4(time_mode="finite").validate()
    with pytest.raises(ValueError, match="time_mode"):
        GameConfig.champions_continuing(time_mode="finite").validate()
    with pytest.raises(ValueError, match="positioning"):
        GameConfig.m4(positioning=False).validate()
    with pytest.raises(ValueError, match="champion_mode"):
        GameConfig.m4(champion_mode=False).validate()
    with pytest.raises(ValueError, match="champion_mode"):
        GameConfig.champions_continuing(champion_mode=False).validate()
    with pytest.raises(ValueError, match="gamma"):
        GameConfig.m4(gamma=1.0).validate()
    with pytest.raises(ValueError, match="time_features"):
        GameConfig.m4(time_features=True).validate()
    with pytest.raises(ValueError, match="plate_mode"):
        GameConfig.m4(plate_mode="crash").validate()


def test_validate_rejects_economies_not_representable_in_micro_resources():
    with pytest.raises(ValueError, match="farm_per_hp"):
        GameConfig.m4(farm_per_hp=0.3333).validate()
    with pytest.raises(ValueError, match="kill_bounty"):
        GameConfig.m4(kill_bounty=0.0000005).validate()
    with pytest.raises(ValueError, match="plate_gold"):
        GameConfig.m4(plate_gold=1.0000001).validate()
    with pytest.raises(ValueError, match="item_prices"):
        GameConfig.m4(item_prices=(4.0000001, 6.0)).validate()
    # the finest default per-HP value is 0.25 * 0.9 * 0.85 * 0.5 = 0.095625 = 95 625 micro-resources: accepted
    GameConfig.m4().validate()


def test_c3_old_run_configurations_load_as_legacy():
    old = GameConfig.champions().to_dict()
    for k in ("ruleset", "time_mode", "time_features", "gamma") + M4_ECONOMY_FIELDS:
        old.pop(k)
    cfg = GameConfig.from_dict(old)
    assert cfg.ruleset == "legacy" and cfg == GameConfig.champions()
    cfg.validate()
