"""WP2: progression tables (design §8.2, §9.1; plan WP2)."""

import pytest

from lanegame import GameConfig
from lanegame import progression as pg

# design §8.2: L, H(L), u(L), AP(L,0), AP(L,1), AP(L,2), dmg(1,L,0), dmg(2,L,0), dmg(1,L,2), dmg(2,L,2)
DESIGN_TABLE = """
1 600 100 10000 12000 14400 100 200 144 288
2 630 105 10500 12600 15120 105 210 151 302
3 662 110 11025 13230 15876 110 221 159 318
4 695 116 11576 13892 16670 116 232 167 333
5 729 122 12155 14586 17503 122 243 175 350
6 766 128 12763 15315 18378 128 255 184 368
7 804 134 13401 16081 19297 134 268 193 386
8 844 141 14071 16885 20262 141 281 203 405
9 886 148 14775 17729 21275 148 296 213 426
10 931 155 15513 18616 22339 155 310 223 447
11 977 163 16289 19547 23456 163 326 235 469
12 1026 171 17103 20524 24629 171 342 246 493
13 1078 180 17959 21550 25860 180 359 259 517
14 1131 188 18856 22628 27153 189 377 272 543
15 1188 198 19799 23759 28511 198 396 285 570
16 1247 208 20789 24947 29937 208 416 299 599
17 1310 218 21829 26194 31433 218 437 314 629
18 1375 229 22920 27504 33005 229 458 330 660
"""
T = pg.DEFAULT


def rows():
    for line in DESIGN_TABLE.strip().splitlines():
        yield [int(x) for x in line.split()]


def test_tables_equal_the_design_table():
    for L, H, u, a0, a1, a2, d10, d20, d12, d22 in rows():
        assert pg.hp_max(L) == H
        assert (pg.attack_power(L, 0), pg.attack_power(L, 1), pg.attack_power(L, 2)) == (a0, a1, a2)
        assert (pg.hit_damage(1, L, 0), pg.hit_damage(2, L, 0), pg.hit_damage(1, L, 2), pg.hit_damage(2, L, 2)) == \
            (d10, d20, d12, d22)
        if L == 14:
            # Specification contradiction (documented in docs/m4/m4_implementation_report.md): the normative formula
            # u(L) = floor(H(L)/6 + 1/2) gives floor(188.5 + 0.5) = 189; the design table's 188 was printed with
            # Python's round() (round half to even). The implementation follows the formula, like every other
            # half-up rounding rule of §8.2.
            assert u == 188 and pg.hp_unit(L) == 189
        else:
            assert pg.hp_unit(L) == u


def test_hp_unit_is_floor_of_a_sixth_plus_half():
    for L in range(1, 19):
        assert pg.hp_unit(L) == (pg.hp_max(L) * 2 + 6) // 12


def test_pressure_damage_values_from_the_design():
    assert pg.pressure_damage(1, 0) == 400 and pg.pressure_damage(6, 1) == 613 and pg.pressure_damage(18, 2) == 1320


def test_xp_thresholds():
    assert [T.xp_cum[L] for L in (2, 3, 6, 9, 11, 13, 16, 18)] == [8, 19, 70, 148, 215, 294, 435, 544]
    for L in range(1, 18):
        assert T.xp_next[L] == 8 + 3 * (L - 1) == T.xp_cum[L + 1] - T.xp_cum[L]
    assert T.xp_next[18] == 0


def test_level_of_boundaries_monotone_and_capped():
    prev = 1
    for x in range(0, 700):
        L = pg.level_of(x)
        assert L >= prev
        prev = L
    for L in range(2, 19):
        assert pg.level_of(T.xp_cum[L]) == L
        assert pg.level_of(T.xp_cum[L] - 1) == L - 1
    for x in (544, 545, 1000, 10 ** 6):
        assert pg.level_of(x) == 18
    with pytest.raises(ValueError):
        pg.level_of(-1)


def test_double_hit_is_twice_a_single_hit_up_to_rounding():
    for L in range(1, 19):
        for N in range(3):
            one = pg.hit_damage(1, L, N)
            assert pg.hit_damage(2, L, N) in (2 * one - 1, 2 * one, 2 * one + 1)
            assert pg.hit_damage(0, L, N) == 0


def test_attack_power_and_hp_are_monotone_and_fit_int32():
    for L in range(1, 19):
        for N in range(3):
            assert 0 < pg.attack_power(L, N) < 2 ** 31
            if N:
                assert pg.attack_power(L, N) > pg.attack_power(L, N - 1)
            if L > 1:
                assert pg.attack_power(L, N) > pg.attack_power(L - 1, N)
        if L > 1:
            assert pg.hp_max(L) > pg.hp_max(L - 1)
    assert pg.attack_power(18, 2) == 33005 and pg.hp_max(18) == 1375


def test_equal_power_reproduces_m2_ratios():
    # at L = 1, N = 0 every damage-to-max-HP ratio equals M2's (6 HP, 1 or 2 HP per hit, unit 1 HP)
    assert pg.hp_max(1) == 600 and pg.hit_damage(1, 1, 0) == 100 and pg.hit_damage(2, 1, 0) == 200
    assert pg.hp_unit(1) == 100


def test_level_up_hp_adds_only_the_max_hp_increase():
    assert pg.level_up_hp(50, 69, 70) == 50 + 766 - 729            # design §15 scenario 8 style
    assert pg.level_up_hp(22, 60, 75) == 22 + (766 - 729)
    assert pg.level_up_hp(100, 0, 19) == 100 + 662 - 600           # two levels in one turn
    assert pg.level_up_hp(100, 30, 31) == 100                      # no level-up


def test_purchase_rules():
    p0, p1 = T.prices
    assert (p0, p1) == (4_000_000, 6_000_000) and T.gold_cap == (10_000_000, 6_000_000, 0)
    assert pg.purchase(0, p0 - 1) == (0, p0 - 1, 0)
    assert pg.purchase(0, p0) == (1, 0, 1)
    assert pg.purchase(0, p0 + p1) == (2, 0, 2)
    assert pg.purchase(0, p0 + p1 - 1) == (1, p1 - 1, 1)             # tier 1 before tier 2
    assert pg.purchase(1, p1) == (2, 0, 1)
    assert pg.purchase(2, 0) == (2, 0, 0)                            # never beyond the last tier
    for items in range(3):
        for gold in range(0, T.gold_cap[items] + 1, 250_000):
            i2, g2, n = pg.purchase(items, gold)
            assert g2 >= 0 and i2 <= 2 and i2 - items == n
            assert pg.purchase(i2, g2) == (i2, g2, 0)               # idempotent once it can no longer buy
    assert pg.next_price(0) == p0 and pg.next_price(1) == p1 and pg.next_price(2) is None


def test_gold_clamp_removes_only_unspendable_gold():
    assert pg.clamp_gold(0, 12_000_000) == (10_000_000, 2_000_000)
    assert pg.clamp_gold(1, 6_000_000) == (6_000_000, 0)
    assert pg.clamp_gold(2, 5) == (0, 5)
    for items in range(3):
        g, removed = pg.clamp_gold(items, 3 * 10 ** 7)
        # whatever was removed could never have been spent: the clamped balance still buys every remaining tier
        assert pg.purchase(items, g)[0] == 2


def test_plates_broken():
    assert [pg.plates_broken(x) for x in (0, 1599, 1600, 3599, 3600, 5999, 6000)] == [0, 0, 1, 1, 2, 2, 3]


def test_tables_follow_the_configuration():
    t = pg.tables(GameConfig.m4(item_tiers=3, item_prices=(3.0, 5.0, 7.0), level_cap=10))
    assert t.item_tiers == 3 and t.level_cap == 10 and t.xp_cap == t.xp_cum[10]
    assert t.prices == (3_000_000, 5_000_000, 7_000_000) and t.gold_cap[0] == 15_000_000
