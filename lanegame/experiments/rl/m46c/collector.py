"""The mixed start distribution of M4.6c training (`train_shared(collector_cls=...)`).

Every training game starts, independently, from an archived state with probability `beta` and from s0 otherwise.
    * s0 games last game_length = 512 turns (M4.5c); archive games last `archive_length` = 150 turns (M4.6a E4). Both
      end in a bootstrapped truncation (a property of the run, not of the game: M4 design §10.4).
    * An archived state is drawn uniformly and mirrored (seats swapped) with probability 1/2, so a leader, a freezer
      or a trailer sits in either seat equally often. In 100% current-policy self-play the network plays both seats,
      so every archive game trains both roles (the stronger and the weaker side) at once.
    * The first game of each environment keeps M4.5c's staggering: an s0 first game has the seeded random length of
      `GameTimes`, an archive first game 1 + (that length - 1) mod archive_length.
With beta = 0.532, 512 and 150 turns, about 25% of agent transitions are played in archive-started games.

Draws use a dedicated generator (SeedSequence([seed, 46003])), so the s0 stream of chance seeds, matchups and first
lengths is that of `SharedCollector`. Counts: games started by kind and archive label, mirrored draws, and the exact
number of joint steps of FINISHED games by kind (`describe_starts`).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def mixed_collector_cls(archive_path: str, beta: float, archive_length: int, mirror_p: float = 0.5):
    """A SharedCollector subclass bound to an archive and its mixture parameters."""
    from ....rl.shared import SharedCollector

    class MixedStartCollector(SharedCollector):
        ARCHIVE_PATH, BETA, ARCHIVE_LENGTH, MIRROR_P = archive_path, float(beta), int(archive_length), float(mirror_p)

        def __init__(self, game_cfg, cfg, sampler, seed: int):
            from ....env.starts import load_archive
            if cfg.start_mix != 0.0:
                raise ValueError("the mixed collector draws archive starts itself: PPOConfig.start_mix must be 0")
            self.archive = load_archive(self.ARCHIVE_PATH, game_cfg)
            if not self.archive:
                raise ValueError("empty start archive")
            self.mix_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 46003]))
            self.game_kind: List[str] = [""] * cfg.n_envs
            self.game_label: List[str] = [""] * cfg.n_envs
            self.counts: Dict[str, int] = {"initial": 0, "archive": 0, "mirrored": 0}
            self.by_label: Dict[str, int] = {}
            self.turns_finished: Dict[str, int] = {"initial": 0, "archive": 0}
            self.turns_by_label: Dict[str, int] = {}
            super().__init__(game_cfg, cfg, sampler, seed)

        def _reset(self, i: int) -> None:
            from ....rl.shared import EpisodeLog, episode_streams
            if self.logs[i] is not None and self.game_kind[i]:        # the game that just ended (collect's order)
                n = len(self.logs[i].rewards)
                self.turns_finished[self.game_kind[i]] += n
                if self.game_kind[i] == "archive":
                    self.turns_by_label[self.game_label[i]] = self.turns_by_label.get(self.game_label[i], 0) + n
            ss = self._episode_seeds.spawn(1)[0]
            _, env_ss, rng_a, rng_b = episode_streams(ss)
            self.match[i] = self.sampler.sample(self.rng)
            skills = (self.game_cfg.skill_a, self.game_cfg.skill_b)
            if self.cfg.skill_levels is not None:
                raise ValueError("the mixed collector does not support skill_levels")
            self.skills[i] = skills
            self.envs[i] = self._env(i, skills)
            self.agent_rng[i] = (rng_a, rng_b)
            self._game_match[i] = self.match[i]
            self.obs[i] = self._start(i, self.envs[i], env_ss)
            names = tuple("learner" if c is None else c.name for c in self.match[i].controllers)
            self.logs[i] = EpisodeLog(seed=None, k0=0, horizon=self.envs[i].horizon, names=names)
            self.payoff[i] = 0.0

        def _start(self, i: int, env, env_ss):
            tm = self.times
            first = tm.games_started[i] == 0
            tm.games_started[i] += 1
            if self.mix_rng.random() < self.BETA:
                st = self.archive[int(self.mix_rng.integers(len(self.archive)))]
                label = st.label.split(" ")[0] if st.label else "archive"
                if self.mix_rng.random() < self.MIRROR_P:
                    st = st.mirror()
                    self.counts["mirrored"] += 1
                env.max_turns = 1 + (tm.first_lengths[i] - 1) % self.ARCHIVE_LENGTH if first else self.ARCHIVE_LENGTH
                self.counts["archive"] += 1
                self.by_label[label] = self.by_label.get(label, 0) + 1
                self.game_kind[i], self.game_label[i] = "archive", label
                tm.start_counts["archive"] = tm.start_counts.get("archive", 0) + 1
                return env.reset(seed=env_ss, root=st.lane, k0=st.phase, champ=st.champ)
            env.max_turns = tm.first_lengths[i] if first else tm.game_length
            self.counts["initial"] += 1
            self.game_kind[i], self.game_label[i] = "initial", ""
            tm.start_counts["initial"] += 1
            return env.reset(seed=env_ss)

        def describe_starts(self) -> dict:
            tot = sum(self.turns_finished.values())
            return {"archive": self.ARCHIVE_PATH, "archive_states": len(self.archive), "beta": self.BETA,
                    "archive_length": self.ARCHIVE_LENGTH, "game_length": self.times.game_length,
                    "mirror_p": self.MIRROR_P, "games_started": dict(self.counts), "archive_by_label": dict(self.by_label),
                    "joint_steps_finished": dict(self.turns_finished),
                    "joint_steps_finished_by_label": dict(self.turns_by_label),
                    "archive_transition_share_finished": self.turns_finished["archive"] / tot if tot else float("nan")}

    return MixedStartCollector


__all__ = ["mixed_collector_cls"]
