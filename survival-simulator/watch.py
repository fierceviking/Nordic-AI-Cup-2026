"""Watch the rule-based hive play, with its internal state drawn on top.

    python watch.py                      # random seed
    python watch.py --seed 11            # a specific map
    python watch.py --seed 11 --speed 4  # 4 sim ticks per rendered frame
    python watch.py --params best_params.json
    python watch.py --set AGING_DUMP=1 --set AGING_COOLDOWN=0.1

Controls
    space        pause / resume
    . or right   single tick (while paused)
    + / -        faster / slower
    tab / shift-tab  follow next / previous agent
    f            stop following
    v            agent labels (id, mode, energy, age)
    t            line from each agent to what it is going for
    m            the hive's remembered map for the followed agent's lineage
    r            restart this seed
    n            new random seed
    h            help
    esc / q      quit

The remembered map is the interesting one: the hive navigates in a per-lineage
frame with an unknown rotation, so pressing `m` re-projects that frame onto the
real world and shows where the hive *believes* trees, fruit and predators are.
Drift between a belief marker and the real object is a dead-reckoning error.
"""

import argparse
import json
import math
import os
import random
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from src.core import SimulationCore

WORLD_VIEW = (1120, 840)
PANEL_W = 430
BG = (18, 18, 22)
FG = (232, 232, 236)
DIM = (150, 150, 158)
ACCENT = (120, 200, 255)
WARN = (255, 170, 90)
BAD = (255, 110, 110)
GOOD = (130, 230, 150)

MODE_COLOUR = {
    "fruit": (120, 230, 140),
    "patrol": (120, 190, 255),
    "hold": (200, 200, 120),
    "flee": (255, 110, 110),
    "explore": (190, 150, 255),
    "lost": (255, 120, 220),
    "init": (150, 150, 150),
}


def load_params(args):
    params = {}
    if args.params:
        with open(args.params) as handle:
            params.update(json.load(handle))
    for item in args.set or []:
        key, _, raw = item.partition("=")
        key = key.strip()
        raw = raw.strip()
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw) if raw.isdigit() or (raw[:1] == "-" and raw[1:].isdigit()) else float(raw)
            except ValueError:
                value = raw
        params[key] = value
    params.pop("__controller__", None)
    return params


def make_hive(controller, params):
    if controller == "burst":
        from src.utils.controllers.burst_agent_policy import BurstHive
        return BurstHive(**params)
    from src.utils.controllers.expert_agent_policy import Hive
    return Hive(**params)


class Watcher:
    def __init__(self, args):
        self.args = args
        self.params = load_params(args)
        self.show_labels = True
        self.show_targets = True
        self.show_memory = False
        self.show_help = False
        self.paused = args.paused
        self.speed = max(1, args.speed)
        self.follow_index = None
        self.step_once = False
        self.births = 0
        self.deaths = 0
        self._prev_ids = set()
        self._next_id = 0

        pygame.init()
        pygame.display.set_caption("survival simulator - hive viewer")
        self.screen = pygame.display.set_mode((WORLD_VIEW[0] + PANEL_W, WORLD_VIEW[1]))
        self.view = self.screen.subsurface(pygame.Rect(0, 0, *WORLD_VIEW))
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.small = pygame.font.SysFont("consolas,menlo,monospace", 13)
        self.big = pygame.font.SysFont("consolas,menlo,monospace", 19, bold=True)
        self.clock = pygame.time.Clock()
        self.reset(args.seed)

    # -- simulation ----------------------------------------------------
    def reset(self, seed):
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.sim = SimulationCore(seed=self.seed)
        self.hive = make_hive(self.args.controller, self.params)
        self.hive.reset()
        self.actions = []
        self.state = {"score": 0.0, "num_agents": 0, "sim_time": 0.0, "observations": []}
        self.zoom = min(WORLD_VIEW[0] / self.sim.env_width, WORLD_VIEW[1] / self.sim.env_height)
        self.births = self.deaths = 0
        self._prev_ids = {a.agent_id for a in self.sim.env.agents}
        self.peak_pop = len(self._prev_ids)
        self.over = False

    def tick(self):
        if self.over:
            return
        self.state = self.sim.step(self.actions)
        ids = {a.agent_id for a in self.sim.env.agents}
        self.births += len(ids - self._prev_ids)
        self.deaths += len(self._prev_ids - ids)
        self._prev_ids = ids
        self.peak_pop = max(self.peak_pop, len(ids))
        requests = self.hive.act(self.state["observations"], self.state["sim_time"])
        self.actions = [(r.agent_id, r) for r in requests]
        if self.state["num_agents"] == 0 or self.sim.env.time > self.args.max_time:
            self.over = True

    # -- geometry ------------------------------------------------------
    def to_screen(self, wx, wy):
        return int(wx * self.zoom), int(wy * self.zoom)

    def followed_agent(self):
        agents = sorted(self.sim.env.agents, key=lambda a: a.agent_id)
        if not agents or self.follow_index is None:
            return None
        return agents[self.follow_index % len(agents)]

    def lineage_transform(self, agent):
        """Map the hive's lineage-local frame onto true world coordinates."""
        mind = getattr(self.hive, "minds", {}).get(agent.agent_id)
        if mind is None:
            return None
        rot = agent.direction - mind.th
        cos_r, sin_r = math.cos(rot), math.sin(rot)

        def project(lx, ly):
            dx, dy = lx - mind.x, ly - mind.y
            return agent.x + cos_r * dx - sin_r * dy, agent.y + sin_r * dx + cos_r * dy

        return mind, project

    # -- drawing -------------------------------------------------------
    def draw_world(self):
        self.sim.env.draw(self.view)
        minds = getattr(self.hive, "minds", {})

        if self.show_memory:
            followed = self.followed_agent()
            if followed is not None:
                transform = self.lineage_transform(followed)
                if transform is not None:
                    mind, project = transform
                    self.draw_memory(mind, project)

        for agent in self.sim.env.agents:
            mind = minds.get(agent.agent_id)
            pos = self.to_screen(agent.x, agent.y)
            mode = getattr(mind, "mode", "?") if mind else "?"
            colour = MODE_COLOUR.get(mode, DIM)

            if self.show_targets and mind is not None:
                self.draw_target_line(agent, mind, pos, colour)

            pygame.draw.circle(self.view, colour, pos, 9, 2)
            if getattr(mind, "aging", False):
                pygame.draw.circle(self.view, WARN, pos, 13, 1)

            if self.show_labels:
                energy = agent.energy
                bar_w = 26
                filled = int(bar_w * max(0.0, min(1.0, energy / max(agent.max_energy, 1.0))))
                bar = pygame.Rect(pos[0] - bar_w // 2, pos[1] - 18, bar_w, 4)
                pygame.draw.rect(self.view, (40, 40, 44), bar)
                pygame.draw.rect(self.view, GOOD if energy > 120 else BAD,
                                 pygame.Rect(bar.x, bar.y, filled, 4))
                label = f"{agent.agent_id}:{mode[:3]} {energy:.0f}"
                self.view.blit(self.small.render(label, True, FG), (pos[0] + 12, pos[1] - 8))

        followed = self.followed_agent()
        if followed is not None:
            pos = self.to_screen(followed.x, followed.y)
            pygame.draw.circle(self.view, ACCENT, pos, 20, 2)
            pygame.draw.line(self.view, ACCENT, (pos[0] - 26, pos[1]), (pos[0] - 14, pos[1]), 2)
            pygame.draw.line(self.view, ACCENT, (pos[0] + 14, pos[1]), (pos[0] + 26, pos[1]), 2)

    def draw_target_line(self, agent, mind, pos, colour):
        transform = self.lineage_transform(agent)
        if transform is None:
            return
        _, project = transform
        world = mind.world
        record = None
        if mind.target_fruit is not None:
            record = world.fruits.get(mind.target_fruit)
        elif mind.target_tree is not None:
            record = world.trees.get(mind.target_tree)
        if record is None:
            return
        tx, ty = project(record[0], record[1])
        pygame.draw.line(self.view, colour, pos, self.to_screen(tx, ty), 1)

    def draw_memory(self, mind, project):
        world = mind.world
        for record in world.trees.values():
            x, y = project(record[0], record[1])
            pygame.draw.circle(self.view, (90, 190, 120), self.to_screen(x, y), 7, 1)
        for record in world.fruits.values():
            x, y = project(record[0], record[1])
            pygame.draw.circle(self.view, (230, 230, 120), self.to_screen(x, y), 4, 1)
        for record in world.predators:
            x, y = project(record[0], record[1])
            point = self.to_screen(x, y)
            pygame.draw.circle(self.view, BAD, point, 11, 1)
            pygame.draw.line(self.view, BAD, (point[0] - 6, point[1] - 6),
                             (point[0] + 6, point[1] + 6), 1)
            pygame.draw.line(self.view, BAD, (point[0] - 6, point[1] + 6),
                             (point[0] + 6, point[1] - 6), 1)

    def draw_panel(self):
        panel = pygame.Rect(WORLD_VIEW[0], 0, PANEL_W, WORLD_VIEW[1])
        pygame.draw.rect(self.screen, BG, panel)
        x = WORLD_VIEW[0] + 14
        y = 12

        def line(text, colour=FG, font=None, gap=18):
            nonlocal y
            self.screen.blit((font or self.font).render(text, True, colour), (x, y))
            y += gap

        env = self.sim.env
        agents = env.agents
        mature = sum(1 for tree in env.trees if tree.age >= 20.0)
        mean_energy = sum(a.energy for a in agents) / len(agents) if agents else 0.0
        mean_age = sum(a.age for a in agents) / len(agents) if agents else 0.0
        ratio = self.births / max(1, self.deaths)

        line(f"seed {self.seed}", ACCENT, self.big, 26)
        line(f"{'PAUSED' if self.paused else 'running'}  x{self.speed}"
             f"{'  GAME OVER' if self.over else ''}",
             WARN if (self.paused or self.over) else DIM)
        y += 6
        line(f"score      {self.state['score']:8.1f}", FG, self.big, 24)
        line(f"sim time   {env.time:8.1f} s")
        line(f"population {len(agents):8d}   peak {self.peak_pop}")
        line(f"births {self.births:4d}  deaths {self.deaths:4d}")
        line(f"births/deaths {ratio:5.2f}", GOOD if ratio >= 1.0 else BAD)
        y += 6
        line(f"trees {len(env.trees):4d}  mature {mature:4d}", DIM)
        line(f"fruits {len(env.fruits):3d}  predators {len(env.predators):3d}", DIM)
        line(f"mean energy {mean_energy:7.1f}", DIM)
        line(f"mean age    {mean_age:7.1f}", DIM)
        y += 8

        modes = {}
        for agent in agents:
            mind = getattr(self.hive, "minds", {}).get(agent.agent_id)
            modes[getattr(mind, "mode", "?")] = modes.get(getattr(mind, "mode", "?"), 0) + 1
        line("mode", ACCENT)
        for mode, count in sorted(modes.items(), key=lambda kv: -kv[1]):
            bar = "#" * min(20, count)
            line(f" {mode:<8}{count:3d} {bar}", MODE_COLOUR.get(mode, DIM), self.small, 16)
        y += 8

        followed = self.followed_agent()
        if followed is not None:
            mind = getattr(self.hive, "minds", {}).get(followed.agent_id)
            line(f"agent {followed.agent_id}", ACCENT, self.big, 24)
            line(f" energy {followed.energy:7.1f} / {followed.max_energy:.0f}")
            line(f" age    {followed.age:7.1f}  max {followed.max_age:.0f}",
                 WARN if followed.age > followed.max_age else FG)
            line(f" speed  {followed.speed:5.1f}  sprint {followed.sprint_speed:5.1f}", DIM)
            line(f" hear   {followed.hearing_radius:5.1f}  vision {followed.vision_radius:5.1f}", DIM)
            if mind is not None:
                line(f" mode   {mind.mode}", MODE_COLOUR.get(mind.mode, FG))
                line(f" aging  {bool(mind.aging)}", WARN if mind.aging else DIM)
                line(f" target fruit {mind.target_fruit}  tree {mind.target_tree}", DIM, self.small, 16)
                line(f" known trees {len(mind.world.trees):3d}"
                     f"  fruit {len(mind.world.fruits):3d}", DIM, self.small, 16)
                line(f" known predators {len(mind.world.predators)}", DIM, self.small, 16)
        else:
            line("tab: follow an agent", DIM)

        debug = getattr(self.hive, "debug", None)
        if debug:
            y += 6
            line("hive", ACCENT)
            for key in ("target", "net", "income"):
                if key in debug:
                    line(f" {key:<7}{debug[key]}", DIM, self.small, 16)

        # __init__ assigns overrides onto the instance, so these are exactly the
        # tunables that differ from the shipped defaults.
        overrides = {k: v for k, v in vars(self.hive).items() if k.isupper()}
        if overrides:
            y += 6
            line("rules active", WARN)
            for key in sorted(overrides):
                line(f" {key} = {overrides[key]}", WARN, self.small, 16)

        if self.show_help:
            self.draw_help()

    def draw_help(self):
        lines = [
            "space  pause          v  labels",
            ". / ->  step          t  target lines",
            "+ / -  speed          m  hive memory",
            "tab    follow next    f  unfollow",
            "r      restart seed   n  new seed",
            "h      hide help      q  quit",
        ]
        box = pygame.Surface((430, 24 + 18 * len(lines)), pygame.SRCALPHA)
        box.fill((10, 10, 14, 235))
        for index, text in enumerate(lines):
            box.blit(self.small.render(text, True, FG), (12, 12 + 18 * index))
        self.view.blit(box, (14, WORLD_VIEW[1] - box.get_height() - 14))

    # -- input ---------------------------------------------------------
    def handle(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type != pygame.KEYDOWN:
            return True
        key = event.key
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_PERIOD, pygame.K_RIGHT):
            self.step_once = True
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(64, self.speed * 2)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(1, self.speed // 2)
        elif key == pygame.K_TAB:
            count = len(self.sim.env.agents)
            if count:
                shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                base = 0 if self.follow_index is None else self.follow_index + (-1 if shift else 1)
                self.follow_index = base % count
        elif key == pygame.K_f:
            self.follow_index = None
        elif key == pygame.K_v:
            self.show_labels = not self.show_labels
        elif key == pygame.K_t:
            self.show_targets = not self.show_targets
        elif key == pygame.K_m:
            self.show_memory = not self.show_memory
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        elif key == pygame.K_r:
            self.reset(self.seed)
        elif key == pygame.K_n:
            self.reset(None)
        return True

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                running = self.handle(event)
                if not running:
                    break
            if not running:
                break
            if not self.paused:
                for _ in range(self.speed):
                    self.tick()
                    if self.over:
                        break
            elif self.step_once:
                self.tick()
                self.step_once = False

            self.screen.fill(BG)
            self.draw_world()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(self.args.fps)

        pygame.quit()
        print(f"seed {self.seed}  score {self.state['score']:.1f}  "
              f"sim_time {self.sim.env.time:.1f}  births {self.births}  deaths {self.deaths}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--speed", type=int, default=2, help="sim ticks per rendered frame")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-time", type=float, default=3000.0)
    parser.add_argument("--controller", choices=["hive", "burst"], default="hive")
    parser.add_argument("--params", help="JSON file of tunable overrides")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override one tunable, repeatable")
    parser.add_argument("--paused", action="store_true", help="start paused")
    args = parser.parse_args()
    Watcher(args).run()


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    os.chdir(_here)
    main()
