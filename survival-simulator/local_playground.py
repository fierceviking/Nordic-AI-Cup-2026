import math
import pygame
import random
import sys
import time
from src.core import SimulationCore
from src.utils.controllers.expert_agent_policy import Hive


def _attach_diagnostics(env, stats):
    """Wrap kill_agent so we can tell starvation apart from predation."""
    original = env.kill_agent

    def patched(agent):
        if agent in env.agents:
            eaten = any(math.hypot(p.x - agent.x, p.y - agent.y) < p.size + agent.size + 2
                        for p in env.predators)
            stats["eaten" if eaten else "starved"] += 1
            stats["ages"].append(agent.age)
            stats["lost_energy"] += agent.energy if eaten else 0.0
        original(agent)

    env.kill_agent = patched


def local_simulation(verbose=True, seed=None, hive=None, max_time=3000, log_every=1,
                     diagnostics=False, **params):
    if seed is None: # If no seed is provided, generate a random one
        seed = random.randint(0, 2**32 - 1)

    sim = SimulationCore(seed=seed)
    if hive is None:
        hive = Hive(**params)
    hive.reset()

    stats = {"eaten": 0, "starved": 0, "ages": [], "lost_energy": 0.0,
             "prev_true": {}, "prev_est": {}, "dr_err": 0.0, "dr_n": 0, "dr_max": 0.0}
    if diagnostics:
        _attach_diagnostics(sim.env, stats)

    pygame.init()
    screen, clock = None, None

    # Optional render
    if verbose:
        info = pygame.display.Info()
        env_ratio = sim.env_width / sim.env_height
        screen_height = int(info.current_h * 0.9) # 90% of screen height
        screen_width = int(screen_height * env_ratio) # Keep aspect ratio
        screen = pygame.display.set_mode((screen_width, int(screen_height)), pygame.SCALED)
        clock = pygame.time.Clock()

    running = True
    actions = []
    state = {"score": 0.0, "num_agents": 0}

    while running:
        if verbose: 
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Check if user closes window
                    running = False

        state = sim.step(actions)

        if diagnostics:
            # Must be sampled *before* act(), while each mind still holds its
            # estimate of the current position.  Each agent dead-reckons in its
            # own frame, rotated by an unknown constant, so compare
            # displacements after undoing that rotation.
            true_now = {a.agent_id: (a.x, a.y) for a in sim.env.agents}
            for aid, (tx, ty) in true_now.items():
                m = hive.minds.get(aid)
                if m is None or aid not in stats["prev_true"] or aid not in stats["prev_est"]:
                    continue
                ptx, pty = stats["prev_true"][aid]
                pex, pey, rot = stats["prev_est"][aid]
                ex, ey = m.x - pex, m.y - pey
                c, s = math.cos(rot), math.sin(rot)
                err = math.hypot(ex * c - ey * s - (tx - ptx),
                                 ex * s + ey * c - (ty - pty))
                stats["dr_err"] += err
                stats["dr_n"] += 1
                stats["dr_max"] = max(stats["dr_max"], err)
            stats["prev_true"] = true_now
            stats["prev_est"] = {a.agent_id: (hive.minds[a.agent_id].x,
                                              hive.minds[a.agent_id].y,
                                              a.direction - hive.minds[a.agent_id].th)
                                 for a in sim.env.agents if a.agent_id in hive.minds}

        requests = hive.act(state["observations"], state["sim_time"])
        actions = [(r.agent_id, r) for r in requests]

        if verbose:
            sim.env.draw(screen)
            font = pygame.font.SysFont(None, 24)
            img = font.render(f'Score: {state["score"]:.2f}', True, (255,255,255))
            screen.blit(img, (20, 20))
            pygame.display.flip()
            clock.tick(60) # Control max FPS
            print(f'Score: {state["score"]:.2f} | Agents alive: {state["num_agents"]:.0f} | Time: {sim.env.time:.2f}')
        elif log_every and int(round(sim.env.time * 10)) % int(log_every * 10) == 0:
            print(f'Score: {state["score"]:.2f} | Agents: {state["num_agents"]:.0f} | '
                  f'Predators: {len(sim.env.predators)} | Trees: {len(sim.env.trees)} | '
                  f'Fruits: {len(sim.env.fruits)} | Time: {sim.env.time:.1f} | {hive.debug}',
                  flush=True)

        if state["num_agents"] == 0 or sim.env.time > max_time:
            print(f"Game over! Final Score: {state['score']}")
            print(f"Seed: {seed}")
            if diagnostics:
                ages = stats["ages"]
                print(f'deaths: eaten={stats["eaten"]} starved={stats["starved"]} '
                      f'mean_age={sum(ages)/max(1,len(ages)):.1f} '
                      f'score_lost_to_predators={stats["lost_energy"]/100:.1f} '
                      f'target_pop={hive.target_pop:.1f} income_ema={hive.income_ema:.2f}')
                by_mode = {}
                young = 0
                for _, age, energy, mode, aging in hive.deaths:
                    by_mode[mode] = by_mode.get(mode, 0) + 1
                    if age < 25:
                        young += 1
                print(f'death modes={by_mode} died_before_age25={young}/{len(hive.deaths)} '
                      f'aged_out={sum(1 for d in hive.deaths if d[4])}')
                print(f'dead-reckoning mean error per tick: '
                      f'{stats["dr_err"]/max(1,stats["dr_n"]):.3f} units '
                      f'(max {stats["dr_max"]:.1f})')
                b = hive.budget
                at = max(1.0, b["agent_ticks"])
                print('energy per agent-second: ' + ' '.join(
                    f'{k}={b[k]/at*10:.2f}' for k in
                    ("income", "move", "turn", "live", "spawn", "age")))
                tot = max(1, sum(hive.mode_ticks.values()))
                print('time in mode: ' + ' '.join(
                    f'{k}={v/tot:.0%}' for k, v in sorted(hive.mode_ticks.items(),
                                                          key=lambda i: -i[1])))
            running = False

    pygame.quit()
    return {"score": state["score"], "time": sim.env.time,
            "agents": state["num_agents"], "seed": seed, "stats": stats}


def benchmark(seeds, max_time=3000, diagnostics=True, **params):
    """Headless multi-seed evaluation."""
    results = []
    for seed in seeds:
        t0 = time.time()
        res = local_simulation(verbose=False, seed=seed, max_time=max_time, log_every=0,
                               diagnostics=diagnostics, **params)
        res["wall"] = time.time() - t0
        results.append(res)
        print(f'seed={seed} score={res["score"]:.1f} time={res["time"]:.1f} '
              f'agents={res["agents"]} wall={res["wall"]:.0f}s', flush=True)
    scores = [r["score"] for r in results]
    print(f'--- {params} mean score {sum(scores)/len(scores):.1f} over {len(scores)} seeds '
          f'(min {min(scores):.1f}, max {max(scores):.1f})', flush=True)
    return results


def sweep(seeds, configs, max_time=3000):
    """Run the same seeds under several tunable settings."""
    summary = []
    for cfg in configs:
        res = benchmark(seeds, max_time=max_time, diagnostics=False, **cfg)
        mean = sum(r["score"] for r in res) / len(res)
        summary.append((mean, cfg))
    summary.sort(reverse=True, key=lambda s: s[0])
    print("=== sweep results ===", flush=True)
    for mean, cfg in summary:
        print(f'{mean:8.1f}  {cfg}', flush=True)
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        seed_list = [int(s) for s in sys.argv[2:]] or [1, 2, 3, 4, 5]
        benchmark(seed_list)
    else:
        local_simulation(verbose=True)
