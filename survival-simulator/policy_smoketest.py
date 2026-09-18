import unittest

from src.utils.controllers.expert_agent_policy import Hive, Mind, World


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.hive = Hive()
        self.state = {
            "energy": 200.0,
            "max_energy": 375.0,
            "speed": 10.0,
            "sprint_speed": 20.0,
            "biome": "grassland",
            "hearing_radius": 50.0,
            "vision_range": 200.0,
            "vision_angle": 1.0,
        }

    def test_duplicate_sightings_preserve_predator_velocity(self):
        world = World()
        world.add_predator(200.0, 0.0, 0.0)
        world.add_predator(185.0, 0.0, 0.1)
        for _ in range(4):
            world.add_predator(185.0, 0.0, 0.1)
        self.assertEqual(len(world.predators), 1)
        self.assertEqual(world.predators[0][3:6], [-15.0, 0.0, 2.0])

    def test_duplicate_sightings_do_not_inflate_confidence(self):
        world = World()
        for _ in range(5):
            world.add_predator(100.0, 0.0, 1.0)
        self.assertEqual(world.predators[0][5], 1.0)

    def test_noisy_duplicate_does_not_shift_next_velocity_sample(self):
        world = World()
        world.add_predator(200.0, 0.0, 0.0)
        world.add_predator(185.0, 0.0, 0.1)
        world.add_predator(190.0, 2.0, 0.1)
        world.add_predator(170.0, 0.0, 0.2)
        self.assertEqual(world.predators[0][3:6], [-15.0, 0.0, 3.0])

    def test_predator_velocity_accounts_for_elapsed_ticks(self):
        world = World()
        world.add_predator(200.0, 0.0, 0.0)
        world.add_predator(170.0, 0.0, 0.2)
        self.assertEqual(world.predators[0][3], -15.0)

    def test_stationary_predator_still_accumulates_confidence(self):
        world = World()
        for tick in range(5):
            world.add_predator(100.0, 0.0, tick * 0.1)
        self.assertEqual(world.predators[0][3:6], [0.0, 0.0, 5.0])

    def test_shared_sighting_of_approaching_predator_triggers_sprint(self):
        world = World()
        world.add_predator(200.0, 0.0, 0.0)
        world.add_predator(185.0, 0.0, 0.1)
        for _ in range(3):
            world.add_predator(185.0, 0.0, 0.1)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        plan = self.hive._plan(mind, self.state, 0.1)
        self.assertEqual(mind.mode, "flee")
        self.assertGreater(plan["cmd"], self.state["speed"])

    def test_independent_maps_do_not_share_fruit_claims(self):
        for agent_id in (1, 2):
            world = World()
            world.add_point(world.fruits, 20.0, 0.0, 1.0, 6.0)
            mind = Mind(agent_id, world, 0.0, 0.0, 0.0)
            self.hive._plan(mind, self.state, 1.0)
            self.assertEqual(mind.mode, "fruit")

    def test_shared_map_does_not_double_claim_fruit(self):
        world = World()
        world.add_point(world.fruits, 20.0, 0.0, 1.0, 6.0)
        first = Mind(1, world, 0.0, 0.0, 0.0)
        second = Mind(2, world, 0.0, 0.0, 0.0)
        self.hive._plan(first, self.state, 1.0)
        self.hive._plan(second, self.state, 1.0)
        self.assertEqual(first.mode, "fruit")
        self.assertNotEqual(second.mode, "fruit")

    def test_independent_maps_do_not_share_tree_claims(self):
        for agent_id in (1, 2):
            world = World()
            nearest = world.add_point(world.trees, 100.0, 0.0, 1.0, 30.0)
            world.add_point(world.trees, 160.0, 0.0, 1.0, 30.0)
            mind = Mind(agent_id, world, 0.0, 0.0, 0.0)
            self.hive._plan(mind, self.state, 1.0)
            self.assertEqual(mind.target_tree, nearest)

    def test_shared_map_spreads_agents_across_trees(self):
        world = World()
        world.add_point(world.trees, 100.0, 0.0, 1.0, 30.0)
        world.add_point(world.trees, 160.0, 0.0, 1.0, 30.0)
        first = Mind(1, world, 0.0, 0.0, 0.0)
        second = Mind(2, world, 0.0, 0.0, 0.0)
        self.hive._plan(first, self.state, 1.0)
        self.hive._plan(second, self.state, 1.0)
        self.assertNotEqual(first.target_tree, second.target_tree)

    def test_hungry_agent_gets_food_regardless_of_request_order(self):
        for order in ((1, 2), (2, 1)):
            with self.subTest(order=order):
                hive = Hive()
                world = World()
                fruit = world.add_point(world.fruits, 20.0, 0.0, 1.0, 6.0)
                hive.minds = {agent_id: Mind(agent_id, world, 0.0, 0.0, 0.0)
                              for agent_id in order}
                states = [dict(self.state, agent_id=agent_id, age=5.0,
                               energy=250.0 if agent_id == 1 else 30.0,
                               hearing_radius=50.0, observations=[])
                          for agent_id in order]
                actions = hive.act_dicts(states, 1.0)
                self.assertEqual(hive.minds[2].target_fruit, fruit)
                self.assertIsNone(hive.minds[1].target_fruit)
                self.assertEqual([action["agent_id"] for action in actions], list(order))

    def test_planning_priority_accounts_for_walking_drain(self):
        slow = Mind(1, World(), 0.0, 0.0, 0.0)
        fast = Mind(2, World(), 0.0, 0.0, 0.0)
        context = [(slow, dict(self.state, speed=5.0, age=5.0)),
                   (fast, dict(self.state, speed=15.0, age=5.0))]
        self.assertIs(self.hive._planning_order(context)[0][0], fast)

    def test_planning_priority_accounts_for_aging_drain(self):
        young = Mind(1, World(), 0.0, 0.0, 0.0)
        old = Mind(2, World(), 0.0, 0.0, 0.0)
        old.aging = True
        context = [(young, dict(self.state, age=5.0)),
                   (old, dict(self.state, age=100.0))]
        self.assertIs(self.hive._planning_order(context)[0][0], old)

    def birth_context(self, count=3, age=70.0, productive=True):
        context, plans = [], {}
        for agent_id in range(1, count + 1):
            world = World()
            mind = Mind(agent_id, world, 0.0, 0.0, 0.0)
            mind.mode = "hold"
            world.add_point(world.trees, 0.0, 0.0, 100.0, 30.0)
            if productive:
                mind.income = 30.0
            context.append((mind, dict(self.state, age=age, energy=350.0)))
            plans[agent_id] = {"after": 350.0, "mdir": 0.0, "d_eff": 0.0, "pen": 1.0}
        return context, plans

    def test_birth_requires_food_not_just_a_tree(self):
        context, plans = self.birth_context(productive=False)
        result = self.hive._guard_births(context, plans, 100.0, {1}, 8)
        self.assertEqual(result, set())

    def test_productive_patch_allows_pre_aging_replacement(self):
        context, plans = self.birth_context()
        result = self.hive._guard_births(context, plans, 100.0, set(), 8)
        self.assertEqual(result, {1, 2, 3})

    def test_young_agents_do_not_trigger_unsolicited_replacement(self):
        context, plans = self.birth_context(age=30.0)
        result = self.hive._guard_births(context, plans, 100.0, set(), 8)
        self.assertEqual(result, set())

    def test_birth_avoids_predator_near_post_move_position(self):
        context, plans = self.birth_context(age=30.0)
        context[0][0].world.add_predator(325.0, 0.0, 100.0)
        plans[1]["d_eff"] = 10.0
        result = self.hive._guard_births(context, plans, 100.0, {1}, 8)
        self.assertEqual(result, set())

    def test_newborn_blocks_another_brood_at_same_patch(self):
        context, plans = self.birth_context(age=30.0)
        parent = context[0][0]
        child = context[1][0]
        child.world = parent.world
        context[1][1]["age"] = 5.0
        result = self.hive._guard_births(context, plans, 100.0, {1}, 8)
        self.assertEqual(result, set())

    def test_only_one_birth_per_patch_per_tick(self):
        context, plans = self.birth_context(age=30.0)
        context[1][0].world = context[0][0].world
        result = self.hive._guard_births(context, plans, 100.0, {1, 2}, 8)
        self.assertEqual(result, {1})

    def test_birth_respects_cooldown_and_population_cap(self):
        context, plans = self.birth_context()
        context[0][0].last_spawn = 99.0
        result = self.hive._guard_births(context, plans, 100.0, {1, 2, 3}, 2)
        self.assertEqual(result, {2})

    def test_dying_last_lineage_can_spawn_without_food(self):
        context, plans = self.birth_context(count=1, age=100.0, productive=False)
        context[0][0].aging = True
        plans[1]["after"] = 130.0
        result = self.hive._guard_births(context, plans, 100.0, {1}, 8)
        self.assertEqual(result, {1})

    def test_fresh_fruit_can_establish_birth_site_productivity(self):
        context, plans = self.birth_context(age=30.0, productive=False)
        world = context[0][0].world
        world.add_point(world.fruits, 30.0, 0.0, 100.0, 6.0)
        world.add_point(world.fruits, 50.0, 0.0, 100.0, 6.0)
        result = self.hive._guard_births(context, plans, 100.0, {1}, 8)
        self.assertEqual(result, {1})

    def test_unaffordable_fruit_remains_available_to_another_agent(self):
        world = World()
        fruit = world.add_point(world.fruits, 150.0, 0.0, 1.0, 6.0)
        hungry = Mind(1, world, 0.0, 0.0, 0.0)
        healthy = Mind(2, world, 0.0, 0.0, 0.0)
        self.hive._plan(hungry, dict(self.state, energy=3.0), 1.0)
        self.hive._plan(healthy, self.state, 1.0)
        self.assertIsNone(hungry.target_fruit)
        self.assertEqual(healthy.target_fruit, fruit)

    def test_fruit_affordability_includes_biome_and_aging(self):
        for biome, aging, expected in (("grassland", False, "fruit"),
                                       ("river", False, "explore"),
                                       ("grassland", True, "explore")):
            with self.subTest(biome=biome, aging=aging):
                hive = Hive()
                world = World()
                world.add_point(world.fruits, 90.0, 0.0, 1.0, 6.0)
                mind = Mind(1, world, 0.0, 0.0, 0.0)
                mind.aging = aging
                hive._plan(mind, dict(self.state, energy=10.0, biome=biome, age=100.0), 1.0)
                self.assertEqual(mind.mode, expected)

    def test_planning_reorder_preserves_spawn_parent_order(self):
        hive = Hive(FOOD_AWARE_BREEDING=True)
        states = []
        for agent_id, energy in ((2, 350.0), (1, 220.0)):
            world = World()
            world.add_point(world.trees, 0.0, 0.0, 100.0, 30.0)
            mind = Mind(agent_id, world, 0.0, 0.0, 0.0)
            mind.income = 30.0
            hive.minds[agent_id] = mind
            states.append(dict(self.state, agent_id=agent_id, energy=energy,
                               age=70.0, observations=[]))
        actions = hive.act_dicts(states, 100.0)
        self.assertEqual([action["agent_id"] for action in actions if action["spawn_agent"]], [2, 1])
        self.assertEqual(hive.pending_spawns, [2, 1])

    def test_birth_requires_fresh_patch_evidence(self):
        context, plans = self.birth_context(age=30.0)
        result = self.hive._guard_births(context, plans, 102.0, {1}, 8)
        self.assertEqual(result, set())

    def test_default_enables_food_priority_without_experimental_breeding(self):
        self.assertTrue(self.hive.PRIORITIZE_FOOD)
        self.assertFalse(self.hive.FOOD_AWARE_BREEDING)
        self.assertFalse(self.hive.TERRITORY)

    def test_territory_gives_each_agent_a_distinct_home(self):
        hive = Hive(TERRITORY=True)
        world = World()
        east = world.add_point(world.trees, 100.0, 0.0, 1.0, 30.0)
        west = world.add_point(world.trees, -100.0, 0.0, 1.0, 30.0)
        first = Mind(1, world, 0.0, 0.0, 0.0)
        second = Mind(2, world, 0.0, 0.0, 0.0)
        hive._assign_homes([(first, self.state), (second, self.state)], 1.0)
        self.assertEqual({first.home, second.home}, {east, west})

    def test_territory_prefers_recently_discovered_tree(self):
        hive = Hive(TERRITORY=True)
        world = World()
        world.add_point(world.trees, 100.0, 0.0, 1.0, 30.0)
        fresh = world.add_point(world.trees, -100.0, 0.0, 50.0, 30.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        hive._assign_homes([(mind, self.state)], 55.0)
        self.assertEqual(mind.home, fresh)

    def test_territory_keeps_home_without_decisive_gain(self):
        hive = Hive(TERRITORY=True)
        world = World()
        home = world.add_point(world.trees, 0.0, 0.0, 1.0, 30.0)
        world.add_point(world.trees, 60.0, 0.0, 5.0, 30.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        mind.home = home
        hive._assign_homes([(mind, self.state)], 10.0)
        self.assertEqual(mind.home, home)

    def test_territory_hands_over_before_an_old_tree_dies(self):
        hive = Hive(TERRITORY=True)
        world = World()
        world.add_point(world.trees, 0.0, 0.0, 1.0, 30.0)
        fresh = world.add_point(world.trees, 40.0, 0.0, 120.0, 30.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        mind.home = 1
        hive._assign_homes([(mind, self.state)], 125.0)
        self.assertEqual(mind.home, fresh)

    def test_territory_releases_home_when_tree_is_gone(self):
        hive = Hive(TERRITORY=True)
        world = World()
        tree = world.add_point(world.trees, 0.0, 0.0, 1.0, 30.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        mind.home = tree
        del world.trees[tree]
        hive._assign_homes([(mind, self.state)], 5.0)
        self.assertIsNone(mind.home)

    def test_territory_ignores_fruit_outside_home_radius(self):
        hive = Hive(TERRITORY=True)
        world = World()
        home = world.add_point(world.trees, 0.0, 0.0, 1.0, 30.0)
        world.add_point(world.fruits, 200.0, 0.0, 1.0, 6.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        mind.home = home
        hive._plan(mind, self.state, 1.0)
        self.assertIsNone(mind.target_fruit)
        self.assertEqual(mind.mode, "hold")

    def test_territory_still_takes_fruit_inside_home_radius(self):
        hive = Hive(TERRITORY=True)
        world = World()
        home = world.add_point(world.trees, 0.0, 0.0, 1.0, 30.0)
        fruit = world.add_point(world.fruits, 60.0, 0.0, 1.0, 6.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        mind.home = home
        hive._plan(mind, self.state, 1.0)
        self.assertEqual(mind.target_fruit, fruit)

    def test_territory_walks_back_to_a_distant_home(self):
        hive = Hive(TERRITORY=True)
        world = World()
        home = world.add_point(world.trees, 300.0, 0.0, 1.0, 30.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        mind.home = home
        plan = hive._plan(mind, self.state, 1.0)
        self.assertEqual(mind.mode, "patrol")
        self.assertGreater(plan["cmd"], 0.0)
        self.assertAlmostEqual(plan["mdir"], 0.0, places=6)

    def test_aging_dump_defaults_match_previous_behaviour(self):
        self.assertEqual(self.hive.AGING_COOLDOWN, self.hive.SPAWN_COOLDOWN)
        self.assertFalse(self.hive.AGING_DUMP)

    def test_aging_agent_dumps_energy_on_its_own_cooldown(self):
        for cooldown, expected in ((4.0, False), (0.1, True)):
            with self.subTest(cooldown=cooldown):
                hive = Hive(AGING_COOLDOWN=cooldown, AGING_DUMP=True)
                context, plans = self.birth_context(count=4, age=100.0)
                aging = context[0][0]
                aging.aging = True
                for mind, _ in context:
                    mind.last_spawn = 99.0
                result = hive._spawn_plan(context, plans, 100.0, len(context))
                self.assertEqual(aging.aid in result, expected)

    def test_healthy_agent_ignores_the_aging_cooldown(self):
        hive = Hive(AGING_COOLDOWN=0.1, AGING_DUMP=True)
        context, plans = self.birth_context(count=4, age=30.0)
        for mind, _ in context:
            mind.last_spawn = 99.0
        result = hive._spawn_plan(context, plans, 100.0, len(context))
        self.assertEqual(result, set())

    def test_aging_dump_lifts_the_population_cap(self):
        for dump, expected in ((False, False), (True, True)):
            with self.subTest(dump=dump):
                hive = Hive(AGING_DUMP=dump, AGING_COOLDOWN=0.1)
                hive.target_pop = 3.0
                context, plans = self.birth_context(count=6, age=100.0)
                aging = context[0][0]
                aging.aging = True
                for mind, _ in context:
                    mind.last_spawn = 99.0
                result = hive._spawn_plan(context, plans, 100.0, len(context))
                self.assertEqual(aging.aid in result, expected)

    def test_nosprint_margin_defaults_to_previous_behaviour(self):
        self.assertEqual(self.hive.NOSPRINT_MARGIN, 1.0)

    def test_agent_that_cannot_sprint_retreats_from_further_out(self):
        # Sits outside BACKOFF_DIST (260) but inside twice it, so only a
        # widened margin should trigger the retreat.
        for margin, energy, expected in ((1.0, 40.0, False), (2.0, 40.0, True),
                                         (2.0, 300.0, False)):
            with self.subTest(margin=margin, energy=energy):
                hive = Hive(NOSPRINT_MARGIN=margin)
                world = World()
                world.add_predator(400.0, 0.0, 0.0)
                world.add_predator(395.0, 0.0, 0.1)
                mind = Mind(1, world, 0.0, 0.0, 0.0)
                hive._plan(mind, dict(self.state, energy=energy), 0.1)
                self.assertEqual(mind.mode == "flee", expected)

    def test_new_rules_default_to_off(self):
        self.assertFalse(self.hive.EARLY_BOOST)
        self.assertFalse(self.hive.UNSTICK)

    def test_early_boost_raises_the_target_only_while_predators_are_absent(self):
        for enabled, now, expect_many in ((False, 10.0, False), (True, 10.0, True),
                                          (True, 500.0, False)):
            with self.subTest(enabled=enabled, now=now):
                hive = Hive(EARLY_BOOST=enabled, EARLY_POP=14, EARLY_COOLDOWN=0.1)
                hive.target_pop = 3.0
                context, plans = self.birth_context(count=3, age=30.0)
                result = hive._spawn_plan(context, plans, now, len(context))
                self.assertEqual(len(result) > 0, expect_many)

    def test_unstick_abandons_a_target_it_keeps_getting_deflected_from(self):
        hive = Hive(UNSTICK=True, STALL_TICKS=3)
        world = World()
        fruit = world.add_point(world.fruits, 60.0, 0.0, 1.0, 6.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        # wall must sit inside the first step (speed 10) for _steer to deflect
        mind.edges[(12, -60, 12, 60)] = 1.0
        for _ in range(3):
            hive._claimed_fruits = set()
            hive._plan(mind, self.state, 1.0)
        self.assertGreater(mind.avoid.get(fruit, 0.0), 1.0)
        self.assertIsNone(mind.target_fruit)

    def test_unstick_leaves_reachable_targets_alone(self):
        hive = Hive(UNSTICK=True, STALL_TICKS=3)
        world = World()
        fruit = world.add_point(world.fruits, 60.0, 0.0, 1.0, 6.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        for _ in range(5):
            hive._claimed_fruits = set()
            hive._plan(mind, self.state, 1.0)
        self.assertEqual(mind.target_fruit, fruit)
        self.assertEqual(mind.avoid, {})

    def test_barren_patience_defaults_to_off(self):
        self.assertEqual(self.hive.BARREN_PATIENCE, 0.0)

    def test_agent_abandons_a_patch_that_never_feeds_it(self):
        hive = Hive(BARREN_PATIENCE=20.0)
        world = World()
        tree = world.add_point(world.trees, 0.0, 0.0, 1.0, 30.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        mind.last_meal = 1.0
        hive._plan(mind, self.state, 10.0)          # 9 s hungry: still patient
        self.assertEqual(mind.mode, "hold")
        self.assertEqual(mind.avoid, {})
        hive._plan(mind, self.state, 40.0)          # 39 s hungry: give up
        self.assertGreater(mind.avoid.get(tree, 0.0), 40.0)
        self.assertIsNone(mind.target_tree)

    def test_a_patch_that_feeds_the_agent_is_kept(self):
        hive = Hive(BARREN_PATIENCE=20.0)
        world = World()
        tree = world.add_point(world.trees, 0.0, 0.0, 1.0, 30.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        for now in (10.0, 40.0, 70.0):
            mind.last_meal = now - 1.0              # eating regularly
            hive._plan(mind, self.state, now)
        self.assertEqual(mind.target_tree, tree)
        self.assertEqual(mind.avoid, {})

    def test_overflow_spawn_defaults_to_off(self):
        self.assertFalse(self.hive.OVERFLOW_SPAWN)

    def test_nearly_full_agent_spawns_even_at_population_target(self):
        for enabled, expected in ((False, False), (True, True)):
            with self.subTest(enabled=enabled):
                hive = Hive(OVERFLOW_SPAWN=enabled)
                hive.target_pop = 3.0                     # already at target
                context, plans = self.birth_context(count=5, age=30.0)
                for (mind, state), _ in zip(context, range(5)):
                    mind.last_spawn = -1e9
                    state["max_energy"] = 500.0
                for aid in plans:                          # headroom 40 < 60
                    plans[aid]["after"] = 460.0
                result = hive._spawn_plan(context, plans, 100.0, len(context))
                self.assertEqual(len(result) > 0, expected)

    def test_overflow_spawn_ignores_agents_with_headroom(self):
        hive = Hive(OVERFLOW_SPAWN=True)
        hive.target_pop = 3.0
        context, plans = self.birth_context(count=5, age=30.0)
        for mind, state in context:
            mind.last_spawn = -1e9
            state["max_energy"] = 500.0
        for aid in plans:                                  # headroom 200 >= 60
            plans[aid]["after"] = 300.0
        self.assertEqual(hive._spawn_plan(context, plans, 100.0, len(context)), set())

    def test_overflow_spawn_respects_the_hard_cap(self):
        hive = Hive(OVERFLOW_SPAWN=True, POP_HARD=4)
        hive.target_pop = 3.0                              # keep routine growth out
        context, plans = self.birth_context(count=5, age=30.0)
        for mind, state in context:
            mind.last_spawn = -1e9
            state["max_energy"] = 500.0
        for aid in plans:
            plans[aid]["after"] = 460.0
        self.assertEqual(hive._spawn_plan(context, plans, 100.0, len(context)), set())

    def test_headroom_pricing_defaults_to_off(self):
        self.assertFalse(self.hive.HEADROOM_PRICING)

    def test_headroom_pricing_declines_fruit_it_cannot_absorb(self):
        # 150 units away: worth the walk at full value, not at 10 of headroom.
        for pricing, energy, should_chase in ((False, 365.0, True),
                                              (True, 365.0, False),
                                              (True, 100.0, True)):
            with self.subTest(pricing=pricing, energy=energy):
                hive = Hive(HEADROOM_PRICING=pricing)
                world = World()
                fruit = world.add_point(world.fruits, 150.0, 0.0, 1.0, 6.0)
                world.add_point(world.trees, 0.0, 0.0, 1.0, 30.0)
                mind = Mind(1, world, 0.0, 0.0, 0.0)
                hive._plan(mind, dict(self.state, energy=energy, max_energy=375.0), 1.0)
                self.assertEqual(mind.target_fruit == fruit, should_chase)

    def test_spread_defaults_to_off(self):
        self.assertEqual(self.hive.SPREAD, 0.0)

    def test_spread_pushes_an_agent_to_the_far_patch(self):
        # two equal patches; a hive-mate is sitting on the near one
        for spread, expected_far in ((0.0, False), (60.0, True)):
            with self.subTest(spread=spread):
                hive = Hive(SPREAD=spread)
                world = World()
                near = world.add_point(world.trees, 120.0, 0.0, 1.0, 30.0)
                far = world.add_point(world.trees, -150.0, 0.0, 1.0, 30.0)
                mind = Mind(1, world, 0.0, 0.0, 0.0)
                mate = Mind(2, world, 120.0, 0.0, 0.0)
                hive.minds = {1: mind, 2: mate}
                hive._plan(mind, self.state, 1.0)
                self.assertEqual(mind.target_tree == far, expected_far)
                if not expected_far:
                    self.assertEqual(mind.target_tree, near)

    def test_spread_ignores_other_lineages(self):
        hive = Hive(SPREAD=60.0)
        world = World()
        near = world.add_point(world.trees, 120.0, 0.0, 1.0, 30.0)
        world.add_point(world.trees, -150.0, 0.0, 1.0, 30.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        stranger = Mind(2, World(), 120.0, 0.0, 0.0)      # different map
        hive.minds = {1: mind, 2: stranger}
        hive._plan(mind, self.state, 1.0)
        self.assertEqual(mind.target_tree, near)

    def test_breeding_guard_is_opt_in(self):
        for enabled, expected in ((False, {1}), (True, set())):
            with self.subTest(enabled=enabled):
                hive = Hive(FOOD_AWARE_BREEDING=enabled)
                context, plans = self.birth_context(age=30.0, productive=False)
                plans[2]["after"] = plans[3]["after"] = 110.0
                result = hive._spawn_plan(context, plans, 100.0, len(context))
                self.assertEqual(result, expected)

    def test_fruit_prior_defaults_to_off(self):
        self.assertEqual(self.hive.FRUIT_PRIOR, 0.0)

    def test_river_trees_are_worthless(self):
        hive = Hive(FRUIT_PRIOR=1.0)
        world = World()
        key = world.add_point(world.trees, 100.0, 0.0, 1.0, 30.0)
        world.biomes[key] = "river"
        self.assertEqual(hive._expected_fruit(world, key, 40.0, 0), 0.0)
        world.biomes[key] = "forest"
        self.assertGreater(hive._expected_fruit(world, key, 40.0, 0), 0.0)

    def test_expected_fruit_grows_with_time_then_saturates(self):
        hive = Hive(FRUIT_PRIOR=1.0)
        world = World()
        key = world.add_point(world.trees, 100.0, 0.0, 1.0, 30.0)
        world.biomes[key] = "forest"
        young = hive._expected_fruit(world, key, 5.0, 0)
        older = hive._expected_fruit(world, key, 25.0, 0)
        ancient = hive._expected_fruit(world, key, 900.0, 0)
        self.assertGreater(older, young)
        # nothing outlives PRIOR_LIFE, and belief is capped
        self.assertEqual(ancient, hive._expected_fruit(world, key, hive.PRIOR_LIFE, 0))
        self.assertLessEqual(ancient, hive.PRIOR_CAP * hive.PRIOR_ENERGY)

    def test_expected_fruit_does_not_double_count_remembered_fruit(self):
        hive = Hive(FRUIT_PRIOR=1.0)
        world = World()
        key = world.add_point(world.trees, 100.0, 0.0, 1.0, 30.0)
        world.biomes[key] = "forest"
        blind = hive._expected_fruit(world, key, 50.0, 0)
        seen = hive._expected_fruit(world, key, 50.0, 3)
        self.assertLess(seen, blind)

    def test_fruitless_dwell_writes_a_patch_off(self):
        hive = Hive(FRUIT_PRIOR=1.0, PRIOR_EVIDENCE=12.0)
        world = World()
        key = world.add_point(world.trees, 100.0, 0.0, 1.0, 30.0)
        world.biomes[key] = "forest"
        full = hive._expected_fruit(world, key, 50.0, 0)
        world.dry[key] = 6.0
        self.assertAlmostEqual(hive._expected_fruit(world, key, 50.0, 0), full * 0.5)
        world.dry[key] = 12.0
        self.assertEqual(hive._expected_fruit(world, key, 50.0, 0), 0.0)

    def test_prior_prefers_the_productive_biome(self):
        # equal distance, equal neglect; only the tagged biome differs
        hive = Hive(FRUIT_PRIOR=1.0)
        world = World()
        desert = world.add_point(world.trees, 140.0, 0.0, 1.0, 30.0)
        forest = world.add_point(world.trees, -140.0, 0.0, 1.0, 30.0)
        world.biomes[desert] = "desert"
        world.biomes[forest] = "forest"
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        hive.minds = {1: mind}
        hive._plan(mind, self.state, 30.0)
        self.assertEqual(mind.target_tree, forest)

    def test_fruit_near_trees_attributes_by_annulus(self):
        world = World()
        key = world.add_point(world.trees, 0.0, 0.0, 1.0, 30.0)
        world.add_point(world.fruits, 40.0, 0.0, 1.0, 6.0)     # inside the annulus
        world.add_point(world.fruits, 400.0, 0.0, 1.0, 6.0)    # someone else's patch
        self.assertEqual(world.fruit_near_trees(1.0, 62.0).get(key), 1)

    def test_forgetting_a_tree_clears_its_belief(self):
        world = World()
        key = world.add_point(world.trees, 0.0, 0.0, 1.0, 30.0)
        world.biomes[key] = "forest"
        world.dry[key] = 5.0
        world.visits[key] = 1.0
        world.forget_tree(key)
        for table in (world.trees, world.biomes, world.dry, world.visits):
            self.assertNotIn(key, table)

    def test_scouting_defaults_to_off(self):
        self.assertEqual(self.hive.SCOUT, 0)

    def scout_crew(self, hive, n, energy=400.0):
        world = World()
        ctx = []
        for i in range(1, n + 1):
            mind = Mind(i, world, i * 30.0, 0.0, 0.0)
            mind.last_state = (1.0, 10.0, energy + i)
            hive.minds[i] = mind
            ctx.append((mind, dict(self.state, agent_id=i, energy=energy + i)))
        return world, ctx

    def test_scout_assignment_respects_minimum_population(self):
        for n, expected in ((4, 0), (6, 1)):
            with self.subTest(population=n):
                hive = Hive(SCOUT=1, SCOUT_MIN_POP=5)
                _, ctx = self.scout_crew(hive, n)
                hive._assign_scouts(ctx, 10.0)
                serving = sum(1 for m, _ in ctx if 10.0 < m.scout_until)
                self.assertEqual(serving, expected)

    def test_scout_job_is_held_then_rotated(self):
        hive = Hive(SCOUT=1, SCOUT_MIN_POP=2, SCOUT_COMMIT=25.0)
        _, ctx = self.scout_crew(hive, 5)
        hive._assign_scouts(ctx, 10.0)
        first = [m for m, _ in ctx if 10.0 < m.scout_until]
        self.assertEqual(len(first), 1)
        # still inside the commit window: nobody else is drafted
        hive._assign_scouts(ctx, 20.0)
        self.assertEqual([m for m, _ in ctx if 20.0 < m.scout_until], first)
        # past it: the post is filled again
        hive._assign_scouts(ctx, 40.0)
        self.assertEqual(len([m for m, _ in ctx if 40.0 < m.scout_until]), 1)

    def test_scout_explores_instead_of_working_a_patch(self):
        for scout, expected in ((0, "patrol"), (1, "scout")):
            with self.subTest(scout=scout):
                hive = Hive(SCOUT=scout, SCOUT_MIN_POP=1)
                world = World()
                world.add_point(world.trees, 120.0, 0.0, 1.0, 30.0)
                mind = Mind(1, world, 0.0, 0.0, 0.0)
                hive.minds = {1: mind}
                mind.scout_until = 99.0
                hive._plan(mind, self.state, 1.0)
                self.assertEqual(mind.mode, expected)

    def test_a_hungry_scout_returns_to_the_economy(self):
        hive = Hive(SCOUT=1, SCOUT_MIN_POP=1, SCOUT_FEED=140.0)
        world = World()
        world.add_point(world.trees, 120.0, 0.0, 1.0, 30.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        hive.minds = {1: mind}
        mind.scout_until = 99.0
        hive._plan(mind, dict(self.state, energy=100.0), 1.0)
        self.assertEqual(mind.mode, "patrol")

    def test_scout_still_takes_fruit_on_its_path(self):
        hive = Hive(SCOUT=1, SCOUT_MIN_POP=1, SCOUT_GRAB=90.0)
        world = World()
        close = world.add_point(world.fruits, 60.0, 0.0, 1.0, 6.0)
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        hive.minds = {1: mind}
        mind.scout_until = 99.0
        hive._plan(mind, self.state, 1.0)
        self.assertEqual(mind.target_fruit, close)

    def test_scout_ignores_fruit_beyond_its_grab_radius(self):
        hive = Hive(SCOUT=1, SCOUT_MIN_POP=1, SCOUT_GRAB=90.0)
        world = World()
        world.add_point(world.fruits, 180.0, 0.0, 1.0, 6.0)   # inside FRUIT_RANGE
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        hive.minds = {1: mind}
        mind.scout_until = 99.0
        hive._plan(mind, self.state, 1.0)
        self.assertIsNone(mind.target_fruit)
        self.assertEqual(mind.mode, "scout")


    def test_chase_budget_defaults_to_off(self):
        self.assertEqual(self.hive.CHASE_BUDGET, 0.0)

    def test_sprinting_predator_starts_and_stops_the_chase_clock(self):
        world = World()
        world.add_predator(500.0, 0.0, 1.0)
        self.assertLess(world.predators[0][6], 0.0)        # standing still
        world.add_predator(486.0, 0.0, 1.1)                # 14 units in one tick
        self.assertAlmostEqual(world.predators[0][6], 1.1)
        started = world.predators[0][6]
        world.add_predator(485.0, 0.0, 1.2)                # slowed to a pivot
        self.assertLess(world.predators[0][6], 0.0)
        self.assertGreater(started, 0.0)

    def test_chase_clock_is_not_restarted_while_still_sprinting(self):
        world = World()
        world.add_predator(500.0, 0.0, 1.0)
        world.add_predator(486.0, 0.0, 1.1)
        first = world.predators[0][6]
        for i, x in enumerate((472.0, 458.0, 444.0), start=2):
            world.add_predator(x, 0.0, 1.0 + i * 0.1)
        self.assertAlmostEqual(world.predators[0][6], first)

    def test_spent_chaser_is_discounted_so_the_agent_stops_sprinting(self):
        # Same geometry, same predator; only the elapsed sprint time differs.
        def run(budget, elapsed):
            hive = Hive(CHASE_BUDGET=budget)
            world = World()
            mind = Mind(1, world, 0.0, 0.0, 0.0)
            hive.minds = {1: mind}
            t = 10.0
            world.predators.append([60.0, 0.0, t, -14.0, 0.0, 5.0, t - elapsed])
            return hive._plan(mind, self.state, t)

        fresh = run(3.5, 0.5)
        spent = run(3.5, 5.0)
        # Flee matches the chaser (near_v * 1.12) rather than maxing sprint,
        # so compare the two cases rather than an absolute speed.
        self.assertEqual(fresh["mode"] if "mode" in fresh else "flee", "flee")
        self.assertGreater(fresh["cmd"], self.state["speed"])
        self.assertLess(spent["cmd"], fresh["cmd"])
        self.assertLess(spent["cost"], fresh["cost"])

    def test_chase_budget_off_ignores_the_clock(self):
        # With the budget disabled, a long-running chaser is treated exactly
        # like a fresh one.
        def run(budget):
            hive = Hive(CHASE_BUDGET=budget)
            world = World()
            mind = Mind(1, world, 0.0, 0.0, 0.0)
            hive.minds = {1: mind}
            t = 10.0
            world.predators.append([60.0, 0.0, t, -14.0, 0.0, 5.0, t - 5.0])
            return hive._plan(mind, self.state, t)["cmd"]

        self.assertGreater(run(0.0), run(3.5))


    def test_breeding_still_requires_holding_by_default(self):
        self.assertTrue(self.hive.BREED_HOLD_ONLY)

    def test_birth_site_accepts_a_holding_agent_either_way(self):
        for hold_only in (True, False):
            with self.subTest(hold_only=hold_only):
                hive = Hive(BREED_HOLD_ONLY=hold_only)
                mind = Mind(1, World(), 0.0, 0.0, 0.0)
                mind.mode = "hold"
                self.assertTrue(hive._good_birth_site(mind))

    def test_birth_site_decouples_eligibility_from_mode(self):
        # Same agent, same spot beside a known tree, but foraging not holding.
        for hold_only, expected in ((True, False), (False, True)):
            with self.subTest(hold_only=hold_only):
                hive = Hive(BREED_HOLD_ONLY=hold_only, BREED_SITE_RADIUS=60.0)
                world = World()
                world.add_point(world.trees, 30.0, 0.0, 1.0, 30.0)
                mind = Mind(1, world, 0.0, 0.0, 0.0)
                mind.mode = "fruit"
                self.assertEqual(hive._good_birth_site(mind), expected)

    def test_birth_site_still_rejects_open_ground(self):
        hive = Hive(BREED_HOLD_ONLY=False, BREED_SITE_RADIUS=60.0)
        world = World()
        world.add_point(world.trees, 400.0, 0.0, 1.0, 30.0)   # far from the agent
        mind = Mind(1, world, 0.0, 0.0, 0.0)
        mind.mode = "patrol"
        self.assertFalse(hive._good_birth_site(mind))


if __name__ == "__main__":
    unittest.main()