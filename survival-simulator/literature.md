# Literature Scout — optimal foraging, multi-agent allocation, and extinction time

Date: 2026-09-18. Scope: published research relevant to the survival-simulator hive policy.
This is a **decision document**, not a reading list. Every quantitative claim carries a title and a
venue/arXiv id. Where I could not confirm a figure I say so explicitly.

## How to read this

- **MEASURED** = the authors ran it and report the number.
- **DERIVED** = arithmetic *I* did from a published formula plus our own measured constants. Not a
  quoted result. Treat as a hypothesis with explicit assumptions.
- **SPECULATED** = the authors assert it in discussion without evidence.
- **UNCONFIRMED** = I could not retrieve the primary number in this session.

Access note: PMC, Springer and OpenReview served bot-challenges during this session. Where a
primary PDF was unreachable I fell back to the arXiv full text, the Europe PMC REST API, or the
Semantic Scholar graph API, and I flag anything that rests only on an abstract.

---

## THEME 1 — Patch leaving: what should replace `MOVE_ON = 35`

### 1.1 The baseline result and what it actually says

Charnov, *Optimal foraging, the marginal value theorem*, **Theoretical Population Biology 9(2):129–136 (1976)**.
The MVT rule: leave the current patch when its **instantaneous** intake rate falls to **R\***, the
long-run average intake rate *of the whole environment including travel*. R\* is not a constant of
the forager; it is a property of the habitat, and it acts exactly as an opportunity cost of time.

This is the first structural problem with our policy. `MOVE_ON = 35` is a **fixed number**. In this
simulator R\* is not fixed: tree spawn carries `0.5 ** (t/300)`, so habitat productivity halves
every 300 s. Under Charnov's condition the leaving threshold must fall with R\*, which means
**optimal residence time must *increase* monotonically over the run**, and a forager should become
progressively *less* choosy. A fixed threshold is correct at exactly one instant and is
increasingly *too impatient* thereafter.

**DERIVED**, from Charnov's condition plus our measured decay constant: the ratio R\*(t)/R\*(0)
tracks the food supply, which our own notes put at ~74 trees at t=300 and ~18 at t=1500. That is a
~4× drop in habitat rate across the middle of a run. A threshold that is right at t=300 is roughly
4× too high at t=1500. Nobody in our 18 experiments has tested a *time-varying* threshold — every
sweep tested different constants.

### 1.2 Does a declining habitat change the MVT prediction?

Calcagno, Mailleret, Wajnberg & Grognard, *How optimal foragers should respond to habitat changes:
a reanalysis of the Marginal Value Theorem*, **Journal of Mathematical Biology (2013)**,
doi:10.1007/s00285-013-0734-y.

This is the paper that matters for us. It derives the **sensitivity of realized fitness and of
optimal residence time to general habitat attributes**, rather than relying on Charnov's graphical
construction. Three findings, all MEASURED (analytically derived and proven, not simulated):

1. **"Knowledge of average characteristics is in general not sufficient to predict the change in
   the average rate of movement."** In a heterogeneous habitat you cannot predict how residence
   time should respond to a habitat change from mean patch quality alone. Our world is
   heterogeneous by construction (forest/grassland `fruit_spawn_rate` 0.1, swamp 0.08, desert 0.05,
   river 0.0).
2. They **prove** a previously conjectured scaling invariance for *homogeneous* habitats, and show
   that **invariance to scaling is not generic in heterogeneous habitats**. Translation: if the
   world were uniform, a global multiplicative decay in productivity would leave the optimal policy
   unchanged (pure rescaling) — and our decay *is* a global multiplicative factor. In a
   heterogeneous world that invariance breaks, and the *relative* ranking of biomes should shift as
   supply decays.
3. They clarify **"the conditions under which it is adaptive to stay longer on poorer patches."**
   This is counterintuitive and is directly relevant: as the habitat empties, staying longer on a
   bad tree can beat commuting to a better one.

**Warning against over-reading point 1:** the homogeneous-habitat invariance result is a reason to
expect our `0.5**(t/300)` decay to be *less* important than it looks. If our effective habitat were
near-homogeneous from the hive's point of view, the global decay would rescale time without
changing the optimal rule, which would explain why MOVE_ON sweeps came out flat.

### 1.3 Incomplete information: Bayesian patch assessment, giving-up rules

Kilpatrick, Davidson & El Hady, *Normative theory of patch foraging decisions*,
**arXiv:2004.10671** (2020), and the journal version Kilpatrick, Davidson & El Hady,
*Uncertainty drives deviations in normative foraging decision strategies*,
**Journal of the Royal Society Interface 18(180):20210337 (2021)**, doi:10.1098/rsif.2021.0337.

Model: the forager **sequentially infers patch yield by Bayesian updating on its encounter
history**, and departs when either the certainty about patch type or the estimated yield falls
below a threshold. Key MEASURED result (from the published abstract, which I retrieved in full via
Europe PMC; I could not open the full text):

> "Uncertainty leads patch-exploiting foragers to **overharvest (underharvest) patches with
> initially low (high) resource yields** in comparison with predictions of the marginal value
> theorem."

Also MEASURED: "the time scale over which uncertainty in resource availability persists strongly
impacts behavioural variables like patch residence times and decision rules." When depletion is
slow (habitat selection regime, which is ours — a tree is not depleted by us, it is depleted by
the global decay), **departures are characterised by a reduction of uncertainty**, i.e. the forager
leaves once it has become confident the patch is poor, not once it has extracted a fixed amount.

Davidson & El Hady, *Foraging as an evidence accumulation process*,
**PLOS Computational Biology 15(7):e1007060 (2019)**, arXiv:1809.05023. MEASURED: they solve the
conditions under which a drift-diffusion accumulator is **exactly equivalent to MVT**, and simulate
deviations when those conditions fail. Two accumulator variants ("increment–decrement" vs
"counting") give **identical decisions in the limiting case but differ in how residence times adapt
when the environment is uncertain** — so the implementation detail is only visible under
uncertainty, which is our regime. They also introduce an **energy-dependent utility** that
**predicts longer-than-optimal residence times when food is plentiful**.

Webb, Steffan, Hayden, Lee, Kemere & McGinley, *Foraging Under Uncertainty Follows the Marginal
Value Theorem with Bayesian Updating of Environment Representations*, **bioRxiv preprint
2024.03.30.587253 (2024)**. MEASURED, in mice, varying travel time and depletion rate both
deterministically and stochastically: residence times were consistent with MVT and **"not
explainable by simple ethologically motivated heuristic strategies"**; behaviour was best accounted
for by **MVT with the environment representation updated by a Bayesian estimator with a dynamic
prior**, driven by *local variations in reward timing*. I could not retrieve the effect sizes (PMC
blocked); treat the model-comparison ranking as confirmed and the magnitudes as **UNCONFIRMED**.

Kilpatrick & El Hady, *Resource depletion accelerates rate learning but not composition learning in
patch foraging*, **arXiv:2607.29476** (2026). This one is unusually on-point and carries a hard
negative result. MEASURED (analytic + simulation):

- **Within a patch**, depletion *accelerates* learning the local rate, because falling encounter
  spacing pins down the initial rate.
- **Across patches**, learning the *composition* of the environment (what fraction of patches are
  high-yield) is **slow, set by the number of patches sampled rather than the time spent in each,
  and unaffected by depletion once rates are known.**
- Reward-maximising and information-seeking policies therefore **diverge**: a forager resolving
  composition **underharvests rich patches**, most sharply early in exposure.
- When a fixed patch set replenishes between visits, **the reward-maximising policy collapses onto
  a stable orbit over the high-yield patches**, and the replenishment rate determines whether the
  forager maps the whole environment or locks onto a rich subset.
- **Which departure rule is best is itself set by patch variability**: it switches from *counting
  prey* to *timing the gaps between them* once richness varies by **more than about a quarter**
  (their phrasing; the exact coefficient of variation threshold is in the paper body, which I did
  not retrieve — treat "about a quarter" as the authors' own summary).

This is the single most useful result in Theme 1 for us, for three reasons:
(a) it says **environment-level knowledge accrues per *patch visited*, not per second spent**,
which is the exact same wall our SCOUT experiment hit from the other side (knowledge limited by
re-sensing rate, not ground covered);
(b) it says that when patches replenish — ours do, trees keep emitting fruit — **the optimal policy
is a trapline orbit over a rich subset**, not exploration. That is consistent with our measurement
that 84.6 % of meals already come from the tree map;
(c) it gives a concrete, testable switch: **count-based vs interval-based giving-up rules are not
equivalent**, and the right one depends on how variable our trees are (which we can measure —
biome-driven `fruit_spawn_rate` varies 0.0/0.05/0.08/0.1, i.e. *much* more than 25 %).

### 1.4 Giving-up density and giving-up time

Brown, *Patch use as an indicator of habitat preference, predation risk, and competition*,
**Behavioral Ecology and Sociobiology 22:37–47 (1988)**. The quitting harvest rate satisfies
**H = C + P + MOC**: metabolic cost of foraging + predation cost + missed-opportunity cost. This is
the MVT generalised to include risk and the value of alternative activities. Two consequences for
us:

- Our `MOVE_ON` is a pure MOC term with C and P set to zero. Our diagnostics say move+turn is
  4.69 of 9.03 energy/agent/s — C is **not** negligible and is not in the rule.
- For a **satiated** agent the marginal value of an additional unit of energy is zero (the clamp
  destroys it), so its quitting harvest rate is reached **immediately**. Brown's framework says a
  full agent should leave a patch at once, without any appeal to altruism. See Theme 3.

### 1.5 Do real foragers actually follow MVT? Conflicting evidence

- **Supports:** Pacheco-Cobos et al., *Nahua mushroom gatherers use area-restricted search
  strategies that conform to marginal value theorem predictions*, **PNAS 116(21):10339–10347
  (2019)** — MVT-consistent, *undiscounted*, in the wild.
- **Contradicts:** Nonacs, *State dependent behavior and the marginal value theorem*,
  **Behavioral Ecology 12(1):71–83 (2001)** — catalogues systematic patch **over-staying**.
  Cash-Padgett & Hayden, *Behavioural variability contributes to over-staying in patchy foraging*,
  **Biology Letters 16(3):20190915 (2020)** — over-staying is partly just decision noise.
  Constantino & Daw, *Learning the opportunity cost of time in a patch-foraging task*,
  **Cognitive, Affective, & Behavioral Neuroscience 15(4):837–853 (2015)** — lab humans over-stay,
  and the discrepancy shrinks once discounting is accounted for.
- **Resolution / conditions:** over-staying appears in *lab* tasks with explicit discounting or
  short horizons; MVT-conformity appears in *field* tasks where the currency is long-run rate.
  Our objective is lineage survival time over a 3000 s horizon with no discounting, which puts us
  on the field side — i.e. **we should expect the undiscounted MVT to be the right target**, and
  our policy's fixed threshold to err in whichever direction the constant happens to sit.
- Note the three "deviation" mechanisms point in **opposite** directions and must not be conflated:
  discounting → over-stay (Wispinski et al., §6); uncertainty about the *current* patch → over-stay
  poor patches, under-stay rich ones (Kilpatrick et al. 2021); learning the *composition* of the
  environment → under-stay rich patches (Kilpatrick & El Hady 2026).

---

## THEME 2 — Multi-forager allocation, and why SPREAD produced nothing

### 2.1 Ideal Free Distribution

Fretwell & Lucas, *On territorial behavior and other factors influencing habitat distribution in
birds. I. Theoretical development*, **Acta Biotheoretica 19:16–36 (1969)**, doi:10.1007/BF01601953.
Foragers distribute so that **realised intake rate is equal across all occupied patches**; patch
occupancy is proportional to patch input rate ("input matching").

The crucial and usually-forgotten precondition is **density-dependent payoff**. IFD produces
spreading only because adding a forager to a patch *lowers* everyone's rate there.

### 2.2 Interference is the mechanism, and we do not have it

Sutherland, *Aggregation and the 'ideal free' distribution*,
**Journal of Animal Ecology 52(3):821–828 (1983)**. Introduces the interference constant *m*:
per-capita intake scales as (competitor density)^(−m). m = 0 means no interference; the IFD
collapses to "everyone goes to the best patch" and spreading has **zero** payoff.

**This is the theoretical explanation of our SPREAD null (experiment #15).** In our simulator:

- Fruit is consumed on contact, so there *is* exploitative competition in principle.
- But our measured densities are 3–8 agents on a 1600×1200 world, and our own coverage probe found
  mean inter-agent separation 459 units, closest pair 78 units. At those densities two agents
  essentially never contest the same 20–60-unit fruit annulus.
- Therefore **m ≈ 0**, and IFD predicts **no benefit from dispersion**. SPREAD moved its target
  metric (separation 459 → 508, closest pair 78 → 113) and produced exactly the payoff theory
  predicts for m ≈ 0: nothing.

**Do not re-run dispersion or repulsion variants.** The theory says the mechanism is absent, and
our measurement agrees. Record this as a *predicted* null, not a mysterious one.

### 2.3 IFD's own empirical failure mode

Kennedy & Gray, *Can ecological theory predict the distribution of foraging animals? A critical
analysis of experiments on the ideal free distribution*, **Oikos 68:158–166 (1993)**.
MEASURED (re-analysis of the published experimental literature using matching-law regression): real
distributions are **systematically undermatched** — "the distribution of organisms is systematically
less extreme than the distribution of resources." So even where IFD *should* apply, animals
under-respond to quality differences.

This is a **negative result about the theory itself** and it cuts our way: a hive that routes
somewhat less aggressively than pure quality-proportional allocation is not obviously wrong.

### 2.4 Social information sharing between foragers — the one place with real numbers

Bidari, El Hady, Davidson & Kilpatrick, *Stochastic dynamics of social patch foraging decisions*,
**Physical Review Research 4(3):033128 (2022)**, arXiv:2202.05761. Each forager runs an evidence
accumulator; beliefs are coupled either **pulsatile** (broadcast only your leave decision) or
**diffusive** (continuously broadcast your belief). MEASURED:

- Foraging efficiency has a **stronger dependence on coupling strength under pulsatile** coupling.
- **Despite employing minimal information transfer, pulsatile coupling can still provide similar or
  higher foraging efficiency than diffusive coupling.**
- **Diffusive coupling is more robust to parameter detuning** and is better when individuals have
  **heterogeneous departure criteria** and heterogeneous weighting of social information.

Blum Moyse & El Hady, *Social patch foraging theory in an egalitarian group*, **arXiv:2412.02381**
(2024). Extends this to large egalitarian groups and **derives optimal agent strategies
analytically** across sharing mechanisms: observing others' rewards, sharing beliefs continuously,
pulsed observation of others' departures *or arrivals*, and simply **counting the number of
individuals in a patch**.

Relevance: our hive already has *perfect* intra-hive sharing (a single centralised controller with
a fused map), so we are at the strong-coupling limit of the diffusive model. The literature's
finding that *minimal* (pulsatile) sharing is competitive is bad news for the "more information
sharing between our agents" family of ideas — **we are already past the point of returns.** Our own
oracle ablation is consistent: the −642 visibility gap is about *sensing the world*, not about
sharing what we sense.

### 2.5 Division of labour / response thresholds

Kang & Theraulaz, *Dynamical models of task organization in social insect colonies*,
**arXiv:1511.04769** (2015). MEASURED (analysis of their model): **"the smaller colony invests its
resource for the colony growth and allocates more workers in the risky tasks such as foraging while
the larger colony shifts more workers to perform the safer tasks inside the colony."**

Fontanari, de Oliveira & Campos, *Evolving division of labor in a response threshold model*,
**Ecological Complexity 58:101083 (2024)**, arXiv:2308.07122. MEASURED: with a structured
metapopulation and winner-take-all group competition, **a substantial fraction of workers
specialise in each task without needing any penalty on task switching**. Also MEASURED: the
mean-field stimulus dynamics exhibit **period-doubling cascades into complex attractors when the
threshold noise is small** — i.e. a deterministic response-threshold controller can oscillate.

Relevance and honest assessment: the colony-size result says specialisation pays *more* in small
colonies, which superficially supports something like SCOUT. But our measured colony is 3–8, and
our SCOUT experiment already showed that the diverted agent does the work and the *knowledge stock*
does not follow. The response-threshold literature offers one variant we have **not** tested:
thresholds keyed to **internal state (energy)** rather than to role. That is Theme 3, and it is the
version with arithmetic behind it.

Goldshtein et al., *Reinforcement learning enables resource partitioning in foraging bats*,
**Current Biology 30(20):4096–4102 (2020)** — cited here because it is the clearest empirical case
of multiple foragers *partitioning* a shared resource landscape via simple learning rules rather
than explicit coordination. I did not retrieve the effect sizes; **UNCONFIRMED** magnitudes.

### 2.6 Market/auction task allocation

I searched arXiv and the general web for market-based or auction-based multi-robot task allocation
reporting **quantitative gains over a centralised greedy assignment on a foraging task** and found
nothing that meets the bar. The standard MRTA auction literature compares against *decentralised*
baselines; our hive is already a centralised assigner with global information, which is the case
auctions are trying to approximate. **I am not recommending this direction and I will not invent a
number for it.**

---

## THEME 3 — Capacity constraints, satiation, and our 22.7 % overflow

This is the theme where the literature is most directly actionable.

### 3.1 Partial loading is a documented, quantified, *optimal* behaviour

Schmid-Hempel, Kacelnik & Houston, **"Honeybees maximize efficiency by not filling their crop"**,
**Behavioral Ecology and Sociobiology 17:61–66 (1985)**. MEASURED: "Honeybees often abandon
non-depleting food sources with a **partially filled crop**." The behaviour is inconsistent with
rate maximisation and consistent with maximising **efficiency** (energy gained per energy spent),
because a heavier load costs more to carry.

The title is literally our problem statement inverted. The currency question is the whole issue:
**rate-maximising says fill up; efficiency-maximising says stop early.** Our objective (survive
longest) is far closer to efficiency than to rate, because energy that overflows the clamp has zero
value while the movement that acquired it had positive cost.

Kacelnik, *Central place foraging in starlings (Sturnus vulgaris). I. Patch residence time*,
**Journal of Animal Ecology, pp. 283–299 (1984)** — the canonical central-place load-size
experiment, showing load size adapts to travel time. **UNCONFIRMED** effect sizes (not retrieved).

### 3.2 Digestive/capacity constraints change the functional response

Jeschke, Kopp & Tollrian, *Predator functional responses: discriminating between handling and
digesting prey*, **Ecological Monographs 72(1):95–112 (2002)**,
doi:10.1890/0012-9615(2002)072[0095:PFRDBH]2.0.CO;2. MEASURED/derived: separating *handling* from
*digesting* yields a "steady-state satiation" model in which **at satiation the relative feeding
time equals the ratio of handling time to digestion time**. The practical content is that a
capacity-limited consumer's optimal *search effort* is not constant — it should fall toward zero as
the internal store fills, because additional intake cannot be assimilated.

Related: Verlinden & Wiley's **Digestive Rate Model** (1989) predicts diet selection under a
digestive-capacity constraint by prioritising items on digestibility and turnover time. I verified
the model's existence and framing but **not** its original venue/pages; treat the citation as
**UNCONFIRMED** and do not quote numbers from it.

### 3.3 Should a satiated forager *avoid* food so it remains for conspecifics?

**Direct answer: I found no paper that reports or models a forager declining food in order to leave
it for conspecifics.** I searched optimal-foraging, social-foraging, producer–scrounger and
resource-partitioning literature. Stating this as a negative search result rather than inventing a
citation.

What the literature *does* supply is a self-interested derivation that reaches the same behaviour:

- **Brown 1988** (above): quit when harvest rate falls to C + P + MOC. For a full agent the
  marginal value of harvest is **zero**, so the quitting threshold is met immediately and the agent
  should leave at once — no altruism required, and no group-selection argument needed.
- **Schmid-Hempel et al. 1985**: real animals do stop short of capacity, for efficiency reasons.
- The nearest social-foraging framework, Barnard & Sibly, *Producers and scroungers: a general
  model and its application to captive flocks of house sparrows*,
  **Animal Behaviour 29(2):543–550 (1981)**, models the *opposite* incentive — individuals
  exploiting food found by others. MEASURED: scroungers obtained most of their food via
  interaction with producers, producers by their own search; whether scroungers persist depends on
  their frequency and on group size. Nothing in it supports voluntary deferral.

**Conclusion for us:** the "leave it for a hungrier sibling" framing is unsupported and unnecessary.
The supported framing is **"a full agent should not spend movement energy acquiring energy it
cannot store."** That is an individual-level efficiency argument with two published anchors, and it
is directly measurable against our 30.2 % of bites / 39,334 destroyed energy.

---

## THEME 4 — Subcritical branching: maximising time to absorption, not growth rate

This theme reframes the whole project and explains several of our nulls.

### 4.1 The size of the population buys almost nothing when R0 < 1

Schreiber, Huang, Jiang & Wang, *Extinction and quasi-stationarity for discrete-time, endemic SIS
and SIR models*, **arXiv:2005.08312** (2020). MEASURED (theorems + large-deviation analysis). The
two regimes are sharply different:

- **R0 > 1:** mean time to extinction **increases exponentially with population size N**.
- **R0 < 1:** "the mean times to extinction are **bounded above by 1/(1−α)** where α < 1 is the
  geometric rate of decrease of the infection when rare"; as N → ∞ the quasi-stationary
  distribution converges to a point mass at the *disease-free* (i.e. extinct) state.

Read that again in our terms. **Below criticality, the mean time to absorption is bounded by a
constant that depends only on the per-capita decay rate α — not on N.** Adding agents does not buy
exponential time; it buys essentially nothing beyond a logarithmic transient.

**DERIVED**, standard subcritical Galton–Watson result plus our measured R0 = 0.93:
starting from N individuals, P(T > n generations) ≈ N·c·R0ⁿ, so
E[T] ≈ log(N·c) / |log R0|. With |log 0.93| = 0.0726 per generation:

| change | extra generations |
|---|---|
| N: 1 → 3 | log 3 / 0.0726 ≈ **15** |
| N: 3 → 8 | log(8/3) / 0.0726 ≈ **13.5** |
| R0: 0.930 → 0.959 | multiplies **all** of E[T] by **1.74** |

This is quantitatively consistent with our own experiment #18: pop_1 570.6 → pop_3 742.5 is +172 s,
and pop_3 → pop_8 is −54 s (population cost starts to bite). It also **predicts** the measurement
that surprised us most: starvation share of deaths was flat (0.57–0.63) across an 8× population
range, because in a subcritical process population size does not change the *per-capita* balance
at all — it only changes how many independent lineages are running down the same clock.

**The actionable statement:** to move the mean from ~740 to the 1291 leaderboard band you need
E[T] × 1.74, which needs R0 from 0.930 to ≈ **0.959**. In energy terms that is moving net
per-agent-second from **−0.76 to about −0.44**, i.e. recovering roughly **0.32 energy/agent/s**.
There is no population configuration that does this, and there is no variance trick that does this.
(Assumptions: generation time roughly constant across configurations; R0 maps monotonically onto
net energy per agent-second. Both are ours, neither is published.)

### 4.2 Optimal control when the objective *is* the absorption time

Claisse & Champagnat, *On the link between infinite horizon control and quasi-stationary
distributions*, **arXiv:1607.08046** (2016). MEASURED (theorems): for controlled continuous-time
branching processes with **almost-sure extinction**, they establish the equivalence between
infinite-horizon control and an optimisation over **quasi-stationary distributions and their
extinction rates**; they characterise the optimal Markov control in the limit where the value
function diverges, and prove convergence to a unique QSD under the optimal control.

The content for us: **the right optimisation target is the extinction rate of the QSD, i.e. the
leading eigenvalue of the killed generator — a single scalar.** All the structure of the policy
collapses into its effect on that one number. This is a strong argument that our policy search has
been optimising the wrong objective surface: we have been sweeping behavioural knobs, when the only
thing any of them can do is shift one eigenvalue that is dominated by the energy budget.

Chazottes, Collet & Méléard, *Sharp asymptotics for the quasi-stationary distribution of
birth-and-death processes*, **Probability Theory and Related Fields 164 (2016)**, arXiv:1406.1742.
MEASURED: for logistic-type birth–death processes with carrying capacity K, the QSD is
asymptotically **Gaussian centred on the deterministic equilibrium** for large K, with explicit
spectral-gap-based estimates of the mean extinction time.

Doering, Sargsyan & Sander, *Extinction times for birth–death processes: exact results, continuum
asymptotics, and the failure of the Fokker–Planck approximation*, **arXiv:q-bio/0401016** (2004).
MEASURED and important as a **methodological negative result**: they give exact discrete
expressions and show that **the Fokker–Planck (diffusion) approximation is valid only quite near
the threshold**. Any continuum/deterministic reasoning about our population of 3–8 is therefore
untrustworthy; the discrete stochastic treatment is required. This retroactively justifies our
n=30 empirical approach over any analytic population model.

Assaf & Meerson, *Extinction of metastable stochastic populations*, **Physical Review E 81:021116
(2010)**, arXiv:0907.0070. MEASURED: WKB treatment giving both the QSD and the (exponentially
long) mean time to extinction, distinguishing the case where n = 0 is a *repelling* point of the
deterministic dynamics from the case where it is *attracting*. Our case is the attracting one —
which is the regime with **no** exponential protection.

### 4.3 Does variance reduction or mean increase matter more near criticality?

Two sources that genuinely conflict, and the conditions under which each holds:

- **Mean matters:** Philippi & Seger, *Hedging one's evolutionary bets, revisited*,
  **Trends in Ecology & Evolution 4(2):41–44 (1989)**, doi:10.1016/0169-5347(89)90138-9. The
  bet-hedging premise is explicit: variance reduction is bought **at the cost of mean
  performance**, and it pays only when fitness is **multiplicative across independent episodes**
  (geometric-mean fitness). Our evaluation is an **arithmetic mean of three runs**, not a product,
  so the bet-hedging argument does not apply and the trade is strictly bad for us. This is an
  independent confirmation of our own P(3-run avg ≥ 1000) analysis.
- **Variance matters:** Gabel, Meerson & Redner, *Survival of the Scarcer*,
  **Physical Review E 87:010101(R) (2013)**, arXiv:1210.0018. MEASURED: in a two-species
  competition model, "for a sizable range of asymmetries in the growth and competition rates, the
  paradoxical situation arises in which the **numerically disadvantaged species according to the
  deterministic rate equations survives much longer**."
- **Resolution:** the "scarcer survives" effect is a *fluctuation* phenomenon in a competitive
  two-species system where the abundant species' own dynamics accelerate its collapse. It does not
  generalise to a single lineage running down an absorption clock. **For us: raise the mean.**

---

## THEME 5 — State-dependent decisions: our 10.6 % refusal rate is a theory error

### 5.1 The canonical framework

Houston & McNamara, **Models of Adaptive Behaviour: An Approach Based on State**, Cambridge
University Press (1999). The book's structure (verified from the published table of contents)
covers "Energy gained from foraging", "Risk-sensitive foraging", "The energy–predation trade-off",
"Behaviour when there is a limit to the time without food", "Risk-sensitive behaviour in a changing
environment", and "Gaining information on foraging options". The core move is to replace
rate-maximisation with **dynamic programming on reserves**, where the terminal payoff is survival.

The central, uncontroversial prediction: **when the optimality criterion is survival probability and
reserves approach the starvation boundary, the optimal policy becomes risk-prone.** A gamble with
negative expected value but positive probability of clearing the threshold beats a certain outcome
that does not clear it, because the value function is convex near the boundary.

Caraco-style reviews frame this as maximising long-term survival probability subject to an upper
limit on reserves (see *Risk-sensitive foraging: a review of the theory*, ScienceDirect
S009282400580031X — **UNCONFIRMED** authorship/venue details, I only reached the search snippet).

### 5.2 The energy budget rule, and the modern correction

Lim, Wittek & Parkinson, *On the origin of risk sensitivity: the energy budget rule revisited*,
**Animal Behaviour 110:69–77 (2015)**, arXiv:1504.04986. MEASURED (normative derivation + fit to
existing empirical data). The classical daily-energy-budget rule says: **risk-averse on a positive
budget, risk-prone on a negative budget.** They note this has been "challenged both empirically and
theoretically", and derive a **gradual budget rule**:

- the conventional rule **holds when the expected reserve is close enough to the overnight-survival
  threshold**, where selection pressure is significant;
- the conventional rule **need not hold when the expected reserve is far from the threshold**,
  where selection pressure is insignificant;
- the gradual rule **"better fits the empirical findings including those that used to challenge the
  conventional budget rule."**

This is precisely the shape of rule we should be implementing: **risk posture as a continuous
function of distance-to-death, not a binary affordability gate.**

### 5.3 Emergent confirmation in a learned controller

Chaturvedi, EL-Gazzar & van Gerven, *Emergence of Internal State-Modulated Swarming in Multi-Agent
Patch Foraging System*, **arXiv:2510.18886**, published version doi:10.1145/3795095.3805114 (2026).
MEASURED, in evolved CTRNN velocity controllers on continuous-2D partially-observed patch foraging:
the strength of emergent aggregation is **inversely proportional to the amount of resource stored
in the forager**, "which supports the risk-sensitive foraging claims". They further **clamped the
recurrent hidden states** to encode a lower stored resource and this **hastened the aggregation
behaviour** — a causal intervention, not a correlation.

### 5.4 Application to our numbers, with the twist our objective demands

Our measurement: agents refuse to walk to food they cannot *provably* reach on **10.6 % of hungry
ticks**, with a **median shortfall of only 6 energy**, and roughly **20 deaths per run beside known
food**. That refusal rule is a **rate-maximising feasibility check applied to a survival-maximising
problem**. Houston & McNamara say it is the wrong rule at low reserves; Lim et al. say the
correction should be graded, not binary.

But there is a twist that the ecology literature does not contain and our simulator does. Because
`score += dt` only stops when **population reaches zero**, an individual death costs score **only
if it is the last agent**. Combining Theme 4 (the objective is time-to-absorption) with Theme 5
(risk posture tracks distance-to-death), the correct rule is two-dimensional:

> Risk-proneness should increase as **individual reserves** fall **and** as **population** falls.
> At population 1, the agent should accept *any* gamble with positive probability of survival.
> At population 8, a marginal agent should be risk-neutral and should not spend hive resources on
> a low-probability rescue.

I have not found this stated in any paper; it is a **DERIVED** synthesis. It is, however, exactly
the structure that Claisse & Champagnat's QSD-control result implies (the optimal Markov control is
a function of the full state, and here the state includes N).

---

## THEME 6 — Learned controllers: the margin, the sample budget, and whether we can afford it

Our constraint, stated up front so every number below can be judged against it:
**per-run sd 130–250, one full episode costs 30–350 s of wall clock, and a resolvable comparison
needs n ≈ 30 per variant.** One data point is ~30 000 environment steps.

### 6.1 The best case: deep RL does reach MVT-optimal — at enormous cost

Wispinski, Butcher, Mathewson, Chapman, Botvinick & Pilarski, *Adaptive patch foraging in deep
reinforcement learning agents*, **Transactions on Machine Learning Research (2023)**,
arXiv:2210.08085. This is the strongest positive result in the area and the numbers are unambiguous.

MEASURED results:
- Task: continuous 3D, 32 × 32 m, **two** patches, 3600 steps/episode, LIDAR observations,
  5-D continuous actions, exponentially decaying in-patch reward with the instantaneous rate
  **visible as patch colour** (i.e. the value estimation problem is handed to the agent).
- Algorithm: MPO, LSTM(256) after a conv + 3-layer MLP trunk.
- **N = 12 agents, 3 per each of 4 discount-rate treatments, each trained for 12e7 = 1.2 × 10⁸
  environment steps**, 16 environments in parallel, Adam, lr 3e-4.
- **Compute: "Each agent was trained on an internal cluster for roughly 13 days, and used
  approximately 40 GiB RAM, 8 CPU, and 8 GPUs."**
- Behaviour: mean patch leaving step 121.7, mean travel 57.7 steps; residence time rises with patch
  distance (b = 9.60 ± 0.87, p = 4.0 × 10⁻²⁸).
- Optimality vs plain MVT: agents **over-stay by 15.8 steps** (se 2.7), t(11) = 5.60,
  p = 1.6 × 10⁻⁴.
- Optimality vs **discount-corrected** MVT: **0.9 steps** (se 2.8), p = 0.74 — statistically
  indistinguishable from optimal.
- Internal dynamics reproduce primate ACC evidence-accumulation signatures (Hayden, Pearson & Platt,
  **Nature Neuroscience 14(7):933–939, 2011**); a single PC carrying 14 % of LSTM variance shows
  the slope/threshold pattern.

**Negative results and limitations the authors themselves record:** only MPO was evaluated ("future
work may find improvements ... with alternative methods"); adaptation across environments was driven
by the **baseline-to-threshold distance**, not the accumulation **slope**, which is the *opposite*
of the primate finding (they flag this as a mechanism mismatch with the same behavioural signature);
and the environment is deliberately impoverished — two patches, a refresh rule, and *visible* patch
value.

**Feasibility verdict for us: NOT FEASIBLE.** 1.2 × 10⁸ steps is **4 000 full 3000-step episodes ×
1 000**, i.e. ~4 million of our episodes per agent. At our fastest (30 s/episode) that is ~3 800
CPU-years per seed, and the paper still needed 8 GPUs × 13 days per agent on a two-patch toy. And
note what it bought: matching a *known closed-form optimum on a task where the optimum was already
computable*. We do not need a network to compute the MVT threshold; we need to know where the food
is, and that is a sensing problem (our −642 visibility ablation), not a policy-class problem.

### 6.2 Model-based RL reaches the same place, framed as a world-model result

Fonseca, Ríos, Quijano & Giraldo, *World Models Unlock Optimal Foraging Strategies in Reinforcement
Learning Agents*, **arXiv:2512.12548** (2025). MEASURED: a model-based RL agent that learns "a
parsimonious predictive representation of its environment" **converges to MVT-aligned patch-leaving
strategies**, whereas standard model-free agents do not show the same biologically-consistent
decision pattern; the authors attribute this to **anticipatory capability rather than reward
maximisation alone**. I did not retrieve per-condition effect sizes or the sample budget —
**UNCONFIRMED magnitudes and budget.**

Relevance: this is the learned-controller literature independently rediscovering our own finding
that "Hive's world model is load-bearing" (our burst/map-free controller scored 389 vs 777). It
argues *against* replacing the hive's explicit map with a model-free policy.

### 6.3 The realistic case: MARL on a *gridworld* foraging benchmark

Papoudakis, Christianos, Schäfer & Albrecht, *Benchmarking Multi-Agent Deep Reinforcement Learning
Algorithms in Cooperative Tasks*, **NeurIPS 2021 Datasets & Benchmarks Track**, arXiv:2006.07869.
This is the calibration point that matters, because Level-Based Foraging (LBF) is a **discrete
grid, ≤ 4 agents, ~0.1 ms/step** — vastly simpler and cheaper than our simulator.

MEASURED budgets and protocol:
- **LBF and MPE: on-policy algorithms trained for 20 million timesteps; off-policy for 2 million.**
  (SMAC and RWARE: 40 M / 4 M.)
- 5 seeds per algorithm-task; 41 evaluation points × 100 episodes each.
- Hyperparameters grid-searched **per environment**, 3 seeds per configuration.
- **Total: 138,916 CPU hours, excluding the hyperparameter search.**

MEASURED outcomes on LBF (maximum returns, parameter sharing, normalised so a solved episode = 1.0):
IQL/IA2C/IPPO/MAA2C/MAPPO/VDN all reach 1.00(0) on `8x8-2p-2f-c`; on the harder `15x15-3p-5f`
IA2C reaches 0.89(4); on `15x15-4p-5f` MAA2C reaches 0.95(1).

MEASURED **negative results**, which are the more useful half:
- **MADDPG and COMA fail across the board** on the discrete grid-world environments; the authors
  attribute MADDPG's failure to the biased Gumbel-Softmax reparameterisation.
- **"A task that we were unable to get any algorithm to learn is the cooperative-only variant with
  three or more agents."**
- On the sparse-reward RWARE, **IQL, COMA, MADDPG, VDN and QMIX "do not exhibit any learning."**
- Independent learning (IA2C) is **competitive with or better than** all the centralised CTDE
  methods on LBF — centralisation only pays under partial observability with real coordination
  requirements.

**Feasibility verdict:** 2 × 10⁶ off-policy steps is **~67 of our episodes' worth of steps** and
looks cheap — but that is the *lower* bound, on a 0.1 ms/step gridworld, for a task with a dense-ish
shaped objective and a 2–4 agent team. The on-policy figure, 2 × 10⁷, is **~667 full episodes per
seed per configuration**, i.e. 5.5–65 hours of wall clock *per seed* at our 30–350 s/episode, before
5 seeds and before any hyperparameter search. A Papoudakis-style protocol on our simulator is
**months of CPU**, and their own total was 139 k CPU-hours.

### 6.4 Learned controllers only beat heuristics when the heuristic is broken

Keddouri, Houhou, Boulmerka & Farhi, *Regime-Conditional Stabilisation of LLM-Augmented Cooperative
Multi-Agent Reinforcement Learning*, **arXiv:2607.04470** (2026). MEASURED with QMIX, 5 seeds,
three environments, and a clean three-regime taxonomy that is directly transferable to our decision:
- **Essential regime (Level-Based Foraging):** the unshaped baseline succeeds **0.1 %** of the time;
  **any** reward shaping unlocks the task (**95.9 %** under EMA-stabilised shaping).
- **Augmentative regime (MPE Simple Spread):** baseline **74.4 %**; EMA shaping **86.7 %**
  (**+12.3 pp, p < 0.01**); **naive** dynamic shaping **collapses it to 15.2 %**.
- **Supplementary regime (SMAC 3m):** baseline **98.8 %**, stabilised shaping **99.9 %**;
  unstabilised shaping "adds variance without gain".

Their conclusion is stated as a design law: **"regime placement is a practical predictor of whether
dynamic shaping helps or harms."** Our hive is emphatically in the *supplementary* regime — it
already achieves 68 % of a perfect-information oracle (772/1141) and the response surface is flat.
The published expectation for that regime is **"preserves performance ... adds variance without
gain."**

Mguni et al., *Fault Tolerant Multi-Agent Learning with Adversarial Budget Constraints*,
**arXiv:2508.08800** (2026). MEASURED: **up to +116.7 % on SMAC, +21.4 % on MPE SimpleTag, +44.6 %
on LBF**. Flagged here precisely because these are large numbers that are **irrelevant to us**: they
are MARL-vs-MARL margins under induced agent faults, not margins over a hand-written heuristic.

Albrecht & Ramamoorthy, *A Game-Theoretic Model and Best-Response Learning Method for Ad Hoc
Coordination in Multiagent Systems*, **arXiv:1506.01170** (2015). MEASURED: their planning method
(HBA) "achieves higher flexibility and efficiency than several alternative algorithms" on LBF. This
is a *planning* method with type reasoning — closer in spirit to our hive than to model-free RL.
Effect sizes **UNCONFIRMED** (not retrieved).

Liao, Wu & Wu, *Dynamic Sight Range Selection in Multi-Agent Reinforcement Learning*,
**AAMAS 2025**, arXiv:2505.12811. MEASURED: dynamically adjusting each agent's **sight range**
during training via UCB improves LBF, RWARE and SMAC across QMIX and MAPPO, and also *accelerates*
training. Included because it is the only MARL paper I found that treats **sensing radius as the
optimisation variable** — which is the variable our own oracle ablation says dominates (−642 for
visibility vs −106.5 for predator state and −47.6 for fruit value).

### 6.5 Evolutionary strategies on a foraging task of roughly our shape

Chaturvedi, EL-Gazzar & van Gerven, **arXiv:2510.18886** (see §5.3). Continuous 2D, stochastic
position updates, partial observability, multiple resource patches, non-cooperative foragers, a
**shared CTRNN velocity controller evolved by an evolution strategy where all policy-distribution
samples are evaluated in the same rollout**. This is the closest published setup to what
`train_es.py` already does, and it worked — but the paper reports *emergent behaviour*, not a
score margin over a hand-written baseline. **No margin is reported; do not infer one.**

### 6.6 The one genuinely encouraging budget number

Suárez et al., *Results of the NeurIPS 2023 Neural MMO Competition on Multi-task Reinforcement
Learning*, **arXiv:2508.12524** (2025). MEASURED: on a massively-multiagent *survival* environment
(procedural maps, resource collection, combat, 128 agents), **"the top solution achieved a score 4×
higher than our baseline within 8 hours of training on a single 4090 GPU."**

This is the only feasible-looking budget in the whole theme, and the reason it is feasible is
**environment throughput**, not algorithmic efficiency — Neural MMO 2.0 was "a complete rewrite of
its predecessor with three-fold improved performance" (arXiv:2311.03736), and the surrounding
tooling (PufferLib arXiv:2406.12905; JaxRobotarium arXiv:2505.06771, reporting **20× training and
150× simulation speedups**) exists precisely to make the step count affordable.

**The transferable lesson is not "train a policy". It is: in this literature, the binding
constraint is always environment steps per second, and the winning move is always to make the
simulator faster, not to make the learner smarter.** Our episode costs 30–350 s. Until that number
drops by 2–3 orders of magnitude, every published learned-controller result in this section is out
of reach, and the 4× Neural MMO result is the exception that proves it.

---

## RANKED RECOMMENDATIONS

Ordered by expected effect on our measured numbers, with the mechanism and an explicit null-risk
flag. Our history is 16+ nulls, so the flags matter more than the ordering.

### 1. Capacity-aware target assignment (route full agents away from fruit, hungry agents toward it)

- **Mechanism.** `agent.energy = min(max_energy, energy + fruit.energy)` destroys the excess on
  contact. A near-full agent that walks to fruit pays the full movement cost and banks a fraction of
  the return. This is an **assignment** error, not a travel-volume error. Brown 1988's quitting rule
  (marginal value zero ⇒ quit immediately) and Schmid-Hempel, Kacelnik & Houston 1985 (bees stop
  short of a full crop because the currency is efficiency, not rate) both say the same thing.
- **Numbers it moves.** Overflow fraction (**22.7 % of harvested energy, 30.2 % of bites, 39,334
  energy destroyed**); net energy per agent-second (**−0.76**); R0 (**0.93**).
- **Why the arithmetic is decisive.** From §4.1, going from 740 to the 1291 band needs ≈ +0.32
  energy/agent/s, which is **≈ 2,070 energy over a 1291 s run at mean population 5**. That is
  **~5 % of the 39,334** we currently destroy. Even a large haircut on that figure leaves the
  conclusion standing — this is the only lever in the project whose measured magnitude exceeds the
  required magnitude by an order of magnitude.
- **Critical framing so this does not become the 17th null.** Do **not** implement it as "travel
  less". Our income/move ratio is 1.9:1 and every travel-reduction idea has failed. Implement it as
  a **hunger-weighted assignment**: total hive travel stays roughly constant, but the *identity* of
  the agent sent to each fruit changes. Concretely, weight a fruit's value to agent *i* by
  `min(fruit.energy, max_energy_i - energy_i) / fruit.energy` rather than by `fruit.energy`.
- **Null risk: MODERATE.** The counter-argument is real: we measured 140,940 energy rotting
  unharvested, so a fruit skipped by a full agent may simply rot rather than feed a sibling. The
  gain is then only the reallocated *trip*, not the saved fruit. Mitigate by measuring the right
  thing: instrument **realised intake per unit movement energy**, not overflow percentage.
- **Companion change with independent arithmetic.** Spawn *before* overflowing rather than on a
  schedule: an agent at 175+ energy that is about to clamp can convert 100 energy into a child
  carrying 75 (25 % loss) instead of losing 100 % of the excess. This is not a population-size
  change — keep the target at 3 — it is a change to *when* the existing spawns fire.

### 2. Two-dimensional risk posture: risk-proneness as a function of reserves **and** population

- **Mechanism.** Houston & McNamara 1999 (convex value function near the starvation boundary) and
  Lim, Wittek & Parkinson 2015 (the *gradual* budget rule; the binary rule only holds near the
  threshold) say a starving forager should accept negative-EV gambles. Our policy does the opposite:
  it refuses trips it cannot *prove* it can complete. Layer on §4.2 — score ends only at population
  zero, so an individual death is free unless it is the last one.
- **Numbers it moves.** The **10.6 % of hungry ticks** spent refusing food; the **median shortfall
  of 6 energy**; the **~20 deaths per run beside known food**; starvation share of deaths (**59 %**).
- **Concrete rule.** Replace the binary affordability gate with a graded one:
  accept a trip if `P(reach) > f(runway, population)`, where f → 0 as population → 1. At population
  1, accept **any** reachable-in-principle target. At population ≥ 5, keep the current strict gate.
- **Null risk: MODERATE.** 20 deaths/run sounds large, but most are not terminal — they only cost
  score when they are the last agent. The population-conditioning is what converts this from a
  cosmetic fix into an absorption-time fix, and it is also the part with **no** published support
  (it is my derivation from Claisse & Champagnat's QSD-control framing). Test the
  population-conditioned version, not the reserves-only version; the reserves-only version is the
  one most likely to be null.

### 3. Time-varying MVT threshold (replace the `MOVE_ON` constant with a tracked R\*)

- **Mechanism.** Charnov 1976: the leaving threshold *is* the environment's current average rate.
  Ours halves every 300 s. Calcagno et al. 2013 give the sensitivity analysis and identify
  conditions where **staying longer on poorer patches** becomes adaptive as the habitat degrades.
  Implementation is an exponential moving average of realised hive-wide intake per second, exactly
  as Constantino & Daw 2015 and Davidson & El Hady 2019 describe.
- **Numbers it moves.** Nothing we currently instrument. It would show up in the late game — we
  measured that agents walk ~30 % of the time in the last 120 s.
- **Null risk: HIGH — flag this as likely null.** We have already measured that (a) MOVE_ON
  constants are flat inside noise, (b) the tree map delivers **84.6 %** of meals at a **2.2 s**
  sight-to-visit latency, so there is no backlog of known-but-unexploited patches, and (c)
  Calcagno et al.'s own **homogeneous-habitat scaling invariance** predicts that a purely
  multiplicative global decay should leave the optimal policy *unchanged*. Three independent
  reasons to expect nothing. Worth one cheap test only because the time-varying form has never been
  tested, and because the EMA is ~5 lines.

### 4. Count-based vs interval-based giving-up rule

- **Mechanism.** Kilpatrick & El Hady, arXiv:2607.29476: the best departure rule **switches from
  counting items to timing the gaps between them** once patch richness varies by more than about a
  quarter. Our tree quality varies by far more than that — biome `fruit_spawn_rate` spans
  0.0/0.05/0.08/0.1, a 2× spread among productive biomes and a hard zero in rivers.
- **Numbers it moves.** Would show up in patch dwell time and in the fraction of dwell-seconds spent
  at unproductive trees (our `PRIOR_EVIDENCE` counter already measures fruitless dwell).
- **Null risk: MODERATE-HIGH.** It is a re-parameterisation of a decision we already make, and the
  FRUIT_PRIOR experiment (#16) showed that better tree *valuation* has almost no headroom. But this
  is a different claim from FRUIT_PRIOR: it is about the *statistic* used to decide departure, not
  about the prior on yield. Cheap to test; test it only after #1 and #2.

### 5. Do not re-run: dispersion, scouts, or more intra-hive information sharing

Recorded here so the theory-backed explanation is on file rather than being rediscovered.

- **Dispersion / soft repulsion (our experiment #15).** Sutherland 1983: spreading pays only through
  the interference constant *m*. At 3–8 agents on a 1600 × 1200 map with mean separation 459 units,
  **m ≈ 0**, so IFD predicts **zero** payoff. Our measurement (coverage +2 points, score
  unresolvable) is exactly the predicted outcome. Additionally, Kennedy & Gray 1993 show that even
  where IFD applies, real distributions **undermatch** — the theory's own effect size is smaller
  than its idealisation.
- **Dedicated scouts (our experiment #17).** Kilpatrick & El Hady 2026: **composition learning is
  set by the number of patches sampled, not the time spent in each, and is unaffected by depletion
  once rates are known.** More ground covered by one agent does not buy environment knowledge. This
  is the same wall we measured from the retention side (TREE_MEMORY 70 s; knowledge is a decaying
  stock). Furthermore, in an environment where patches **replenish**, the same paper finds the
  reward-maximising policy **collapses onto a stable orbit over a rich subset** — i.e. traplining,
  not exploring. Our hive already does this.
- **More information sharing between hive agents.** Bidari et al., PRR 4:033128 (2022): **minimal
  pulsatile coupling matches or beats continuous diffusive coupling.** We already run at the
  strong-coupling limit (one controller, one fused map). There is no headroom above full sharing.
  The −642 visibility ablation is about *sensing*, not *sharing*.

### 6. Do not attempt a learned controller under the current simulator throughput

- **Evidence.** Wispinski et al. (TMLR 2023): **1.2 × 10⁸ steps and ~13 days on 8 GPUs per agent**,
  on a **two-patch** task with the patch value made directly observable, to match a closed-form
  optimum. Papoudakis et al. (NeurIPS 2021 D&B): **2 × 10⁶ off-policy / 2 × 10⁷ on-policy steps** on
  a ~0.1 ms/step **gridworld**, **138,916 CPU hours** in total, with **five of nine algorithms
  showing no learning at all** on the sparse-reward variant.
- **Our budget.** 30–350 s per 30,000-step episode. The Papoudakis on-policy figure alone is
  ~667 episodes per seed per configuration — 5.5 to 65 hours of wall clock **per seed**, before
  5 seeds and before hyperparameter search.
- **Regime argument.** Keddouri et al. (arXiv:2607.04470) classify environments by baseline
  competence and report that in the **supplementary regime** (strong baseline) learned augmentation
  "preserves performance while unstabilised shaping adds variance without gain". Our hive sits at
  772 against a 1141 perfect-information oracle — the supplementary regime.
- **If this is ever revisited, the prerequisite is throughput, not algorithm.** Every feasible
  budget in this literature came from a faster simulator (Neural MMO 2.0's 3× rewrite and the 4×
  score improvement in 8 GPU-hours; JaxRobotarium's 20×/150×; PufferLib). Our own note that env
  creation was once ~10 s because of 1.92 M `Surface.set_at` calls is the same category of problem.

### 7. Sensing radius as an optimisation target — the only idea aligned with our largest ablation

- **Mechanism.** Our oracle ladder says **visibility −642**, dwarfing predator state (−106.5), fruit
  value (−47.6) and tree age (−0.6). Liao, Wu & Wu (AAMAS 2025, arXiv:2505.12811) is the only paper
  I found that treats **sight range as the variable to optimise**, and it improves three benchmarks
  across two algorithms while *accelerating* training.
- **Why this is listed last despite the biggest ablation.** We have already established that traits
  **do not evolve within a run** (after 45 births, hearing was still exactly 50.0 and vision exactly
  200.0), so we cannot breed the sensor. The sensing budget is fixed by the simulator. The only
  remaining lever is *inference* — turning fewer observations into more knowledge — and FRUIT_PRIOR
  (#16) already tested the obvious inference and came back null.
- **Null risk: HIGH, and currently unactionable.** Logged as the standing explanation for why the
  gap to 1291 exists at all: **the leaders beat our omniscient greedy oracle (1141), so their
  advantage is not better use of information we already have.** Nothing in the reviewed literature
  explains a policy that exceeds a perfect-information greedy bound; that gap most likely comes
  from a mechanic or an evaluation-side effect we have not identified, not from foraging theory.

---

## Summary of the single most important cross-theme claim

Theme 4 says the objective is one scalar: the extinction rate of the quasi-stationary distribution,
which for a subcritical process is governed by the **per-capita** energy balance and is **bounded
independently of population size** (Schreiber et al., arXiv:2005.08312). Theme 3 says our largest
measured per-capita leak is a **capacity clamp destroying 22.7 % of all harvested energy**, and that
partial loading is the documented optimum under an efficiency currency (Schmid-Hempel, Kacelnik &
Houston, BES 17:61–66, 1985). Those two facts intersect at a single number: recovering ~0.32
energy/agent/s moves R0 from 0.930 to ~0.959 and multiplies expected survival time by ~1.74. Every
other suggestion in this document is smaller than that, and most of them are flagged likely-null.
