"""Print progress of the running Optuna study."""

import os
import sqlite3
import sys

DEFAULT_DB = "optuna_s6.db"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    db = os.path.join(here, sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB)
    if not os.path.exists(db):
        print(f"no study database at {db}")
        return
    con = sqlite3.connect("file:" + db + "?mode=ro", uri=True)
    rows = con.execute(
        "select t.number, t.state, v.value from trials t "
        "left join trial_values v on v.trial_id = t.trial_id order by t.number"
    ).fetchall()
    done = [(n, v) for n, s, v in rows if v is not None]
    print(f"trials recorded: {len(rows)}   completed: {len(done)}")
    for n, s, v in rows[-15:]:
        print(f"  {n:3d} {s:10s} {'' if v is None else round(v, 1)}")
    if done:
        best_n, best_v = max(done, key=lambda r: r[1])
        print(f"\nbest so far: trial {best_n} = {best_v:.1f}")
        params = con.execute(
            "select p.param_name, p.param_value, p.distribution_json from trial_params p "
            "join trials t on t.trial_id = p.trial_id where t.number = ?", (best_n,)
        ).fetchall()
        for name, value, dist in sorted(params):
            kind = "int" if '"IntDistribution"' in dist else "float"
            print(f"  {name} = {int(value) if kind == 'int' else round(value, 3)}")
    top = sorted(done, key=lambda r: -r[1])[:8]
    print("\ntop trials: " + ", ".join(f"#{n}={v:.0f}" for n, v in top))

    names = [r[0] for r in con.execute(
        "select distinct param_name from trial_params order by param_name")]
    print("\n" + "  ".join(f"{n[:11]:>11s}" for n in ["trial", "score"] + names))
    for n, v in top:
        row = dict(con.execute(
            "select p.param_name, p.param_value from trial_params p "
            "join trials t on t.trial_id = p.trial_id where t.number = ?", (n,)).fetchall())
        cells = "  ".join(f"{row.get(k, float('nan')):11.2f}" for k in names)
        print(f"{n:>11d}  {v:11.1f}  {cells}")


if __name__ == "__main__":
    sys.exit(main())
