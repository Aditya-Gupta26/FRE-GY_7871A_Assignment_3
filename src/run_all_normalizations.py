"""Driver: run the full econometric core (omegas -> regressions -> variance
decomposition -> robustness checks) once for each normalizing variable in
config.NORMALIZING_VARIABLES_TO_RUN. Assumes build_master_dataset.py has already been
run (master_panel.parquet exists); the serial-correlation filtering step is shared and
runs once regardless of how many normalizing variables are evaluated.
"""
from config import NORMALIZING_VARIABLES_TO_RUN
import event_study
import run_regressions
import variance_decomposition
import robustness_checks


def main():
    event_study.run_filtering()
    for nv in NORMALIZING_VARIABLES_TO_RUN:
        print(f"\n{'=' * 70}\nNormalizing variable: {nv}\n{'=' * 70}")
        event_study.compute_omegas(nv)
        run_regressions.main(nv)
        variance_decomposition.main(nv)
        robustness_checks.main(nv)


if __name__ == "__main__":
    main()
