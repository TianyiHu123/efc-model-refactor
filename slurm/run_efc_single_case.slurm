#!/bin/bash
# =============================================================================
# Example: run the ELM EFC engine for ONE case as a Perlmutter Slurm job.
#
# Convention (AGENTS.md §1): the .slurm script, the Slurm --output/--error logs,
# and the scientific outputs all live together in ONE attempt folder:
#   {output-root}/{site}/{case-name}/{test|operational}/attempt{N}/
# Because the engine would otherwise auto-select attempt{N}, we PRE-CREATE the
# attempt dir, copy this script there, submit from there, and pass the same dir
# to the engine via --output-dir.
#
# Stage 2 (prediction) is CPU-bound and embarrassingly parallel; --jobs sets the
# RandomForest thread count to the cores allocated here. Operational runs must
# go through Slurm (not the login node). One case per job; use a job array for
# many sites (not included here).
#
# Typical usage (helper does the pre-create + copy + submit):
#   see the "prepare & submit" snippet at the bottom of this file.
# =============================================================================
#SBATCH --account=m4803
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48GB
#SBATCH --time=00:30:00
#SBATCH --job-name=efc_single
#SBATCH --output=efc_%x_%j.out
#SBATCH --error=efc_%x_%j.err

set -euo pipefail

# --- interpreter (OLMT Perlmutter env; has xarray/sklearn/joblib/...) --------
readonly PYTHON=/global/common/software/m4803/conda/envs/OLMT_pm_TianyiPY311/bin/python

# --- repo root (directory containing run_efc.py) -----------------------------
readonly REPO_ROOT=/pscratch/sd/t/tianyihu/efc-model-refactor

# --- case definition (override via environment; defaults = verified ABBY) ----
SITE="${SITE:-ABBY}"
CASE_NAME="${CASE_NAME:-ABBY_ctrlvertsoc}"
CONFIG="${CONFIG:-/pscratch/sd/t/tianyihu/E3SM_out/SOIL_project/NEON_ctrl/ctrlvertsoc/${SITE}/config/${CASE_NAME}.cfg}"
RUN_DIR="${RUN_DIR:-/pscratch/sd/t/tianyihu/E3SM_out/SOIL_project/NEON_ctrl/ctrlvertsoc/${SITE}/${CASE_NAME}_I20TRCNPRDCTCBC/run}"
NEON_CSV_ROOT="${NEON_CSV_ROOT:-/pscratch/sd/t/tianyihu/CTSM_Hu/neon_surffiles}"
COMPSET="${COMPSET:-I20TRCNPRDCTCBC}"

# Attempt dir: this script runs FROM it (sbatch submitted there), so logs land
# here too. Pass it to the engine so scientific outputs co-locate.
readonly OUTPUT_DIR="${OUTPUT_DIR:-$SLURM_SUBMIT_DIR}"

# Match RandomForest threads to the cores Slurm gave us.
readonly JOBS="${SLURM_CPUS_PER_TASK:-1}"

echo "[slurm] job ${SLURM_JOB_ID:-<none>} on $(hostname); cpus-per-task=${JOBS}"
echo "[slurm] site=${SITE} case=${CASE_NAME} output_dir=${OUTPUT_DIR}"

cd "${REPO_ROOT}"

/usr/bin/time -v "${PYTHON}" run_efc.py \
  --site "${SITE}" \
  --case-name "${CASE_NAME}" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --neon-csv-root "${NEON_CSV_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --compset "${COMPSET}" \
  --mode operational \
  --use-vertsoilc true \
  --stage all \
  --jobs "${JOBS}"

echo "[slurm] done."

# =============================================================================
# Prepare & submit (run on a login node), picking the next free attempt dir:
#
#   OUT=/pscratch/sd/t/tianyihu/efc-prediction/ABBY/ABBY_ctrlvertsoc/operational
#   N=1; while [ -e "$OUT/attempt$N" ]; do N=$((N+1)); done
#   ATT="$OUT/attempt$N"; mkdir -p "$ATT"
#   cp slurm/run_efc_single_case.slurm "$ATT/"
#   ( cd "$ATT" && sbatch run_efc_single_case.slurm )
#
# The script, efc_*.out/err logs, and the engine outputs all land in $ATT.
# =============================================================================
