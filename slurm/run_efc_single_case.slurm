#!/bin/bash
# =============================================================================
# Example: run the ELM EFC engine for ONE case as a Perlmutter Slurm job.
#
# Stage 2 (prediction) is CPU-bound and embarrassingly parallel; --jobs sets the
# RandomForest thread count to the cores allocated here. Operational runs must
# go through Slurm (not the login node). One case per job; use a job array for
# many sites (not included here).
#
# Submit:
#   sbatch slurm/run_efc_single_case.slurm
# or override the case via environment variables:
#   SITE=ABBY CASE_NAME=ABBY_ctrlvertsoc \
#   CONFIG=/path/to/ABBY_ctrlvertsoc.cfg \
#   RUN_DIR=/path/to/ABBY_ctrlvertsoc_I20TRCNPRDCTCBC/run \
#   sbatch slurm/run_efc_single_case.slurm
# =============================================================================
#SBATCH --account=m4803
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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
OUTPUT_ROOT="${OUTPUT_ROOT:-/pscratch/sd/t/tianyihu/efc-prediction}"
COMPSET="${COMPSET:-I20TRCNPRDCTCBC}"

# Match RandomForest threads to the cores Slurm gave us.
readonly JOBS="${SLURM_CPUS_PER_TASK:-1}"

echo "[slurm] job ${SLURM_JOB_ID:-<none>} on $(hostname); cpus-per-task=${JOBS}"
echo "[slurm] site=${SITE} case=${CASE_NAME}"

cd "${REPO_ROOT}"

/usr/bin/time -v "${PYTHON}" run_efc.py \
  --site "${SITE}" \
  --case-name "${CASE_NAME}" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --neon-csv-root "${NEON_CSV_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --compset "${COMPSET}" \
  --mode operational \
  --use-vertsoilc true \
  --stage all \
  --jobs "${JOBS}"

echo "[slurm] done."
