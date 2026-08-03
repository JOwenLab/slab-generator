#!/usr/bin/env bash
#
# regenerate.sh - rebuild every derived artifact in this repository from
# committed inputs, in dependency order.
#
# WHY THIS EXISTS
# ---------------
# Several artifacts could only be reproduced with flags that were recorded
# nowhere. Recovering them meant reading the committed output and working
# backwards -- dividing a GHz column by a strain column to find a coupling
# constant, reading a scan's own first row to find the range it was run over.
# An artifact whose invocation is unrecorded is not reproducible in any useful
# sense, however deterministic the code behind it is.
#
# Every command below is the exact one that reproduces the committed file. If a
# command here stops reproducing its artifact byte-for-byte, either an input
# changed or a default moved, and that is worth knowing immediately.
#
# WHAT THIS DOES NOT DO
# ---------------------
# It does not run any DFT. Every pw.out under results/ is an input to this
# script, not an output of it. Regenerating those requires Quantum ESPRESSO,
# the pseudopotentials (not distributed -- see README), and cluster time.
#
# Usage:
#   ./regenerate.sh            # regenerate everything
#   ./regenerate.sh --check    # regenerate, then fail if anything changed
#
set -euo pipefail
cd "$(dirname "$0")"

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

say() { printf '\n=== %s ===\n' "$1"; }

# --------------------------------------------------------------- parsers
say "parse_convergence.py -> results/convergence/convergence_summary.csv"
# Bare default = results/convergence only. Passing results/thickness_stress as
# well is valid and produces a LARGER file (59 rows vs 44); the committed
# artifact is the bare form. The thickness ladder reaches tau_inf through
# fit_tau_infinity.py, which reads the run directories directly.
python3 parse_convergence.py

say "parse_reference.py -> reference summaries"
python3 parse_reference.py --root results/reference_diamond
python3 parse_reference.py --root results/reference_90_720

# ------------------------------------------------------- bulk reference
say "fit_bulk_reference.py -> bulk fits"
# --strain-type any: the 90/720 run folders are named <prefix>~eps_<value>, so
# parse_reference records the folder name in strain_type rather than
# "hydrostatic", and the default filter would drop every row.
python3 fit_bulk_reference.py \
    --input results/reference_90_720/reference_summary.csv \
    --outdir results/reference_90_720 --strain-type any
python3 fit_bulk_reference.py \
    --input results/reference_diamond/reference_summary.csv \
    --outdir results/reference_diamond
# NOTE: config/reference_pbe_sssp.json is NOT rewritten here. Updating it needs
# --update-config plus --supersede-reason, and overwriting the live a0/B is a
# deliberate act, not a side effect of regenerating reports.

# ------------------------------------------------------- slab pipeline
say "analyze_slab_stress.py -> results/slabs/slab_stress_summary.*"
python3 analyze_slab_stress.py

say "fit_slab_strain.py -> results/slabs/slab_strain_fit_summary.*"
python3 fit_slab_strain.py

# --------------------------------------------------------- surface stress
say "fit_tau_infinity.py -> results/production/tau_infinity*"
python3 fit_tau_infinity.py --runs-dir results/production --out-dir results/production

say "layer_profile.py -> results/production/layer_profile*"
python3 layer_profile.py

say "surface_energy.py -> reports AND config/surface_energies_h.json"
# --write-config is REQUIRED: a bare run writes the reports but leaves the
# config untouched, so the config silently keeps whatever it had.
python3 surface_energy.py --write-config

# ------------------------------------------------------------ particles
say "particle_strain.py -> results/production/particle_strain*"
for shape in octahedron cube rhombic-dodecahedron mixture; do
    python3 particle_strain.py --shape "$shape"
done
# mixture takes no --fraction: without one it derives the facet fractions from
# the surface energies (stability-weighted, offset-invariant). wulff is omitted
# on purpose -- it refuses, because gamma < 0 across the whole allowed mu_H
# range makes the equilibrium shape undefined.

say "particle_strain.py --scan-mu-h -> mu_H scan and sign-change curve"
# The committed scan is 0.0 -> 3.0 eV in 121 points. The lower bound is clamped
# internally to the delta_mu at which a Wulff construction first exists
# (+0.336793 eV), which is why the output starts there and not at 0.
python3 particle_strain.py --scan-mu-h 0.0 3.0 --scan-points 121

# ------------------------------------------------------------------ NV
say "nv_spin_strain.py -> results/nv/nv_predictions*"
python3 nv_spin_strain.py

say "nv_strain_model.py -> results/nv/nv_strain_*"
# The two coupling constants are CLI-only. A bare invocation nulls
# estimated_delta_D_GHz and estimated_E_GHz -- it does not fail, it silently
# drops two columns. 13 and 5 GHz/strain are the provisional values recorded in
# CLAUDE.md section 3; they are literature order-of-magnitude figures, not
# calibrated, and the emitted report now records them.
python3 nv_strain_model.py \
    --d-shift-ghz-per-strain 13 \
    --e-splitting-ghz-per-strain 5

say "nv_local_strain.py -> results/production/nv_local_strain_*"
# Position-resolved interior strain. Needs scikit-fem; skipped rather than
# failed if it is absent, since every other artifact is independent of it.
if python3 -c "import skfem" 2>/dev/null; then
    python3 nv_local_strain.py --refine 3 --depths 0.5 1.0 1.5
else
    echo "SKIPPED: scikit-fem not installed (pip install -r requirements.txt)"
fi

# -------------------------------------------------------------- figures
say "plot_convergence.py + plot_surface_stress.py -> results/figures/"
python3 plot_convergence.py
# N1 and N2 are retired here, not merely unbuilt: this run also DELETES their
# stale outputs from results/figures/surface-stress/. N1 is absorbed into F1(b)
# below; N2 asserted that the (100) anisotropy drives the E channel, which
# nv_local_strain.py contradicts.
python3 plot_surface_stress.py

say "plot_h_results.py -> results/figures/h-results/ (F1-F5)"
# The H-terminated result set. F1(a) is a deliberately empty slot: rendering
# the relaxed structures needs VESTA or OVITO and neither is a dependency of
# this repository, so the panel is marked rather than faked. F2's
# dehydrogenation ceiling is likewise not drawn -- it needs a bare-facet ladder
# that is not committed here. Both are reported as GAPs on stdout.
python3 plot_h_results.py

# --------------------------------------------------------------- tables
say "make_tables.py -> results/tables/ (T1-T4, markdown + booktabs)"
# T1's convergence column is read from plot_convergence's own reductions, so
# the table cannot disagree with the figures. T3's dehydrogenation-ceiling
# column is NOT COMPUTED for the same missing-ladder reason as F2.
python3 make_tables.py

# ----------------------------------------------------------------- gate
say "preflight.py -> structural gate (advisory here, not a rebuild)"
# --skip pseudopotentials_exist: UPF files are deliberately not committed, so
# an archived results directory legitimately fails that one check. Never skip
# it when gating an actual submission.
python3 preflight.py results/production/thick_a0corr~* \
    --skip pseudopotentials_exist -q \
    && echo "production structures pass" \
    || echo "PRODUCTION STRUCTURES FAILED THE GATE"

echo
if [ "$CHECK" = "1" ]; then
    # Two files can never match byte-for-byte and are not evidence of anything:
    #
    #   queue_status.csv is a run log, not a derived artifact. Nothing here
    #   regenerates it.
    #
    #   convergence_summary.csv stamps the git commit into every row. That is
    #   deliberate provenance and worth keeping, but it means the file differs
    #   whenever HEAD moves or the tree is dirty. It is therefore compared with
    #   that one column removed, and only a difference in the REST counts.
    #
    # Exempting them is the point. A check that reports the same two files on
    # every run teaches the reader to skip its output, which is how a real
    # regression gets waived.
    conv=results/convergence/convergence_summary.csv
    changed=$(git status --porcelain \
              | grep -v 'queue_status.csv$' \
              | grep -v "$conv\$" || true)

    if ! diff -q <(git show "HEAD:$conv" | cut -d, -f1-4,6-) \
                 <(cut -d, -f1-4,6- "$conv") >/dev/null 2>&1; then
        changed="$changed
 M $conv (differs BEYOND the git_commit column)"
    fi

    if [ -n "$(printf '%s' "$changed" | tr -d '[:space:]')" ]; then
        echo "REGENERATION CHANGED FILES:"
        printf '%s\n' "$changed"
        echo
        echo "Either an input changed, or a default moved. Both are worth"
        echo "understanding before committing."
        exit 1
    fi
    echo "clean: every artifact regenerated byte-identically"
    echo "  (git_commit column in $conv excluded -- provenance stamp;"
    echo "   queue_status.csv excluded -- run log, not a derived artifact)"
fi
echo "done."
