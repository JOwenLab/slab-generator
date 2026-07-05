# Batch: differential-shift test (symmetric slabs)

Purpose: first entries of the termination-difference matrix
ΔD(H) − ΔD(X), ΔE(H) − ΔE(X) at matched thickness, per
PLAN_nv_strain_campaign.md ("Differential framing").

## Jobs (relax, fixed cell at a0 = 3.573641 Å DFT reference)

Matched at 6L:
  C100_2x1_H_6L_sym        (12 C + 4 H)   } symmetric H/H — clean re-baseline;
  C110_1x1_H_6L_sym        (12 C + 4 H)   } the ORIGINAL 6L H runs were
  C111_1x1_H_6L_sym        ( 6 C + 2 H)   } asymmetric (bottom=bare)!
  C100_2x1_bare_6L_sym     (12 C)
  C110_1x1_bare_6L_sym     (12 C)
  C100_1x1_O_ether_6L_sym  ( 6 C + 2 O)
  C110_O_ether_6L_sym      (24 C + 4 O)

## Pandey (111) — REMOVED pending symmetric support
  C111_2x1_pandey_10L_Hbot (20 C + 2 H; H-capped bottom, 2 layers frozen)
  C111_1x1_H_10L_sym       (10 C + 2 H)

## Known caveats

1. The pre-existing C1xx_*_H_6L results are asymmetric (H top / bare bottom):
   their tau and eps* mix two different surfaces. Use the *_sym runs for all
   termination attribution; keep old runs as diagnostics only.
2. C111_2x1_pandey_10L_Hbot is asymmetric by necessity (single stored face).
   Its surface quantities require subtracting the H contribution using
   C111_1x1_H_10L_sym from the same batch. NOTE: slabgen currently emits NO
   dipole-correction flags (tefield/dipfield/edir) for asymmetric slabs —
   known open issue; assess the slab dipole from the relax output before
   trusting the stress numbers.
3. Bare C(111) 1x1 (unreconstructed radical surface) intentionally omitted;
   Pandey 2x1 is the physical bare (111).
4. Bare/Pandey surfaces may have (near-)metallic surface states: MV smearing
   with degauss 0.01 Ry is retained; check occupations and consider nspin=2
   spot-checks if the relax shows instability.

## After relaxations complete (same workflow as the H series)

  python3 run_queue.py --run                # local, one at a time
  python3 make_slab_stress_scf.py --only <name> ...
  python3 make_slab_strain_series.py --mode biaxial \
      --strain -0.01 --strain -0.005 --strain 0.0 --strain 0.005 --strain 0.01 \
      --only <name>_stress_scf ...
  python3 update_slab_analysis.py
  python3 nv_spin_strain.py                 # differential matrix inputs
