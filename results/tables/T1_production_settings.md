# Table T1 - Production settings and what was demonstrated about each

| parameter              | production value             | convergence evidence                                                                                    | level |
| ---------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------- | ----- |
| functional             | PBE                          | read from pw.out, not from the input                                                                    | --    |
| pseudopotential (C)    | C.pbe-n-kjpaw_psl.1.0.0.UPF  | as opened by pw.x (CLAUDE.md invariant 7)                                                               | --    |
| pseudopotential (H)    | H_ONCV_PBE-1.0.oncvpsp.upf   | as opened by pw.x (CLAUDE.md invariant 7)                                                               | --    |
| ecutwfc                | 90 Ry                        | vs densest cutoff computed: 0.07 kbar in mean sigma, 0.01 kbar in anisotropy                            | L2    |
| ecutrho                | 720 Ry                       | ecutrho/ecutwfc = 8                                                                                     | L2    |
| k-mesh, (100)          | 9x5x1 (n_k a = 22.7, 25.3 A) | 20.2 A sits within 0.035 kbar of the densest swept (50.5 A) (production is finer than this swept point) | L2    |
| k-mesh, (110)          | 9x7x1 (n_k a = 22.7, 25.0 A) | 22.7 A sits within 0.010 kbar of the densest swept (37.9 A)                                             | L2    |
| k-mesh, (111)          | 9x9x1 (n_k a = 22.7 A)       | 22.7 A sits within 0.030 kbar of the densest swept (45.5 A)                                             | L2    |
| vacuum (thinnest slab) | 9.8 A                        | tau moves 0.0024 N/m over 8-24 A; sigma alone is not converged                                          | L2    |
| thickness ladder       | 8L 10L 12L 16L               | 6L excluded as outside the asymptotic regime                                                            | L2    |
| lattice constant a0    | 3.572997 A                   | birch murnaghan 3rd order EV, B = 433.66 GPa                                                            | L2    |
| smearing               | Marzari-Vanderbilt, 0.01 Ry  | metallic-style smearing on an insulator; a numerical device, no occupation is fractional                | --    |
| SCF threshold          | 1e-08 Ry                     | tighter than the default 1e-6 because stress converges after energy                                     | --    |
| stress run type        | scf                          | single point on the relaxed geometry; ions are NOT re-relaxed                                           | --    |

**Table T1.** Production settings for the 15 stress SCFs that the surface-stress results are built from, together with what was demonstrated about each one. The 'evidence' column is the point of the table: a setting with no sweep behind it says so rather than being listed as though it were converged. Values are read from the committed pw.in and pw.out of the runs themselves -- pseudopotentials from pw.out, which is ground truth for the file pw.x actually opened. The convergence numbers are taken from the same reduction that draws the convergence figures, so table and figure cannot disagree. Levels are per CLAUDE.md section 4 and apply to the CONVERGENCE CLAIM in that row, not to the setting.


- Positive sigma means the cell is COMPRESSED (CLAUDE.md sec 2).
- Smearing and threshold rows carry no level: they are inputs, not claims.
