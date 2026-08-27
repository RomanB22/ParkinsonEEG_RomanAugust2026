# Analysis modules

All scientific implementation lives under `src/analyses/`.
Users normally run these modules through `bash run_pipeline.sh`; the modules
are separated here so their ownership and dependencies remain easy to follow.

| Module | Purpose | Method notes |
| --- | --- | --- |
| `psd` | Welch PSD and relative band power | [PSD](psd.md) |
| `ordinal` | Shannon, Fisher, and Rényi ordinal quantities | [Ordinal](ordinal.md) |
| `scale_free` | fixed/knee specparam fits and eBOSC bouts | [Scale free](scale_free.md) |
| `bouts` | ordinal quantities within detected bouts | [Within-bout ordinal](bouts.md) |
| `bycycle` | independent burst-detector sensitivity | [Bycycle](bycycle.md) |
| `exploration` | explainable PD classification | [Exploration](exploration.md) |
| `behavioral` | MOCA associations | [Behavioral](behavioral.md) |
| `progression` | MOCA and UPDRS progression axes | [Progression](progression.md) |
| `duration_qc` | accepted-duration sensitivity | [Duration QC](duration_qc.md) |
| `eight_electrode` | prespecified electrode subset | [Eight electrodes](eight_electrode.md) |
| `matching` | shared age/sex matched cohort | [Matching](matching.md) |

Analysis configuration files use the same short names under
`config/analyses/`. Generated artifacts use them under `outputs/full/` and
`outputs/matched/`.
