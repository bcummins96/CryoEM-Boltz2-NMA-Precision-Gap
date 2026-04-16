#!/usr/bin/env python3
"""
5-HT2A NMA Replicate Variance — TM-Core vs Full Structure Comparison
Benjamin Cummins

Runs the replicate variance analysis TWICE:
  1. Full structure (all common residues — reproduces previous result)
  2. TM-core only (structured helical regions, excluding disordered
     N-term, C-term, and ICL3)

If the between-ligand noise was driven by disordered loop variance,
the TM-core analysis should show within-ligand RMSIP > between-ligand.

Requirements:
    pip install prody matplotlib numpy scipy seaborn
"""

import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings('ignore')

from prody import (
    parsePDB, parseMMCIF, writePDB,
    ANM, GNM, calcSqFlucts, calcSubspaceOverlap,
    calcRMSD, superpose, confProDy
)

confProDy(verbosity='warning')



# CONFIGURATION


OUTPUT_DIR = "5HT2A_NMA_TM_rescue"

ANM_CUTOFF = 15.0
N_MODES = 20
RMSIP_MODES = 10
N_REPLICATES = 5

REPLICATE_DIR = r"C:\Users\Ben\boltz2_replicates_firstrun_nosteering"

# ── TM Region Definitions ──────────────────────────────────────────────
# Human 5-HT2A (UniProt P28223) TM boundaries.
# These define the STRUCTURED CORE — everything else is excluded in
# the TM-trimmed analysis.
#
# Adjust these if Boltz2 structures use different
# residue numbering (e.g., starting from 1 instead of UniProt numbering).
#
# If Boltz2 numbers from 1 (full sequence = 471 aa), these UniProt
# numbers are correct since the full sequence starts at Met1.

TM_CORE_RESIDUES = (
    list(range(75, 103))   +   # TM1
    list(range(103, 109))  +   # ICL1 (short, structured)
    list(range(109, 141))  +   # TM2
    list(range(141, 154))  +   # ECL1
    list(range(154, 188))  +   # TM3
    list(range(188, 200))  +   # ICL2 (G-protein contact, structured)
    list(range(200, 229))  +   # TM4
    list(range(229, 245))  +   # ECL2 (structured lid)
    list(range(245, 278))  +   # TM5
    # ICL3 (278-310) EXCLUDED — disordered in most structures
    list(range(311, 346))  +   # TM6
    list(range(346, 352))  +   # ECL3
    list(range(352, 379))  +   # TM7
    list(range(379, 396))      # H8
)
# N-terminus (1-74) EXCLUDED
# ICL3 (278-310) EXCLUDED
# C-terminus (396-471) EXCLUDED

TM_CORE_SET = set(TM_CORE_RESIDUES)

LIGANDS = [
    {'name': 'Serotonin',    'subdir': 'serotonin',    'pattern': '*.cif', 'color': '#e74c3c'},
    {'name': 'Psilocin',     'subdir': 'psilocin',     'pattern': '*.cif', 'color': '#c0392b'},
    {'name': 'DMT',          'subdir': 'dmt',           'pattern': '*.cif', 'color': '#e67e22'},
    {'name': 'LSD',          'subdir': 'lsd',           'pattern': '*.cif', 'color': '#9b59b6'},
    {'name': 'BOL-148',      'subdir': 'bol148',        'pattern': '*.cif', 'color': '#8e44ad'},
    {'name': 'Lisuride',     'subdir': 'lisuride',      'pattern': '*.cif', 'color': '#6c3483'},
    {'name': '25CN-NBOH',    'subdir': '25cn_nboh',     'pattern': '*.cif', 'color': '#27ae60'},
    {'name': 'Mescaline',    'subdir': 'mescaline',     'pattern': '*.cif', 'color': '#2ecc71'},
    {'name': 'RS130-180',    'subdir': 'rs130180',      'pattern': '*.cif', 'color': '#1abc9c'},
    {'name': 'Risperidone',  'subdir': 'risperidone',   'pattern': '*.cif', 'color': '#2c3e50'},
    {'name': 'Methiothepin', 'subdir': 'methiothepin',  'pattern': '*.cif', 'color': '#7f8c8d'},
]



# HELPER FUNCTIONS


def load_structure(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in ('.cif', '.mmcif'):
            structure = parseMMCIF(filepath)
        else:
            structure = parsePDB(filepath)
    except Exception as e:
        return None
    if structure is None or structure.numAtoms() == 0:
        return None
    all_ca = structure.select('protein and name CA')
    if all_ca is None:
        return None
    chains = sorted(set(all_ca.getChids()))
    best_chain, best_count = None, 0
    for ch in chains:
        sel = all_ca.select(f'chain {ch}')
        if sel is not None and len(sel) > best_count:
            best_count = len(sel)
            best_chain = ch
    if best_chain and best_count >= 150:
        return all_ca.select(f'chain {best_chain}')
    elif len(all_ca) >= 150:
        return all_ca
    return None


def find_common_residues(all_atoms_list):
    residue_sets = [set(atoms.getResnums()) for atoms in all_atoms_list]
    common = residue_sets[0]
    for rs in residue_sets[1:]:
        common = common.intersection(rs)
    return sorted(common)


def trim_to_residues(atoms, residue_list):
    resnum_str = ' '.join(str(r) for r in residue_list)
    return atoms.select(f'resnum {resnum_str}')


def run_anm(atoms, label):
    anm = ANM(label)
    anm.buildHessian(atoms, cutoff=ANM_CUTOFF)
    anm.calcModes(n_modes=N_MODES, zeros=False)
    return anm


def run_variance_analysis(trimmed_structures, analysis_label):
    """
    Core variance analysis: runs ANM, computes within/between RMSIP,
    runs statistics. Returns a results dict.
    """
    print(f"\n  Running ANM...")
    anm_results = {}
    for name, reps in trimmed_structures.items():
        anms = []
        for i, atoms in enumerate(reps):
            anm = run_anm(atoms, f"{name}_rep{i+1}")
            anms.append(anm)
        anm_results[name] = anms

    ligand_names = list(anm_results.keys())

    # Within-ligand RMSIP
    within_rmsip = {}
    for name, anms in anm_results.items():
        pairs = []
        for i in range(len(anms)):
            for j in range(i + 1, len(anms)):
                ov = calcSubspaceOverlap(anms[i][:RMSIP_MODES], anms[j][:RMSIP_MODES])
                pairs.append(ov)
        within_rmsip[name] = pairs

    # Between-ligand RMSIP
    between_rmsip = []
    between_detail = {}
    for i, name_a in enumerate(ligand_names):
        for j, name_b in enumerate(ligand_names):
            if i >= j:
                continue
            pair_vals = []
            for anm_a in anm_results[name_a]:
                for anm_b in anm_results[name_b]:
                    ov = calcSubspaceOverlap(anm_a[:RMSIP_MODES], anm_b[:RMSIP_MODES])
                    pair_vals.append(ov)
                    between_rmsip.append(ov)
            between_detail[(name_a, name_b)] = pair_vals

    all_within = []
    for vals in within_rmsip.values():
        all_within.extend(vals)

    # Statistics (built in! wow-wee!)
    u_stat, p_value = stats.mannwhitneyu(all_within, between_rmsip, alternative='greater')
    pooled_std = np.sqrt((np.std(all_within)**2 + np.std(between_rmsip)**2) / 2)
    cohens_d = (np.mean(all_within) - np.mean(between_rmsip)) / pooled_std if pooled_std > 0 else 0

    results = {
        'label': analysis_label,
        'within_rmsip': within_rmsip,
        'between_rmsip': between_rmsip,
        'between_detail': between_detail,
        'all_within': all_within,
        'anm_results': anm_results,
        'ligand_names': ligand_names,
        'u_stat': u_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'within_mean': np.mean(all_within),
        'within_std': np.std(all_within),
        'between_mean': np.mean(between_rmsip),
        'between_std': np.std(between_rmsip),
    }

    print(f"  {analysis_label}:")
    print(f"    Within-ligand:  {results['within_mean']:.4f} ± {results['within_std']:.4f}")
    print(f"    Between-ligand: {results['between_mean']:.4f} ± {results['between_std']:.4f}")
    print(f"    Mann-Whitney p = {p_value:.2e}, Cohen's d = {cohens_d:.3f}")

    return results



# PLOTTING


def plot_comparison(results_full, results_tm):
    """Side-by-side comparison figure."""
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel A: Full structure distributions ──
    ax = axes[0]
    r = results_full
    ax.hist(r['all_within'], bins=25, alpha=0.4, color='#2196F3', density=True,
            edgecolor='white', linewidth=0.5, label='Within-ligand')
    ax.hist(r['between_rmsip'], bins=25, alpha=0.4, color='#FF5722', density=True,
            edgecolor='white', linewidth=0.5, label='Between-ligand')
    ax.axvline(r['within_mean'], color='#1565C0', linestyle='--', linewidth=1.5)
    ax.axvline(r['between_mean'], color='#D84315', linestyle='--', linewidth=1.5)
    ax.set_xlabel('RMSIP', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Full Structure\np = {r["p_value"]:.2e}, d = {r["cohens_d"]:.2f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(0.3, 1.0)

    # ── Panel B: TM-core distributions ──
    ax = axes[1]
    r = results_tm
    ax.hist(r['all_within'], bins=25, alpha=0.4, color='#2196F3', density=True,
            edgecolor='white', linewidth=0.5, label='Within-ligand')
    ax.hist(r['between_rmsip'], bins=25, alpha=0.4, color='#FF5722', density=True,
            edgecolor='white', linewidth=0.5, label='Between-ligand')
    ax.axvline(r['within_mean'], color='#1565C0', linestyle='--', linewidth=1.5)
    ax.axvline(r['between_mean'], color='#D84315', linestyle='--', linewidth=1.5)
    ax.set_xlabel('RMSIP', fontsize=10)
    ax.set_title(f'TM Core Only\np = {r["p_value"]:.2e}, d = {r["cohens_d"]:.2f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(0.3, 1.0)

    # Color the title based on significance
    if r['p_value'] < 0.001:
        axes[1].title.set_color('#1B5E20')

    # ── Panel C: Per-ligand box plots for TM-core ──
    ax = axes[2]
    lig_names_sorted = sorted(r['within_rmsip'].keys(),
                               key=lambda x: np.mean(r['within_rmsip'][x]) if r['within_rmsip'][x] else 0,
                               reverse=True)
    box_data = [r['within_rmsip'][name] for name in lig_names_sorted if r['within_rmsip'][name]]
    box_labels = [name for name in lig_names_sorted if r['within_rmsip'][name]]
    box_colors = []
    lig_color_map = {lig['name']: lig['color'] for lig in LIGANDS}
    for name in box_labels:
        box_colors.append(lig_color_map.get(name, '#999999'))

    bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(box_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Within-Ligand RMSIP', fontsize=10)
    ax.set_title('TM Core Replicate Consistency', fontsize=11, fontweight='bold')
    ax.axhline(r['between_mean'], color='#D84315', linestyle='--', linewidth=1,
               label=f'Between mean ({r["between_mean"]:.3f})')
    ax.legend(fontsize=8, loc='lower left')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures',
                             'TM_rescue_comparison.png'), dpi=300)
    plt.close(fig)
    print("  Saved: TM_rescue_comparison.png")


def plot_between_ligand_heatmap(results, suffix=""):
    """Mean between-ligand RMSIP heatmap from replicate averages."""
    r = results
    names = r['ligand_names']
    n = len(names)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = np.mean(r['within_rmsip'][names[i]]) if r['within_rmsip'][names[i]] else 1.0
        for j in range(i+1, n):
            key = (names[i], names[j]) if (names[i], names[j]) in r['between_detail'] else (names[j], names[i])
            if key in r['between_detail']:
                val = np.mean(r['between_detail'][key])
            else:
                val = 0
            matrix[i, j] = val
            matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=names, yticklabels=names,
                annot=True, fmt='.3f', cmap='RdYlBu_r',
                vmin=0.4, vmax=1.0, square=True, ax=ax,
                cbar_kws={'label': 'Mean RMSIP (10 modes)'})
    ax.set_title(f'Replicate-Averaged RMSIP — {r["label"]}',
                 fontsize=12, fontweight='bold')
    plt.xticks(fontsize=8, rotation=45, ha='right')
    plt.yticks(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures',
                             f'mean_rmsip_heatmap_{suffix}.png'), dpi=300)
    plt.close(fig)
    print(f"  Saved: mean_rmsip_heatmap_{suffix}.png")



# MAIN


def main():
    print("=" * 70)
    print("  5-HT2A NMA — TM-Core Rescue Analysis")
    print("=" * 70)

    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'data'), exist_ok=True)

    # ── Step 1: Load all structures ──
    print("\n[Step 1] Loading replicate structures...")
    all_structures = {}
    all_flat = []

    for lig in LIGANDS:
        name = lig['name']
        search_dir = os.path.join(REPLICATE_DIR, lig['subdir'])
        files = sorted(glob.glob(os.path.join(search_dir, lig['pattern'])))
        if not files:
            flat_pattern = os.path.join(REPLICATE_DIR, f"{lig['subdir']}*")
            files = sorted(glob.glob(flat_pattern))

        print(f"  {name}: {len(files)} files found")
        structures = []
        for i, fpath in enumerate(files[:N_REPLICATES]):
            atoms = load_structure(fpath)
            if atoms is not None:
                structures.append(atoms)
                all_flat.append((name, i, atoms))
        if len(structures) >= 2:
            all_structures[name] = structures

    # ── Step 2: Common residues ──
    print(f"\n[Step 2] Finding common residues...")
    all_atoms_list = [atoms for _, _, atoms in all_flat]
    common_residues = find_common_residues(all_atoms_list)
    print(f"  Full common residues: {len(common_residues)} "
          f"(range {common_residues[0]}-{common_residues[-1]})")

    # TM-core residues = intersection of common residues with TM regions
    tm_residues = sorted([r for r in common_residues if r in TM_CORE_SET])
    print(f"  TM-core residues:     {len(tm_residues)} "
          f"(range {tm_residues[0]}-{tm_residues[-1]})")
    print(f"  Excluded:             {len(common_residues) - len(tm_residues)} residues "
          f"(N-term, ICL3, C-term)")

    # ── Step 3: Prepare both datasets ──
    print(f"\n[Step 3] Preparing structures for both analyses...")
    ref_name = list(all_structures.keys())[0]

    # Full dataset
    full_structures = {}
    ref_full = trim_to_residues(all_structures[ref_name][0], common_residues)
    for name, reps in all_structures.items():
        trimmed = []
        for i, atoms in enumerate(reps):
            t = trim_to_residues(atoms, common_residues)
            if t is not None and len(t) == len(common_residues):
                if not (name == ref_name and i == 0):
                    superpose(t, ref_full)
                trimmed.append(t)
        full_structures[name] = trimmed
    print(f"  Full structures prepared: {sum(len(v) for v in full_structures.values())} total")

    # TM-core dataset
    tm_structures = {}
    ref_tm = trim_to_residues(all_structures[ref_name][0], tm_residues)
    for name, reps in all_structures.items():
        trimmed = []
        for i, atoms in enumerate(reps):
            t = trim_to_residues(atoms, tm_residues)
            if t is not None and len(t) == len(tm_residues):
                if not (name == ref_name and i == 0):
                    superpose(t, ref_tm)
                trimmed.append(t)
        tm_structures[name] = trimmed
    print(f"  TM-core structures prepared: {sum(len(v) for v in tm_structures.values())} total")

    # ── Step 4: Run both analyses ──
    print(f"\n{'='*70}")
    print(f"  ANALYSIS A: FULL STRUCTURE ({len(common_residues)} residues)")
    print(f"{'='*70}")
    results_full = run_variance_analysis(full_structures, f"Full ({len(common_residues)} res)")

    print(f"\n{'='*70}")
    print(f"  ANALYSIS B: TM CORE ONLY ({len(tm_residues)} residues)")
    print(f"{'='*70}")
    results_tm = run_variance_analysis(tm_structures, f"TM Core ({len(tm_residues)} res)")

    # ── Step 5: Comparison ──
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'Metric':30s}  {'Full':>12s}  {'TM Core':>12s}")
    print(f"  {'─'*30}  {'─'*12}  {'─'*12}")
    print(f"  {'Residues':30s}  {len(common_residues):12d}  {len(tm_residues):12d}")
    print(f"  {'Within-ligand RMSIP':30s}  {results_full['within_mean']:12.4f}  {results_tm['within_mean']:12.4f}")
    print(f"  {'Between-ligand RMSIP':30s}  {results_full['between_mean']:12.4f}  {results_tm['between_mean']:12.4f}")
    print(f"  {'Separation (W - B)':30s}  {results_full['within_mean']-results_full['between_mean']:+12.4f}  {results_tm['within_mean']-results_tm['between_mean']:+12.4f}")
    print(f"  {'Mann-Whitney p':30s}  {results_full['p_value']:12.2e}  {results_tm['p_value']:12.2e}")
    cd = "Cohen's d"
    print(f"  {cd:30s}  {results_full['cohens_d']:12.3f}  {results_tm['cohens_d']:12.3f}")

    # ── Verdict ──
    print(f"\n  VERDICT:")
    if results_tm['p_value'] < 0.001 and results_tm['cohens_d'] > 0.5:
        print(f"  ✓ TM-core trimming RESCUES ligand-specific signal!")
        print(f"    Disordered regions were the primary noise source.")
        print(f"    Between-ligand dynamics from Boltz2 are interpretable")
        print(f"    when restricted to the structured TM core.")
    elif results_tm['p_value'] < 0.05:
        print(f"  ~ TM-core trimming shows a MARGINAL improvement.")
        print(f"    Some signal may be present but effect size is modest.")
    else:
        print(f"  ✗ TM-core trimming does NOT rescue the signal.")
        print(f"    Prediction noise extends into the structured core,")
        print(f"    not just disordered loops.")

    # ── Step 6: Figures ──
    print(f"\n[Step 6] Generating figures...")
    plot_comparison(results_full, results_tm)
    plot_between_ligand_heatmap(results_tm, "tm_core")
    plot_between_ligand_heatmap(results_full, "full")

    # ── Step 7: Export summary ──
    with open(os.path.join(OUTPUT_DIR, 'data', 'rescue_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("5-HT2A NMA TM-Core Rescue Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Full structure: {len(common_residues)} residues\n")
        f.write(f"TM core:        {len(tm_residues)} residues\n")
        f.write(f"Excluded:       {len(common_residues) - len(tm_residues)} residues\n\n")
        f.write(f"{'Metric':30s}  {'Full':>12s}  {'TM Core':>12s}\n")
        f.write(f"{'─'*30}  {'─'*12}  {'─'*12}\n")
        f.write(f"{'Within-ligand RMSIP':30s}  {results_full['within_mean']:12.4f}  {results_tm['within_mean']:12.4f}\n")
        f.write(f"{'Between-ligand RMSIP':30s}  {results_full['between_mean']:12.4f}  {results_tm['between_mean']:12.4f}\n")
        f.write(f"{'Separation (W - B)':30s}  {results_full['within_mean']-results_full['between_mean']:+12.4f}  {results_tm['within_mean']-results_tm['between_mean']:+12.4f}\n")
        f.write(f"{'Mann-Whitney p':30s}  {results_full['p_value']:12.2e}  {results_tm['p_value']:12.2e}\n")
        f.write(f"{'Cohens d':30s}  {results_full['cohens_d']:12.3f}  {results_tm['cohens_d']:12.3f}\n")
    print("  Saved: rescue_summary.txt")

    print(f"\n  Output: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
