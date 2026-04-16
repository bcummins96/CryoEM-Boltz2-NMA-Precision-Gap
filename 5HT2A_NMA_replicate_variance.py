#!/usr/bin/env python3
"""
5-HT2A NMA Replicate Variance Analysis
Benjamin Cummins

Runs NMA on multiple Boltz2 replicates per ligand and computes:
  - Within-ligand RMSIP (same ligand, different Boltz2 seeds)
  - Between-ligand RMSIP (different ligands)
  - Statistical comparison (Mann-Whitney U + Cohen's d)

Requirements:
    pip install prody matplotlib numpy scipy seaborn

Generate 5 Boltz2 replicates per ligand (different random seeds)
Organize files as: boltz2_replicates/<ligand_name>/rep1.cif, rep2.cif, ...
Update REPLICATE_DIR and LIGANDS below
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


OUTPUT_DIR = "5HT2A_NMA_replicate_analysis"

# ANM parameters (must match the main pipeline)
ANM_CUTOFF = 15.0
N_MODES = 20
RMSIP_MODES = 10  # Number of modes for RMSIP calculation

# Number of replicates expected per ligand
N_REPLICATES = 5

# ── Replicate file organization ────────────────────────────────────────
# Option A: Organized in subdirectories per ligand
#   boltz2_replicates/
#     serotonin/
#       rep1.cif, rep2.cif, rep3.cif, rep4.cif, rep5.cif
#     psilocin/
#       rep1.cif, rep2.cif, ...
#
# Option B: Flat directory with naming convention
#   boltz2_replicates/
#     serotonin_rep1.cif, serotonin_rep2.cif, ...
#     psilocin_rep1.cif, psilocin_rep2.cif, ...
#
# Set REPLICATE_DIR and the file pattern below.

REPLICATE_DIR = r"C:\Users\Ben\boltz2_replicates"

# Ligand definitions: name, subdirectory or filename pattern, color
# Adjust 'pattern' to match your file naming convention.
# For subdirectory organization, pattern is the glob within the subdir.
# For flat organization, pattern includes the ligand prefix.

LIGANDS = [
    {'name': 'Serotonin',    'subdir': 'serotonin',    'pattern': '*.cif', 'color': '#e74c3c', 'chemotype': 'Tryptamine'},
    {'name': 'Psilocin',     'subdir': 'psilocin',     'pattern': '*.cif', 'color': '#c0392b', 'chemotype': 'Tryptamine'},
    {'name': 'DMT',          'subdir': 'dmt',           'pattern': '*.cif', 'color': '#e67e22', 'chemotype': 'Tryptamine'},
    {'name': 'LSD',          'subdir': 'lsd',           'pattern': '*.cif', 'color': '#9b59b6', 'chemotype': 'Ergoline'},
    {'name': 'BOL-148',      'subdir': 'bol148',        'pattern': '*.cif', 'color': '#8e44ad', 'chemotype': 'Ergoline'},
    {'name': 'Lisuride',     'subdir': 'lisuride',      'pattern': '*.cif', 'color': '#6c3483', 'chemotype': 'Ergoline'},
    {'name': '25CN-NBOH',    'subdir': '25cn_nboh',     'pattern': '*.cif', 'color': '#27ae60', 'chemotype': 'Phenethylamine'},
    {'name': 'Mescaline',    'subdir': 'mescaline',     'pattern': '*.cif', 'color': '#2ecc71', 'chemotype': 'Phenethylamine'},
    {'name': 'RS130-180',    'subdir': 'rs130180',      'pattern': '*.cif', 'color': '#1abc9c', 'chemotype': 'Phenethylamine'},
    {'name': 'Risperidone',  'subdir': 'risperidone',   'pattern': '*.cif', 'color': '#2c3e50', 'chemotype': 'Antipsychotic'},
    {'name': 'Methiothepin', 'subdir': 'methiothepin',  'pattern': '*.cif', 'color': '#7f8c8d', 'chemotype': 'Inverse Agonist'},
]


# HELPER FUNCTIONS


def load_structure(filepath):
    """Parse a local PDB or CIF file and return Cα atoms of longest chain."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in ('.cif', '.mmcif'):
            structure = parseMMCIF(filepath)
        else:
            structure = parsePDB(filepath)
    except Exception as e:
        print(f"    ERROR parsing {filepath}: {e}")
        return None

    if structure is None or structure.numAtoms() == 0:
        return None

    all_ca = structure.select('protein and name CA')
    if all_ca is None:
        return None

    # Auto-detect longest chain
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
    else:
        return None


def find_common_residues(all_atoms_list):
    """Find intersection of residue numbers across all structures."""
    residue_sets = [set(atoms.getResnums()) for atoms in all_atoms_list]
    common = residue_sets[0]
    for rs in residue_sets[1:]:
        common = common.intersection(rs)
    return sorted(common)


def trim_to_common(atoms, common_residues):
    """Select only common residues."""
    resnum_str = ' '.join(str(r) for r in common_residues)
    return atoms.select(f'resnum {resnum_str}')


def run_anm(atoms, label):
    """Run ANM and return the ANM object."""
    anm = ANM(label)
    anm.buildHessian(atoms, cutoff=ANM_CUTOFF)
    anm.calcModes(n_modes=N_MODES, zeros=False)
    return anm



# MAIN PIPELINE

def main():
    print("=" * 70)
    print("  5-HT2A NMA Replicate Variance Analysis")
    print("=" * 70)

    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'data'), exist_ok=True)

    # ── Step 1: Load all replicate structures ──
    print("\n[Step 1] Loading replicate structures...")

    # Dict: ligand_name -> list of AtomGroups
    all_structures = {}
    all_flat = []  # (ligand_name, rep_idx, atoms) for common-residue finding

    for lig in LIGANDS:
        name = lig['name']
        search_dir = os.path.join(REPLICATE_DIR, lig['subdir'])
        files = sorted(glob.glob(os.path.join(search_dir, lig['pattern'])))

        if not files:
            # Try flat organization
            flat_pattern = os.path.join(REPLICATE_DIR, f"{lig['subdir']}*")
            files = sorted(glob.glob(flat_pattern))

        print(f"  {name}: found {len(files)} files")

        if len(files) < 2:
            print(f"    WARNING: Need at least 2 replicates, skipping {name}")
            continue

        structures = []
        for i, fpath in enumerate(files[:N_REPLICATES]):
            atoms = load_structure(fpath)
            if atoms is not None:
                structures.append(atoms)
                all_flat.append((name, i, atoms))
                print(f"    Rep {i+1}: {len(atoms)} Cα atoms")
            else:
                print(f"    Rep {i+1}: FAILED to load")

        if len(structures) >= 2:
            all_structures[name] = structures

    if len(all_structures) < 2:
        print("\nERROR: Need at least 2 ligands with replicates. Exiting.")
        return

    # ── Step 2: Find common residues across ALL structures ──
    print(f"\n[Step 2] Finding common residues across all {len(all_flat)} structures...")
    all_atoms_list = [atoms for _, _, atoms in all_flat]
    common_residues = find_common_residues(all_atoms_list)
    print(f"  Common residues: {len(common_residues)} (range {common_residues[0]}-{common_residues[-1]})")

    # ── Step 3: Trim and align ──
    print("\n[Step 3] Trimming and aligning...")
    # Use first replicate of first ligand as reference
    ref_name = list(all_structures.keys())[0]
    ref_atoms = trim_to_common(all_structures[ref_name][0], common_residues)

    trimmed_structures = {}  # ligand -> list of trimmed AtomGroups
    anm_results = {}         # ligand -> list of ANM objects

    for name, reps in all_structures.items():
        trimmed = []
        for i, atoms in enumerate(reps):
            t = trim_to_common(atoms, common_residues)
            if t is not None and len(t) == len(common_residues):
                if not (name == ref_name and i == 0):
                    superpose(t, ref_atoms)
                trimmed.append(t)
        trimmed_structures[name] = trimmed
        print(f"  {name}: {len(trimmed)} replicates aligned")

    # ── Step 4: Run ANM on all replicates ──
    print(f"\n[Step 4] Running ANM on {sum(len(v) for v in trimmed_structures.values())} structures...")
    for name, reps in trimmed_structures.items():
        anms = []
        for i, atoms in enumerate(reps):
            anm = run_anm(atoms, f"{name}_rep{i+1}")
            anms.append(anm)
        anm_results[name] = anms
        print(f"  {name}: {len(anms)} ANM calculations complete")

    # ── Step 5: Compute RMSIP matrices ──
    print("\n[Step 5] Computing within- and between-ligand RMSIP...")

    within_rmsip = {}   # ligand -> list of within-ligand RMSIP values
    between_rmsip = []  # all between-ligand RMSIP values
    between_detail = {} # (lig_a, lig_b) -> list of RMSIP values

    ligand_names = list(anm_results.keys())

    # Within-ligand RMSIP
    for name, anms in anm_results.items():
        pairs = []
        for i in range(len(anms)):
            for j in range(i + 1, len(anms)):
                ov = calcSubspaceOverlap(anms[i][:RMSIP_MODES], anms[j][:RMSIP_MODES])
                pairs.append(ov)
        within_rmsip[name] = pairs
        if pairs:
            print(f"  {name:15s} within-ligand: {np.mean(pairs):.4f} ± {np.std(pairs):.4f} "
                  f"(n={len(pairs)} pairs)")

    # Between-ligand RMSIP
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

    print(f"\n  Overall within-ligand RMSIP:  {np.mean(all_within):.4f} ± {np.std(all_within):.4f} "
          f"(n={len(all_within)})")
    print(f"  Overall between-ligand RMSIP: {np.mean(between_rmsip):.4f} ± {np.std(between_rmsip):.4f} "
          f"(n={len(between_rmsip)})")

    # ── Step 6: Statistical tests ──
    print(f"\n[Step 6] Statistical analysis...")

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(all_within, between_rmsip, alternative='greater')
    print(f"  Mann-Whitney U test (within > between):")
    print(f"    U = {u_stat:.1f}, p = {p_value:.2e}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(all_within)**2 + np.std(between_rmsip)**2) / 2)
    cohens_d = (np.mean(all_within) - np.mean(between_rmsip)) / pooled_std if pooled_std > 0 else 0
    print(f"  Cohen's d = {cohens_d:.3f}", end="")
    if abs(cohens_d) >= 0.8:
        print(" (large effect)")
    elif abs(cohens_d) >= 0.5:
        print(" (medium effect)")
    else:
        print(" (small effect)")

    # Separation ratio
    sep = np.mean(all_within) / np.mean(between_rmsip) if np.mean(between_rmsip) > 0 else float('inf')
    print(f"  Separation ratio (within/between): {sep:.3f}")

    # ── Step 7: Generate figures ──
    print(f"\n[Step 7] Generating figures...")

    # Figure 1: Distribution comparison (the key figure)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

    # Panel A: Overlapping histograms / KDE
    ax = axes[0]
    ax.hist(all_within, bins=25, alpha=0.5, color='#2196F3', label='Within-ligand',
            density=True, edgecolor='white', linewidth=0.5)
    ax.hist(between_rmsip, bins=25, alpha=0.5, color='#FF5722', label='Between-ligand',
            density=True, edgecolor='white', linewidth=0.5)

    # KDE overlay
    if len(all_within) > 5:
        from scipy.stats import gaussian_kde
        x_range = np.linspace(min(min(all_within), min(between_rmsip)) - 0.05,
                              max(max(all_within), max(between_rmsip)) + 0.05, 200)
        kde_w = gaussian_kde(all_within)
        kde_b = gaussian_kde(between_rmsip)
        ax.plot(x_range, kde_w(x_range), color='#1565C0', linewidth=2)
        ax.plot(x_range, kde_b(x_range), color='#D84315', linewidth=2)

    ax.axvline(np.mean(all_within), color='#1565C0', linestyle='--', linewidth=1.5,
               label=f'Within mean = {np.mean(all_within):.3f}')
    ax.axvline(np.mean(between_rmsip), color='#D84315', linestyle='--', linewidth=1.5,
               label=f'Between mean = {np.mean(between_rmsip):.3f}')

    ax.set_xlabel('RMSIP (10 modes)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Within-Ligand vs Between-Ligand Dynamic Similarity', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)

    # Add stats annotation
    sig_text = f"Mann-Whitney p = {p_value:.2e}\nCohen's d = {cohens_d:.2f}"
    ax.text(0.02, 0.95, sig_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel B: Per-ligand within-RMSIP box plot
    ax2 = axes[1]
    lig_names_sorted = sorted(within_rmsip.keys(),
                               key=lambda x: np.mean(within_rmsip[x]) if within_rmsip[x] else 0,
                               reverse=True)
    box_data = [within_rmsip[name] for name in lig_names_sorted if within_rmsip[name]]
    box_labels = [name for name in lig_names_sorted if within_rmsip[name]]
    box_colors = []
    for name in box_labels:
        for lig in LIGANDS:
            if lig['name'] == name:
                box_colors.append(lig['color'])
                break

    bp = ax2.boxplot(box_data, vert=True, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xticklabels(box_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Within-Ligand RMSIP', fontsize=11)
    ax2.set_title('Replicate Consistency', fontsize=12, fontweight='bold')

    # Add horizontal line for between-ligand mean
    ax2.axhline(np.mean(between_rmsip), color='#D84315', linestyle='--', linewidth=1,
                label=f'Between-ligand mean ({np.mean(between_rmsip):.3f})')
    ax2.legend(fontsize=8, loc='lower left')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'replicate_variance_analysis.png'), dpi=300)
    plt.close(fig)
    print("  Saved: replicate_variance_analysis.png")

    # Figure 2: Full RMSIP heatmap with replicates grouped
    n_total = sum(len(v) for v in anm_results.values())
    full_rmsip = np.zeros((n_total, n_total))
    full_labels = []
    idx = 0
    idx_map = {}  # (ligand, rep) -> index
    for name in ligand_names:
        for i in range(len(anm_results[name])):
            full_labels.append(f"{name} r{i+1}")
            idx_map[(name, i)] = idx
            idx += 1

    for (name_a, i_a), idx_a in idx_map.items():
        for (name_b, i_b), idx_b in idx_map.items():
            if idx_a == idx_b:
                full_rmsip[idx_a, idx_b] = 1.0
            elif idx_a < idx_b:
                ov = calcSubspaceOverlap(
                    anm_results[name_a][i_a][:RMSIP_MODES],
                    anm_results[name_b][i_b][:RMSIP_MODES]
                )
                full_rmsip[idx_a, idx_b] = ov
                full_rmsip[idx_b, idx_a] = ov

    fig2, ax3 = plt.subplots(figsize=(16, 14))
    sns.heatmap(full_rmsip, xticklabels=full_labels, yticklabels=full_labels,
                cmap='RdYlBu_r', vmin=0.3, vmax=1.0, square=True, ax=ax3,
                cbar_kws={'label': 'RMSIP (10 modes)'})
    ax3.set_title('Full Replicate RMSIP Matrix', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)

    # Draw boxes around within-ligand blocks
    offset = 0
    for name in ligand_names:
        n = len(anm_results[name])
        rect = plt.Rectangle((offset, offset), n, n, fill=False,
                              edgecolor='black', linewidth=2)
        ax3.add_patch(rect)
        offset += n

    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'figures', 'full_replicate_heatmap.png'), dpi=300)
    plt.close(fig2)
    print("  Saved: full_replicate_heatmap.png")

    # ── Step 8: Export data ──
    print(f"\n[Step 8] Exporting data...")

    import csv

    # Within-ligand stats
    with open(os.path.join(OUTPUT_DIR, 'data', 'within_ligand_rmsip.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ligand', 'N_replicates', 'N_pairs', 'Mean_RMSIP', 'Std_RMSIP', 'Min', 'Max'])
        for name in ligand_names:
            vals = within_rmsip.get(name, [])
            if vals:
                writer.writerow([name, len(anm_results[name]), len(vals),
                                 f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}",
                                 f"{np.min(vals):.4f}", f"{np.max(vals):.4f}"])
    print("  Saved: within_ligand_rmsip.csv")

    # Between-ligand stats
    with open(os.path.join(OUTPUT_DIR, 'data', 'between_ligand_rmsip.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ligand_A', 'Ligand_B', 'N_pairs', 'Mean_RMSIP', 'Std_RMSIP'])
        for (a, b), vals in sorted(between_detail.items()):
            writer.writerow([a, b, len(vals), f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}"])
    print("  Saved: between_ligand_rmsip.csv")

    # Full summary
    with open(os.path.join(OUTPUT_DIR, 'data', 'variance_analysis_summary.txt'), 'w') as f:
        f.write("5-HT2A NMA Replicate Variance Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Ligands analyzed: {len(ligand_names)}\n")
        f.write(f"Total structures: {n_total}\n")
        f.write(f"Common residues: {len(common_residues)}\n\n")
        f.write(f"Within-ligand RMSIP:  {np.mean(all_within):.4f} ± {np.std(all_within):.4f} (n={len(all_within)})\n")
        f.write(f"Between-ligand RMSIP: {np.mean(between_rmsip):.4f} ± {np.std(between_rmsip):.4f} (n={len(between_rmsip)})\n\n")
        f.write(f"Mann-Whitney U = {u_stat:.1f}, p = {p_value:.2e}\n")
        f.write(f"Cohen's d = {cohens_d:.3f}\n")
        f.write(f"Separation ratio = {sep:.3f}\n\n")
        if p_value < 0.001 and cohens_d > 0.8:
            f.write("CONCLUSION: Within-ligand RMSIP is significantly greater than\n")
            f.write("between-ligand RMSIP with a large effect size. The between-ligand\n")
            f.write("dynamic differences observed in the main analysis are real signal,\n")
            f.write("not Boltz2 prediction noise.\n")
        elif p_value < 0.05:
            f.write("CONCLUSION: Statistically significant difference detected, but\n")
            f.write("effect size is moderate. Some caution warranted in interpreting\n")
            f.write("between-ligand differences.\n")
        else:
            f.write("CONCLUSION: No significant difference between within- and between-\n")
            f.write("ligand RMSIP. The between-ligand differences may be largely\n")
            f.write("attributable to Boltz2 prediction variance.\n")
    print("  Saved: variance_analysis_summary.txt")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  RESULT SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Within-ligand RMSIP:  {np.mean(all_within):.4f} ± {np.std(all_within):.4f}")
    print(f"  Between-ligand RMSIP: {np.mean(between_rmsip):.4f} ± {np.std(between_rmsip):.4f}")
    print(f"  Separation:           {np.mean(all_within) - np.mean(between_rmsip):.4f}")
    print(f"  Mann-Whitney p:       {p_value:.2e}")
    print(f"  Cohen's d:            {cohens_d:.3f}")
    print(f"\n  Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
