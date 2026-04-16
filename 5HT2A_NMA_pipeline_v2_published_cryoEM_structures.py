#!/usr/bin/env python3
"""
5-HT2A Receptor Comparative Normal Mode Analysis Pipeline
Benjamin R. Cummins

Systematic NMA comparison across activation states and ligand chemotypes
using ProDy 2.0 ANM/GNM with SignDy ensemble analysis.

Requirements:
    pip install prody matplotlib numpy scipy seaborn

Output:
    - Figures (PNG) for all comparative analyses
    - CSV files with numerical results
    - NMD files for visualization in VMD/NMWiz
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

warnings.filterwarnings('ignore')

# ── ProDy imports ──────────────────────────────────────────────────────────
from prody import (
    fetchPDB, parsePDB, parseMMCIF, writePDB,
    ANM, GNM, calcANM, calcGNM,
    matchChains, mapOntoChain,
    calcOverlap, calcSubspaceOverlap,
    calcSqFlucts, calcCrossCorr,
    calcRMSD, superpose,
    PDBEnsemble, confProDy
)
from scipy.spatial.distance import pdist, squareform

# Suppress ProDy verbosity (set to 'info' or 'debug' for troubleshooting)
confProDy(verbosity='warning')


# CONFIGURATION - SETUP IN THIS FILE TO USE PUBLISHED CRYOEM PDB STRUCTURES

# Output directory for all results
OUTPUT_DIR = "5HT2A_NMA_results_cryoEM"

# ANM/GNM parameters
ANM_CUTOFF = 15.0   # Angstroms (standard for Cα ANM)
GNM_CUTOFF = 7.3    # Angstroms (standard for Cα GNM)
N_MODES = 20        # Number of modes to compute and analyze

# Reference structure for alignment and transition vector
REFERENCE_PDB = "6A93"  # Risperidone-bound inactive state

# Structure Panel
# Each entry: {
#   'pdb':       PDB ID,
#   'chain':     receptor chain ID in the PDB file,
#   'ligand':    ligand name (for labels),
#   'category':  pharmacological category,
#   'chemotype': chemical class,
#   'color':     color for plots
# }
#
# IMPORTANT: Verify chain IDs before running! Open each PDB in ChimeraX or
# on rcsb.org to confirm which chain is the receptor vs G-protein/nanobody.
# Chain IDs are set to 'auto' — the script will automatically detect
# the receptor chain by picking the longest protein chain in each
# structure. The 5-HT2A receptor (~370 residues) will always be longer
# than G-protein subunits, nanobodies, or scFv fragments.
# You can override with a specific chain letter if auto-detect picks wrong.

STRUCTURES = [
    # ── Inactive / Inverse Agonist ──────────────────────────────────────
    {
        'pdb': '6A93', 'chain': 'auto',
        'ligand': 'Risperidone', 'category': 'Inactive',
        'chemotype': 'Antipsychotic', 'color': '#2c3e50'
    },
    {
        'pdb': '6WH4', 'chain': 'auto',
        'ligand': 'Methiothepin', 'category': 'Inverse Agonist',
        'chemotype': 'Inverse Agonist', 'color': '#7f8c8d'
    },

    # ── Active: Tryptamines ─────────────────────────────────────────────
    {
        'pdb': '9ARX', 'chain': 'auto',
        'ligand': 'Serotonin', 'category': 'Full Agonist',
        'chemotype': 'Tryptamine', 'color': '#e74c3c'
    },
    {
        'pdb': '9AS7', 'chain': 'auto',
        'ligand': 'Psilocin', 'category': 'Full Agonist (Psychedelic)',
        'chemotype': 'Tryptamine', 'color': '#c0392b'
    },
    {
        'pdb': '9AS1', 'chain': 'auto',
        'ligand': 'DMT', 'category': 'Full Agonist (Psychedelic)',
        'chemotype': 'Tryptamine', 'color': '#e67e22'
    },

    # ── Active: Ergolines ───────────────────────────────────────────────
    {
        'pdb': '9AS3', 'chain': 'auto',
        'ligand': 'LSD', 'category': 'Full Agonist (Psychedelic)',
        'chemotype': 'Ergoline', 'color': '#9b59b6'
    },
    {
        'pdb': '9ARZ', 'chain': 'auto',
        'ligand': 'BOL-148', 'category': 'Partial Agonist (Non-hallucinogenic)',
        'chemotype': 'Ergoline', 'color': '#8e44ad'
    },
    {
        'pdb': '8UWL', 'chain': 'auto',
        'ligand': 'Lisuride', 'category': 'Agonist (Non-hallucinogenic)',
        'chemotype': 'Ergoline', 'color': '#6c3483'
    },

    # ── Active: Phenethylamines ─────────────────────────────────────────
    {
        'pdb': '6WHA', 'chain': 'auto',
        'ligand': '25CN-NBOH', 'category': 'Full Agonist (Psychedelic)',
        'chemotype': 'Phenethylamine', 'color': '#27ae60'
    },
    {
        'pdb': '9AS5', 'chain': 'auto',
        'ligand': 'Mescaline', 'category': 'Full Agonist (Psychedelic)',
        'chemotype': 'Phenethylamine', 'color': '#2ecc71'
    },
    {
        'pdb': '9AS9', 'chain': 'auto',
        'ligand': 'RS130-180', 'category': 'βArr-biased Agonist',
        'chemotype': 'Phenethylamine', 'color': '#1abc9c'
    },
]

# TM helix boundaries (approximate, Ballesteros-Weinstein based)
# Adjust these to match the actual residue numbering in your reference PDB.
# These are for human 5-HT2A (UniProt P28223).
TM_REGIONS = {
    'TM1': (75, 102),
    'ICL1': (103, 108),
    'TM2': (109, 140),
    'ECL1': (141, 153),
    'TM3': (154, 187),
    'ICL2': (188, 199),
    'TM4': (200, 228),
    'ECL2': (229, 244),
    'TM5': (245, 277),
    'ICL3': (278, 310),
    'TM6': (311, 345),
    'ECL3': (346, 351),
    'TM7': (352, 378),
    'H8':  (379, 395),
}

# Key functional residues to highlight in plots
KEY_RESIDUES = {
    'D155 (3.32)':  155,   # Conserved anchor for amine
    'S242 (5.46)':  242,   # Primate-specific H-bond donor
    'W336 (6.48)':  336,   # Toggle switch
    'F340 (6.52)':  340,   # Aromatic lid
    'N343 (6.55)':  343,   # Your dissertation residue — included for
                            # context but not the focus of this paper
    'L229 (ECL2)':  229,   # ECL2 lid / mescaline contact
    'Y370 (7.43)':  370,   # Conserved NPxxY motif region
}



# HELPER FUNCTIONS

def setup_output_dirs():
    """Create output directory structure."""
    subdirs = ['figures', 'data', 'nmd_files', 'prepared_structures']
    for sub in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


def fetch_and_prepare(entry):
    """
    Fetch a PDB/mmCIF, auto-detect the receptor chain (longest protein
    chain), extract Cα atoms, and return the AtomGroup.

    Handles:
      - PDB-format files (older structures like 6A93, 6WH4, 6WHA)
      - mmCIF-only files (newer cryo-EM like 9ARX, 9AS7, etc.)
      - Automatic receptor chain detection (picks longest protein chain)
    """
    pdb_id = entry['pdb']
    chain_id = entry.get('chain', 'auto')
    label = f"{entry['ligand']} ({pdb_id})"

    print(f"  Fetching {pdb_id}...", end=" ")

    # ── Step 1: Parse the structure ────────────────────────────────────
    structure = None

    # Try PDB format first
    try:
        structure = parsePDB(pdb_id)
        if structure is not None and structure.numAtoms() == 0:
            structure = None
    except Exception:
        structure = None

    # If PDB failed or returned None/empty, try mmCIF
    if structure is None:
        try:
            structure = parseMMCIF(pdb_id)
            if structure is not None and structure.numAtoms() == 0:
                structure = None
        except Exception as e:
            print(f"ERROR: Could not parse {pdb_id} in PDB or mmCIF format!")
            print(f"       {e}")
            return None

    if structure is None:
        print(f"ERROR: No atoms found in {pdb_id}!")
        return None

    # ── Step 2: Identify the receptor chain ────────────────────────────
    all_ca = structure.select('protein and name CA')
    if all_ca is None:
        print(f"ERROR: No protein Cα atoms in {pdb_id}!")
        return None

    if chain_id != 'auto':
        # User specified a chain — try it
        ca_atoms = structure.select(f'protein and name CA and chain {chain_id}')
        if ca_atoms is not None and len(ca_atoms) > 200:
            n_res = len(ca_atoms)
            print(f"OK ({n_res} Cα, chain {chain_id}, "
                  f"res {ca_atoms.getResnums()[0]}-{ca_atoms.getResnums()[-1]})")
            out_path = os.path.join(OUTPUT_DIR, 'prepared_structures',
                                    f'{pdb_id}_{chain_id}_CA.pdb')
            writePDB(out_path, ca_atoms)
            return ca_atoms
        else:
            print(f"chain {chain_id} not found or too short, auto-detecting...",
                  end=" ")

    # Auto-detect: find the longest protein chain (= the receptor)
    chains = sorted(set(all_ca.getChids()))
    best_chain = None
    best_count = 0

    for ch in chains:
        sel = all_ca.select(f'chain {ch}')
        if sel is not None and len(sel) > best_count:
            best_count = len(sel)
            best_chain = ch

    if best_chain is None or best_count < 150:
        print(f"ERROR: No suitable receptor chain found in {pdb_id}!")
        print(f"       Chains found: {chains}, largest has {best_count} Cα atoms")
        return None

    ca_atoms = all_ca.select(f'chain {best_chain}')
    n_res = len(ca_atoms)
    resnums = ca_atoms.getResnums()

    # Print all chains so user can verify the auto-detection
    chain_summary = ", ".join(
        [f"{ch}({len(all_ca.select(f'chain {ch}'))})"
         for ch in chains
         if all_ca.select(f'chain {ch}') is not None]
    )
    print(f"OK (chain {best_chain}, {n_res} Cα, "
          f"res {resnums[0]}-{resnums[-1]}) "
          f"[all chains: {chain_summary}]")

    # Save prepared structure
    out_path = os.path.join(OUTPUT_DIR, 'prepared_structures',
                            f'{pdb_id}_{best_chain}_CA.pdb')
    writePDB(out_path, ca_atoms)

    return ca_atoms


def find_common_residues(all_atoms):
    """
    Find the intersection of residue numbers across all structures.
    Returns a sorted list of residue numbers present in every structure.
    """
    residue_sets = []
    for entry, atoms in all_atoms:
        resnums = set(atoms.getResnums())
        residue_sets.append(resnums)

    common = residue_sets[0]
    for rs in residue_sets[1:]:
        common = common.intersection(rs)

    common = sorted(common)
    print(f"\n  Common residues across all structures: {len(common)}")
    print(f"  Range: {common[0]} - {common[-1]}")
    return common


def trim_to_common(atoms, common_residues):
    """Select only the common residues from an AtomGroup."""
    # Build selection string
    resnum_str = ' '.join(str(r) for r in common_residues)
    sel = atoms.select(f'resnum {resnum_str}')
    return sel


def align_structures(all_atoms_trimmed, ref_idx=0):
    """Superpose all structures onto the reference (first by default)."""
    ref = all_atoms_trimmed[ref_idx][1]
    ref_coords = ref.getCoords()

    for i, (entry, atoms) in enumerate(all_atoms_trimmed):
        if i == ref_idx:
            continue
        # Mobile onto target
        superpose(atoms, ref)
        rmsd = calcRMSD(atoms, ref)
        print(f"  Aligned {entry['ligand']:15s} → ref  RMSD = {rmsd:.2f} Å")



# CORE NMA FUNCTIONS

def run_anm(atoms, label, cutoff=ANM_CUTOFF, n_modes=N_MODES):
    """Run ANM on a Cα AtomGroup. Returns the ANM object."""
    anm = ANM(label)
    anm.buildHessian(atoms, cutoff=cutoff)
    anm.calcModes(n_modes=n_modes, zeros=False)
    return anm


def run_gnm(atoms, label, cutoff=GNM_CUTOFF, n_modes=N_MODES):
    """
    Run GNM on a Cα AtomGroup using scipy for distance calculation.
    This avoids the ProDy KDTree C extension bug on some Windows installs.
    """
    coords = atoms.getCoords()
    n_atoms = len(coords)

    # Build Kirchhoff matrix manually using scipy
    dist_matrix = squareform(pdist(coords))
    contact = (dist_matrix <= cutoff).astype(float)
    np.fill_diagonal(contact, 0)

    kirchhoff = -contact
    np.fill_diagonal(kirchhoff, contact.sum(axis=1))

    # Create GNM and inject the Kirchhoff matrix
    gnm = GNM(label)
    gnm._kirchhoff = kirchhoff
    gnm._n_atoms = n_atoms
    gnm._dof = n_atoms

    gnm.calcModes(n_modes=n_modes, zeros=False)
    return gnm


# ANALYSIS FUNCTIONS

def compute_sq_fluctuations(anm_results):
    """Compute per-residue mean-square fluctuations from ANM modes."""
    flucts = {}
    for entry, anm in anm_results:
        label = entry['ligand']
        sq = calcSqFlucts(anm)
        flucts[label] = sq
    return flucts


def compute_rmsip_matrix(anm_results, n_modes=10):
    """
    Compute RMSIP (Root Mean Square Inner Product) between the first
    n_modes of each pair of structures. RMSIP ranges from 0 (orthogonal
    subspaces) to 1 (identical subspaces).
    """
    n = len(anm_results)
    labels = [e['ligand'] for e, _ in anm_results]
    rmsip = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            overlap = calcSubspaceOverlap(
                anm_results[i][1][:n_modes],
                anm_results[j][1][:n_modes]
            )
            rmsip[i, j] = overlap

    return rmsip, labels


def compute_transition_overlaps(anm_results, all_atoms_trimmed, ref_idx=0):
    """
    Compute overlap of each structure's ANM modes with the
    inactive→active transition vector (displacement from reference).
    Returns dict of {ligand: overlap_array} for the first N_MODES.
    """
    ref_coords = all_atoms_trimmed[ref_idx][1].getCoords()
    overlaps = {}

    for i, (entry, anm) in enumerate(anm_results):
        if i == ref_idx:
            continue

        # Transition vector: displacement from inactive to this active state
        active_coords = all_atoms_trimmed[i][1].getCoords()
        disp = (active_coords - ref_coords).flatten()
        disp_norm = disp / np.linalg.norm(disp)

        # Overlap of each mode with the transition vector
        mode_overlaps = []
        for mode_idx in range(min(N_MODES, anm.numModes())):
            mode_vec = anm[mode_idx].getEigvec()
            ov = abs(np.dot(mode_vec, disp_norm))
            mode_overlaps.append(ov)

        overlaps[entry['ligand']] = np.array(mode_overlaps)

    return overlaps


def compute_cross_correlations(anm_results):
    """Compute normalized cross-correlation matrices from ANM modes."""
    cc_maps = {}
    for entry, anm in anm_results:
        cc = calcCrossCorr(anm)
        cc_maps[entry['ligand']] = cc
    return cc_maps


def compute_gnm_flexibility(gnm_results):
    """Extract GNM slow-mode fluctuations (stiffness profiles)."""
    profiles = {}
    for entry, gnm in gnm_results:
        sq = calcSqFlucts(gnm)
        profiles[entry['ligand']] = sq
    return profiles



# PLOTTING FUNCTIONS

def add_tm_shading(ax, common_residues, alpha=0.08):
    """Add subtle TM helix shading to a residue-based plot."""
    res_array = np.array(common_residues)
    for name, (start, end) in TM_REGIONS.items():
        mask = (res_array >= start) & (res_array <= end)
        if mask.any():
            idx_start = np.where(mask)[0][0]
            idx_end = np.where(mask)[0][-1]
            if 'TM' in name:
                ax.axvspan(idx_start, idx_end, alpha=alpha, color='steelblue')
                # Label at top
                mid = (idx_start + idx_end) / 2
                ax.text(mid, ax.get_ylim()[1] * 0.95, name,
                        ha='center', va='top', fontsize=6, color='steelblue',
                        fontweight='bold')


def add_key_residue_markers(ax, common_residues, y_pos=None):
    """Add vertical lines and labels for key functional residues."""
    res_array = np.array(common_residues)
    for label, resnum in KEY_RESIDUES.items():
        idx = np.where(res_array == resnum)[0]
        if len(idx) > 0:
            ax.axvline(idx[0], color='red', alpha=0.3, linewidth=0.5,
                       linestyle='--')


def plot_fluctuation_profiles(flucts, structures, common_residues):
    """
    Plot per-residue ANM square fluctuations for all structures,
    grouped by category.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    groups = {
        'Inactive / Inverse Agonist': [],
        'Psychedelic Agonists': [],
        'Non-hallucinogenic / Biased': [],
    }

    for entry in structures:
        cat = entry['category']
        if 'Inactive' in cat or 'Inverse' in cat:
            groups['Inactive / Inverse Agonist'].append(entry)
        elif 'Non-hallucinogenic' in cat or 'biased' in cat.lower():
            groups['Non-hallucinogenic / Biased'].append(entry)
        else:
            groups['Psychedelic Agonists'].append(entry)

    x = np.arange(len(common_residues))

    for ax, (group_name, entries) in zip(axes, groups.items()):
        # Always plot inactive baseline in gray on all panels
        for e in structures:
            if e['pdb'] == REFERENCE_PDB and group_name != 'Inactive / Inverse Agonist':
                if e['ligand'] in flucts:
                    ax.fill_between(x, flucts[e['ligand']], alpha=0.15,
                                    color='gray', label=f"{e['ligand']} (ref)")
                    ax.plot(x, flucts[e['ligand']], color='gray', alpha=0.4,
                            linewidth=0.8)

        for entry in entries:
            lig = entry['ligand']
            if lig in flucts:
                ax.plot(x, flucts[lig], color=entry['color'],
                        label=f"{lig} ({entry['pdb']})", linewidth=1.2)

        add_tm_shading(ax, common_residues)
        add_key_residue_markers(ax, common_residues)

        ax.set_ylabel('Sq. Fluctuation (a.u.)')
        ax.set_title(group_name, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

    # X-axis labels — show every 20th residue number
    tick_step = 20
    tick_positions = list(range(0, len(common_residues), tick_step))
    tick_labels = [str(common_residues[i]) for i in tick_positions]
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels, fontsize=7, rotation=45)
    axes[-1].set_xlabel('Residue Number')

    fig.suptitle('5-HT2A ANM Per-Residue Fluctuation Profiles',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures',
                             'fluctuation_profiles.png'), dpi=300)
    plt.close(fig)
    print("  Saved: fluctuation_profiles.png")


def plot_differential_fluctuations(flucts, structures, common_residues):
    """
    Plot Δfluctuation = (agonist - inactive) for each active structure.
    Positive values = more flexible than inactive; negative = stiffer.
    """
    ref_lig = None
    for e in structures:
        if e['pdb'] == REFERENCE_PDB:
            ref_lig = e['ligand']
            break

    if ref_lig not in flucts:
        print("  WARNING: Reference fluctuations not found, skipping Δ plot.")
        return

    ref_fluct = flucts[ref_lig]
    x = np.arange(len(common_residues))

    fig, ax = plt.subplots(figsize=(14, 5))

    for entry in structures:
        lig = entry['ligand']
        if lig == ref_lig or lig not in flucts:
            continue
        delta = flucts[lig] - ref_fluct
        ax.plot(x, delta, color=entry['color'],
                label=f"{lig}", linewidth=0.9, alpha=0.8)

    ax.axhline(0, color='black', linewidth=0.5)
    add_tm_shading(ax, common_residues)
    add_key_residue_markers(ax, common_residues)

    ax.set_ylabel('ΔSq. Fluctuation (agonist − inactive)')
    ax.set_xlabel('Residue Number')
    ax.set_title(f'Differential Flexibility Relative to {ref_lig} ({REFERENCE_PDB})',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.9)

    tick_step = 20
    tick_positions = list(range(0, len(common_residues), tick_step))
    tick_labels = [str(common_residues[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures',
                             'differential_fluctuations.png'), dpi=300)
    plt.close(fig)
    print("  Saved: differential_fluctuations.png")


def plot_rmsip_heatmap(rmsip, labels):
    """Plot RMSIP matrix as a clustered heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(rmsip, xticklabels=labels, yticklabels=labels,
                annot=True, fmt='.2f', cmap='RdYlBu_r',
                vmin=0, vmax=1, square=True, ax=ax,
                cbar_kws={'label': 'RMSIP (10 modes)'})
    ax.set_title('Subspace Overlap (RMSIP) Between Structures',
                 fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures',
                             'rmsip_heatmap.png'), dpi=300)
    plt.close(fig)
    print("  Saved: rmsip_heatmap.png")


def plot_transition_overlaps(overlaps, structures):
    """
    Bar chart: cumulative overlap of first N modes with the
    inactive→active transition vector for each structure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(overlaps))
    bar_labels = []
    cumulative = []
    colors = []

    for entry in structures:
        lig = entry['ligand']
        if lig in overlaps:
            # Cumulative overlap = sqrt(sum of squared individual overlaps)
            cum = np.sqrt(np.sum(overlaps[lig][:10]**2))
            cumulative.append(cum)
            bar_labels.append(lig)
            colors.append(entry['color'])

    x_pos = np.arange(len(bar_labels))
    bars = ax.bar(x_pos, cumulative, color=colors, edgecolor='black',
                  linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Cumulative Overlap (10 modes)')
    ax.set_title('ANM Mode Overlap with Inactive→Active Transition Vector',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, val in zip(bars, cumulative):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures',
                             'transition_overlap.png'), dpi=300)
    plt.close(fig)
    print("  Saved: transition_overlap.png")


def plot_cross_correlation_comparison(cc_maps, structures, common_residues):
    """
    Plot cross-correlation maps for a selected subset and their
    differential maps relative to inactive.
    """
    # Pick representative structures to compare
    show_list = [REFERENCE_PDB]
    # Add one from each chemotype
    for chemotype in ['Tryptamine', 'Ergoline', 'Phenethylamine']:
        for e in structures:
            if e['chemotype'] == chemotype and 'Psychedelic' in e['category']:
                show_list.append(e['pdb'])
                break
    # Add one non-hallucinogenic
    for e in structures:
        if 'Non-hallucinogenic' in e['category']:
            show_list.append(e['pdb'])
            break

    # Map PDB IDs to ligand names
    pdb_to_lig = {e['pdb']: e['ligand'] for e in structures}
    show_ligs = [pdb_to_lig.get(p) for p in show_list if pdb_to_lig.get(p) in cc_maps]

    if len(show_ligs) < 2:
        print("  WARNING: Not enough structures for cross-correlation comparison.")
        return

    n_panels = len(show_ligs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, lig in zip(axes, show_ligs):
        cc = cc_maps[lig]
        im = ax.imshow(cc, cmap='RdBu_r', vmin=-1, vmax=1,
                       origin='lower', aspect='equal')
        ax.set_title(lig, fontsize=10, fontweight='bold')

        # Sparse tick labels
        tick_step = 30
        ticks = list(range(0, len(common_residues), tick_step))
        tlabels = [str(common_residues[t]) for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels, fontsize=6, rotation=45)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tlabels, fontsize=6)

    fig.colorbar(im, ax=axes, shrink=0.8, label='Cross-correlation')
    fig.suptitle('ANM Cross-Correlation Maps', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures',
                             'cross_correlations.png'), dpi=300)
    plt.close(fig)
    print("  Saved: cross_correlations.png")

    # ── Differential cross-correlation maps ──
    ref_lig = pdb_to_lig.get(REFERENCE_PDB)
    if ref_lig not in cc_maps:
        return

    ref_cc = cc_maps[ref_lig]
    diff_ligs = [l for l in show_ligs if l != ref_lig]

    if len(diff_ligs) == 0:
        return

    fig2, axes2 = plt.subplots(1, len(diff_ligs),
                               figsize=(5 * len(diff_ligs), 4.5))
    if len(diff_ligs) == 1:
        axes2 = [axes2]

    for ax, lig in zip(axes2, diff_ligs):
        diff = cc_maps[lig] - ref_cc
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                       origin='lower', aspect='equal')
        ax.set_title(f'{lig} − {ref_lig}', fontsize=10, fontweight='bold')

        ticks = list(range(0, len(common_residues), tick_step))
        tlabels = [str(common_residues[t]) for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabels, fontsize=6, rotation=45)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tlabels, fontsize=6)

    fig2.colorbar(im, ax=axes2, shrink=0.8, label='ΔCross-correlation')
    fig2.suptitle(f'Differential Cross-Correlation (vs. {ref_lig})',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'figures',
                              'diff_cross_correlations.png'), dpi=300)
    plt.close(fig2)
    print("  Saved: diff_cross_correlations.png")


def plot_gnm_stiffness(gnm_profiles, structures, common_residues):
    """Plot GNM-derived flexibility profiles (validates ANM results)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(common_residues))

    for entry in structures:
        lig = entry['ligand']
        if lig in gnm_profiles:
            ax.plot(x, gnm_profiles[lig], color=entry['color'],
                    label=lig, linewidth=0.9, alpha=0.8)

    add_tm_shading(ax, common_residues)
    add_key_residue_markers(ax, common_residues)

    ax.set_ylabel('GNM Sq. Fluctuation (a.u.)')
    ax.set_xlabel('Residue Number')
    ax.set_title('GNM Flexibility Profiles (Validation)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6, ncol=2, framealpha=0.9)

    tick_step = 20
    tick_positions = list(range(0, len(common_residues), tick_step))
    tick_labels = [str(common_residues[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures',
                             'gnm_flexibility.png'), dpi=300)
    plt.close(fig)
    print("  Saved: gnm_flexibility.png")


def plot_eigenvalue_spectra(anm_results):
    """Compare eigenvalue spectra (mode stiffness) across structures."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for entry, anm in anm_results:
        eigvals = anm.getEigvals()
        ax.plot(range(1, len(eigvals) + 1), eigvals,
                color=entry['color'], marker='o', markersize=3,
                label=entry['ligand'], linewidth=1)

    ax.set_xlabel('Mode Index')
    ax.set_ylabel('Eigenvalue (ω²)')
    ax.set_title('ANM Eigenvalue Spectra', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.set_xlim(0, N_MODES + 1)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures',
                             'eigenvalue_spectra.png'), dpi=300)
    plt.close(fig)
    print("  Saved: eigenvalue_spectra.png")


def plot_collectivity(anm_results):
    """
    Compare collectivity of the first few modes across structures.
    Collectivity measures how many residues participate in a mode
    (ranges from 1/N for localized to 1 for fully collective).
    """
    from prody import calcCollectivity

    fig, ax = plt.subplots(figsize=(10, 5))
    n_show = min(10, N_MODES)

    for entry, anm in anm_results:
        coll = []
        for i in range(n_show):
            c = calcCollectivity(anm[i])
            coll.append(c)
        ax.plot(range(1, n_show + 1), coll, color=entry['color'],
                marker='s', markersize=4, label=entry['ligand'], linewidth=1)

    ax.set_xlabel('Mode Index')
    ax.set_ylabel('Collectivity')
    ax.set_title('Mode Collectivity (fraction of residues participating)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.set_xlim(0, n_show + 1)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures',
                             'collectivity.png'), dpi=300)
    plt.close(fig)
    print("  Saved: collectivity.png")



# DATA EXPORT


def export_fluctuations_csv(flucts, common_residues):
    """Export all fluctuation data to CSV for downstream use."""
    import csv
    path = os.path.join(OUTPUT_DIR, 'data', 'anm_fluctuations.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['ResNum'] + list(flucts.keys())
        writer.writerow(header)
        for i, resnum in enumerate(common_residues):
            row = [resnum]
            for lig in flucts:
                row.append(f"{flucts[lig][i]:.6f}")
            writer.writerow(row)
    print(f"  Saved: anm_fluctuations.csv")


def export_rmsip_csv(rmsip, labels):
    """Export RMSIP matrix to CSV."""
    import csv
    path = os.path.join(OUTPUT_DIR, 'data', 'rmsip_matrix.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [f"{rmsip[i,j]:.4f}" for j in range(len(labels))])
    print(f"  Saved: rmsip_matrix.csv")


def export_nmd_files(anm_results, all_atoms_trimmed):
    """Export NMD files for visualization in VMD/NMWiz."""
    from prody import writeNMD
    for (entry, anm), (_, atoms) in zip(anm_results, all_atoms_trimmed):
        path = os.path.join(OUTPUT_DIR, 'nmd_files',
                            f'{entry["pdb"]}_{entry["ligand"]}_anm.nmd')
        writeNMD(path, anm, atoms)
    print(f"  Saved NMD files for VMD visualization")



# MAIN PIPELINE


def main():
    print("=" * 70)
    print("  5-HT2A Comparative NMA Pipeline")
    print("=" * 70)

    # ── Setup ──
    setup_output_dirs()

    # ── Step 1: Fetch and prepare structures ──
    print("\n[Step 1] Fetching and preparing structures...")
    all_atoms = []
    failed = []
    for entry in STRUCTURES:
        atoms = fetch_and_prepare(entry)
        if atoms is not None:
            all_atoms.append((entry, atoms))
        else:
            failed.append(entry['pdb'])

    if failed:
        print(f"\n  ⚠ Failed structures: {failed}")
        print("    Check chain IDs in the STRUCTURES config and re-run.")

    if len(all_atoms) < 2:
        print("\n  ERROR: Need at least 2 structures to compare. Exiting.")
        return

    # ── Step 2: Find common residues and trim ──
    print("\n[Step 2] Finding common residues...")
    common_residues = find_common_residues(all_atoms)

    print("\n  Trimming to common residues...")
    all_atoms_trimmed = []
    for entry, atoms in all_atoms:
        trimmed = trim_to_common(atoms, common_residues)
        if trimmed is not None and len(trimmed) == len(common_residues):
            all_atoms_trimmed.append((entry, trimmed))
        else:
            actual = len(trimmed) if trimmed else 0
            print(f"  WARNING: {entry['ligand']} has {actual} atoms after "
                  f"trimming (expected {len(common_residues)}), skipping.")

    # ── Step 3: Align ──
    print("\n[Step 3] Aligning structures to reference...")
    # Find index of reference
    ref_idx = 0
    for i, (entry, _) in enumerate(all_atoms_trimmed):
        if entry['pdb'] == REFERENCE_PDB:
            ref_idx = i
            break
    align_structures(all_atoms_trimmed, ref_idx)

    # ── Step 4: Run ANM ──
    print(f"\n[Step 4] Running ANM (cutoff={ANM_CUTOFF} Å, "
          f"{N_MODES} modes)...")
    anm_results = []
    for entry, atoms in all_atoms_trimmed:
        label = f"{entry['ligand']}_{entry['pdb']}"
        anm = run_anm(atoms, label)
        anm_results.append((entry, anm))
        print(f"  {entry['ligand']:15s}  "
              f"eigenvalues: {anm.getEigvals()[0]:.4f} - "
              f"{anm.getEigvals()[-1]:.4f}")

    # ── Step 5: Run GNM ──
    print(f"\n[Step 5] Running GNM (cutoff={GNM_CUTOFF} Å, "
          f"{N_MODES} modes)...")
    gnm_results = []
    for entry, atoms in all_atoms_trimmed:
        label = f"{entry['ligand']}_{entry['pdb']}"
        gnm = run_gnm(atoms, label)
        gnm_results.append((entry, gnm))

    # ── Step 6: Compute analyses ──
    print("\n[Step 6] Computing analyses...")
    active_structures = [e for e in STRUCTURES if e['pdb'] != REFERENCE_PDB]

    print("  Computing fluctuation profiles...")
    flucts = compute_sq_fluctuations(anm_results)

    print("  Computing RMSIP matrix...")
    rmsip, rmsip_labels = compute_rmsip_matrix(anm_results)

    print("  Computing transition vector overlaps...")
    trans_overlaps = compute_transition_overlaps(
        anm_results, all_atoms_trimmed, ref_idx)

    print("  Computing cross-correlations...")
    cc_maps = compute_cross_correlations(anm_results)

    print("  Computing GNM profiles...")
    gnm_profiles = compute_gnm_flexibility(gnm_results)

    # ── Step 7: Generate figures ──
    print("\n[Step 7] Generating figures...")
    plot_fluctuation_profiles(flucts, STRUCTURES, common_residues)
    plot_differential_fluctuations(flucts, STRUCTURES, common_residues)
    plot_rmsip_heatmap(rmsip, rmsip_labels)
    plot_transition_overlaps(trans_overlaps, STRUCTURES)
    plot_cross_correlation_comparison(cc_maps, STRUCTURES, common_residues)
    plot_gnm_stiffness(gnm_profiles, STRUCTURES, common_residues)
    plot_eigenvalue_spectra(anm_results)
    plot_collectivity(anm_results)

    # ── Step 8: Export data ──
    print("\n[Step 8] Exporting data...")
    export_fluctuations_csv(flucts, common_residues)
    export_rmsip_csv(rmsip, rmsip_labels)
    export_nmd_files(anm_results, all_atoms_trimmed)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Structures analyzed: {len(all_atoms_trimmed)}")
    print(f"  Common residues:     {len(common_residues)}")
    print(f"  Output directory:    {os.path.abspath(OUTPUT_DIR)}")
    print(f"\n  Figures:  {OUTPUT_DIR}/figures/")
    print(f"  Data:     {OUTPUT_DIR}/data/")
    print(f"  NMD:      {OUTPUT_DIR}/nmd_files/")
    print(f"\n  Load NMD files in VMD → Extensions → Analysis → "
          f"Normal Mode Wizard")
    print("=" * 70)


if __name__ == '__main__':
    main()
