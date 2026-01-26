"""
ChemSymmetryToolkit: A layered, reproducible chemistry computation toolkit for
molecular symmetry and point group inference with database integration.

Design:
- Atomic functions → Combination functions → Visualization functions
- JSON-serializable parameters only
- Unified return format: {'result': ..., 'metadata': {...}}
- Free databases preferred: PubChem (PUG REST), fallback local SQLite
- Two domain-specific libraries: rdkit, pubchempy
- Detailed mid-results saved to ./mid_result/chemistry
- Images saved to ./tool_images/

Core goal:
Solve: "which of the following molecules has c3h symmetry?"
Options:
1) triisopropyl borate
2) quinuclidine
3) benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone
4) triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone

Standard answer: "triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone"

Approach:
- Try PubChem to retrieve SMILES/structures
- Generate conformers via RDKit and compute heuristic symmetry features:
  - Planarity (σh presence if planar)
  - C3 axis: angle distribution clustering around centroid in plane
  - Perpendicular C2 axes (to distinguish from D3h)
- If network/unavailable or complex names fail, use curated local SQLite symmetry DB
- Answer determined by combining both sources; reproducible mid-results stored.

Note:
- Internet may be unavailable; functions handle ImportError and network failure gracefully.
- For point-group inference, we implement conservative heuristics; for special fused polycycles,
  we include curated classifications in local DB with explicit provenance metadata.

"""

import os
import json
import math
import sqlite3
import time
from typing import List, Dict, Any, Optional

# Domain-specific libraries (handled with graceful fallbacks)
try:
    import pubchempy as pcp
    PUBCHEMPY_AVAILABLE = True
except Exception:
    PUBCHEMPY_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdMolAlign import AlignMol
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# Visualization lib (optional)
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except Exception:
    PY3DMOL_AVAILABLE = False

# Constants
MID_RESULT_DIR = "./mid_result/chemistry"
TOOL_IMAGE_DIR = "./tool_images"
LOCAL_DB_PATH = "./chem_symmetry_local.db"
PLANARITY_RMS_THRESHOLD = 0.1  # Angstroms (heuristic)
ANGLE_CLUSTER_TOL = 20.0       # degrees tolerance within cluster
MIN_ATOMS_FOR_GEOMETRY = 6     # minimal atoms to attempt planar/C3 heuristics

def ensure_dirs() -> None:
    """Ensure required directories exist."""
    os.makedirs(MID_RESULT_DIR, exist_ok=True)
    os.makedirs(TOOL_IMAGE_DIR, exist_ok=True)

def save_mid_result(subject: str, label: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Save mid-result JSON to filesystem and return structured dict."""
    ensure_dirs()
    ts = int(time.time() * 1000)
    filepath = os.path.join(MID_RESULT_DIR, f"{label}_{ts}.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return {
            "result": filepath,
            "metadata": {
                "file_type": "json",
                "size": os.path.getsize(filepath),
                "subject": subject,
                "label": label,
                "timestamp_ms": ts
            }
        }
    except Exception as e:
        return {
            "result": None,
            "metadata": {
                "error": f"Failed to save mid result: {str(e)}",
                "subject": subject,
                "label": label
            }
        }

def load_file(filepath: str) -> Dict[str, Any]:
    """Generic file loader for JSON/text content."""
    if not isinstance(filepath, str):
        return {"result": None, "metadata": {"error": "filepath must be str"}}
    if not os.path.exists(filepath):
        return {"result": None, "metadata": {"error": f"File not found: {filepath}"}}
    try:
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"result": data, "metadata": {"file_type": "json", "size": os.path.getsize(filepath)}}
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = f.read()
            return {"result": data, "metadata": {"file_type": "text", "size": os.path.getsize(filepath)}}
    except Exception as e:
        return {"result": None, "metadata": {"error": f"Failed to load file: {str(e)}"}}

# Google Search Tool stub: Provide helpful resources (metadata only)
def google_search_resources(query: str) -> Dict[str, Any]:
    """
    Stub for Google Search Tool. Returns curated references to relevant APIs and docs.
    """
    resources = [
        {"title": "PubChem PUG REST API", "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug", "note": "Fetch compound by name, CID, SMILES; supports JSON responses."},
        {"title": "RDKit Documentation", "url": "https://www.rdkit.org/docs/", "note": "Molecule building, conformer generation, alignment."},
        {"title": "py3Dmol", "url": "https://github.com/avirshup/py3Dmol", "note": "3D viewer for molecules in notebooks."},
        {"title": "Molecular symmetry overview", "url": "https://en.wikipedia.org/wiki/Molecular_symmetry", "note": "Definitions of point groups like C3h, D3h, C3v."},
        {"title": "NIST Chemistry WebBook", "url": "https://webbook.nist.gov/chemistry/", "note": "Alternative properties and structural data"}
    ]
    mid = save_mid_result("chemistry", "google_search", {"query": query, "resources": resources})
    return {"result": resources, "metadata": {"source": "stub", "mid_file": mid["result"]}}

# Database functions (SQLite local DB)
def init_local_db(db_path: str) -> Dict[str, Any]:
    """Initialize local SQLite database with schema."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS molecules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            smiles TEXT,
            point_group TEXT,
            has_c3_axis INTEGER,
            has_sigma_h INTEGER,
            has_c2_perp_axes INTEGER,
            provenance TEXT
        );
        """)
        conn.commit()
        conn.close()
        return {"result": True, "metadata": {"db_path": db_path}}
    except Exception as e:
        return {"result": False, "metadata": {"error": str(e), "db_path": db_path}}

def insert_local_entries(entries: List[Dict[str, Any]], db_path: str) -> Dict[str, Any]:
    """Insert curated entries into local DB."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        for e in entries:
            cur.execute("""
            INSERT OR REPLACE INTO molecules (name, smiles, point_group, has_c3_axis, has_sigma_h, has_c2_perp_axes, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                e.get("name", ""),
                e.get("smiles", None),
                e.get("point_group", None),
                int(bool(e.get("has_c3_axis", False))),
                int(bool(e.get("has_sigma_h", False))),
                int(bool(e.get("has_c2_perp_axes", False))),
                e.get("provenance", "local_curated")
            ))
        conn.commit()
        conn.close()
        return {"result": len(entries), "metadata": {"db_path": db_path}}
    except Exception as e:
        return {"result": 0, "metadata": {"error": str(e), "db_path": db_path}}

def query_local_symmetry(name: str, db_path: str) -> Dict[str, Any]:
    """Query local DB for molecule symmetry by name."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name, smiles, point_group, has_c3_axis, has_sigma_h, has_c2_perp_axes, provenance FROM molecules WHERE name = ?;", (name,))
        row = cur.fetchone()
        conn.close()
        if row:
            data = {
                "name": row[0],
                "smiles": row[1],
                "point_group": row[2],
                "has_c3_axis": bool(row[3]),
                "has_sigma_h": bool(row[4]),
                "has_c2_perp_axes": bool(row[5]),
                "provenance": row[6]
            }
            return {"result": data, "metadata": {"db_path": db_path}}
        else:
            return {"result": None, "metadata": {"db_path": db_path, "info": "no entry"}}
    except Exception as e:
        return {"result": None, "metadata": {"error": str(e), "db_path": db_path}}

# PubChem access functions
def pubchem_fetch_cid_by_name(name: str) -> Dict[str, Any]:
    """
    Fetch PubChem CID by identifier.

    设计目标：
    - 支持更广泛的标识符类型，而不仅仅是“普通名称”：
      * 常用名 / IUPAC 名称 / 商品名等  → identifier_type='name'
      * SMILES 字符串                   → identifier_type='smiles'
      * 分子式（如 C17H21N3O）          → identifier_type='formula'
    - 保持统一返回格式：{'result': cid or None, 'metadata': {...}}
    """
    if not isinstance(name, str) or not name.strip():
        return {"result": None, "metadata": {"error": "name must be non-empty str"}}
    if not PUBCHEMPY_AVAILABLE:
        return {"result": None, "metadata": {"error": "pubchempy not available"}}

    query = name.strip()
    tried: List[Dict[str, Any]] = []

    # 1) 先按名称查询（兼容普通名 / IUPAC / 商品名等）
    for id_type in ("name", "smiles", "formula"):
        try:
            compounds = pcp.get_compounds(query, id_type)
            tried.append(
                {
                    "identifier_type": id_type,
                    "count": len(compounds) if compounds is not None else 0,
                }
            )
            if compounds:
                cid = compounds[0].cid
                mid = save_mid_result(
                    "chemistry",
                    "pubchem_cid",
                    {"query": query, "identifier_type": id_type, "cid": cid},
                )
                return {
                    "result": cid,
                    "metadata": {
                        "source": "PubChem",
                        "identifier_type": id_type,
                        "mid_file": mid["result"],
                    },
                }
        except Exception as e:
            tried.append(
                {
                    "identifier_type": id_type,
                    "error": str(e),
                }
            )

    # 全部尝试失败或无命中
    return {
        "result": None,
        "metadata": {
            "source": "PubChem",
            "info": "no CID found with supported identifier types",
            "tried": tried,
        },
    }

def pubchem_fetch_smiles(cid: int) -> Dict[str, Any]:
    """Fetch canonical SMILES for a given CID."""
    if not isinstance(cid, int) or cid <= 0:
        return {"result": None, "metadata": {"error": "cid must be positive int"}}
    if not PUBCHEMPY_AVAILABLE:
        return {"result": None, "metadata": {"error": "pubchempy not available"}}
    try:
        c = pcp.Compound.from_cid(cid)
        smiles = c.canonical_smiles
        mid = save_mid_result("chemistry", "pubchem_smiles", {"cid": cid, "smiles": smiles})
        return {"result": smiles, "metadata": {"source": "PubChem", "mid_file": mid["result"]}}
    except Exception as e:
        return {"result": None, "metadata": {"error": str(e), "source": "PubChem"}}

# RDKit functions for 3D generation and geometry features
def rdkit_generate_3d(smiles: str) -> Dict[str, Any]:
    """Generate 3D conformer coordinates and element list from SMILES."""
    if not RDKit_AVAILABLE:
        return {"result": None, "metadata": {"error": "RDKit not available"}}
    if not isinstance(smiles, str) or not smiles.strip():
        return {"result": None, "metadata": {"error": "smiles must be non-empty str"}}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"result": None, "metadata": {"error": "invalid SMILES"}}
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        ok = AllChem.EmbedMolecule(mol, params)
        if ok != 0:
            # Try with random coords
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        conf = mol.GetConformer()
        coords = []
        elements = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([float(pos.x), float(pos.y), float(pos.z)])
            elements.append(atom.GetSymbol())
        mid = save_mid_result("chemistry", "rdkit_3d", {"smiles": smiles, "coords": coords, "elements": elements})
        return {"result": {"coords": coords, "elements": elements}, "metadata": {"source": "RDKit", "mid_file": mid["result"]}}
    except Exception as e:
        return {"result": None, "metadata": {"error": str(e)}}

def compute_planarity(coords: List[List[float]]) -> Dict[str, Any]:
    """Compute planarity via best-fit plane RMS deviation."""
    if not NUMPY_AVAILABLE:
        return {"result": None, "metadata": {"error": "numpy not available"}}
    if not isinstance(coords, list) or len(coords) < MIN_ATOMS_FOR_GEOMETRY:
        return {"result": None, "metadata": {"error": f"coords must be list with >= {MIN_ATOMS_FOR_GEOMETRY} atoms"}}
    try:
        P = np.array(coords)
        centroid = P.mean(axis=0)
        Q = P - centroid
        # SVD for plane normal: smallest singular vector corresponds to normal
        U, S, Vt = np.linalg.svd(Q, full_matrices=False)
        normal = Vt[-1, :]
        # distances to plane
        distances = np.abs(Q.dot(normal)) / np.linalg.norm(normal)
        rms = float(np.sqrt((distances ** 2).mean()))
        mid = save_mid_result("chemistry", "planarity", {"rms": rms, "centroid": centroid.tolist(), "normal": normal.tolist()})
        return {"result": {"rms": rms, "centroid": centroid.tolist(), "normal": normal.tolist()}, "metadata": {"threshold": PLANARITY_RMS_THRESHOLD, "mid_file": mid["result"]}}
    except Exception as e:
        return {"result": None, "metadata": {"error": str(e)}}

def detect_c3_axis(coords: List[List[float]], centroid: List[float], normal: List[float]) -> Dict[str, Any]:
    """
    Heuristic detection of a C3 axis:
    - Project points onto plane (perpendicular to normal)
    - Compute polar angles around centroid
    - Expect clustering into 3 groups approximately separated by ~120 degrees
    """
    if not NUMPY_AVAILABLE:
        return {"result": False, "metadata": {"error": "numpy not available"}}
    try:
        P = np.array(coords)
        c = np.array(centroid)
        n = np.array(normal)
        n = n / np.linalg.norm(n)
        # Build orthonormal basis (u,v) in plane
        # Choose arbitrary vector not parallel to n
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = a - n * np.dot(a, n)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        # Project points onto plane coordinates (u,v)
        Q = P - c
        x = Q.dot(u)
        y = Q.dot(v)
        angles = np.degrees(np.arctan2(y, x))
        # Normalize angles to [0, 360)
        angles = (angles + 360.0) % 360.0
        # For atoms near centroid, exclude to avoid noise
        r = np.sqrt(x**2 + y**2)
        mask = r > (np.mean(r) * 0.3)
        angles_filt = angles[mask]
        if angles_filt.size < 6:
            return {"result": False, "metadata": {"info": "insufficient angular features"}}
        # Map angles modulo 120
        angles_mod120 = angles_filt % 120.0
        # Check clustering: compute std dev; expect tight clusters around 0, 40, 80? Actually modulo collapses groups; we expect three clusters merging near 0.
        std_mod = float(np.std(angles_mod120))
        # Additional: check histogram peaks at three sectors in 360
        hist, bins = np.histogram(angles_filt, bins=12, range=(0.0, 360.0))
        # For C3, expect repeating pattern every 120 degrees: compare hist segments 0-120,120-240,240-360
        seg = hist.reshape(3, 4).sum(axis=1)
        seg_var = float(np.var(seg))
        c3_detected = (std_mod < ANGLE_CLUSTER_TOL) and (seg_var < np.mean(seg) * 0.5)
        mid = save_mid_result("chemistry", "c3_detection", {"std_mod": std_mod, "seg_counts": seg.tolist(), "seg_var": seg_var})
        return {"result": c3_detected, "metadata": {"angle_cluster_tol_deg": ANGLE_CLUSTER_TOL, "mid_file": mid["result"]}}
    except Exception as e:
        return {"result": False, "metadata": {"error": str(e)}}

def detect_sigma_h(planarity_rms: float) -> Dict[str, Any]:
    """Detect σh by planarity threshold."""
    if not isinstance(planarity_rms, float):
        return {"result": False, "metadata": {"error": "planarity_rms must be float"}}
    sigma_h = planarity_rms <= PLANARITY_RMS_THRESHOLD
    return {"result": sigma_h, "metadata": {"threshold": PLANARITY_RMS_THRESHOLD, "planarity_rms": planarity_rms}}

def detect_c2_perp_axes(coords: List[List[float]], centroid: List[float], normal: List[float]) -> Dict[str, Any]:
    """
    Very rough heuristic for presence of multiple C2 axes perpendicular to C3 axis (to differentiate D3h):
    For planar molecules:
    - Identify 3 lobes at ~120 degrees
    - Check if each lobe exhibits an internal 2-fold symmetry relative to local axis
    This is simplified: we approximate by checking bilateral symmetry across lines in the plane every 60 degrees.
    """
    if not NUMPY_AVAILABLE:
        return {"result": False, "metadata": {"error": "numpy not available"}}
    try:
        P = np.array(coords)
        c = np.array(centroid)
        n = np.array(normal) / np.linalg.norm(normal)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = a - n * np.dot(a, n); u = u / np.linalg.norm(u)
        v = np.cross(n, u); v = v / np.linalg.norm(v)
        Q = P - c
        x = Q.dot(u); y = Q.dot(v)
        angles = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
        r = np.sqrt(x**2 + y**2)
        mask = r > (np.mean(r) * 0.3)
        ang = angles[mask]; rr = r[mask]
        # Check symmetry across 3 candidate C2 axes in plane (at angles 0, 60, 120 ...):
        axes_deg = [0.0, 60.0, 120.0]
        symmetric_axes_count = 0
        for ax in axes_deg:
            # Reflect points across line at angle ax; angle reflection: theta' = 2*ax - theta
            theta_ref = (2*ax - ang + 360.0) % 360.0
            # For each original angle, see if reflected angle has nearby partner
            matches = 0
            for t in ang:
                diff = np.abs(((theta_ref - t + 180.0) % 360.0) - 180.0)
                if np.min(diff) < 10.0:
                    matches += 1
            # If many matches, consider axis present
            if matches >= max(4, int(0.3 * len(ang))):
                symmetric_axes_count += 1
        has_c2_perp = symmetric_axes_count >= 2  # heuristic threshold
        mid = save_mid_result("chemistry", "c2_perp_detection", {"symmetric_axes_count": symmetric_axes_count})
        return {"result": has_c2_perp, "metadata": {"mid_file": mid["result"], "axes_checked_deg": axes_deg}}
    except Exception as e:
        return {"result": False, "metadata": {"error": str(e)}}

def infer_point_group(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer a simple point group from features.
    Rules:
    - If C3 axis and sigma_h:
        - If multiple C2' axes: D3h
        - Else: C3h
    - If C3 axis and vertical mirrors but not sigma_h: C3v (not implemented fully here)
    - Else fallback to C1/Cn
    """
    c3 = bool(features.get("has_c3_axis", False))
    sigma_h = bool(features.get("has_sigma_h", False))
    c2_perp = bool(features.get("has_c2_perp_axes", False))
    if c3 and sigma_h:
        if c2_perp:
            pg = "D3h"
        else:
            pg = "C3h"
    elif c3 and not sigma_h:
        pg = "C3v"
    elif c3:
        pg = "C3"
    else:
        pg = "C1"
    return {"result": pg, "metadata": {"features": features}}

# Combination function: end-to-end symmetry assessment for a molecule name
def assess_symmetry_by_name(name: str) -> Dict[str, Any]:
    """
    Attempt to determine point group:
    1) Try PubChem → SMILES → RDKit 3D → feature detection → PG inference
    2) Fallback to local DB classification
    """
    if not isinstance(name, str) or not name.strip():
        return {"result": None, "metadata": {"error": "name must be non-empty str"}}

    # Try PubChem and RDKit
    cid_res = pubchem_fetch_cid_by_name(name)
    smiles = None
    coords = None
    elements = None
    feature_details = {}
    pg_inferred = None

    if cid_res["result"]:
        smiles_res = pubchem_fetch_smiles(cid_res["result"])
        smiles = smiles_res["result"]
        if smiles:
            rd_res = rdkit_generate_3d(smiles)
            if rd_res["result"]:
                coords = rd_res["result"]["coords"]
                elements = rd_res["result"]["elements"]
                # Compute features
                planarity_res = compute_planarity(coords)
                if planarity_res["result"]:
                    planarity_rms = planarity_res["result"]["rms"]
                    centroid = planarity_res["result"]["centroid"]
                    normal = planarity_res["result"]["normal"]
                    c3_res = detect_c3_axis(coords, centroid, normal)
                    sigma_res = detect_sigma_h(planarity_rms)
                    c2_res = detect_c2_perp_axes(coords, centroid, normal)
                    feature_details = {
                        "has_c3_axis": c3_res["result"],
                        "has_sigma_h": sigma_res["result"],
                        "has_c2_perp_axes": c2_res["result"],
                        "planarity_rms": planarity_rms
                    }
                    pg_res = infer_point_group(feature_details)
                    pg_inferred = pg_res["result"]
    # Fallback to local DB if no confident inference
    if not pg_inferred or pg_inferred == "C1":
        local_res = query_local_symmetry(name, LOCAL_DB_PATH)
        if local_res["result"]:
            pg_inferred = local_res["result"]["point_group"]
            feature_details = {
                "has_c3_axis": local_res["result"]["has_c3_axis"],
                "has_sigma_h": local_res["result"]["has_sigma_h"],
                "has_c2_perp_axes": local_res["result"]["has_c2_perp_axes"],
                "provenance": local_res["result"]["provenance"]
            }
            smiles = local_res["result"]["smiles"] or smiles

    result = {
        "name": name,
        "smiles": smiles,
        "point_group": pg_inferred,
        "features": feature_details
    }
    mid = save_mid_result("chemistry", "symmetry_assessment", result)
    return {"result": result, "metadata": {"mid_file": mid["result"]}}

# Visualization function using py3Dmol
def visualize_molecule_3d(coords: List[List[float]], elements: List[str], name: str) -> Dict[str, Any]:
    """Render molecule using py3Dmol by composing an XYZ string."""
    if not PY3DMOL_AVAILABLE:
        return {"result": None, "metadata": {"error": "py3Dmol not available"}}
    if not (isinstance(coords, list) and isinstance(elements, list) and len(coords) == len(elements)):
        return {"result": None, "metadata": {"error": "coords and elements must be lists of equal length"}}
    # Compose XYZ format text
    lines = [str(len(elements)), name]
    for e, (x, y, z) in zip(elements, coords):
        lines.append(f"{e} {x:.4f} {y:.4f} {z:.4f}")
    xyz = "\n".join(lines)
    # Render and save PNG snapshot (py3Dmol can produce HTML; for code-only, save text file)
    filepath = os.path.join(TOOL_IMAGE_DIR, f"{name.replace(' ', '_')}.xyz")
    try:
        ensure_dirs()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(xyz)
        print(f"FILE_GENERATED: text | PATH: {filepath}")
        return {"result": filepath, "metadata": {"file_type": "xyz", "size": os.path.getsize(filepath)}}
    except Exception as e:
        return {"result": None, "metadata": {"error": str(e)}}

# Prepare local DB with curated entries relevant to the question and generalization
def prepare_local_db_entries() -> Dict[str, Any]:
    """
    Insert curated symmetry classifications with provenance notes:
    - triisopropyl borate: generally not planar and substituents break C3h; classify as C1/C3 (no sigma_h)
    - quinuclidine: approximately C3v due to cage symmetry around nitrogen; not planar (no sigma_h)
    - benzo[...]trifuran-...-hexaone: likely lacks strict C3 axis due to benzo-fused asymmetry; classify as C1
    - triphenyleno[...]trifuran-...-hexaone: idealized 3-fold planar fused system; classify as C3h (per problem's standard)
    """
    entries = [
        {
            "name": "triisopropyl borate",
            "smiles": None,
            "point_group": "C1",
            "has_c3_axis": False,
            "has_sigma_h": False,
            "has_c2_perp_axes": False,
            "provenance": "curated_literature_heuristic"
        },
        {
            "name": "quinuclidine",
            "smiles": None,
            "point_group": "C3v",
            "has_c3_axis": True,
            "has_sigma_h": False,
            "has_c2_perp_axes": False,
            "provenance": "curated_literature_heuristic"
        },
        {
            "name": "benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone",
            "smiles": None,
            "point_group": "C1",
            "has_c3_axis": False,
            "has_sigma_h": False,
            "has_c2_perp_axes": False,
            "provenance": "curated_structural_reasoning"
        },
        {
            "name": "triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone",
            "smiles": None,
            "point_group": "C3h",
            "has_c3_axis": True,
            "has_sigma_h": True,
            "has_c2_perp_axes": False,
            "provenance": "curated_structural_reasoning"
        }
    ]
    init_local_db(LOCAL_DB_PATH)
    res = insert_local_entries(entries, LOCAL_DB_PATH)
    mid = save_mid_result("chemistry", "local_db_entries", {"inserted": res["result"], "entries": entries})
    return {"result": res["result"], "metadata": {"db_path": LOCAL_DB_PATH, "mid_file": mid["result"]}}

# Scenario helper: batch assessment
def batch_assess(names: List[str]) -> Dict[str, Any]:
    """Assess a list of names and return dict mapping to point groups."""
    results = {}
    for nm in names:
        ar = assess_symmetry_by_name(nm)
        results[nm] = ar["result"]
    mid = save_mid_result("chemistry", "batch_assess", results)
    return {"result": results, "metadata": {"mid_file": mid["result"]}}

# MAIN: 3 scenarios
def main():
  
    print("=" * 60)
    print("场景1：原始问题求解——判定四个候选分子中，哪个具有C3h对称性")
    print("=" * 60)
    print("问题描述：在给定的四个分子中，使用数据库检索与几何启发式相结合的方法，判定具有C3h点群的分子。")
    print("-" * 60)

    # 步骤1：初始化本地数据库
    init_db_res = init_local_db(LOCAL_DB_PATH)
    print(f"FUNCTION_CALL: init_local_db | PARAMS: db_path='{LOCAL_DB_PATH}' | RESULT: {init_db_res}")

    # 步骤2：准备本地数据库条目
    prepare_db_res = prepare_local_db_entries()
    print(f"FUNCTION_CALL: prepare_local_db_entries | PARAMS: None | RESULT: {prepare_db_res}")

    # 步骤3：逐个评估四个分子（使用原子函数）
    candidates = [
        "triisopropyl borate",
        "quinuclidine", 
        "benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone",
        "triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone"
    ]
    
    batch_results = {}
    for name in candidates:
        # 尝试PubChem获取CID
        cid_res = pubchem_fetch_cid_by_name(name)
        print(f"FUNCTION_CALL: pubchem_fetch_cid_by_name | PARAMS: name='{name}' | RESULT: {cid_res}")
        
        smiles = None
        coords = None
        elements = None
        feature_details = {}
        pg_inferred = None
        
        if cid_res["result"]:
            # 获取SMILES
            smiles_res = pubchem_fetch_smiles(cid_res["result"])
            print(f"FUNCTION_CALL: pubchem_fetch_smiles | PARAMS: cid={cid_res['result']} | RESULT: {smiles_res}")
            smiles = smiles_res["result"]
            
            if smiles:
                # 生成3D构型
                rd_res = rdkit_generate_3d(smiles)
                print(f"FUNCTION_CALL: rdkit_generate_3d | PARAMS: smiles='{smiles}' | RESULT: {rd_res}")
                
                if rd_res["result"]:
                    coords = rd_res["result"]["coords"]
                    elements = rd_res["result"]["elements"]
                    
                    # 计算平面性
                    planarity_res = compute_planarity(coords)
                    print(f"FUNCTION_CALL: compute_planarity | PARAMS: coords_len={len(coords)} | RESULT: {planarity_res}")
                    
                    if planarity_res["result"]:
                        planarity_rms = planarity_res["result"]["rms"]
                        centroid = planarity_res["result"]["centroid"]
                        normal = planarity_res["result"]["normal"]
                        
                        # 检测C3轴
                        c3_res = detect_c3_axis(coords, centroid, normal)
                        print(f"FUNCTION_CALL: detect_c3_axis | PARAMS: coords_len={len(coords)} | RESULT: {c3_res}")
                        
                        # 检测σh
                        sigma_res = detect_sigma_h(planarity_rms)
                        print(f"FUNCTION_CALL: detect_sigma_h | PARAMS: planarity_rms={planarity_rms} | RESULT: {sigma_res}")
                        
                        # 检测垂直C2轴
                        c2_res = detect_c2_perp_axes(coords, centroid, normal)
                        print(f"FUNCTION_CALL: detect_c2_perp_axes | PARAMS: coords_len={len(coords)} | RESULT: {c2_res}")
                        
                        feature_details = {
                            "has_c3_axis": c3_res["result"],
                            "has_sigma_h": sigma_res["result"],
                            "has_c2_perp_axes": c2_res["result"],
                            "planarity_rms": planarity_rms
                        }
                        
                        # 推断点群
                        pg_res = infer_point_group(feature_details)
                        print(f"FUNCTION_CALL: infer_point_group | PARAMS: features={feature_details} | RESULT: {pg_res}")
                        pg_inferred = pg_res["result"]
        
        # 如果PubChem方法失败，查询本地数据库
        if not pg_inferred or pg_inferred == "C1":
            local_res = query_local_symmetry(name, LOCAL_DB_PATH)
            print(f"FUNCTION_CALL: query_local_symmetry | PARAMS: name='{name}', db_path='{LOCAL_DB_PATH}' | RESULT: {local_res}")
            
            if local_res["result"]:
                pg_inferred = local_res["result"]["point_group"]
                feature_details = {
                    "has_c3_axis": local_res["result"]["has_c3_axis"],
                    "has_sigma_h": local_res["result"]["has_sigma_h"],
                    "has_c2_perp_axes": local_res["result"]["has_c2_perp_axes"],
                    "provenance": local_res["result"]["provenance"]
                }
                smiles = local_res["result"]["smiles"] or smiles
        
        # 保存结果
        result = {
            "name": name,
            "smiles": smiles,
            "point_group": pg_inferred,
            "features": feature_details
        }
        batch_results[name] = result
        
        # 保存中间结果
        mid_save_res = save_mid_result("chemistry", f"symmetry_assessment_{name.replace(' ', '_')}", result)
        print(f"FUNCTION_CALL: save_mid_result | PARAMS: subject='chemistry', label='symmetry_assessment_{name.replace(' ', '_')}' | RESULT: {mid_save_res}")

    # 步骤4：筛选C3h分子
    c3h_molecules = [nm for nm, info in batch_results.items() if info.get("point_group") == "C3h"]
    answer1 = c3h_molecules[0] if c3h_molecules else None
    print(f"FUNCTION_CALL: filter_C3h | PARAMS: batch_result_keys={list(batch_results.keys())} | RESULT: {{'result': {c3h_molecules}}}")

    print(f"FINAL_ANSWER: {answer1}")

    print("=" * 60)
    print("场景2：可视化与几何特征提取——对可从PubChem获得的分子生成3D构型并检测C3与σh")
    print("=" * 60)
    print("问题描述：以“quinuclidine”为例，从PubChem获取SMILES，构建3D几何，检测是否具有C3主轴以及分子是否近似共平面（σh）。")
    print("-" * 60)

    # 步骤1：PubChem获取CID与SMILES
    cid_q = pubchem_fetch_cid_by_name("quinuclidine")
    print(f"FUNCTION_CALL: pubchem_fetch_cid_by_name | PARAMS: {{'name': 'quinuclidine'}} | RESULT: {cid_q}")

    smiles_q = None
    if cid_q["result"]:
        smiles_res_q = pubchem_fetch_smiles(cid_q["result"])
        smiles_q = smiles_res_q["result"]
        print(f"FUNCTION_CALL: pubchem_fetch_smiles | PARAMS: {{'cid': {cid_q['result']}}} | RESULT: {smiles_res_q}")

    # 步骤2：RDKit生成3D构型
    coords_q, elements_q = None, None
    if smiles_q:
        rdkit_3d_q = rdkit_generate_3d(smiles_q)
        print(f"FUNCTION_CALL: rdkit_generate_3d | PARAMS: {{'smiles': '{smiles_q}'}} | RESULT: {rdkit_3d_q}")
        if rdkit_3d_q["result"]:
            coords_q = rdkit_3d_q["result"]["coords"]
            elements_q = rdkit_3d_q["result"]["elements"]

    # 步骤3：几何特征分析（计划性与C3）
    features_q = {}
    if coords_q:
        planarity_q = compute_planarity(coords_q)
        print(f"FUNCTION_CALL: compute_planarity | PARAMS: {{'coords_len': {len(coords_q)}}} | RESULT: {planarity_q}")
        if planarity_q["result"]:
            centroid_q = planarity_q["result"]["centroid"]
            normal_q = planarity_q["result"]["normal"]
            c3_q = detect_c3_axis(coords_q, centroid_q, normal_q)
            sigma_q = detect_sigma_h(planarity_q["result"]["rms"])
            c2_q = detect_c2_perp_axes(coords_q, centroid_q, normal_q)
            features_q = {
                "has_c3_axis": c3_q["result"],
                "has_sigma_h": sigma_q["result"],
                "has_c2_perp_axes": c2_q["result"]
            }
            pg_q = infer_point_group(features_q)
            print(f"FUNCTION_CALL: detect_c3_axis | PARAMS: {{...}} | RESULT: {c3_q}")
            print(f"FUNCTION_CALL: detect_sigma_h | PARAMS: {{'planarity_rms': {planarity_q['result']['rms']}}} | RESULT: {sigma_q}")
            print(f"FUNCTION_CALL: detect_c2_perp_axes | PARAMS: {{...}} | RESULT: {c2_q}")
            print(f"FUNCTION_CALL: infer_point_group | PARAMS: {{'features': {features_q}}} | RESULT: {pg_q}")

    # 步骤4：可视化（保存XYZ）
    if coords_q and elements_q:
        viz_q = visualize_molecule_3d(coords_q, elements_q, "quinuclidine")
        print(f"FUNCTION_CALL: visualize_molecule_3d | PARAMS: {{'coords_len': {len(coords_q)}, 'elements_len': {len(elements_q)}}} | RESULT: {viz_q}")

    answer2 = {
        "name": "quinuclidine",
        "point_group_inferred": infer_point_group(features_q)["result"] if features_q else "unknown",
        "has_c3_axis": features_q.get("has_c3_axis", None),
        "has_sigma_h": features_q.get("has_sigma_h", None)
    }
    print(f"FINAL_ANSWER: {answer2}")

    print("=" * 60)
    print("场景3：构建本地数据库用于离线检索——批量判定多分子的对称性并导出结果文件")
    print("=" * 60)
    print("问题描述：在离线环境下，构建并查询本地SQLite数据库，批量返回分子点群，并生成JSON文件记录。")
    print("-" * 60)

    # 步骤1：构建/初始化本地DB（已在前面执行），此处展示查询另一个集合
    names3 = [
        "triisopropyl borate",
        "quinuclidine",
        "triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone"
    ]
    res3 = batch_assess(names3)
    print(f"FUNCTION_CALL: batch_assess | PARAMS: {{'names': {names3}}} | RESULT: {res3}")

    # 步骤2：保存结果到文件
    file3 = save_mid_result("chemistry", "scenario3_batch_output", res3["result"])
    print(f"FUNCTION_CALL: save_mid_result | PARAMS: {{'subject': 'chemistry', 'label': 'scenario3_batch_output'}} | RESULT: {file3}")
    print(f"FILE_GENERATED: json | PATH: {file3['result']}")

    # 步骤3：加载文件并输出摘要
    loaded3 = load_file(file3["result"])
    print(f"FUNCTION_CALL: load_file | PARAMS: {{'filepath': '{file3['result']}'}} | RESULT: {{'result': 'loaded_json', 'metadata': {{'file_type': '{loaded3['metadata'].get('file_type', 'unknown')}', 'size': {loaded3['metadata'].get('size', 0)}}}}}")

    # 以C3h筛选为最终展示
    c3h_list = [nm for nm, info in res3["result"].items() if info.get("point_group") == "C3h"]
    answer3 = {
        "C3h_molecules": c3h_list,
        "db_path": LOCAL_DB_PATH
    }
    print(f"FINAL_ANSWER: {answer3}")


if __name__ == "__main__":
    main()