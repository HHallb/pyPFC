"""
Example 06: HDF5 file I/O in pyPFC

This example shows how to map a typical legacy pickle payload list into the newer
save_hdf5() group structure.

Legacy payload style (common in earlier examples):
    [step, total_time, ndiv, domain_size, den, state_output]

New HDF5 layout used here:
    /meta         -> scalar metadata and run info
    /simulation   -> simulation control values, defaults to the pyPFC configuration parameters
    /grid         -> grid/domain data
    /fields       -> field_data array and matching field_labels
    /atoms        -> coords, atom_data, atom_data_labels
    /diagnostics  -> time series and scalar diagnostics
    /user         -> optional custom data
"""

import os
import numpy as np
import pypfc

# Simulation-specific settings
# ============================
output_path = "./examples/ex06_output/"
filename    = output_path + "step_0001000"

# Define domain and create simulation object
# ==========================================
domain_size = np.array([16.0, 16.0, 16.0], dtype=float)
ndiv        = np.array([64, 64, 64], dtype=int)
sim         = pypfc.setup_simulation(domain_size, ndiv, verbose=False, device_type="cpu")

# Ensure output directory exists
# ==============================
os.makedirs(output_path, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) Build a typical legacy payload (old pickle style)
# -----------------------------------------------------------------------------
step       = 1000
total_time = 1.0

# Example density field and simple diagnostics array
rng = np.random.default_rng(42)
den = rng.standard_normal(ndiv)
state_output = np.array([
    [0.0, 100.0, -0.15],
    [0.5, 110.0, -0.16],
    [1.0, 120.0, -0.17],
], dtype=float)

legacy_payload = [step, total_time, ndiv, domain_size, den, state_output]

# -----------------------------------------------------------------------------
# 2) Map legacy payload to save_hdf5() groups
# -----------------------------------------------------------------------------
legacy_step, legacy_total_time, legacy_ndiv, legacy_domain_size, legacy_den, legacy_state = legacy_payload

meta = {
    "step": int(legacy_step),
    "simulation_time": float(legacy_total_time),
    "run_id": "ex06_demo",
}

simulation = {
    "dtime": float(sim.get_dtime()),
    "update_scheme": str(sim.get_update_scheme()),
}

grid = {
    "ndiv": np.asarray(legacy_ndiv, dtype=int),
    "domain_size": np.asarray(legacy_domain_size, dtype=float),
    "ddiv": np.asarray(legacy_domain_size, dtype=float) / np.asarray(legacy_ndiv, dtype=float),
}

fields = {
    "density": np.asarray(legacy_den, dtype=float),
}

# Atoms are optional. Here we create synthetic data for the /atoms layout.
atom_coords = np.array([
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0],
], dtype=float)

atom_data = np.array([
    [0.45, -0.10],
    [0.50, -0.09],
    [0.55, -0.08],
], dtype=float)

atom_data_labels = ["den", "ene"]
atoms = {
    "coords": atom_coords,
    "atom_data": atom_data,
    "atom_data_labels": atom_data_labels,
}

diagnostics = {
    "state_output": np.asarray(legacy_state, dtype=float),
}

user_data = {
    "notes": "Mapped from legacy pickle-style payload list.",
}

# -----------------------------------------------------------------------------
# 3) Save and load one HDF5 file
# -----------------------------------------------------------------------------
sim.save_hdf5(
    filename,
    meta=meta,
    simulation=simulation,
    grid=grid,
    fields=fields,
    atoms=atoms,
    diagnostics=diagnostics,
    user_data=user_data,
)

loaded = sim.load_hdf5(filename)

# -----------------------------------------------------------------------------
# 4) Print a compact verification summary
# -----------------------------------------------------------------------------
print("Saved file:", filename + ".h5")
print("Schema:", loaded["root_attrs"].get("schema_name"), loaded["root_attrs"].get("schema_version"))
print("meta.step:", loaded["meta"].get("step"))
print("fields.field_data shape:", loaded["fields"]["field_data"].shape)
print("fields.field_labels:", loaded["fields"]["field_labels"])
density_index = loaded["fields"]["field_labels"].index("density")
print("density field shape:", loaded["fields"]["field_data"][density_index].shape)
print("atoms.coords shape:", loaded["atoms"]["coords"].shape)
print("atoms.atom_data shape:", loaded["atoms"]["atom_data"].shape)
print("atoms.atom_data_labels:", loaded["atoms"]["atom_data_labels"])

# Cleanup
# =======
sim.cleanup()
