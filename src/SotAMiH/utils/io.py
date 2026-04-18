import pandas as pd

from pathlib import Path

from src.SotAMiH.core.simulation import Simulation
from src.SotAMiH.core.mesh import Mesh1D

def write_1D_simulation_results(simulation: Simulation, output_path: Path, name: str):
    if not isinstance(simulation.mesh, Mesh1D):
        raise TypeError("Please provide a simulation with a 1D Mesh")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_headers = "Name,Length,Resolution"
    metadata_values = f"{name},{simulation.mesh.length},{simulation.mesh.dx}"

    dfs = []
    
    for time, Q_array in simulation.saved_times.items():
        eta = Q_array[1:-1,0]
        q = Q_array[1:-1,1]

        temp_df = pd.DataFrame({
            "Time" : time,
            "x" : simulation.mesh.x_vals,
            "eta" : eta,
            "q" : q
        })

        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)

    with open(output_path, "w", newline="") as file:
        file.write(metadata_headers+"\n")
        file.write(metadata_values+"\n")
        file.write("\n")
        df.to_csv(file, index=False)