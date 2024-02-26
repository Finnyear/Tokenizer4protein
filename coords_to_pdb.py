import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

aa_1_to_3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP',
    'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY',
    'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER',
    'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

def write_coords_to_pdb(coords: np.ndarray, res_names: list, out_fname: str) -> str:
    """
    Write the coordinates to the given pdb fname
    """
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    # assert len(coords) % 3 == 0

    atoms = []
    for i, (ca_coord, res_names) in enumerate(zip(coords, res_names)):

        atom = struc.Atom(
            ca_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i + 1,
            res_name=res_names,
            atom_name="CA",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )

        atoms.extend([atom])
    full_structure = struc.array(atoms)


    sink = PDBFile()
    sink.set_structure(full_structure)
    sink.write(out_fname)
    return out_fname

def process_input_data(file_path):
    num = 0
    with open(file_path, 'r') as file:
        for line in file:
            coords = []
            res_names = []
            num = num + 1
            line = line.split('\t')[0]
            parts = line.split()
            for i in range(0, len(parts), 4):  # Process every block of 4 elements (1 letter code + 3 coordinates)
                if i + 3 >= len(parts):
                    break  # Ensure there are enough parts for another amino acid block
                aa_code, x, y, z = parts[i], float(parts[i+1]), float(parts[i+2]), float(parts[i+3])
                res_name = aa_1_to_3.get(aa_code)  # Convert 1 letter code to 3 letter code
                if res_name:  # If the amino acid code is valid
                    coords.append([x, y, z])
                    res_names.append(res_name)
            out_fname = "EC/EC_train_PDB/" + str(num) + ".pdb"
            write_coords_to_pdb(coords, res_names, out_fname)

process_input_data("EC/EC_new_train.txt")

