import os
import sys
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Structure as BioStructure, Model as BioModel
from Bio.PDB import Polypeptide
import warnings

try:
    import freesasa
except ImportError:
    print("Error: The 'freesasa' library is not installed.", file=sys.stderr)
    print("Please install it using: pip install freesasa", file=sys.stderr)
    sys.exit(1)

from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter('ignore', PDBConstructionWarning)

DELTA_SASA_THRESHOLD = 1.0  # 根据你的定义，保持1.0 Å²


def get_sasa(structure, target_chain_id=None):
    """
    Calculates SASA for each residue in a structure using the FreeSASA library,
    following the officially recommended integration method with Biopython.

    Args:
        structure: Biopython Structure object
        target_chain_id: If specified, only return SASA for this chain, but calculate
                        considering all chains in the structure
    """
    try:
        # 1. Create a FreeSASA Structure object from the Biopython Structure.
        fs_structure = freesasa.structureFromBioPDB(structure)

        # 2. Run the SASA calculation considering ALL chains in the structure
        result = freesasa.calc(fs_structure)

        # 3. Extract per-residue SASA values from the result.
        residue_areas = result.residueAreas()

        sasa_map = {}
        for chain_id, residues in residue_areas.items():
            # 如果指定了目标链，只返回该链的数据
            if target_chain_id is not None and chain_id != target_chain_id:
                continue

            for res_num, area in residues.items():
                sasa_map[(chain_id, res_num)] = area.total

        return sasa_map

    except Exception as e:
        print(f"FreeSASA execution failed: {e}", file=sys.stderr)
        return None


def main():
    input_file = '../cleaned_dataset.csv' # Use the high-resolution dataset
    pdb_dir = '../pdb'
    output_file = '../sasa_labeled_dataset.csv'
    temp_pdb_file = '../temp_monomer.pdb'

    if not os.path.exists(input_file):
        print(f"Error: Input file '{os.path.basename(input_file)}' not found.", file=sys.stderr)
        print("Please run the resolution filtering script first.", file=sys.stderr)
        sys.exit(1)

    parser = PDBParser()
    pdb_io = PDBIO()
    final_data = []

    input_df = pd.read_csv(input_file)

    print("Starting label generation using PDB files and FreeSASA...")
    print(f"Using SASA reduction threshold: {DELTA_SASA_THRESHOLD} Å²")

    for idx, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Processing Complexes"):
        pid = row['PID']
        pdb_id = row['PDB_ID']
        chain_id = str(row['Chain_ID'])

        pdb_path = os.path.join(pdb_dir, f"{pdb_id.upper()}.pdb")
        if not os.path.exists(pdb_path):
            continue

        try:
            structure = parser.get_structure(pdb_id, pdb_path)
            model = structure[0]
            target_chain = model[chain_id]
        except Exception:
            continue

        # 计算复合物中目标链的SASA（考虑其他链的遮蔽效应）
        sasa_complex = get_sasa(structure, target_chain_id=chain_id)
        if sasa_complex is None: continue

        # 创建单体结构的临时PDB文件
        monomer_structure_builder = BioStructure.Structure("monomer")
        model_builder = BioModel.Model(0)

        # 复制目标链到新结构中
        target_chain_copy = target_chain.copy()
        model_builder.add(target_chain_copy)
        monomer_structure_builder.add(model_builder)

        pdb_io.set_structure(monomer_structure_builder)
        pdb_io.save(temp_pdb_file)

        # 计算单体中的SASA（只有目标链，无遮蔽）
        try:
            monomer_structure_parser = parser.get_structure(f"{pdb_id}_{chain_id}_monomer", temp_pdb_file)
            sasa_monomer = get_sasa(monomer_structure_parser, target_chain_id=chain_id)
        finally:
            if os.path.exists(temp_pdb_file): os.remove(temp_pdb_file)

        if sasa_monomer is None: continue

        labels = []
        sequence = ""
        positive_labels = 0

        for residue in target_chain:
            res_name = residue.get_resname()
            if res_name in Polypeptide.protein_letters_3to1:
                sequence += Polypeptide.protein_letters_3to1[res_name]
                res_id = residue.id[1]

                # 尝试多种可能的key格式
                possible_keys = [
                    (chain_id, str(res_id)),
                    (chain_id, res_id),
                    (chain_id.upper(), str(res_id)),
                    (chain_id.lower(), str(res_id)),
                    str(res_id),
                    res_id
                ]

                monomer_sasa = None
                complex_sasa = None

                # 在单体中找匹配的key
                for key in possible_keys:
                    if key in sasa_monomer:
                        monomer_sasa = sasa_monomer[key]
                        break

                # 在复合物中找匹配的key
                for key in possible_keys:
                    if key in sasa_complex:
                        complex_sasa = sasa_complex[key]
                        break

                if monomer_sasa is not None and complex_sasa is not None:
                    # SASA减少量 = 单体SASA - 复合物SASA
                    sasa_reduction = monomer_sasa - complex_sasa

                    # 根据定义：SASA减少≥1.0 Å²的残基为相互作用位点
                    label = 1 if sasa_reduction >= DELTA_SASA_THRESHOLD else 0
                    labels.append(label)

                    if label == 1:
                        positive_labels += 1
                else:
                    labels.append(0)

        if len(sequence) != len(labels): continue

        final_data.append({
            "PID": pid,
            "Sequence": sequence,
            "Labels": "".join(map(str, labels)),
            "PDB_ID": pdb_id,
            "Chain_ID": chain_id
        })

    final_df = pd.DataFrame(final_data)
    final_df.to_csv(output_file, index=False)

    print(f"\n--- Report ---")
    print(f"Successfully generated labels for {len(final_df)} protein chains.")
    print(f"Final dataset saved to: {output_file}")
    print("----------------")


if __name__ == '__main__':
    main()