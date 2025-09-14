import os
import numpy as np
import pickle
import pandas as pd
from Bio.PDB import MMCIFParser, StructureBuilder, PDBIO
import torch
from tqdm import tqdm
import argparse
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

class StructureProcessor:
    def __init__(self, save_dir=None, csv_path=None):
        """
        初始化结构处理器
        Args:
            save_dir: 保存处理后数据的目录
            csv_path: 包含PID、Chain ID和序列信息的CSV文件路径
        """
        self.parser = MMCIFParser(QUIET=True)
        self.save_dir = save_dir or 'processed_data'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 读取CSV文件
        self.chain_info = {}
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                pid = row['PID'].lower()
                if pid not in self.chain_info:
                    self.chain_info[pid] = {}
                self.chain_info[pid][row['Chain_ID']] = {
                    'sequence': row['Sequence']
                }
    
    def process_cif(self, cif_path, pdb_id=None):
        """从CIF文件中提取原子和残基的3D坐标"""
        try:
            # 如果没有提供pdb_id，从文件名中提取
            if pdb_id is None:
                pdb_id = os.path.splitext(os.path.basename(cif_path))[0]
            
            # 解析CIF文件
            structure = self.parser.get_structure(pdb_id, cif_path)
            
            # 存储数据的列表
            atom_coords = []      # 原子坐标 [N, 3]
            residue_coords = []   # 残基坐标 [M, 3]
            chain_ids = []        # 链ID列表
            residue_names = []    # 残基名称列表
            
            residue_count = 0
            atom_count = 0
            
            # 遍历结构
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # 计算残基中心坐标
                        res_coord = np.zeros(3)
                        res_atoms = []
                        
                        # 首先收集残基中的所有原子
                        for atom in residue:
                            res_coord += atom.get_coord()
                            res_atoms.append(atom)
                        
                        if len(res_atoms) > 0:  # 确保残基有原子
                            res_coord = res_coord / len(res_atoms)
                            residue_coords.append(res_coord)
                            
                            # 记录残基信息
                            chain_ids.append(chain.id)
                            residue_names.append(residue.resname)
                            
                            # 处理残基中的每个原子
                            for atom in res_atoms:
                                # 记录原子坐标
                                atom_pos = atom.get_coord()
                                atom_coords.append(atom_pos)
                                
                                atom_count += 1
                            residue_count += 1
            
            # 获取链信息
            chain_data = {}
            if pdb_id.lower() in self.chain_info:
                chain_data = self.chain_info[pdb_id.lower()]
            
            # 转换为张量
            atom_coords = torch.tensor(atom_coords, dtype=torch.float)
            residue_coords = torch.tensor(residue_coords, dtype=torch.float)
            
            data = {
                'pdb_id': pdb_id,
                'atom_pos': atom_coords,
                'residue_pos': residue_coords,
                'chain_ids': chain_ids,
            }
            
            return data
            
        except Exception as e:
            print(f"处理{cif_path}时出错: {str(e)}")
            return None
    
    def process_directory(self, cif_dir):
        """批量处理目录中的所有CIF文件并保存到一个pkl文件中"""
        # 获取所有CIF文件
        cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
        
        # 使用tqdm显示进度
        processed_data = {}
        for cif_file in tqdm(cif_files, desc="处理CIF文件"):
            cif_path = os.path.join(cif_dir, cif_file)
            data = self.process_cif(cif_path)
            if data is not None:
                pdb_id = os.path.splitext(cif_file)[0]
                processed_data[pdb_id] = data
        
        # 保存所有处理后的数据到一个pkl文件
        if self.save_dir:
            save_path = os.path.join(self.save_dir, "Tuning_test_POS.pkl")
            print(f"正在保存所有数据到：{save_path}")
            with open(save_path, 'wb') as f:
                pickle.dump(processed_data, f)
            print(f"数据保存完成！共处理了 {len(processed_data)} 个结构")
        
        return processed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='处理CIF文件提取3D坐标信息')
    parser.add_argument('--input', type=str, default='../cif/Tuning',
                      help='输入CIF文件路径或包含CIF文件的目录路径')
    parser.add_argument('--output', type=str, default='processed_data',
                      help='输出目录路径 (默认: processed_data)')
    parser.add_argument('--csv', type=str, default='../Test_label.csv',
                      help='包含PID、Chain ID和序列信息的CSV文件路径')
    args = parser.parse_args()
    
    # 初始化处理器
    processor = StructureProcessor(save_dir=args.output, csv_path=args.csv)
    
    # 检查输入路径是文件还是目录
    if os.path.isfile(args.input):
        if not args.input.endswith('.cif'):
            print(f"错误：输入文件必须是CIF格式 ({args.input})")
        else:
            print(f"处理单个CIF文件: {args.input}")
            data = processor.process_cif(args.input, distance_threshold=args.distance_threshold)
            if data is not None:
                print(f"处理完成，数据已保存至 {args.output}")
    
    elif os.path.isdir(args.input):
        print(f"处理目录中的CIF文件: {args.input}")
        processed_data = processor.process_directory(args.input)
        print(f"共处理 {len(processed_data)} 个文件")
        print(f"数据已保存至 {args.output}")
    
    else:
        print(f"错误：输入路径不存在 ({args.input})")
