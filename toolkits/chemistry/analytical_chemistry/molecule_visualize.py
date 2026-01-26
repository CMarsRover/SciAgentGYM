from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



class BestMolecule3DVisualizer:
    """最佳3D分子可视化方案"""
    
    def __init__(self):
        pass
    
    def visualize(self, smiles: str, output_prefix: str, 
                 methods=['2d', '3d_projection', '3d_matplotlib']):
        """
        综合可视化
        
        methods: 
            '2d' - 标准2D结构
            '3d_projection' - 3D结构的2D投影
            '3d_matplotlib' - Matplotlib 3D图
        """
        
        results = {}
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"无效的SMILES: {smiles}")
            return results
        
        # 1. 标准2D结构
        if '2d' in methods:
            output_2d = f'{output_prefix}_2d.png'
            AllChem.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(600, 600))
            img.save(output_2d)
            results['2d'] = output_2d
            print(f"✓ 完成2D结构的绘制: {output_2d}")
        
        # 生成3D结构（用于后续方法）
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d)
        
        # 2. 3D结构的2D投影
        if '3d_projection' in methods:
            output_3d_proj = f'{output_prefix}_3d_projection.png'
            img = Draw.MolToImage(mol_3d, size=(600, 600))
            img.save(output_3d_proj)
            results['3d_projection'] = output_3d_proj
            print(f"✓ 完成3D投影的绘制: {output_3d_proj}")
        
        # 3. Matplotlib 3D可视化
        if '3d_matplotlib' in methods:
            output_3d_mpl = f'{output_prefix}_3d_matplotlib.png'
            self._draw_3d_matplotlib(mol_3d, output_3d_mpl, smiles)
            results['3d_matplotlib'] = output_3d_mpl
            print(f"✓ 完成3D图的绘制: {output_3d_mpl}")
        
        return results
    
    def _draw_3d_matplotlib(self, mol, output_file, smiles):
        """Matplotlib 3D绘图"""
        
        conf = mol.GetConformer()
        atoms = mol.GetAtoms()
        
        coords = []
        colors = []
        
        element_colors = {
            'C': 'gray', 'H': 'lightgray', 'O': 'red',
            'N': 'blue', 'S': 'yellow', 'P': 'orange',
        }
        
        for atom in atoms:
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
            colors.append(element_colors.get(atom.GetSymbol(), 'pink'))
        
        coords = np.array(coords)
        
        bonds = []
        for bond in mol.GetBonds():
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制键
        for bond in bonds:
            start, end = bond
            ax.plot([coords[start, 0], coords[end, 0]],
                   [coords[start, 1], coords[end, 1]],
                   [coords[start, 2], coords[end, 2]],
                   'k-', linewidth=2, alpha=0.6)
        
        # 绘制原子
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                  c=colors, s=300, edgecolors='black', linewidth=2, alpha=0.9)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'3D Structure\n{smiles}', fontsize=14, pad=20)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white')
        plt.close()


def visualize(smiles: str, output_prefix: str, methods=None):
    """
    综合可视化分子结构，支持2D结构、3D结构的2D投影和Matplotlib 3D图三种方式。
    
    Parameters:
    -----------
    smiles : str
        分子的SMILES字符串表示
    output_prefix : str
        输出文件的前缀路径
    methods : list, optional
        可视化方法列表，可选值：
        - '2d': 标准2D结构
        - '3d_projection': 3D结构的2D投影
        - '3d_matplotlib': Matplotlib 3D图
        默认为全部三种方法
    
    Returns:
    --------
    dict
        包含生成的文件路径的字典，键为方法名，值为文件路径
        
    Examples:
    ---------
    >>> # 生成所有类型的可视化
    >>> results = visualize('CCO', 'ethanol')
    >>> print(results)
    {'2d': 'ethanol_2d.png', '3d_projection': 'ethanol_3d_projection.png', ...}
    
    >>> # 只生成2D结构
    >>> results = visualize('c1ccccc1', 'benzene', methods=['2d'])
    """
    if methods is None:
        methods = ['2d', '3d_projection', '3d_matplotlib']
    
    visualizer = BestMolecule3DVisualizer()
    return visualizer.visualize(smiles, output_prefix, methods)


# ============ 使用示例 ============
def chem_visualizer(molecules):
    """
    分子可视化工具，支持批量生成分子的2D结构图和3D结构图
    
    基于RDKit的绘图功能，可以生成多种格式的分子结构图像：
    - 2D标准结构式
    - 3D结构的2D投影图
    - 基于Matplotlib的交互式3D图
    
    Parameters:
    -----------
    molecules : list of dict or dict
        分子列表（推荐）或分子字典
        - 列表格式: [{'name': '分子名称', 'smiles': 'SMILES字符串'}, ...]
          例如: [{'name': 'ethanol', 'smiles': 'CCO'}, {'name': 'benzene', 'smiles': 'c1ccccc1'}]
        - 字典格式: {分子名称: SMILES字符串}
          例如: {'ethanol': 'CCO', 'benzene': 'c1ccccc1'}
        
    Returns:
    --------
    None
        直接生成图片文件到当前目录，文件命名格式为：
        - {分子名称}_2d.png
        - {分子名称}_3d_projection.png  
        - {分子名称}_3d_matplotlib.png
        
    Examples:
    ---------
    >>> # 可视化多个分子（列表格式）
    >>> molecules = [
    ...     {'name': 'ethanol', 'smiles': 'CCO'},
    ...     {'name': 'benzene', 'smiles': 'c1ccccc1'},
    ...     {'name': 'aspirin', 'smiles': 'CC(=O)Oc1ccccc1C(=O)O'}
    ... ]
    >>> chem_visualizer(molecules=molecules)
    
    >>> # 可视化多个分子（字典格式）
    >>> molecules = {
    ...     'ethanol': 'CCO',
    ...     'benzene': 'c1ccccc1',
    ...     'aspirin': 'CC(=O)Oc1ccccc1C(=O)O'
    ... }
    >>> chem_visualizer(molecules=molecules)
    """
    visualizer = BestMolecule3DVisualizer()
    
    # 兼容两种格式：列表和字典
    if isinstance(molecules, list):
        # 列表格式：[{'name': 'ethanol', 'smiles': 'CCO'}, ...]
        mol_dict = {item['name']: item['smiles'] for item in molecules}
    elif isinstance(molecules, dict):
        # 字典格式：{'ethanol': 'CCO', ...}
        mol_dict = molecules
    else:
        raise ValueError("molecules参数必须是列表或字典格式")
    
    for name, smiles in mol_dict.items():
        print(f"\n{'='*60}")
        print(f"正在处理: {name}")
        print(f"{'='*60}")
        results = visualizer.visualize(smiles, name, methods=['2d', '3d_projection', '3d_matplotlib'])
        
        print(f"生成的文件:")
        for method, file in results.items():
            print(f"  {method}: {file}") 

if __name__ == "__main__":
    # 测试多个分子
    molecules = {
            'ethanol': 'CCO',
            'benzene': 'c1ccccc1',
            'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'aspirin': 'CC(=O)Oc1ccccc1C(=O)O'
        }
    chem_visualizer(molecules=molecules)