import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from typing import Optional, List

def generate_mutation_heatmap(
    csv_file: str,
    output_file: str,
    top_n_positions: int = 100,
    split_charts: bool = True,
    max_positions_per_chart: int = 30
):
    """
    从现有的突变预测CSV文件生成优化的热图可视化。
    
    Args:
        csv_file: 包含突变预测结果的CSV文件路径
        output_file: 输出热图的文件路径
        top_n_positions: 要可视化的顶部位点数量
        split_charts: 是否将大图分割成多个文件
        max_positions_per_chart: 每个图表中显示的最大位点数
    """
    print(f"从文件 {csv_file} 加载突变预测数据...")
    df = pd.read_csv(csv_file)
    
    # 按位点分组并获取顶部突变的概率总和
    position_scores = df.groupby('position')['probability'].sum().reset_index()
    position_scores = position_scores.sort_values('probability', ascending=False).head(top_n_positions)
    
    # 获取顶部位点
    top_positions = position_scores['position'].tolist()
    
    # 添加原始氨基酸信息到position_scores
    position_to_original = df[['position', 'original']].drop_duplicates().set_index('position')['original']
    position_scores['original'] = position_scores['position'].map(position_to_original)
    
    # 过滤数据框，只包含顶部位点
    df_filtered = df[df['position'].isin(top_positions)]
    
    # 创建热图的数据透视表
    pivot_data = df_filtered.pivot_table(
        index='position', 
        columns='mutated_to', 
        values='probability',
        fill_value=0
    )
    
    # 按总概率对位点进行排序
    pivot_data = pivot_data.loc[position_scores['position']]
    
    # 创建更好的行标签，包含原始氨基酸
    row_labels = [f"{int(pos)} ({orig})" for pos, orig in 
                 zip(position_scores['position'], position_scores['original'])]
    
    # 如果位点太多，将图表分割成多个
    if split_charts and top_n_positions > max_positions_per_chart:
        num_charts = (top_n_positions + max_positions_per_chart - 1) // max_positions_per_chart
        charts_data = []
        
        for i in range(num_charts):
            start_idx = i * max_positions_per_chart
            end_idx = min((i + 1) * max_positions_per_chart, top_n_positions)
            
            # 获取此图表的位点
            chart_positions = top_positions[start_idx:end_idx]
            chart_labels = row_labels[start_idx:end_idx]
            
            # 过滤此图表的数据
            chart_data = pivot_data.loc[chart_positions]
            
            # 创建图表
            plt.figure(figsize=(20, min(15, 0.4 * len(chart_positions) + 5)))
            ax = sns.heatmap(chart_data, cmap='viridis', annot=True, fmt='.3f', 
                          linewidths=0.5, annot_kws={"size": 9})
            
            # 设置更好的标签
            ax.set_yticklabels(chart_labels, rotation=0)
            
            plt.title(f'FHase突变概率热图 - 顶部位点 (第{i+1}/{num_charts}部分)')
            plt.xlabel('突变氨基酸')
            plt.ylabel('位置 (原始氨基酸)')
            plt.tight_layout()
            
            # 保存此图表
            base_name, ext = os.path.splitext(output_file)
            chart_file = f"{base_name}_part{i+1}{ext}"
            plt.savefig(chart_file, bbox_inches='tight', dpi=300)
            print(f"可视化部分 {i+1} 已保存至 {chart_file}")
            charts_data.append(chart_file)
            
            plt.close()
    else:
        # 为所有位点创建单个图表
        plt.figure(figsize=(20, min(20, 0.4 * len(top_positions) + 5)))
        ax = sns.heatmap(pivot_data, cmap='viridis', annot=True, fmt='.3f', 
                      linewidths=0.5, annot_kws={"size": 9})
        
        # 设置更好的标签
        ax.set_yticklabels(row_labels, rotation=0)
        
        plt.title('FHase突变概率热图 - 顶部位点')
        plt.xlabel('突变氨基酸')
        plt.ylabel('位置 (原始氨基酸)')
        plt.tight_layout()
        
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"可视化已保存至 {output_file}")
        
        plt.close()
    
    print("热图生成完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从CSV文件生成突变热图')
    parser.add_argument('--csv_file', type=str, default='fhase_mutations_full.csv', 
                        help='包含突变预测结果的CSV文件路径')
    parser.add_argument('--output', type=str, default='fhase_mutations_heatmap.png', 
                        help='输出热图的文件路径')
    parser.add_argument('--top_n', type=int, default=100, 
                        help='要可视化的顶部位点数量')
    parser.add_argument('--split', action='store_true', default=True,
                        help='是否将大图分割成多个文件')
    parser.add_argument('--max_per_chart', type=int, default=30,
                        help='每个图表中显示的最大位点数')
    
    args = parser.parse_args()
    
    generate_mutation_heatmap(
        csv_file=args.csv_file,
        output_file=args.output,
        top_n_positions=args.top_n,
        split_charts=args.split,
        max_positions_per_chart=args.max_per_chart
    )
