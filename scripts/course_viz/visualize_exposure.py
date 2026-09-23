"""
因子暴露分析可视化脚本

功能：
读取 exposure_coefficients.csv，绘制以下图表：
1. R2 走势图：判断因子被风险解释的程度随时间的变化。
2. 风格暴露时序图：Beta (市场) 和 Size (市值) 的暴露系数随时间变化。
3. 行业暴露均值图：因子在各行业上的平均暴露，判断是否存在特定的行业偏好。

输出：
在 outputs/exposure_analysis 目录下生成 .png 图片。
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 设置中文字体 (尝试几种常见的中文字体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_exposure_analysis(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 获取唯一的因子列表
    factors = df['factor'].unique()
    
    for factor in factors:
        sub_df = df[df['factor'] == factor]
        
        # 1. 绘制 R2 走势图
        plt.figure(figsize=(12, 6))
        plt.plot(sub_df['date'], sub_df['r2'], label='R-squared', color='navy')
        plt.title(f'Factor Risk Explanation (R2) - {factor}')
        plt.ylabel('R-squared')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 格式化日期轴
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gcf().autofmt_xdate()
        
        plt.savefig(out_path / f'{factor}_r2_timeseries.png')
        plt.close()
        
        # 2. 绘制 Size 和 Beta 暴露时序图
        # 列名可能存在变化，先尝试查找
        beta_col = 'beta_beta' if 'beta_beta' in sub_df.columns else None
        size_col = 'beta_size' if 'beta_size' in sub_df.columns else None
        
        if beta_col or size_col:
            plt.figure(figsize=(12, 6))
            if beta_col:
                plt.plot(sub_df['date'], sub_df[beta_col], label='Market Beta Exposure', alpha=0.8)
            if size_col:
                plt.plot(sub_df['date'], sub_df[size_col], label='Size Exposure', alpha=0.8)
                
            plt.title(f'Style Factor Exposures - {factor}')
            plt.ylabel('Exposure Coefficient')
            plt.xlabel('Date')
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gcf().autofmt_xdate()
            
            plt.savefig(out_path / f'{factor}_style_exposure.png')
            plt.close()
            
        # 3. 绘制行业暴露均值图 (Horizontal Bar Chart)
        ind_cols = [c for c in sub_df.columns if c.startswith('beta_ind_')]
        if ind_cols:
            mean_exposures = sub_df[ind_cols].mean().sort_values()
            # 去掉前缀以便显示更好看
            labels = [c.replace('beta_ind_', '') for c in mean_exposures.index]
            
            plt.figure(figsize=(10, 8))
            plt.barh(labels, mean_exposures.values, color='teal', alpha=0.7)
            plt.title(f'Average Industry Exposure - {factor}')
            plt.xlabel('Average Coefficient')
            plt.grid(axis='x', alpha=0.3)
            plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
            
            plt.tight_layout()
            plt.savefig(out_path / f'{factor}_industry_exposure.png')
            plt.close()

    print(f"Visualization saved to {output_dir}")

if __name__ == "__main__":
    csv_file = "./outputs/exposure_analysis/exposure_coefficients.csv"
    out_dir = "./outputs/exposure_analysis"
    
    if Path(csv_file).exists():
        plot_exposure_analysis(csv_file, out_dir)
    else:
        print(f"File not found: {csv_file}")
