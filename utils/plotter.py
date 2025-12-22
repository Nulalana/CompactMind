import matplotlib.pyplot as plt
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def generate_performance_plot(original_ppl, final_ppl, best_config, save_path):
    """
    生成性能对比可视化图表
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # 数据准备
        labels = ['Original', 'Compressed']
        values = [original_ppl, final_ppl]
        colors = ['#1f77b4', '#ff7f0e'] # 蓝橙配色
        
        # 绘制柱状图
        bars = plt.bar(labels, values, color=colors, width=0.5)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=12)
        
        # 添加标题和标签
        # 修复 KeyError: 'method' - best_config 可能包含 'pipeline' 而不是 'method'
        if best_config:
            if 'method' in best_config:
                method_name = best_config['method']
            elif 'pipeline' in best_config:
                # 简化显示 pipeline 名称
                methods = [step['method'] for step in best_config['pipeline']]
                method_name = " + ".join(methods)
            else:
                method_name = "Unknown"
        else:
            method_name = "None"

        plt.title(f'Model Compression Performance\nMethod: {method_name}', fontsize=14)
        plt.ylabel('Perplexity (Lower is Better)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 计算变化率
        if original_ppl > 0:
            change_pct = ((final_ppl - original_ppl) / original_ppl) * 100
            plt.text(0.5, max(values) * 0.5, 
                    f'PPL Change:\n{change_pct:+.2f}%', 
                    ha='center', fontsize=14, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate performance plot: {e}")

def generate_search_history_plot(history, original_ppl, save_path):
    """
    生成搜索空间分析图：压缩比 vs PPL，含 Pareto Frontier
    自适应支持 Broken Axis (断轴) 或 Log Scale 展示
    """
    try:
        # 过滤掉 PPL 为 inf 的点
        valid_history = [h for h in history if h['ppl'] != float('inf')]
        if not valid_history:
            print("No valid history to plot.")
            return

        ratios = [h['ratio'] for h in valid_history]
        ppls = [h['ppl'] for h in valid_history]
        types = [h['type'] for h in valid_history]
        
        # === 自适应显示逻辑 ===
        y_min, y_max = min(ppls), max(ppls)
        sorted_ppls = np.sort(ppls)
        diffs = np.diff(sorted_ppls)
        
        mode = 'linear'
        break_params = None
        
        # 如果跨度极大 (> 2个数量级)，考虑优化显示
        if y_max / (y_min + 1e-6) > 100 and len(ppls) > 1:
            max_gap_idx = np.argmax(diffs)
            max_gap = diffs[max_gap_idx]
            gap_start = sorted_ppls[max_gap_idx]
            gap_end = sorted_ppls[max_gap_idx+1]
            
            # 如果最大空白占据了 > 60% 的线性空间，使用断轴 (Broken Axis)
            if max_gap > 0.6 * (y_max - y_min):
                mode = 'broken'
                # 留出一些余量
                margin_bottom = (gap_start - y_min) * 0.2 if gap_start > y_min else 1.0
                margin_top = (y_max - gap_end) * 0.2
                
                # 底部区间 [min - margin, gap_start + margin]
                # 顶部区间 [gap_end - margin, max + margin]
                ylim_bottom = (max(0, y_min - margin_bottom), gap_start + margin_bottom)
                ylim_top = (gap_end - margin_top, y_max + margin_top)
                break_params = (ylim_top, ylim_bottom)
            else:
                # 否则使用对数坐标
                mode = 'log'
                
        # === 创建画布 ===
        if mode == 'broken':
            # 创建两个子图，高度比 1:2 (假设低值区更重要，或者根据点分布决定？这里简单给低值区更多空间)
            # 统计两边的点数
            low_count = sum(1 for p in ppls if p <= break_params[1][1])
            high_count = sum(1 for p in ppls if p >= break_params[0][0])
            ratio_h = [1, 1] if high_count > low_count else [1, 2] 
            
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), 
                                          gridspec_kw={'height_ratios': ratio_h})
            fig.subplots_adjust(hspace=0.1)
            axes_list = [ax1, ax2]
            
            # 设置坐标轴范围
            ax1.set_ylim(*break_params[0])  # Top
            ax2.set_ylim(*break_params[1])  # Bottom
            
            # 隐藏脊线制造断裂感
            ax1.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop=False) 
            ax2.xaxis.tick_bottom()
            
            # 添加波浪线/斜线
            d = .015 
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=ax2.transAxes) 
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) # bottom-right diagonal

        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            axes_list = [ax]
            if mode == 'log':
                ax.set_yscale('log')

        # === 绘图函数 ===
        def draw_on_ax(ax):
            # 1. 绘制散点
            single_indices = [i for i, t in enumerate(types) if t == 'single']
            pipeline_indices = [i for i, t in enumerate(types) if t == 'hybrid' or t == 'pipeline']
            
            if single_indices:
                ax.scatter([ratios[i] for i in single_indices], [ppls[i] for i in single_indices], 
                           c='blue', label='Single Method', alpha=0.6, s=80, marker='o', edgecolors='k')
                
            if pipeline_indices:
                ax.scatter([ratios[i] for i in pipeline_indices], [ppls[i] for i in pipeline_indices], 
                           c='red', label='Hybrid Method', alpha=0.6, s=100, marker='^', edgecolors='k')
            
            # 2. 绘制 Pareto Frontier
            sorted_points = sorted(valid_history, key=lambda x: x['ratio'])
            pareto_points = []
            current_min_ppl = float('inf')
            
            for point in sorted_points:
                if point['ppl'] < current_min_ppl:
                    pareto_points.append(point)
                    current_min_ppl = point['ppl']
            
            if pareto_points:
                p_ratios = [p['ratio'] for p in pareto_points]
                p_ppls = [p['ppl'] for p in pareto_points]
                ax.plot(p_ratios, p_ppls, 'g--', linewidth=2, label='Pareto Frontier', alpha=0.8)
                ax.scatter(p_ratios, p_ppls, c='gold', s=150, marker='*', zorder=10, label='Pareto Optimal')

            # 辅助线
            ax.axhline(y=original_ppl, color='green', linestyle=':', label='Original PPL')
            # 移除 target_ratio 线
            # ax.axvline(x=target_ratio, color='gray', linestyle='--', label='Target Ratio Limit')
            
            ax.grid(True, alpha=0.3)
            
            # 标注最佳点
            if ppls:
                best_idx = ppls.index(min(ppls))
                best_x = ratios[best_idx]
                best_y = ppls[best_idx]
                # 只有当点在当前坐标轴范围内时才标注，防止标注飞出
                ylim = ax.get_ylim()
                if ylim[0] <= best_y <= ylim[1]:
                    ax.annotate('Best Found', xy=(best_x, best_y), xytext=(best_x, best_y*1.15),
                                arrowprops=dict(facecolor='black', shrink=0.05),
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

        # === 执行绘图 ===
        for ax in axes_list:
            draw_on_ax(ax)
            
        # === 设置标题和标签 ===
        if mode == 'broken':
            # 统一 Legend (只在第一个图显示，去重)
            handles, labels = axes_list[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes_list[0].legend(by_label.values(), by_label.keys(), loc='upper right')
            
            axes_list[1].set_xlabel('Estimated Compression Ratio (Lower is More Compressed)')
            axes_list[0].set_ylabel('Perplexity')
            plt.suptitle('Search Space Analysis: Compression Ratio vs PPL (Broken Axis)', y=0.95)
            
        elif mode == 'log':
            plt.xlabel('Estimated Compression Ratio (Lower is More Compressed)')
            plt.ylabel('Perplexity (Log Scale)')
            plt.title('Search Space Analysis: Compression Ratio vs PPL (Log Scale)')
            plt.legend()
        else:
            plt.xlabel('Estimated Compression Ratio (Lower is More Compressed)')
            plt.ylabel('Perplexity (Lower is Better)')
            plt.title('Search Space Analysis: Compression Ratio vs PPL')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        logger.error(f"Failed to generate search history plot: {e}")
