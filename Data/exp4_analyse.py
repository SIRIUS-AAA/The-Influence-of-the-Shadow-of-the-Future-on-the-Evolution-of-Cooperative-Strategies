import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings('ignore')

# Unified plotting configuration
plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Unified color palette
COLORS = {
    'primary': '#2E86C1',
    'secondary': '#E74C3C',
    'tertiary': '#28B463',
    'quaternary': '#F39C12',
    'accent': '#8E44AD',
    'neutral': '#85929E'
}

sns.set_style("whitegrid")
sns.set_palette([COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], 
                 COLORS['quaternary'], COLORS['accent'], COLORS['neutral']])

class Experiment4Analyzer:
    """Analyzer for Experiment 4: SCoT vs CoT reasoning effects on cooperation"""
    
    def __init__(self, experiment_folder='Experiment4', summary_file='exp4_summary.xlsx'):
        self.experiment_folder = experiment_folder
        self.experiments = {}
        self.experiment_configs = []
        
        self.summary_data = self._load_summary_table(summary_file)
        self.load_all_experiments()
        
    def _load_summary_table(self, summary_file):
        """Load summary table data"""
        try:
            df = pd.read_excel(summary_file, sheet_name=0)
            print(f"Loaded {len(df)} experiment configurations from {summary_file}")
            return df
        except Exception as e:
            print(f"Cannot read summary file: {e}")
            return pd.DataFrame()
    
    def load_all_experiments(self):
        """Load all experiment data"""
        possible_paths = [
            Path('./Experiment4'),
            Path('Experiment4'),
            Path('../Experiment4'),
            Path(self.experiment_folder),
            Path('.'),
        ]
        
        experiment_path = None
        for path in possible_paths:
            if path.exists():
                csv_pattern = list(path.glob('exp4*.csv'))
                if csv_pattern:
                    experiment_path = path
                    print(f"Found {len(csv_pattern)} exp4*.csv files in {path.absolute()}")
                    break
        
        if experiment_path is None:
            print("Error: No exp4*.csv files found")
            return
        
        print(f"Loading data from {experiment_path.absolute()}...")
        loaded_count = 0
        
        if not self.summary_data.empty:
            print("Available columns:", list(self.summary_data.columns))
            
            for _, row in self.summary_data.iterrows():
                exp_filename = row['实验结果'] if '实验结果' in row else row.iloc[-1]
                csv_file = experiment_path / exp_filename
                
                if csv_file.exists():
                    try:
                        data = pd.read_csv(csv_file)
                        
                        if 'round' in data.columns:
                            data = data.drop_duplicates(subset=['match_id', 'round'], keep='first')
                            data = data.sort_values(['match_id', 'round']).reset_index(drop=True)
                        
                        if 'answer1' in data.columns and 'answer2' in data.columns:
                            data['both_cooperate'] = (data['answer1'] == 'J') & (data['answer2'] == 'J')
                            data['both_defect'] = (data['answer1'] == 'F') & (data['answer2'] == 'F')
                            data['coordination'] = (data['answer1'] == data['answer2'])
                            data['player1_cooperate'] = data['answer1'] == 'J'
                            data['player2_cooperate'] = data['answer2'] == 'J'
                        
                        if 'p_now' in data.columns:
                            data['p_change'] = data.groupby('match_id')['p_now'].diff()
                            data['p_trend'] = data['p_change'].apply(lambda x: 'increasing' if x > 0 else 'decreasing' if x < 0 else 'stable')
                        
                        exp_name = csv_file.stem
                        self.experiments[exp_name] = data
                        
                        config = self._parse_config_from_summary(exp_name, row, data)
                        self.experiment_configs.append(config)
                        
                        print(f"✓ {exp_filename} ({len(data)} rounds)")
                        loaded_count += 1
                        
                    except Exception as e:
                        print(f"✗ {exp_filename}: {str(e)}")
                else:
                    print(f"- {exp_filename}: Not found")
        
        print(f"\nLoaded {loaded_count} experiments")
    
    def _parse_config_from_summary(self, exp_name, summary_row, data):
        """Parse experiment configuration from summary table"""
        config = {
            'name': exp_name,
            'experiment_id': summary_row['实验序号'] if '实验序号' in summary_row else summary_row.iloc[0],
            'p_start': summary_row['start'] if 'start' in summary_row else summary_row.iloc[1],
            'p_end': summary_row['end'] if 'end' in summary_row else summary_row.iloc[2],
            'trend': summary_row['趋势'] if '趋势' in summary_row else summary_row.iloc[3],
            'repetitions': summary_row['重复'] if '重复' in summary_row else summary_row.iloc[4],
            'seed': summary_row['seed'] if 'seed' in summary_row else summary_row.iloc[5],
            'has_memory': (summary_row['是否记忆'] if '是否记忆' in summary_row else summary_row.iloc[6]) == 'y',
            'memory_window': summary_row['记忆窗口长度'] if '记忆窗口长度' in summary_row else summary_row.iloc[7],
            'player1': summary_row['模型a'] if '模型a' in summary_row else summary_row.iloc[8],
            'player2': summary_row['模型b'] if '模型b' in summary_row else summary_row.iloc[9],
            'scenario': summary_row['场景'] if '场景' in summary_row else summary_row.iloc[10],
            'reasoning_mode': summary_row['mode'] if 'mode' in summary_row else summary_row.iloc[11],
            'total_matches': data['match_id'].nunique() if 'match_id' in data.columns else 1
        }
        return config
    
    def create_summary_dataframe(self):
        """Create summary dataframe for reasoning mode analysis"""
        summary_data = []
        
        for config in self.experiment_configs:
            exp_data = self.experiments[config['name']]
            
            if 'reasoning_mode' in exp_data.columns:
                grouped = exp_data.groupby('reasoning_mode')
            else:
                grouped = [(config.get('reasoning_mode', 'baseline'), exp_data)]
            
            for reasoning_mode, group in grouped:
                if len(group) == 0:
                    continue
                
                if config['scenario'].lower() == 'pd':
                    cooperation_rate = group['both_cooperate'].mean()
                    success_metric = cooperation_rate
                else:
                    coordination_rate = group['coordination'].mean()
                    success_metric = coordination_rate
                
                p_start = group['p_now'].iloc[0] if 'p_now' in group.columns and len(group) > 0 else config.get('p_start', 0)
                p_end = group['p_now'].iloc[-1] if 'p_now' in group.columns and len(group) > 0 else config.get('p_end', 0)
                p_change = abs(p_end - p_start) if p_start and p_end else 0
                
                summary = {
                    'experiment_name': config['name'],
                    'experiment_id': config.get('experiment_id', 0),
                    'game_scenario': config['scenario'].upper(),
                    'reasoning_mode': reasoning_mode,
                    'p_trend': config.get('trend', 'unknown'),
                    'p_start': p_start,
                    'p_end': p_end,
                    'p_change': p_change,
                    'player1': config.get('player1', 'unknown'),
                    'player2': config.get('player2', 'unknown'),
                    'success_rate': success_metric,
                    'cooperation_rate': group['both_cooperate'].mean(),
                    'coordination_rate': group['coordination'].mean(),
                    'avg_game_length': len(group) / group['match_id'].nunique() if 'match_id' in group.columns else len(group),
                    'total_games': group['match_id'].nunique() if 'match_id' in group.columns else 1,
                    'avg_score_p1': group['points1'].mean(),
                    'avg_score_p2': group['points2'].mean(),
                    'has_memory': config.get('has_memory', False),
                    'memory_window': config.get('memory_window', 0)
                }
                summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def plot_reasoning_mode_comparison(self, save_path=None):
        """Compare different reasoning modes (baseline, CoT, SCoT)"""
        summary_df = self.create_summary_dataframe()
        
        if len(summary_df) == 0:
            print("No data available for analysis")
            return
        
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Success rates by reasoning mode
            ax1 = axes[0, 0]
            reasoning_comparison = summary_df.groupby('reasoning_mode')['success_rate'].agg(['mean', 'std', 'count'])
            bars = ax1.bar(reasoning_comparison.index, reasoning_comparison['mean'], 
                          yerr=reasoning_comparison['std'], capsize=5, alpha=0.8,
                          color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']])
            ax1.set_title('Success Rate by Reasoning Mode')
            ax1.set_ylabel('Average Success Rate')
            ax1.set_xlabel('Reasoning Mode')
            
            for bar, mean, count in zip(bars, reasoning_comparison['mean'], reasoning_comparison['count']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.1%}\n(n={count})', ha='center', va='bottom')
            
            # Game scenario breakdown
            ax2 = axes[0, 1]
            scenario_reasoning = summary_df.groupby(['game_scenario', 'reasoning_mode'])['success_rate'].mean().unstack(fill_value=0)
            scenario_reasoning.plot(kind='bar', ax=ax2, alpha=0.8, 
                                   color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']])
            ax2.set_title('Success Rate by Game Scenario and Reasoning Mode')
            ax2.set_ylabel('Success Rate')
            ax2.set_xlabel('Game Scenario')
            ax2.legend(title='Reasoning Mode')
            ax2.tick_params(axis='x', rotation=0)
            
            # P-value trend effects
            ax3 = axes[1, 0]
            trend_effects = summary_df.groupby(['p_trend', 'reasoning_mode'])['success_rate'].mean().unstack(fill_value=0)
            trend_effects.plot(kind='bar', ax=ax3, alpha=0.8,
                              color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']])
            ax3.set_title('Success Rate by P-value Trend and Reasoning Mode')
            ax3.set_ylabel('Success Rate')
            ax3.set_xlabel('P-value Trend')
            ax3.legend(title='Reasoning Mode')
            ax3.tick_params(axis='x', rotation=45)
            
            # Score efficiency by reasoning mode
            ax4 = axes[1, 1]
            summary_df['total_welfare'] = summary_df['avg_score_p1'] + summary_df['avg_score_p2']
            welfare_comparison = summary_df.groupby('reasoning_mode')['total_welfare'].agg(['mean', 'std'])
            
            bars = ax4.bar(welfare_comparison.index, welfare_comparison['mean'],
                          yerr=welfare_comparison['std'], capsize=5, alpha=0.8,
                          color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']])
            ax4.set_title('Total Welfare by Reasoning Mode')
            ax4.set_ylabel('Average Total Welfare')
            ax4.set_xlabel('Reasoning Mode')
            
            for bar, mean in zip(bars, welfare_comparison['mean']):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{mean:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def plot_p_value_dynamics(self, save_path=None):
        """Analyze p-value changes and their effects"""
        if len(self.experiments) == 0:
            print("No experiments to analyze")
            return
        
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot p-value trajectories for different reasoning modes
            ax1 = axes[0, 0]
            colors = {'baseline': COLORS['primary'], 'cot': COLORS['secondary'], 'scot': COLORS['tertiary']}
            
            for exp_name, exp_data in self.experiments.items():
                if 'p_now' in exp_data.columns and 'reasoning_mode' in exp_data.columns:
                    for reasoning_mode in exp_data['reasoning_mode'].unique():
                        mode_data = exp_data[exp_data['reasoning_mode'] == reasoning_mode]
                        if len(mode_data) > 1:
                            ax1.plot(mode_data['round'], mode_data['p_now'], 
                                   alpha=0.7, color=colors.get(reasoning_mode, COLORS['neutral']),
                                   label=f'{reasoning_mode}' if reasoning_mode not in ax1.get_legend_handles_labels()[1] else "")
            
            ax1.set_title('P-value Trajectories by Reasoning Mode')
            ax1.set_xlabel('Round')
            ax1.set_ylabel('P-value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Success rate vs p-value change
            ax2 = axes[0, 1]
            summary_df = self.create_summary_dataframe()
            
            if len(summary_df) > 0:
                for reasoning_mode, color in colors.items():
                    mode_data = summary_df[summary_df['reasoning_mode'] == reasoning_mode]
                    if len(mode_data) > 0:
                        ax2.scatter(mode_data['p_change'], mode_data['success_rate'], 
                                  alpha=0.7, label=reasoning_mode, color=color, s=60)
            
            ax2.set_title('Success Rate vs P-value Change')
            ax2.set_xlabel('P-value Change Magnitude')
            ax2.set_ylabel('Success Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Cooperation over time for different reasoning modes
            ax3 = axes[1, 0]
            
            for exp_name, exp_data in list(self.experiments.items())[:3]:
                if 'reasoning_mode' in exp_data.columns:
                    for reasoning_mode in exp_data['reasoning_mode'].unique():
                        mode_data = exp_data[exp_data['reasoning_mode'] == reasoning_mode]
                        if len(mode_data) > 5:
                            window = max(3, len(mode_data) // 10)
                            coop_rate = mode_data['both_cooperate'].rolling(window=window, center=True, min_periods=1).mean()
                            ax3.plot(mode_data['round'], coop_rate, 
                                   alpha=0.8, color=colors.get(reasoning_mode, COLORS['neutral']),
                                   linewidth=2, label=f'{reasoning_mode}')
            
            ax3.set_title('Cooperation Rate Evolution')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Cooperation Rate')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Memory window effects
            ax4 = axes[1, 1]
            if 'memory_window' in summary_df.columns:
                memory_effects = summary_df.groupby(['memory_window', 'reasoning_mode'])['success_rate'].mean().unstack(fill_value=0)
                memory_effects.plot(kind='bar', ax=ax4, alpha=0.8,
                                   color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']])
                ax4.set_title('Memory Window Effects by Reasoning Mode')
                ax4.set_ylabel('Success Rate')
                ax4.set_xlabel('Memory Window Size')
                ax4.legend(title='Reasoning Mode')
                ax4.tick_params(axis='x', rotation=0)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def plot_strategic_reasoning_analysis(self, save_path=None):
        """Analyze strategic reasoning effects (SCoT focus)"""
        summary_df = self.create_summary_dataframe()
        
        if len(summary_df) == 0:
            print("No data for strategic reasoning analysis")
            return
        
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # SCoT vs others comparison
            ax1 = axes[0, 0]
            summary_df['is_scot'] = summary_df['reasoning_mode'] == 'scot'
            scot_comparison = summary_df.groupby('is_scot')['success_rate'].agg(['mean', 'std', 'count'])
            
            labels = ['Non-SCoT', 'SCoT']
            bars = ax1.bar(labels, scot_comparison['mean'], 
                          yerr=scot_comparison['std'], capsize=5, alpha=0.8,
                          color=[COLORS['neutral'], COLORS['accent']])
            ax1.set_title('SCoT vs Non-SCoT Performance')
            ax1.set_ylabel('Success Rate')
            
            for bar, mean, count in zip(bars, scot_comparison['mean'], scot_comparison['count']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.1%}\n(n={count})', ha='center', va='bottom')
            
            # Player pairing effects with reasoning modes
            ax2 = axes[0, 1]
            if 'player1' in summary_df.columns and 'player2' in summary_df.columns:
                summary_df['player_pair'] = summary_df['player1'] + ' vs ' + summary_df['player2']
                pairing_reasoning = summary_df.groupby(['player_pair', 'reasoning_mode'])['success_rate'].mean().unstack(fill_value=0)
                pairing_reasoning.plot(kind='bar', ax=ax2, alpha=0.8,
                                      color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']])
                ax2.set_title('Player Pairing Performance by Reasoning Mode')
                ax2.set_ylabel('Success Rate')
                ax2.set_xlabel('Player Pairing')
                ax2.legend(title='Reasoning Mode', bbox_to_anchor=(1.05, 1))
                ax2.tick_params(axis='x', rotation=45)
            
            # Game length vs success for different reasoning modes
            ax3 = axes[1, 0]
            for reasoning_mode, color in {'baseline': COLORS['primary'], 'cot': COLORS['secondary'], 'scot': COLORS['tertiary']}.items():
                mode_data = summary_df[summary_df['reasoning_mode'] == reasoning_mode]
                if len(mode_data) > 0:
                    ax3.scatter(mode_data['avg_game_length'], mode_data['success_rate'], 
                              alpha=0.7, label=reasoning_mode, color=color, s=60)
            
            ax3.set_title('Game Length vs Success Rate')
            ax3.set_xlabel('Average Game Length')
            ax3.set_ylabel('Success Rate')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Reasoning mode effectiveness by game scenario
            ax4 = axes[1, 1]
            effectiveness_heatmap = summary_df.pivot_table(
                values='success_rate', 
                index='reasoning_mode', 
                columns='game_scenario', 
                aggfunc='mean'
            )
            
            if not effectiveness_heatmap.empty:
                sns.heatmap(effectiveness_heatmap, annot=True, fmt='.2f', cmap='Blues', ax=ax4)
                ax4.set_title('Reasoning Mode Effectiveness by Game Scenario')
                ax4.set_xlabel('Game Scenario')
                ax4.set_ylabel('Reasoning Mode')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("="*70)
        print("Experiment 4: SCoT vs CoT Strategic Reasoning Analysis")
        print("="*70)
        
        summary_df = self.create_summary_dataframe()
        
        if len(summary_df) == 0:
            print("Warning: No experiment data loaded")
            return
        
        print(f"\n【Data Overview】")
        print(f"  Total experiments: {len(self.experiments)}")
        print(f"  Reasoning modes: {summary_df['reasoning_mode'].unique()}")
        print(f"  Game scenarios: {summary_df['game_scenario'].unique()}")
        
        print(f"\n【Reasoning Mode Effects】")
        reasoning_effects = summary_df.groupby('reasoning_mode')['success_rate'].agg(['mean', 'std', 'count'])
        for mode, stats in reasoning_effects.iterrows():
            print(f"  {mode}: {stats['mean']:.1%} ± {stats['std']:.1%} ({stats['count']} experiments)")
        
        scot_data = summary_df[summary_df['reasoning_mode'] == 'scot']
        non_scot_data = summary_df[summary_df['reasoning_mode'] != 'scot']
        
        if len(scot_data) > 0 and len(non_scot_data) > 0:
            scot_performance = scot_data['success_rate'].mean()
            non_scot_performance = non_scot_data['success_rate'].mean()
            improvement = scot_performance - non_scot_performance
            
            print(f"\n【SCoT Strategic Analysis】")
            print(f"  SCoT average performance: {scot_performance:.1%}")
            print(f"  Non-SCoT average performance: {non_scot_performance:.1%}")
            print(f"  SCoT improvement: {improvement:.1%}")
            
            try:
                from scipy import stats
                t_stat, p_val = stats.ttest_ind(scot_data['success_rate'], non_scot_data['success_rate'])
                print(f"  Statistical significance: p={p_val:.4f}")
            except ImportError:
                print("  (scipy not available for significance testing)")
        
        print(f"\n【Game Scenario Analysis】")
        for scenario in summary_df['game_scenario'].unique():
            scenario_data = summary_df[summary_df['game_scenario'] == scenario]
            scenario_performance = scenario_data['success_rate'].mean()
            print(f"  {scenario}: {scenario_performance:.1%} average success rate")
        
        if 'p_trend' in summary_df.columns:
            print(f"\n【P-value Trend Effects】")
            trend_effects = summary_df.groupby('p_trend')['success_rate'].mean()
            for trend, performance in trend_effects.items():
                print(f"  {trend} trend: {performance:.1%} success rate")
    
    def save_all_analysis(self, output_dir='experiment4_analysis'):
        """Save all analysis results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\nGenerating analysis results...")
        
        if len(self.experiments) == 0:
            print("Warning: No experiments loaded")
            return
        
        summary_df = self.create_summary_dataframe()
        if len(summary_df) > 0:
            summary_df.to_csv(f'{output_dir}/exp4_summary_analysis.csv', index=False, encoding='utf-8-sig')
            
            self.plot_reasoning_mode_comparison(f'{output_dir}/reasoning_mode_comparison.pdf')
            self.plot_p_value_dynamics(f'{output_dir}/p_value_dynamics.pdf')
            self.plot_strategic_reasoning_analysis(f'{output_dir}/strategic_reasoning_analysis.pdf')
            
            print(f"✓ Analysis results saved to {output_dir}/")
        else:
            print("Warning: Empty summary data")

if __name__ == "__main__":
    analyzer = Experiment4Analyzer('Experiment4', 'exp4_summary.xlsx')
    analyzer.generate_comprehensive_report()
    analyzer.save_all_analysis()
    print("\n🎉 Experiment 4 analysis complete!")