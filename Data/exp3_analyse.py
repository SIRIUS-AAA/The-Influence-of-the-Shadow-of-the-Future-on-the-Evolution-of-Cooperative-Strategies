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
sns.set_palette("Set2")

class MultiExperimentAnalyzer:
    """Game theory multi-experiment analyzer based on summary table"""
    
    def __init__(self, experiment_folder='Experiment3', summary_file='exp3_summary.xlsx'):
        self.experiment_folder = experiment_folder
        self.experiments = {}
        self.experiment_configs = []
        
        self.summary_data = self._load_summary_table(summary_file)
        self.load_all_experiments()
        
    def _load_summary_table(self, summary_file):
        """Load summary table data"""
        try:
            import openpyxl
            df = pd.read_excel(summary_file, sheet_name=0)
            print(f"Loaded {len(df)} experiment configurations from {summary_file}")
            return df
        except Exception as e:
            print(f"Cannot read summary file: {e}")
            return pd.DataFrame()
    
    def load_all_experiments(self):
        """Load all experiment data based on summary table"""
        possible_paths = [
            Path(self.experiment_folder),
            Path('.'),
            Path('./Experiment3'),
            Path('../Experiment3')
        ]
        
        experiment_path = None
        for path in possible_paths:
            if path.exists() and any(path.glob('*.csv')):
                experiment_path = path
                break
        
        if experiment_path is None:
            print(f"Error: No CSV files found")
            return
        
        print(f"Loading experiment data from {experiment_path.absolute()}...")
        
        loaded_count = 0
        
        if not self.summary_data.empty:
            for _, row in self.summary_data.iterrows():
                exp_filename = row['实验结果']
                csv_file = experiment_path / exp_filename
                
                if csv_file.exists():
                    try:
                        data = pd.read_csv(csv_file)
                        
                        if 'round' in data.columns:
                            data = data.drop_duplicates(subset=['round'], keep='first')
                            data = data.sort_values('round').reset_index(drop=True)
                        
                        if 'answer1' in data.columns and 'answer2' in data.columns:
                            data['both_cooperate'] = (data['answer1'] == 'J') & (data['answer2'] == 'J')
                            data['both_defect'] = (data['answer1'] == 'F') & (data['answer2'] == 'F')
                            data['mixed_strategy'] = ~(data['both_cooperate'] | data['both_defect'])
                            data['player1_cooperate'] = data['answer1'] == 'J'
                            data['player2_cooperate'] = data['answer2'] == 'J'
                        
                        exp_name = csv_file.stem
                        self.experiments[exp_name] = data
                        
                        config = self._parse_config_from_summary(exp_name, row, data)
                        self.experiment_configs.append(config)
                        
                        print(f"✓ {exp_filename} ({len(data)} rounds)")
                        loaded_count += 1
                        
                    except Exception as e:
                        print(f"✗ {exp_filename}: Loading failed - {str(e)}")
                else:
                    print(f"- {exp_filename}: File not found at {experiment_path}")
        else:
            csv_files = list(experiment_path.glob("*.csv"))
            for csv_file in csv_files:
                self._load_single_experiment(csv_file)
                loaded_count += 1
        
        print(f"\nSuccessfully loaded {loaded_count} experiments")
        
    def _load_single_experiment(self, csv_file):
        """Load single experiment file"""
        try:
            exp_name = csv_file.stem
            data = pd.read_csv(csv_file)
            
            if not data.empty and 'round' in data.columns:
                data = data.drop_duplicates(subset=['round'], keep='first')
                data = data.sort_values('round').reset_index(drop=True)
                
                if 'answer1' in data.columns and 'answer2' in data.columns:
                    data['both_cooperate'] = (data['answer1'] == 'J') & (data['answer2'] == 'J')
                    data['both_defect'] = (data['answer1'] == 'F') & (data['answer2'] == 'F')
                    data['mixed_strategy'] = ~(data['both_cooperate'] | data['both_defect'])
                    data['player1_cooperate'] = data['answer1'] == 'J'
                    data['player2_cooperate'] = data['answer2'] == 'J'
                
                self.experiments[exp_name] = data
                
                config = {
                    'name': exp_name,
                    'scenario': 'bos' if 'bos' in exp_name.lower() else 'pd',
                    'player1': 'unknown',
                    'player2': 'unknown', 
                    'path': 'unknown',
                    'rounds': len(data),
                    'cooperation_rate': data['both_cooperate'].mean() if 'both_cooperate' in data else None,
                    'completed': True
                }
                self.experiment_configs.append(config)
                
                print(f"✓ {exp_name} ({len(data)} rounds)")
                
        except Exception as e:
            print(f"✗ {csv_file.name}: {str(e)}")
    
    def _parse_config_from_summary(self, exp_name, summary_row, data):
        """Parse experiment configuration from summary table"""
        config = {
            'name': exp_name,
            'experiment_id': summary_row['实验序号'],
            'scenario': summary_row['场景'],
            'player1': summary_row['模型a'],
            'player2': summary_row['模型b'],
            'path': summary_row['趋势'],
            'p_start': summary_row['start'],
            'p_end': summary_row['end'],
            'seed': summary_row['seed'],
            'memory': summary_row['是否记忆'],
            'window_size': summary_row['记忆窗口长度'],
            'rounds': len(data),
            'cooperation_rate': data['both_cooperate'].mean() if 'both_cooperate' in data else None,
            'completed': summary_row['是否做完'] == '√'
        }
        return config
    
    def create_summary_dataframe(self):
        """Create experiment summary table"""
        summary_data = []
        
        for config in self.experiment_configs:
            exp_data = self.experiments[config['name']]
            
            summary = {
                'experiment_id': config.get('experiment_id', 'N/A'),
                'experiment_name': config['name'].replace('exp3_', ''),
                'game_scenario': config['scenario'].upper(),
                'player1': config['player1'],
                'player2': config['player2'],
                'p_value_path': config['path'],
                'p_start': config.get('p_start'),
                'p_end': config.get('p_end'),
                'p_value_change': (config['p_start'] - config['p_end']) if config.get('p_start') and config.get('p_end') else None,
                'total_rounds': config['rounds'],
                'overall_cooperation_rate': config['cooperation_rate'],
                'player1_cooperation_rate': exp_data['player1_cooperate'].mean() if 'player1_cooperate' in exp_data else None,
                'player2_cooperation_rate': exp_data['player2_cooperate'].mean() if 'player2_cooperate' in exp_data else None,
                'player1_avg_score': exp_data['points1'].mean() if 'points1' in exp_data.columns else None,
                'player2_avg_score': exp_data['points2'].mean() if 'points2' in exp_data.columns else None,
                'strategy_changes': self._count_strategy_changes(exp_data),
                'completed': config.get('completed', True)
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def _count_strategy_changes(self, data):
        """Count strategy changes"""
        if 'both_cooperate' not in data.columns or len(data) < 2:
            return 0
        
        changes = (data['both_cooperate'].shift() != data['both_cooperate']).sum()
        return max(0, changes - 1)
    
    def plot_scenario_comparison(self, save_path=None):
        """Scenario comparison plot (PD vs BOS)"""
        summary_df = self.create_summary_dataframe()
        
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            pd_data = summary_df[summary_df['game_scenario'] == 'PD']
            bos_data = summary_df[summary_df['game_scenario'] == 'BOS']
            
            # Cooperation rate comparison
            if len(pd_data) > 0 and len(bos_data) > 0:
                ax1 = axes[0, 0]
                scenarios = ['PD', 'BOS']
                coop_rates = [pd_data['overall_cooperation_rate'].mean(), bos_data['overall_cooperation_rate'].mean()]
                coop_std = [pd_data['overall_cooperation_rate'].std(), bos_data['overall_cooperation_rate'].std()]
                
                bars = ax1.bar(scenarios, coop_rates, yerr=coop_std, capsize=5, 
                              color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
                ax1.set_title('Cooperation Rate by Game Scenario')
                ax1.set_ylabel('Average Cooperation Rate')
                ax1.set_ylim(0, 1.1)
                
                for bar, rate in zip(bars, coop_rates):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{rate:.1%}', ha='center', va='bottom')
            
            # P-value sensitivity comparison
            ax2 = axes[0, 1]
            for scenario, color in [('PD', COLORS['primary']), ('BOS', COLORS['secondary'])]:
                scenario_data = summary_df[summary_df['game_scenario'] == scenario]
                if len(scenario_data) > 0:
                    valid_data = scenario_data.dropna(subset=['p_value_change', 'overall_cooperation_rate'])
                    if len(valid_data) > 1:
                        ax2.scatter(valid_data['p_value_change'], valid_data['overall_cooperation_rate'], 
                                  alpha=0.7, s=80, label=scenario, color=color)
            
            ax2.set_xlabel('P-value Change')
            ax2.set_ylabel('Cooperation Rate')
            ax2.set_title('P-value Sensitivity by Scenario')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Player comparison
            ax3 = axes[1, 0]
            player_stats = summary_df.groupby(['player1', 'player2']).agg({
                'overall_cooperation_rate': 'mean',
                'experiment_name': 'count'
            }).reset_index()
            
            if len(player_stats) > 0:
                player_pairs = player_stats['player1'] + ' vs ' + player_stats['player2']
                bars = ax3.barh(range(len(player_pairs)), player_stats['overall_cooperation_rate'], 
                               color=COLORS['tertiary'], alpha=0.8)
                ax3.set_yticks(range(len(player_pairs)))
                ax3.set_yticklabels(player_pairs)
                ax3.set_xlabel('Average Cooperation Rate')
                ax3.set_title('AI Pairing Performance')
            
            # Path type comparison
            ax4 = axes[1, 1]
            path_stats = summary_df.groupby('p_value_path')['overall_cooperation_rate'].agg(['mean', 'std', 'count'])
            
            if len(path_stats) > 0:
                bars = ax4.bar(path_stats.index, path_stats['mean'], 
                              yerr=path_stats['std'], capsize=5, 
                              color=COLORS['quaternary'], alpha=0.8)
                ax4.set_title('Cooperation Rate by P-value Path')
                ax4.set_ylabel('Average Cooperation Rate')
                ax4.set_xlabel('P-value Change Path')
                
                for i, (path, stats) in enumerate(path_stats.iterrows()):
                    ax4.text(i, stats['mean'] + 0.02, 
                            f'n={stats["count"]}', ha='center', va='bottom')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def plot_detailed_p_trajectories(self, save_path=None):
        """Detailed p-value trajectory plot"""
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            path_types = ['up', 'down', 'random']
            colors = [COLORS['tertiary'], COLORS['secondary'], COLORS['accent']]
            
            for i, path_type in enumerate(path_types):
                ax = axes[i]
                
                path_experiments = [(name, data) for name, data in self.experiments.items() 
                                  if path_type in name and 'p_now' in data.columns]
                
                if path_experiments:
                    for j, (exp_name, exp_data) in enumerate(path_experiments):
                        label = exp_name.replace('exp3_', '').replace(f'_{path_type}', '')
                        ax.plot(exp_data['round'], exp_data['p_now'], 
                               marker='o', label=label, linewidth=2, alpha=0.8, color=colors[i])
                    
                    ax.set_title(f'P-value Trajectory: {path_type.upper()}', fontweight='bold', fontsize=12)
                    ax.set_xlabel('Round')
                    ax.set_ylabel('P-value')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'No {path_type} type experiment data', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'P-value Trajectory: {path_type.upper()}')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def plot_cooperation_evolution_matrix(self, save_path=None):
        """Cooperation evolution matrix plot"""
        if len(self.experiments) == 0:
            print("No experiment data to plot")
            return
        
        with PdfPages(save_path) as pdf:
            n_experiments = min(len(self.experiments), 9)
            cols = 3
            rows = (n_experiments + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            experiments_list = list(self.experiments.items())[:n_experiments]
            
            for idx, (exp_name, exp_data) in enumerate(experiments_list):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col]
                
                config = next((c for c in self.experiment_configs if c['name'] == exp_name), None)
                
                if config and 'both_cooperate' in exp_data.columns:
                    window_size = max(3, len(exp_data) // 10)
                    if config['scenario'].lower() == 'pd':
                        metric = exp_data['both_cooperate'].rolling(window=window_size, center=True, min_periods=1).mean()
                        metric_name = 'Cooperation Rate'
                    else:
                        metric = exp_data['both_cooperate'].rolling(window=window_size, center=True, min_periods=1).mean()
                        metric_name = 'Cooperation Rate'
                    
                    ax.plot(exp_data['round'], metric, linewidth=3, color=COLORS['primary'], label='Cooperation Rate')
                    ax.fill_between(exp_data['round'], metric, alpha=0.3, color=COLORS['primary'])
                    
                    if 'p_now' in exp_data.columns:
                        ax2 = ax.twinx()
                        ax2.plot(exp_data['round'], exp_data['p_now'], 
                               'r--', linewidth=2, alpha=0.7, label='P-value')
                        ax2.set_ylabel('P-value', color='red', fontsize=10)
                        ax2.tick_params(axis='y', colors='red')
                    
                    avg_coop = exp_data['both_cooperate'].mean()
                    ax.text(0.02, 0.98, f'Avg Cooperation: {avg_coop:.1%}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           fontsize=9)
                
                ax.set_title(exp_name.replace('exp3_', ''), fontsize=10, fontweight='bold')
                ax.set_xlabel('Round', fontsize=9)
                ax.set_ylabel('Cooperation Rate', fontsize=9, color='blue')
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3)
            
            for idx in range(n_experiments, rows * cols):
                row = idx // cols
                col = idx % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("="*70)
        print("Experiment 3: 18-Configuration AI Decision Behavior Analysis Report")
        print("="*70)
        
        summary_df = self.create_summary_dataframe()
        
        print(f"\n【Data Overview】")
        print(f"  Successfully loaded experiments: {len(self.experiments)}")
        
        if len(summary_df) == 0:
            print("  Warning: No experiment data loaded successfully")
            return
        
        print(f"  PD scenario experiments: {len(summary_df[summary_df['game_scenario'] == 'PD'])}")
        print(f"  BOS scenario experiments: {len(summary_df[summary_df['game_scenario'] == 'BOS'])}")
        
        print(f"\n【Overall Statistics】")
        print(f"  Average cooperation rate: {summary_df['overall_cooperation_rate'].mean():.1%}")
        print(f"  Cooperation rate std dev: {summary_df['overall_cooperation_rate'].std():.1%}")
        print(f"  Highest cooperation rate: {summary_df['overall_cooperation_rate'].max():.1%}")
        print(f"  Lowest cooperation rate: {summary_df['overall_cooperation_rate'].min():.1%}")
        
        if 'p_value_path' in summary_df.columns:
            print(f"\n【Path Type Analysis】")
            for path in summary_df['p_value_path'].unique():
                if pd.notna(path):
                    path_data = summary_df[summary_df['p_value_path'] == path]
                    print(f"  {path}: {len(path_data)} experiments, avg cooperation rate {path_data['overall_cooperation_rate'].mean():.1%}")
        
        print(f"\n【AI Pairing Analysis】")
        pairing_stats = summary_df.groupby(['player1', 'player2'])['overall_cooperation_rate'].agg(['mean', 'count'])
        for (p1, p2), stats in pairing_stats.iterrows():
            print(f"  {p1} vs {p2}: {stats['count']} experiments, avg cooperation rate {stats['mean']:.1%}")
        
        print(f"\n【Key Findings】")
        max_coop_exp = summary_df.loc[summary_df['overall_cooperation_rate'].idxmax()]
        min_coop_exp = summary_df.loc[summary_df['overall_cooperation_rate'].idxmin()]
        
        print(f"  • Most cooperative config: {max_coop_exp['experiment_name']} (cooperation rate: {max_coop_exp['overall_cooperation_rate']:.1%})")
        print(f"  • Least cooperative config: {min_coop_exp['experiment_name']} (cooperation rate: {min_coop_exp['overall_cooperation_rate']:.1%})")
        
        sensitivity_data = summary_df.dropna(subset=['p_value_change', 'overall_cooperation_rate'])
        if len(sensitivity_data) > 1:
            correlation = sensitivity_data['p_value_change'].corr(sensitivity_data['overall_cooperation_rate'])
            print(f"  • P-value change vs cooperation rate correlation: {correlation:.3f}")
    
    def save_all_analysis(self, output_dir='experiment3_analysis'):
        """Save all analysis results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\nGenerating analysis results...")
        
        summary_df = self.create_summary_dataframe()
        summary_df.to_csv(f'{output_dir}/experiment_summary.csv', index=False, encoding='utf-8-sig')
        
        self.plot_scenario_comparison(f'{output_dir}/scenario_comparison.pdf')
        self.plot_detailed_p_trajectories(f'{output_dir}/p_value_trajectories.pdf')
        self.plot_cooperation_evolution_matrix(f'{output_dir}/cooperation_evolution_matrix.pdf')
        
        print(f"✓ Analysis results saved to {output_dir}/ directory")

if __name__ == "__main__":
    analyzer = MultiExperimentAnalyzer('Experiment3')
    analyzer.generate_comprehensive_report()
    analyzer.save_all_analysis()
    print("\n🎉 Multi-experiment analysis complete!")