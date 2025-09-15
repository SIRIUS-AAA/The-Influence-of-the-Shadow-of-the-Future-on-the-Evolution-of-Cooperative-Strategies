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

# Read data from csv files, load the files
class Experiment1Analyzer:
    """Analyzer for Experiment 1: P-value disclosure effect in finite games"""
    
    def __init__(self, experiment_folder='Experiment1', summary_file='exp1_summary.xlsx'):
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
        """Load all experiment data based on summary table"""
        possible_paths = [
            Path(self.experiment_folder),
            Path('.'),
            Path('./Experiment1'),
            Path('../Experiment1'),
            Path('./Data'),
            Path('../Data')
        ]
        
        experiment_path = None
        csv_files_found = []
        
        for path in possible_paths:
            if path.exists():
                csv_pattern = list(path.glob('exp1*.csv'))
                if csv_pattern:
                    experiment_path = path
                    csv_files_found = csv_pattern
                    break
        
        if experiment_path is None:
            print(f"Error: No exp1*.csv files found")
            return
        
        print(f"Found {len(csv_files_found)} CSV files in {experiment_path.absolute()}")
        loaded_count = 0
        
        if not self.summary_data.empty:
            for _, row in self.summary_data.iterrows():
                exp_filename = row.iloc[-1]
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
                            data['coordination'] = (data['answer1'] == data['answer2'])
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
                    print(f"- {exp_filename}: File not found")
        
        print(f"Successfully loaded {loaded_count} experiments")
    
    def _parse_config_from_summary(self, exp_name, summary_row, data):
        """Parse experiment configuration from summary row"""
        config = {
            'name': exp_name,
            'experiment_id': summary_row.iloc[0],
            'game_type': summary_row.iloc[1],
            'round_type': summary_row.iloc[2], 
            'total_rounds': summary_row.iloc[3],
            'has_p_value': summary_row.iloc[4] == 'y',
            'p_value': summary_row.iloc[5] if pd.notna(summary_row.iloc[5]) else None,
            'player1': summary_row.iloc[6],
            'player2': summary_row.iloc[7],
            'seed': summary_row.iloc[8],
            'rounds': len(data)
        }
        return config
    
    def create_summary_dataframe(self):
        """Create comprehensive summary dataframe"""
        summary_data = []
        
        for config in self.experiment_configs:
            exp_data = self.experiments[config['name']]
            
            if config['game_type'].lower() == 'pd':
                cooperation_rate = exp_data['both_cooperate'].mean()
                player1_coop_rate = exp_data['player1_cooperate'].mean()
                player2_coop_rate = exp_data['player2_cooperate'].mean()
                coordination_rate = cooperation_rate
            else:
                coordination_rate = exp_data['coordination'].mean()
                player1_coop_rate = exp_data['player1_cooperate'].mean()
                player2_coop_rate = exp_data['player2_cooperate'].mean()
                cooperation_rate = coordination_rate
            
            summary = {
                'experiment_id': config['experiment_id'],
                'experiment_name': config['name'].replace('exp1_', ''),
                'game_type': config['game_type'].upper(),
                'has_p_value': config['has_p_value'],
                'p_value': config['p_value'],
                'player1': config['player1'],
                'player2': config['player2'], 
                'total_rounds': config['rounds'],
                'cooperation_rate': cooperation_rate,
                'coordination_rate': coordination_rate,
                'player1_cooperation_rate': player1_coop_rate,
                'player2_cooperation_rate': player2_coop_rate,
                'avg_score_p1': exp_data['points1'].mean(),
                'avg_score_p2': exp_data['points2'].mean(),
                'final_score_p1': exp_data['total1'].iloc[-1] if len(exp_data) > 0 else 0,
                'final_score_p2': exp_data['total2'].iloc[-1] if len(exp_data) > 0 else 0
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    # Analyze the effect of p-value on the cooperation rate of agents
    def plot_p_value_effect_comparison(self, save_path=None):
        """Compare performance with vs without p-value disclosure"""
        summary_df = self.create_summary_dataframe()
        
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            with_p = summary_df[summary_df['has_p_value'] == True]
            without_p = summary_df[summary_df['has_p_value'] == False]
            
            # Cooperation rate comparison
            ax1 = axes[0, 0]
            categories = ['Without P-value', 'With P-value']
            coop_rates = [without_p['cooperation_rate'].mean(), with_p['cooperation_rate'].mean()]
            coop_std = [without_p['cooperation_rate'].std(), with_p['cooperation_rate'].std()]
            
            bars = ax1.bar(categories, coop_rates, yerr=coop_std, capsize=5, 
                          color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
            ax1.set_title('Cooperation Rate: P-value Disclosure Effect')
            ax1.set_ylabel('Average Cooperation Rate')
            ax1.set_ylim(0, 1.1)
            
            for bar, rate in zip(bars, coop_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{rate:.1%}', ha='center', va='bottom')
            
            # Game type breakdown
            ax2 = axes[0, 1]
            game_comparison = summary_df.groupby(['game_type', 'has_p_value'])['cooperation_rate'].mean().unstack()
            game_comparison.plot(kind='bar', ax=ax2, color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
            ax2.set_title('Cooperation by Game Type and P-value')
            ax2.set_ylabel('Cooperation Rate')
            ax2.set_xlabel('Game Type')
            ax2.legend(['Without P-value', 'With P-value'])
            ax2.tick_params(axis='x', rotation=0)
            
            # P-value levels effect
            ax3 = axes[1, 0]
            if len(with_p) > 0:
                p_value_effect = with_p.groupby('p_value')['cooperation_rate'].agg(['mean', 'std']).reset_index()
                ax3.errorbar(p_value_effect['p_value'], p_value_effect['mean'], 
                            yerr=p_value_effect['std'], marker='o', capsize=5, 
                            color=COLORS['tertiary'], linewidth=2, markersize=8)
                ax3.set_title('Effect of Different P-values on Cooperation')
                ax3.set_xlabel('P-value')
                ax3.set_ylabel('Cooperation Rate')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No P-value experiments found', ha='center', va='center')
            
            # Player pairing comparison
            ax4 = axes[1, 1]
            pairing_stats = summary_df.groupby(['player1', 'player2', 'has_p_value'])['cooperation_rate'].mean().unstack()
            if not pairing_stats.empty:
                pairing_stats.plot(kind='bar', ax=ax4, color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
                ax4.set_title('Player Pairing Performance')
                ax4.set_ylabel('Cooperation Rate')
                ax4.legend(['Without P-value', 'With P-value'])
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def plot_strategy_evolution(self, save_path=None):
        """Plot strategy evolution over rounds"""
        n_experiments = min(len(self.experiments), 6)
        
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            experiments_list = list(self.experiments.items())[:n_experiments]
            
            for idx, (exp_name, exp_data) in enumerate(experiments_list):
                ax = axes[idx]
                
                config = next((c for c in self.experiment_configs if c['name'] == exp_name), None)
                
                if config and 'coordination' in exp_data.columns:
                    window_size = max(3, len(exp_data) // 10)
                    if config['game_type'].lower() == 'pd':
                        metric = exp_data['both_cooperate'].rolling(window=window_size, center=True, min_periods=1).mean()
                        metric_name = 'Cooperation Rate'
                    else:
                        metric = exp_data['coordination'].rolling(window=window_size, center=True, min_periods=1).mean()
                        metric_name = 'Coordination Rate'
                    
                    ax.plot(exp_data['round'], metric, linewidth=2, color=COLORS['primary'], alpha=0.8)
                    ax.fill_between(exp_data['round'], metric, alpha=0.3, color=COLORS['primary'])
                    
                    p_info = f"p={config['p_value']}" if config['has_p_value'] else "No P-value"
                    ax.set_title(f"{exp_name.replace('exp1_', '')}\n{config['game_type'].upper()} - {p_info}")
                    ax.set_xlabel('Round')
                    ax.set_ylabel(metric_name)
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(True, alpha=0.3)
            
            for idx in range(n_experiments, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def plot_score_analysis(self, save_path=None):
        """Analyze score patterns"""
        summary_df = self.create_summary_dataframe()
        
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Average scores with/without p-value
            ax1 = axes[0, 0]
            score_comparison = summary_df.groupby('has_p_value').agg({
                'avg_score_p1': ['mean', 'std'],
                'avg_score_p2': ['mean', 'std']
            })
            
            x_pos = np.arange(2)
            width = 0.35
            
            ax1.bar(x_pos - width/2, score_comparison['avg_score_p1']['mean'], width, 
                   yerr=score_comparison['avg_score_p1']['std'], capsize=5, 
                   label='Player 1', alpha=0.8, color=COLORS['primary'])
            ax1.bar(x_pos + width/2, score_comparison['avg_score_p2']['mean'], width,
                   yerr=score_comparison['avg_score_p2']['std'], capsize=5, 
                   label='Player 2', alpha=0.8, color=COLORS['secondary'])
            
            ax1.set_title('Average Scores by P-value Disclosure')
            ax1.set_ylabel('Average Score per Round')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(['Without P-value', 'With P-value'])
            ax1.legend()
            
            # Final cumulative scores
            ax2 = axes[0, 1]
            colors = [COLORS['primary'] if not has_p else COLORS['secondary'] for has_p in summary_df['has_p_value']]
            ax2.scatter(summary_df['final_score_p1'], summary_df['final_score_p2'], 
                       c=colors, alpha=0.7, s=100)
            ax2.set_xlabel('Player 1 Final Score')
            ax2.set_ylabel('Player 2 Final Score') 
            ax2.set_title('Final Score Distribution')
            max_score = max(summary_df['final_score_p1'].max(), summary_df['final_score_p2'].max())
            ax2.plot([0, max_score], [0, max_score], 'r--', alpha=0.5, label='Equal scores')
            ax2.legend()
            
            # Score efficiency (total welfare)
            ax3 = axes[1, 0]
            summary_df['total_welfare'] = summary_df['final_score_p1'] + summary_df['final_score_p2']
            welfare_comparison = summary_df.groupby('has_p_value')['total_welfare'].agg(['mean', 'std'])
            
            bars = ax3.bar(['Without P-value', 'With P-value'], welfare_comparison['mean'],
                          yerr=welfare_comparison['std'], capsize=5, 
                          color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
            ax3.set_title('Total Welfare (Sum of Scores)')
            ax3.set_ylabel('Total Score')
            
            for bar, value in zip(bars, welfare_comparison['mean']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{value:.1f}', ha='center', va='bottom')
            
            # Game type welfare comparison
            ax4 = axes[1, 1]
            welfare_by_game = summary_df.groupby(['game_type', 'has_p_value'])['total_welfare'].mean().unstack()
            welfare_by_game.plot(kind='bar', ax=ax4, color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
            ax4.set_title('Welfare by Game Type')
            ax4.set_ylabel('Total Welfare')
            ax4.legend(['Without P-value', 'With P-value'])
            ax4.tick_params(axis='x', rotation=0)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("="*70)
        print("Experiment 1: P-value Disclosure Effect Analysis Report")
        print("="*70)
        
        summary_df = self.create_summary_dataframe()
        
        if len(summary_df) == 0:
            print("Warning: No experiment data loaded")
            return
            
        print(f"\n【Data Overview】")
        print(f"  Total experiments loaded: {len(self.experiments)}")
        print(f"  Experiments with P-value: {len(summary_df[summary_df['has_p_value'] == True])}")
        print(f"  Experiments without P-value: {len(summary_df[summary_df['has_p_value'] == False])}")
        
        game_breakdown = summary_df['game_type'].value_counts()
        print(f"  Game types: {dict(game_breakdown)}")
        
        print(f"\n【P-value Disclosure Effect】")
        
        with_p = summary_df[summary_df['has_p_value'] == True]
        without_p = summary_df[summary_df['has_p_value'] == False]
        
        if len(with_p) > 0 and len(without_p) > 0:
            coop_with_p = with_p['cooperation_rate'].mean()
            coop_without_p = without_p['cooperation_rate'].mean()
            
            print(f"  Cooperation with P-value: {coop_with_p:.1%}")
            print(f"  Cooperation without P-value: {coop_without_p:.1%}")
            print(f"  Difference: {(coop_with_p - coop_without_p):.1%}")
            
            try:
                from scipy import stats
                t_stat, p_val = stats.ttest_ind(with_p['cooperation_rate'], without_p['cooperation_rate'])
                print(f"  T-test p-value: {p_val:.4f} {'(significant)' if p_val < 0.05 else '(not significant)'}")
            except ImportError:
                print("  Statistical test unavailable (scipy not installed)")
        
        print(f"\n【P-value Level Effects】")
        if len(with_p) > 0:
            p_value_effects = with_p.groupby('p_value')['cooperation_rate'].agg(['mean', 'count'])
            for p_val, stats in p_value_effects.iterrows():
                print(f"  P-value {p_val}: {stats['mean']:.1%} cooperation ({stats['count']} experiments)")
        
        print(f"\n【Game Type Analysis】")
        for game_type in summary_df['game_type'].unique():
            game_data = summary_df[summary_df['game_type'] == game_type]
            print(f"  {game_type}:")
            
            game_with_p = game_data[game_data['has_p_value'] == True]
            game_without_p = game_data[game_data['has_p_value'] == False]
            
            if len(game_with_p) > 0:
                print(f"    With P-value: {game_with_p['cooperation_rate'].mean():.1%} cooperation")
            if len(game_without_p) > 0:
                print(f"    Without P-value: {game_without_p['cooperation_rate'].mean():.1%} cooperation")
        
        print(f"\n【Performance Analysis】")
        if len(with_p) > 0 and len(without_p) > 0:
            print(f"  Average welfare with P-value: {(with_p['avg_score_p1'] + with_p['avg_score_p2']).mean():.1f}")
            print(f"  Average welfare without P-value: {(without_p['avg_score_p1'] + without_p['avg_score_p2']).mean():.1f}")
    
    def save_all_analysis(self, output_dir='experiment1_analysis'):
        """Save all analysis results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\nGenerating analysis results...")
        
        summary_df = self.create_summary_dataframe()
        summary_df.to_csv(f'{output_dir}/exp1_summary_analysis.csv', index=False, encoding='utf-8-sig')
        
        self.plot_p_value_effect_comparison(f'{output_dir}/p_value_effect_comparison.pdf')
        self.plot_strategy_evolution(f'{output_dir}/strategy_evolution.pdf')
        self.plot_score_analysis(f'{output_dir}/score_analysis.pdf')
        
        print(f"✓ Analysis results saved to {output_dir}/ directory")

if __name__ == "__main__":
    analyzer = Experiment1Analyzer('Experiment1', 'exp1_summary.xlsx')
    analyzer.generate_comprehensive_report()
    analyzer.save_all_analysis()
    print("\n🎉 Experiment 1 analysis complete!")