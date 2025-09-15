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

class Experiment2Analyzer:
    """Analyzer for Experiment 2: Finite vs Infinite Games with T and P parameter effects"""
    
    def __init__(self, experiment_folder='Experiment2', summary_file='exp2_summary.xlsx'):
        self.experiment_folder = experiment_folder
        self.experiments = {}
        self.experiment_configs = []
        self.summary_data = self._load_summary_table(summary_file)
        self.load_all_experiments()
        
    def _load_summary_table(self, summary_file):
        """Load experiment summary table"""
        try:
            df = pd.read_excel(summary_file, sheet_name=0)
            print(f"Loaded {len(df)} experiment configurations")
            return df
        except Exception as e:
            print(f"Cannot read summary file: {e}")
            return pd.DataFrame()
    
    def load_all_experiments(self):
        """Load all experiment CSV files"""
        possible_paths = [
            Path('./Experiment2'), 
            Path('Experiment2'), 
            Path('../Experiment2'), 
            Path('.')
        ]
        
        experiment_path = None
        for path in possible_paths:
            if path.exists() and list(path.glob('exp2*.csv')):
                experiment_path = path
                break
        
        if not experiment_path:
            print("Error: No exp2*.csv files found")
            return
        
        print(f"Loading from {experiment_path.absolute()}")
        loaded_count = 0
        
        for _, row in self.summary_data.iterrows():
            exp_filename = row.iloc[-1]
            csv_file = experiment_path / exp_filename
            
            if csv_file.exists():
                try:
                    data = pd.read_csv(csv_file)
                    
                    if 'round' in data.columns and 'match_id' in data.columns:
                        data = data.drop_duplicates(subset=['match_id', 'round'], keep='first')
                        data = data.sort_values(['match_id', 'round']).reset_index(drop=True)
                    
                    if 'answer1' in data.columns and 'answer2' in data.columns:
                        data['both_cooperate'] = (data['answer1'] == 'J') & (data['answer2'] == 'J')
                        data['coordination'] = (data['answer1'] == data['answer2'])
                    
                    exp_name = csv_file.stem
                    self.experiments[exp_name] = data
                    self.experiment_configs.append(self._parse_config(exp_name, row))
                    
                    print(f"✓ {exp_filename} ({len(data)} rounds, {data['match_id'].nunique()} games)")
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"✗ {exp_filename}: {str(e)}")
        
        print(f"Successfully loaded {loaded_count} experiments")
    
    def _parse_config(self, exp_name, row):
        """Parse experiment configuration from summary row"""
        return {
            'name': exp_name,
            'game_type': row.iloc[1],
            'player1': row.iloc[8],
            'player2': row.iloc[9],
        }

    def create_summary_dataframe(self):
        """Create summary dataframe with all parameter combinations"""
        summary_data = []
        
        for config in self.experiment_configs:
            exp_data = self.experiments[config['name']]
            
            finite_data = exp_data[exp_data['horizon_type'] == 'fixed']
            if len(finite_data) > 0:
                self._process_horizon_data(finite_data, config, summary_data)
            
            infinite_data = exp_data[exp_data['horizon_type'] == 'continuation']
            if len(infinite_data) > 0:
                self._process_horizon_data(infinite_data, config, summary_data)
        
        return pd.DataFrame(summary_data)
    
    def _process_horizon_data(self, data, config, summary_data):
        """Process data for one horizon type"""
        group_cols = ['horizon_type']
        
        if 'T' in data.columns and not data['T'].isna().all():
            group_cols.append('T')
        if 'p' in data.columns and not data['p'].isna().all():
            group_cols.append('p')
        if 'T_in_prompt' in data.columns:
            group_cols.append('T_in_prompt')
        if 'p_in_prompt' in data.columns:
            group_cols.append('p_in_prompt')
        
        for group_key, group_data in data.groupby(group_cols):
            if len(group_data) == 0:
                continue
                
            horizon = group_key[0] if isinstance(group_key, tuple) else group_key
            T_val = p_val = T_prompt = p_prompt = None
            
            if isinstance(group_key, tuple):
                for i, val in enumerate(group_key[1:], 1):
                    col_name = group_cols[i]
                    if col_name == 'T':
                        T_val = val
                    elif col_name == 'p':
                        p_val = val
                    elif col_name == 'T_in_prompt':
                        T_prompt = val
                    elif col_name == 'p_in_prompt':
                        p_prompt = val
            
            if config['game_type'].lower() == 'pd':
                cooperation_rate = group_data['both_cooperate'].mean() if 'both_cooperate' in group_data.columns else 0
                coordination_rate = 0
                success_rate = cooperation_rate
            else:
                cooperation_rate = 0
                coordination_rate = group_data['coordination'].mean() if 'coordination' in group_data.columns else 0
                success_rate = coordination_rate
            
            if 'match_id' in group_data.columns:
                match_lengths = group_data.groupby('match_id')['round'].max()
                avg_game_length = match_lengths.mean()
                total_games = group_data['match_id'].nunique()
            else:
                avg_game_length = len(group_data)
                total_games = 1
            
            summary = {
                'experiment_name': config['name'],
                'game_type': config['game_type'].upper(),
                'horizon_type': horizon,
                'T_value': T_val,
                'P_value': p_val,
                'T_in_prompt': T_prompt if T_prompt is not None else 0,
                'p_in_prompt': p_prompt if p_prompt is not None else 0,
                'player1': config['player1'],
                'player2': config['player2'],
                'success_rate': success_rate,
                'cooperation_rate': cooperation_rate,
                'coordination_rate': coordination_rate,
                'avg_game_length': avg_game_length,
                'total_games': total_games,
                'avg_score_p1': group_data['points1'].mean() if 'points1' in group_data.columns else 0,
                'avg_score_p2': group_data['points2'].mean() if 'points2' in group_data.columns else 0,
            }
            summary_data.append(summary)

    def plot_tp_joint_heatmaps(self, save_path=None):
        """Plot T and P joint effects on strategy choices"""
        summary_df = self.create_summary_dataframe()
        
        if len(summary_df) == 0:
            print("No data available")
            return
        
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            joint_data = []
            
            for _, row in summary_df.iterrows():
                if row['horizon_type'] == 'fixed':
                    T_val = row['T_value'] if pd.notna(row['T_value']) else 0
                    P_val = 0
                else:
                    T_val = 0
                    P_val = row['P_value'] if pd.notna(row['P_value']) else 0
                
                joint_data.append({
                    'T_value': T_val,
                    'P_value': P_val,
                    'game_type': row['game_type'],
                    'cooperation_rate': row['cooperation_rate'],
                    'coordination_rate': row['coordination_rate'],
                    'player_combo': f"{row['player1']}_vs_{row['player2']}"
                })
            
            joint_df = pd.DataFrame(joint_data)
            
            # PD Game: Joint T and P effects on cooperation strategy
            ax1 = axes[0]
            pd_data = joint_df[joint_df['game_type'] == 'PD']
            if len(pd_data) > 0:
                tp_pd_pivot = pd_data.pivot_table(
                    values='cooperation_rate',
                    index='T_value',
                    columns='P_value',
                    aggfunc='mean'
                )
                sns.heatmap(tp_pd_pivot, annot=True, fmt='.3f', cmap='Blues', ax=ax1, 
                           cbar_kws={'label': 'Cooperation Rate'})
                ax1.set_title('PD Game: Joint T and P Effects on Cooperation Strategy')
                ax1.set_xlabel('P Value (Infinite Game Continuation Probability)')
                ax1.set_ylabel('T Value (Finite Game Rounds)')
            
            # BoS Game: Joint T and P effects on coordination strategy
            ax2 = axes[1]
            bos_data = joint_df[joint_df['game_type'] == 'BOS']
            if len(bos_data) > 0:
                tp_bos_pivot = bos_data.pivot_table(
                    values='coordination_rate',
                    index='T_value', 
                    columns='P_value',
                    aggfunc='mean'
                )
                sns.heatmap(tp_bos_pivot, annot=True, fmt='.3f', cmap='Reds', ax=ax2, 
                           cbar_kws={'label': 'Coordination Rate'})
                ax2.set_title('BoS Game: Joint T and P Effects on Coordination Strategy')
                ax2.set_xlabel('P Value (Infinite Game Continuation Probability)')
                ax2.set_ylabel('T Value (Finite Game Rounds)')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def plot_performance_comparison(self, save_path=None):
        """Plot key performance comparisons"""
        summary_df = self.create_summary_dataframe()
        
        if len(summary_df) == 0:
            print("No data available")
            return
        
        with PdfPages(save_path) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Success rates by horizon type
            ax1 = axes[0, 0]
            horizon_stats = summary_df.groupby('horizon_type')['success_rate'].agg(['mean', 'std', 'count'])
            bars = ax1.bar(horizon_stats.index, horizon_stats['mean'], 
                          yerr=horizon_stats['std'], capsize=5, 
                          color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
            ax1.set_title('Success Rate: Finite vs Infinite Games')
            ax1.set_ylabel('Success Rate')
            
            for bar, mean, count in zip(bars, horizon_stats['mean'], horizon_stats['count']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.1%}\n(n={count})', ha='center', va='bottom')
            
            # Game type comparison
            ax2 = axes[0, 1]
            game_stats = summary_df.groupby('game_type')['success_rate'].agg(['mean', 'std', 'count'])
            bars = ax2.bar(game_stats.index, game_stats['mean'], 
                          yerr=game_stats['std'], capsize=5, alpha=0.8, 
                          color=[COLORS['tertiary'], COLORS['quaternary']])
            ax2.set_title('Success Rate by Game Type')
            ax2.set_ylabel('Success Rate')
            
            for bar, mean, count in zip(bars, game_stats['mean'], game_stats['count']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.1%}\n(n={count})', ha='center', va='bottom')
            
            # Parameter disclosure effects
            ax3 = axes[1, 0]
            disclosure_data = []
            labels = []
            
            if 'T_in_prompt' in summary_df.columns:
                t_disclosed = summary_df[summary_df['T_in_prompt'] == 1]['success_rate']
                t_hidden = summary_df[summary_df['T_in_prompt'] == 0]['success_rate']
                if len(t_disclosed) > 0 and len(t_hidden) > 0:
                    disclosure_data.extend([t_hidden, t_disclosed])
                    labels.extend(['T Hidden', 'T Disclosed'])
            
            if 'p_in_prompt' in summary_df.columns:
                p_disclosed = summary_df[summary_df['p_in_prompt'] == 1]['success_rate']
                p_hidden = summary_df[summary_df['p_in_prompt'] == 0]['success_rate']
                if len(p_disclosed) > 0 and len(p_hidden) > 0:
                    disclosure_data.extend([p_hidden, p_disclosed])
                    labels.extend(['P Hidden', 'P Disclosed'])
            
            if disclosure_data:
                box_plot = ax3.boxplot(disclosure_data, labels=labels, patch_artist=True)
                for patch, color in zip(box_plot['boxes'], [COLORS['primary'], COLORS['secondary']] * 2):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                ax3.set_title('Parameter Disclosure Effects')
                ax3.set_ylabel('Success Rate')
                ax3.tick_params(axis='x', rotation=45)
            
            # Game length vs success
            ax4 = axes[1, 1]
            colors = {'fixed': COLORS['primary'], 'continuation': COLORS['secondary']}
            for horizon in summary_df['horizon_type'].unique():
                data_subset = summary_df[summary_df['horizon_type'] == horizon]
                ax4.scatter(data_subset['avg_game_length'], data_subset['success_rate'], 
                           alpha=0.7, c=colors.get(horizon, COLORS['neutral']), label=horizon, s=60)
            
            ax4.set_xlabel('Average Game Length')
            ax4.set_ylabel('Success Rate')
            ax4.set_title('Game Length vs Success Rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def generate_report(self):
        """Generate analysis report"""
        print("=" * 60)
        print("Experiment 2: Finite vs Infinite Game Analysis")
        print("=" * 60)
        
        summary_df = self.create_summary_dataframe()
        
        if len(summary_df) == 0:
            print("No data available")
            return
        
        print(f"\nData Overview:")
        print(f"  Experiment configurations: {len(self.experiments)}")
        print(f"  Parameter combinations: {len(summary_df)}")
        print(f"  Game types: {', '.join(summary_df['game_type'].unique())}")
        
        print(f"\nKey Findings:")
        
        # Horizon comparison
        horizon_means = summary_df.groupby('horizon_type')['success_rate'].mean()
        if len(horizon_means) >= 2:
            finite_rate = horizon_means.get('fixed', 0)
            infinite_rate = horizon_means.get('continuation', 0)
            print(f"  Finite game success rate: {finite_rate:.1%}")
            print(f"  Infinite game success rate: {infinite_rate:.1%}")
            print(f"  Difference: {(infinite_rate - finite_rate):+.1%}")
        
        # Game type comparison
        game_means = summary_df.groupby('game_type')['success_rate'].mean()
        for game_type, rate in game_means.items():
            print(f"  {game_type} game success rate: {rate:.1%}")
        
        # Parameter effects
        if 'T_in_prompt' in summary_df.columns:
            t_effect = summary_df.groupby('T_in_prompt')['success_rate'].mean()
            if len(t_effect) >= 2:
                print(f"  T parameter disclosure effect: {t_effect.get(1, 0) - t_effect.get(0, 0):+.1%}")
        
        if 'p_in_prompt' in summary_df.columns:
            p_effect = summary_df.groupby('p_in_prompt')['success_rate'].mean()
            if len(p_effect) >= 2:
                print(f"  P parameter disclosure effect: {p_effect.get(1, 0) - p_effect.get(0, 0):+.1%}")
    
    def save_analysis(self, output_dir='experiment2_analysis'):
        """Save all analysis results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        summary_df = self.create_summary_dataframe()
        if len(summary_df) > 0:
            summary_df.to_csv(f'{output_dir}/exp2_summary.csv', index=False, encoding='utf-8-sig')
            
            try:
                self.plot_tp_joint_heatmaps(f'{output_dir}/tp_joint_heatmaps.pdf')
                self.plot_performance_comparison(f'{output_dir}/performance_comparison.pdf')
                print(f"✓ Results saved to {output_dir}/")
            except Exception as e:
                print(f"Plotting error: {str(e)}")
        else:
            print("No data to save")

if __name__ == "__main__":
    analyzer = Experiment2Analyzer('Experiment2', 'exp2_summary.xlsx')
    analyzer.generate_report()
    analyzer.save_analysis()