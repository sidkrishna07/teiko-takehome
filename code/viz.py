#!/usr/bin/env python3
"""
Visualization and Statistical Testing for TR1 Response Analysis
==============================================================

Functions for creating boxplots and running statistical tests
comparing responders vs non-responders for each cell population.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, Any
import warnings

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("Set2")


def plot_tr1_response_comparison(tr1_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create boxplots comparing responders vs non-responders for each cell population.
    
    Args:
        tr1_df: DataFrame from compare_tr1_response() with columns:
                ['sample_id', 'population', 'relative_frequency', 'response', ...]
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib.pyplot.Figure: Figure object containing the boxplots
        
    Raises:
        ValueError: If required columns are missing or no data provided
    """
    
    # Validate input
    if tr1_df.empty:
        raise ValueError("Input DataFrame is empty")
    
    required_cols = ['population', 'relative_frequency', 'response']
    missing_cols = [col for col in required_cols if col not in tr1_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get unique populations
    populations = sorted(tr1_df['population'].unique())
    n_populations = len(populations)
    
    if n_populations == 0:
        raise ValueError("No populations found in data")
    
    # Calculate subplot layout (prefer rectangular layout)
    n_cols = min(3, n_populations)  # Max 3 columns
    n_rows = (n_populations + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_populations == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    # Create boxplot for each population
    for i, population in enumerate(populations):
        ax = axes[i]
        
        # Filter data for this population
        pop_data = tr1_df[tr1_df['population'] == population]
        
        # Create boxplot
        sns.boxplot(
            data=pop_data, 
            x='response', 
            y='relative_frequency',
            ax=ax,
            palette=['lightcoral', 'lightblue']  # Red for non-responders, blue for responders
        )
        
        # Customize subplot
        ax.set_title(f'{population.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Response Status', fontsize=10)
        ax.set_ylabel('Relative Frequency (%)', fontsize=10)
        
        # Customize x-tick labels
        ax.set_xticklabels(['Non-responder', 'Responder'])
        
        # Add sample size annotations
        response_counts = pop_data['response'].value_counts()
        n_nonresp = response_counts.get('n', 0)
        n_resp = response_counts.get('y', 0)
        
        # Add sample sizes to x-axis labels
        ax.text(0, ax.get_ylim()[0], f'n={n_nonresp}', ha='center', va='top', fontsize=9, color='gray')
        ax.text(1, ax.get_ylim()[0], f'n={n_resp}', ha='center', va='top', fontsize=9, color='gray')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots if any
    for i in range(n_populations, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle('Cell Population Frequencies: TR1 Responders vs Non-Responders\n(Melanoma PBMC Samples)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    
    return fig


def calculate_tr1_statistics(tr1_df: pd.DataFrame, test_type: str = 'mannwhitney') -> pd.DataFrame:
    """
    Calculate statistical tests comparing responders vs non-responders for each population.
    
    Args:
        tr1_df: DataFrame from compare_tr1_response()
        test_type: Statistical test to use ('mannwhitney' or 'ttest')
        
    Returns:
        pd.DataFrame: Statistics summary with columns:
                     ['population', 'test_statistic', 'p_value', 'responder_mean', 
                      'nonresponder_mean', 'responder_n', 'nonresponder_n', 'test_type']
    """
    
    if tr1_df.empty:
        return pd.DataFrame(columns=['population', 'test_statistic', 'p_value', 'responder_mean', 
                                   'nonresponder_mean', 'responder_n', 'nonresponder_n', 'test_type'])
    
    populations = sorted(tr1_df['population'].unique())
    results = []
    
    for population in populations:
        # Filter data for this population
        pop_data = tr1_df[tr1_df['population'] == population]
        
        # Split by response status
        responder_freq = pop_data[pop_data['response'] == 'y']['relative_frequency']
        nonresponder_freq = pop_data[pop_data['response'] == 'n']['relative_frequency']
        
        # Calculate basic statistics
        resp_mean = responder_freq.mean() if len(responder_freq) > 0 else np.nan
        nonresp_mean = nonresponder_freq.mean() if len(nonresponder_freq) > 0 else np.nan
        resp_n = len(responder_freq)
        nonresp_n = len(nonresponder_freq)
        
        # Perform statistical test
        if resp_n == 0 or nonresp_n == 0:
            # Cannot perform test with empty groups
            test_stat = np.nan
            p_value = np.nan
        elif resp_n == 1 and nonresp_n == 1:
            # Cannot perform meaningful test with single observations
            test_stat = np.nan
            p_value = np.nan
        else:
            try:
                if test_type.lower() == 'mannwhitney':
                    # Mann-Whitney U test (non-parametric)
                    test_stat, p_value = stats.mannwhitneyu(
                        responder_freq, nonresponder_freq, 
                        alternative='two-sided'
                    )
                elif test_type.lower() == 'ttest':
                    # Independent t-test
                    test_stat, p_value = stats.ttest_ind(
                        responder_freq, nonresponder_freq,
                        equal_var=False  # Welch's t-test (unequal variances)
                    )
                else:
                    raise ValueError(f"Unknown test_type: {test_type}")
            except Exception as e:
                warnings.warn(f"Statistical test failed for {population}: {e}")
                test_stat = np.nan
                p_value = np.nan
        
        results.append({
            'population': population,
            'test_statistic': test_stat,
            'p_value': p_value,
            'responder_mean': resp_mean,
            'nonresponder_mean': nonresp_mean,
            'responder_n': resp_n,
            'nonresponder_n': nonresp_n,
            'test_type': test_type
        })
    
    stats_df = pd.DataFrame(results)
    
    # Round numeric columns for better readability
    numeric_cols = ['test_statistic', 'p_value', 'responder_mean', 'nonresponder_mean']
    for col in numeric_cols:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].round(4)
    
    return stats_df


def analyze_tr1_visualization(tr1_df: pd.DataFrame, test_type: str = 'mannwhitney', 
                             figsize: Tuple[int, int] = (15, 10), 
                             save_plots: bool = False, 
                             plot_filename: str = 'tr1_boxplots.png') -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Complete visualization and statistical analysis of TR1 response data.
    
    Args:
        tr1_df: DataFrame from compare_tr1_response()
        test_type: Statistical test to use ('mannwhitney' or 'ttest')
        figsize: Figure size for plots
        save_plots: Whether to save the plots to file
        plot_filename: Filename for saved plots
        
    Returns:
        Tuple[plt.Figure, pd.DataFrame]: (figure object, statistics DataFrame)
    """
    
    # Create boxplots
    fig = plot_tr1_response_comparison(tr1_df, figsize=figsize)
    
    # Calculate statistics
    stats_df = calculate_tr1_statistics(tr1_df, test_type=test_type)
    
    # Save plots if requested
    if save_plots:
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_filename}")
    
    # Print summary
    if not stats_df.empty:
        print(f"\n📈 Statistical Analysis Summary ({test_type.title()} Test):")
        print("=" * 80)
        
        # Show results with significance indicators
        display_df = stats_df.copy()
        
        # Add significance indicators
        def add_significance(p_val):
            if pd.isna(p_val):
                return "N/A"
            elif p_val < 0.001:
                return f"{p_val:.4f} ***"
            elif p_val < 0.01:
                return f"{p_val:.4f} **"
            elif p_val < 0.05:
                return f"{p_val:.4f} *"
            else:
                return f"{p_val:.4f}"
        
        display_df['p_value_sig'] = display_df['p_value'].apply(add_significance)
        
        # Select columns for display
        display_cols = ['population', 'responder_mean', 'nonresponder_mean', 
                       'responder_n', 'nonresponder_n', 'p_value_sig']
        
        print(display_df[display_cols].to_string(index=False))
        print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")
        
        # Count significant results
        significant = stats_df['p_value'] < 0.05
        n_significant = significant.sum() if not stats_df['p_value'].isna().all() else 0
        print(f"\n🎯 Significant differences found: {n_significant}/{len(stats_df)} populations")
    
    return fig, stats_df


# Convenience functions for backward compatibility
def plot_tr1_boxplots(tr1_df: pd.DataFrame, **kwargs) -> plt.Figure:
    """Alias for plot_tr1_response_comparison for backward compatibility."""
    return plot_tr1_response_comparison(tr1_df, **kwargs)


def tr1_statistical_tests(tr1_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Alias for calculate_tr1_statistics for backward compatibility."""
    return calculate_tr1_statistics(tr1_df, **kwargs) 