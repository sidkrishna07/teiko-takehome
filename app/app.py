#!/usr/bin/env python3
"""
TR1 Clinical Trial Analysis Dashboard
====================================

Interactive Streamlit application for Bob Loblaw to analyze TR1 treatment
effectiveness without touching code.

Features:
- CSV data loading and database management
- Interactive filtering and exploration
- Frequency analysis with downloads
- Responder vs non-responder comparisons
- Publication-ready visualizations
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import io
from datetime import datetime
import warnings

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

try:
    from loader import load_csv, get_sample_summary
    from analysis import (
        analyze_frequencies, 
        get_frequency_summary_with_metadata,
        analyze_tr1_response,
        compare_tr1_response
    )
    from viz import analyze_tr1_visualization, calculate_tr1_statistics
    from schema import init_db, get_session
except ImportError as e:
    st.error(f"Failed to import analysis modules: {e}")
    st.error("Please ensure you're running from the project root directory")
    st.stop()

# Configure page
st.set_page_config(
    page_title="TR1 Clinical Trial Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings for cleaner display
warnings.filterwarnings('ignore')

def init_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'last_reload' not in st.session_state:
        st.session_state.last_reload = None
    if 'database_url' not in st.session_state:
        st.session_state.database_url = "sqlite:///streamlit_cells.db"

def load_data_summary():
    """Get summary of loaded data."""
    try:
        summary = get_sample_summary(st.session_state.database_url)
        return summary
    except Exception as e:
        return {"error": str(e)}

def get_filter_options():
    """Get available filter options from the database."""
    try:
        # Get metadata to populate filter options
        engine, SessionLocal = init_db(st.session_state.database_url)
        session = get_session(SessionLocal)
        
        # Get frequency data with metadata for filter options
        freq_df = get_frequency_summary_with_metadata(session)
        session.close()
        
        if freq_df.empty:
            return {}
        
        # Helper function to safely sort unique values, filtering out None
        def safe_sort_unique(series):
            unique_vals = series.dropna().unique()
            return sorted([str(x) for x in unique_vals if x is not None])
        
        options = {
            'projects': safe_sort_unique(freq_df['project']),
            'conditions': safe_sort_unique(freq_df['condition']),
            'treatments': safe_sort_unique(freq_df['treatment']),
            'sample_types': safe_sort_unique(freq_df['sample_type']),
            'responses': safe_sort_unique(freq_df['response']),
            'timepoints': safe_sort_unique(freq_df['time_from_treatment_start']),
            'populations': safe_sort_unique(freq_df['population'])
        }
        return options
    except Exception as e:
        st.error(f"Error getting filter options: {e}")
        return {}

def main():
    """Main application function."""
    
    init_session_state()
    
    # Header
    st.title("🧬 TR1 Clinical Trial Analysis Dashboard")
    st.markdown("**Interactive analysis for Bob Loblaw @ Loblaw Bio**")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📊 Data Management")
        
        # CSV Upload/Reload Section
        st.subheader("📁 Upload CSV Data")
        
        csv_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="Upload any CSV file with clinical trial data"
        )
        
        if st.button("🔄 Load CSV to Database", type="primary"):
            if csv_file is not None:
                with st.spinner("Loading CSV data into database..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = f"temp_{csv_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(csv_file.getbuffer())
                        
                        stats = load_csv(temp_path, st.session_state.database_url)
                        os.remove(temp_path)  # Clean up
                        
                        st.session_state.data_loaded = True
                        st.session_state.last_reload = datetime.now()
                        st.success(f"✅ Successfully loaded {csv_file.name}")
                        st.info(f"📊 Processed: {stats['rows_processed']} rows")
                        st.info(f"👥 Added: {stats['subjects_added']} subjects, {stats['samples_added']} samples")
                        st.info(f"🔢 Created: {stats['counts_added']} cell count records")
                        
                        if stats.get('errors'):
                            st.warning(f"⚠️ {len(stats['errors'])} rows had issues - check data format")
                            
                    except Exception as e:
                        st.error(f"❌ Error loading CSV: {e}")
                        st.info("💡 Make sure your CSV has the required columns: project, subject, condition, age, sex, treatment, response, sample, sample_type, time_from_treatment_start, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte")
            else:
                st.error("📤 Please select a CSV file to upload")
        
        # Data Status
        if st.session_state.data_loaded:
            st.success("✅ Data loaded successfully")
            if st.session_state.last_reload:
                st.caption(f"⏰ Last loaded: {st.session_state.last_reload.strftime('%H:%M:%S')}")
            
            # Show data summary
            summary = load_data_summary()
            if 'error' not in summary:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Samples", summary.get('total_samples', 0))
                with col2:
                    st.metric("Total Subjects", summary.get('total_subjects', 0))
        else:
            st.info("📋 Upload a CSV file to begin analysis")
        
        st.divider()
        
        # Filters Section
        st.subheader("🔍 Data Filters")
        st.caption("Apply filters to focus your analysis (leave empty to see all data)")
        
        if st.session_state.data_loaded:
            filter_options = get_filter_options()
            
            if filter_options:
                # Treatment filter
                treatments = st.multiselect(
                    "Treatment", 
                    filter_options.get('treatments', []),
                    default=[],  # No default filters
                    help="Select specific treatments (leave empty for all)"
                )
                
                # Condition filter
                conditions = st.multiselect(
                    "Condition",
                    filter_options.get('conditions', []),
                    default=[],  # No default filters
                    help="Select disease conditions (leave empty for all)"
                )
                
                # Sample type filter
                sample_types = st.multiselect(
                    "Sample Type",
                    filter_options.get('sample_types', []),
                    default=[],  # No default filters
                    help="Select sample types (leave empty for all)"
                )
                
                # Response filter
                responses = st.multiselect(
                    "Response",
                    filter_options.get('responses', []),
                    default=[],  # No default filters
                    help="Select response categories (leave empty for all)"
                )
                
                # Timepoint filter
                timepoints = st.multiselect(
                    "Timepoints",
                    filter_options.get('timepoints', []),
                    default=[],  # No default filters
                    help="Select timepoints (leave empty for all)"
                )
                
                # Store filters in session state
                st.session_state.filters = {
                    'treatments': treatments,
                    'conditions': conditions,
                    'sample_types': sample_types,
                    'responses': responses,
                    'timepoints': timepoints
                }
                
                # Show filter summary
                active_filters = sum(1 for f in st.session_state.filters.values() if f)
                if active_filters > 0:
                    st.info(f"🔍 {active_filters} filter(s) active - click 'Clear All' to see full dataset")
                    if st.button("🗑️ Clear All Filters"):
                        st.session_state.filters = {
                            'treatments': [],
                            'conditions': [],
                            'sample_types': [],
                            'responses': [],
                            'timepoints': []
                        }
                        st.rerun()
                else:
                    st.success("📊 Showing all data (no filters applied)")
                    
            else:
                st.warning("No filter options available")
        else:
            st.info("Load data to enable filtering")
    
    # Main content area
    if not st.session_state.data_loaded:
        # Welcome screen
        st.info("👈 Upload your CSV file using the sidebar to begin analysis")
        
        st.markdown("""
        ### Welcome to the Clinical Trial Analysis Dashboard!
        
        This tool helps you analyze clinical trial data by:
        - **Loading any CSV** into a relational database
        - **Calculating frequency summaries** for cell populations
        - **Comparing treatment responses** with statistical analysis
        - **Exploring baseline demographics** and sample characteristics
        
        **To Get Started:**
        1. 📤 Upload your CSV file using the sidebar
        2. 🔄 Click "Load CSV to Database" to process the data  
        3. 📊 Use the tabs below to explore your analysis
        4. 🔍 Apply filters to focus on specific data subsets
        
        **Expected CSV Format:**
        Your file should contain columns like: `project`, `subject`, `condition`, `treatment`, `response`, `sample`, `sample_type`, `time_from_treatment_start`, and cell count columns (`b_cell`, `cd8_t_cell`, etc.)
        """)
        
        return
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Frequency Summary", 
        "🔬 Responder Comparison", 
        "📊 Baseline Analysis", 
        "📋 Data Overview",
        "🔍 Query & Manage"
    ])
    
    with tab1:
        frequency_analysis_tab()
    
    with tab2:
        responder_comparison_tab()
    
    with tab3:
        baseline_analysis_tab()
    
    with tab4:
        data_overview_tab()
        
    with tab5:
        query_and_manage_tab()

def frequency_analysis_tab():
    """Frequency analysis tab content."""
    st.header("📈 Cell Population Frequency Analysis")
    
    try:
        with st.spinner("Calculating frequencies..."):
            # Get frequency data with metadata
            freq_df = analyze_frequencies(st.session_state.database_url, include_metadata=True)
        
        if freq_df.empty:
            st.warning("No frequency data available")
            return
        
        # Apply filters if they exist
        if hasattr(st.session_state, 'filters'):
            filters = st.session_state.filters
            
            # Apply filters
            if filters['treatments']:
                freq_df = freq_df[freq_df['treatment'].isin(filters['treatments'])]
            if filters['conditions']:
                freq_df = freq_df[freq_df['condition'].isin(filters['conditions'])]
            if filters['sample_types']:
                freq_df = freq_df[freq_df['sample_type'].isin(filters['sample_types'])]
            if filters['responses']:
                freq_df = freq_df[freq_df['response'].isin(filters['responses'])]
            if filters['timepoints']:
                freq_df = freq_df[freq_df['time_from_treatment_start'].isin(filters['timepoints'])]
        
        if freq_df.empty:
            st.warning("No data matches the current filters")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", freq_df['sample_id'].nunique())
        with col2:
            st.metric("Subjects", freq_df['subject_id'].nunique())
        with col3:
            st.metric("Populations", freq_df['population'].nunique())
        with col4:
            st.metric("Records", len(freq_df))
        
        # Frequency table options
        st.subheader("📋 Frequency Table")
        
        # Table display options
        col1, col2 = st.columns(2)
        with col1:
            show_metadata = st.checkbox("Include metadata", value=True)
        with col2:
            max_rows = st.selectbox("Max rows to display", [50, 100, 500, 1000], index=1)
        
        # Prepare table for display
        if show_metadata:
            display_df = freq_df.copy()
        else:
            # Bob's original format
            display_df = freq_df[['sample_id', 'population', 'count', 'total_count', 'relative_frequency']].copy()
        
        # Display table
        st.dataframe(
            display_df.head(max_rows),
            use_container_width=True,
            height=400
        )
        
        if len(display_df) > max_rows:
            st.info(f"Showing first {max_rows} of {len(display_df)} records")
        
        # Download options
        st.subheader("💾 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_buffer = io.StringIO()
            display_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"frequency_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel download
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                display_df.to_excel(writer, sheet_name='Frequency_Analysis', index=False)
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"frequency_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Interactive visualizations
        st.subheader("📊 Interactive Visualizations")
        
        # Population distribution plot
        if st.checkbox("Show population distribution"):
            
            # Population means across all samples
            pop_means = freq_df.groupby('population')['relative_frequency'].agg(['mean', 'std']).reset_index()
            
            fig = px.bar(
                pop_means, 
                x='population', 
                y='mean',
                error_y='std',
                title="Average Cell Population Frequencies",
                labels={'mean': 'Mean Frequency (%)', 'population': 'Cell Population'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample-level heatmap
        if st.checkbox("Show sample heatmap"):
            # Create pivot table for heatmap
            pivot_df = freq_df.pivot_table(
                index='sample_id', 
                columns='population', 
                values='relative_frequency'
            )
            
            # Limit to first 50 samples for readability
            pivot_display = pivot_df.head(50)
            
            fig = px.imshow(
                pivot_display.values,
                x=pivot_display.columns,
                y=pivot_display.index,
                title="Cell Population Frequencies by Sample",
                labels={'color': 'Frequency (%)'},
                aspect='auto'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            if len(pivot_df) > 50:
                st.info(f"Showing first 50 of {len(pivot_df)} samples")
        
    except Exception as e:
        st.error(f"Error in frequency analysis: {e}")

def responder_comparison_tab():
    """Responder comparison tab content."""
    st.header("🔬 TR1 Responder vs Non-Responder Analysis")
    
    try:
        # Auto-filter for TR1 analysis or allow custom
        analysis_mode = st.radio(
            "Analysis Mode",
            ["TR1 Melanoma PBMC (Auto)", "Custom Filtering"],
            help="Choose preset TR1 analysis or custom filtering"
        )
        
        if analysis_mode == "TR1 Melanoma PBMC (Auto)":
            # Use the TR1 analysis function
            with st.spinner("Getting TR1 comparison data..."):
                tr1_df = analyze_tr1_response(st.session_state.database_url)
            
            if tr1_df.empty:
                st.warning("No TR1 melanoma PBMC samples found with response data")
                st.info("Try the Custom Filtering mode or check your data filters")
                return
            
            comparison_df = tr1_df
            
        else:
            # Custom filtering
            with st.spinner("Applying custom filters..."):
                # Get frequency data with metadata
                freq_df = analyze_frequencies(st.session_state.database_url, include_metadata=True)
                
                if freq_df.empty:
                    st.warning("No data available for custom filtering")
                    return
                
                # Apply current filters
                if hasattr(st.session_state, 'filters'):
                    filters = st.session_state.filters
                    
                    if filters['treatments']:
                        freq_df = freq_df[freq_df['treatment'].isin(filters['treatments'])]
                    if filters['conditions']:
                        freq_df = freq_df[freq_df['condition'].isin(filters['conditions'])]
                    if filters['sample_types']:
                        freq_df = freq_df[freq_df['sample_type'].isin(filters['sample_types'])]
                    if filters['responses']:
                        freq_df = freq_df[freq_df['response'].isin(filters['responses'])]
                    if filters['timepoints']:
                        freq_df = freq_df[freq_df['time_from_treatment_start'].isin(filters['timepoints'])]
                
                # Filter for samples with response data
                comparison_df = freq_df[freq_df['response'].isin(['y', 'n'])].copy()
                
                if comparison_df.empty:
                    st.warning("No samples with response data found")
                    return
        
        # Show summary
        responders = comparison_df[comparison_df['response'] == 'y']['sample_id'].nunique()
        non_responders = comparison_df[comparison_df['response'] == 'n']['sample_id'].nunique()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Responders", responders)
        with col2:
            st.metric("Non-responders", non_responders)
        with col3:
            st.metric("Total Samples", responders + non_responders)
        
        if responders == 0 or non_responders == 0:
            st.warning("Need both responders and non-responders for comparison")
            return
        
        # Statistical Analysis
        st.subheader("📊 Statistical Analysis")
        
        test_type = st.selectbox(
            "Statistical Test",
            ["mannwhitney", "ttest"],
            help="Choose between Mann-Whitney U (non-parametric) or t-test"
        )
        
        with st.spinner("Running statistical tests..."):
            # Calculate statistics
            from viz import calculate_tr1_statistics
            stats_df = calculate_tr1_statistics(comparison_df, test_type=test_type)
        
        if not stats_df.empty:
            # Format statistics table for display
            display_stats = stats_df.copy()
            display_stats['p_value_formatted'] = display_stats['p_value'].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
            display_stats['significance'] = display_stats['p_value'].apply(
                lambda x: "***" if pd.notna(x) and x < 0.001 
                         else "**" if pd.notna(x) and x < 0.01
                         else "*" if pd.notna(x) and x < 0.05
                         else "ns"
            )
            
            st.dataframe(
                display_stats[['population', 'responder_mean', 'nonresponder_mean', 
                             'responder_n', 'nonresponder_n', 'p_value_formatted', 'significance']],
                use_container_width=True
            )
            
            # Count significant results
            significant = (display_stats['p_value'] < 0.05).sum()
            st.info(f"Significant differences found: {significant}/{len(display_stats)} populations")
        
        # Visualizations
        st.subheader("📈 Comparison Visualizations")
        
        viz_type = st.selectbox(
            "Visualization Type",
            ["Boxplots", "Bar Chart", "Violin Plots"],
            help="Choose visualization style"
        )
        
        if viz_type == "Boxplots":
            # Create interactive boxplots
            populations = comparison_df['population'].unique()
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=len(populations),
                subplot_titles=[pop.replace('_', ' ').title() for pop in populations]
            )
            
            for i, pop in enumerate(populations):
                pop_data = comparison_df[comparison_df['population'] == pop]
                
                # Responders
                resp_data = pop_data[pop_data['response'] == 'y']['relative_frequency']
                fig.add_trace(
                    go.Box(y=resp_data, name='Responder', 
                          boxpoints='all', jitter=0.3, pointpos=-1.8),
                    row=1, col=i+1
                )
                
                # Non-responders
                nonresp_data = pop_data[pop_data['response'] == 'n']['relative_frequency']
                fig.add_trace(
                    go.Box(y=nonresp_data, name='Non-responder',
                          boxpoints='all', jitter=0.3, pointpos=1.8),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                height=500,
                title_text="Cell Population Frequencies: Responders vs Non-Responders",
                showlegend=True
            )
            fig.update_yaxes(title_text="Relative Frequency (%)")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart":
            # Mean comparison bar chart
            means_df = comparison_df.groupby(['population', 'response'])['relative_frequency'].agg(['mean', 'std']).reset_index()
            means_df.columns = ['population', 'response', 'mean_freq', 'std_freq']
            
            fig = px.bar(
                means_df,
                x='population',
                y='mean_freq',
                color='response',
                error_y='std_freq',
                title="Mean Cell Population Frequencies by Response",
                labels={'mean_freq': 'Mean Frequency (%)', 'population': 'Cell Population'},
                barmode='group'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Violin Plots":
            # Create violin plots for each population
            populations = comparison_df['population'].unique()
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=len(populations),
                subplot_titles=[pop.replace('_', ' ').title() for pop in populations]
            )
            
            colors = {'y': '#1f77b4', 'n': '#ff7f0e'}  # Blue for responders, orange for non-responders
            
            for i, pop in enumerate(populations):
                pop_data = comparison_df[comparison_df['population'] == pop]
                
                # Check if we have data for both groups
                resp_data = pop_data[pop_data['response'] == 'y']['relative_frequency']
                nonresp_data = pop_data[pop_data['response'] == 'n']['relative_frequency']
                
                if len(resp_data) > 0:
                    fig.add_trace(
                        go.Violin(
                            y=resp_data, 
                            name='Responder',
                            side='negative',
                            line_color=colors['y'],
                            fillcolor=colors['y'],
                            opacity=0.6,
                            showlegend=(i == 0)  # Only show legend for first subplot
                        ),
                        row=1, col=i+1
                    )
                
                if len(nonresp_data) > 0:
                    fig.add_trace(
                        go.Violin(
                            y=nonresp_data, 
                            name='Non-responder',
                            side='positive',
                            line_color=colors['n'],
                            fillcolor=colors['n'],
                            opacity=0.6,
                            showlegend=(i == 0)  # Only show legend for first subplot
                        ),
                        row=1, col=i+1
                    )
            
            fig.update_layout(
                height=500,
                title_text="Cell Population Frequency Distributions: Responders vs Non-responders",
                violinmode='overlay'
            )
            fig.update_yaxes(title_text="Relative Frequency (%)")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download comparison data
            csv_buffer = io.StringIO()
            comparison_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Comparison Data",
                data=csv_data,
                file_name=f"responder_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download statistics
            if not stats_df.empty:
                stats_csv = io.StringIO()
                stats_df.to_csv(stats_csv, index=False)
                stats_data = stats_csv.getvalue()
                
                st.download_button(
                    label="📥 Download Statistics",
                    data=stats_data,
                    file_name=f"statistical_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
    except Exception as e:
        st.error(f"Error in responder comparison: {e}")

def baseline_analysis_tab():
    """Baseline analysis tab content."""
    st.header("📊 Baseline Analysis")
    
    try:
        # Get frequency data with metadata
        with st.spinner("Loading baseline analysis data..."):
            freq_df = analyze_frequencies(st.session_state.database_url, include_metadata=True)
        
        if freq_df.empty:
            st.warning("No data available for baseline analysis")
            return
        
        # Overall summary
        st.subheader("📈 Dataset Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Subjects", freq_df['subject_id'].nunique())
        with col2:
            st.metric("Total Samples", freq_df['sample_id'].nunique())
        with col3:
            st.metric("Cell Populations", freq_df['population'].nunique())
        with col4:
            st.metric("Projects", freq_df['project'].nunique())
        
        # Breakdown by categories
        st.subheader("🔍 Breakdown by Categories")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Treatment breakdown
            treatment_counts = freq_df.groupby('treatment')['sample_id'].nunique().reset_index()
            treatment_counts.columns = ['Treatment', 'Sample Count']
            
            st.write("**By Treatment:**")
            st.dataframe(treatment_counts, use_container_width=True)
        
        with col2:
            # Condition breakdown
            condition_counts = freq_df.groupby('condition')['sample_id'].nunique().reset_index()
            condition_counts.columns = ['Condition', 'Sample Count']
            
            st.write("**By Condition:**")
            st.dataframe(condition_counts, use_container_width=True)
        
        # Sample type and response breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample type breakdown
            sample_type_counts = freq_df.groupby('sample_type')['sample_id'].nunique().reset_index()
            sample_type_counts.columns = ['Sample Type', 'Sample Count']
            
            st.write("**By Sample Type:**")
            st.dataframe(sample_type_counts, use_container_width=True)
        
        with col2:
            # Response breakdown
            response_counts = freq_df.groupby('response')['sample_id'].nunique().reset_index()
            response_counts.columns = ['Response', 'Sample Count']
            
            st.write("**By Response:**")
            st.dataframe(response_counts, use_container_width=True)
        
        # Gender breakdown
        st.subheader("👥 Demographics")
        gender_counts = freq_df.groupby('sex')['subject_id'].nunique().reset_index()
        gender_counts.columns = ['Gender', 'Subject Count']
        st.dataframe(gender_counts, use_container_width=True)
        
        # Cross-tabulations
        st.subheader("📋 Cross-Tabulations")
        
        # Treatment x Response
        if st.checkbox("Treatment × Response", key="baseline_treatment_response"):
            crosstab = pd.crosstab(
                freq_df['treatment'], 
                freq_df['response'], 
                values=freq_df['sample_id'],
                aggfunc='nunique',
                margins=True
            )
            st.write("**Sample counts by Treatment and Response:**")
            st.dataframe(crosstab)
        
        # Condition x Treatment
        if st.checkbox("Condition × Treatment", key="baseline_condition_treatment"):
            crosstab2 = pd.crosstab(
                freq_df['condition'],
                freq_df['treatment'],
                values=freq_df['sample_id'],
                aggfunc='nunique',
                margins=True
            )
            st.write("**Sample counts by Condition and Treatment:**")
            st.dataframe(crosstab2)
        
        # Timepoint analysis
        st.subheader("⏰ Timepoint Analysis")
        timepoint_summary = freq_df.groupby('time_from_treatment_start')['sample_id'].nunique().reset_index()
        timepoint_summary.columns = ['Timepoint', 'Sample Count']
        
        fig = px.bar(
            timepoint_summary,
            x='Timepoint',
            y='Sample Count',
            title="Samples by Timepoint"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # TR1 baseline analysis
        st.subheader("🎯 TR1 Baseline Analysis")
        st.markdown("""
        **Analysis:** Identify all melanoma PBMC samples at baseline (time_from_treatment_start = 0) 
        from patients who have treatment TR1. The analysis determines:
        - How many samples from each project
        - How many subjects were responders/non-responders  
        - How many subjects were males/females
        """)
        
        # Filter for TR1 baseline samples as specified
        tr1_baseline = freq_df[
            (freq_df['condition'] == 'melanoma') &
            (freq_df['treatment'] == 'tr1') &
            (freq_df['sample_type'] == 'PBMC') &
            (freq_df['time_from_treatment_start'] == 0)
        ]
        
        if not tr1_baseline.empty:
            # Get unique samples (remove population duplicates)
            unique_baseline = tr1_baseline[['sample_id', 'subject_id', 'project', 'response', 'sex']].drop_duplicates()
            
            st.write(f"**Found {len(unique_baseline)} TR1 baseline melanoma PBMC samples**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Samples by project
                project_baseline = unique_baseline['project'].value_counts().reset_index()
                project_baseline.columns = ['Project', 'Sample Count']
                st.write("**Samples by Project:**")
                st.dataframe(project_baseline, use_container_width=True)
            
            with col2:
                # Subjects by response
                response_baseline = unique_baseline['response'].value_counts().reset_index()
                response_baseline.columns = ['Response', 'Subject Count']
                st.write("**Subjects by Response:**")
                st.dataframe(response_baseline, use_container_width=True)
            
            with col3:
                # Subjects by gender
                gender_baseline = unique_baseline['sex'].value_counts().reset_index()
                gender_baseline.columns = ['Gender', 'Subject Count']
                st.write("**Subjects by Gender:**")
                st.dataframe(gender_baseline, use_container_width=True)
            
            # Show the actual baseline samples
            st.write("**TR1 Baseline Sample Details:**")
            st.dataframe(unique_baseline[['sample_id', 'subject_id', 'project', 'response', 'sex']], use_container_width=True)
            
        else:
            st.warning("No TR1 baseline melanoma PBMC samples found in the dataset")
        
        # Age distribution
        if 'age' in freq_df.columns:
            st.subheader("📊 Age Distribution")
            
            # Get unique subjects for age analysis
            subject_df = freq_df[['subject_id', 'age', 'sex', 'condition', 'treatment']].drop_duplicates()
            
            fig = px.histogram(
                subject_df,
                x='age',
                color='sex',
                title="Age Distribution by Gender",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in baseline analysis: {e}")

def data_overview_tab():
    """Data overview and baseline counts tab."""
    st.header("📊 Data Overview & Baseline Counts")
    
    try:
        # Get frequency data with metadata
        with st.spinner("Loading data overview..."):
            freq_df = analyze_frequencies(st.session_state.database_url, include_metadata=True)
        
        if freq_df.empty:
            st.warning("No data available for overview")
            return
        
        # Overall summary
        st.subheader("📈 Dataset Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Subjects", freq_df['subject_id'].nunique())
        with col2:
            st.metric("Total Samples", freq_df['sample_id'].nunique())
        with col3:
            st.metric("Cell Populations", freq_df['population'].nunique())
        with col4:
            st.metric("Projects", freq_df['project'].nunique())
        
        # Breakdown by categories
        st.subheader("🔍 Breakdown by Categories")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Treatment breakdown
            treatment_counts = freq_df.groupby('treatment')['sample_id'].nunique().reset_index()
            treatment_counts.columns = ['Treatment', 'Sample Count']
            
            st.write("**By Treatment:**")
            st.dataframe(treatment_counts, use_container_width=True)
        
        with col2:
            # Condition breakdown
            condition_counts = freq_df.groupby('condition')['sample_id'].nunique().reset_index()
            condition_counts.columns = ['Condition', 'Sample Count']
            
            st.write("**By Condition:**")
            st.dataframe(condition_counts, use_container_width=True)
        
        # Sample type and response breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample type breakdown
            sample_type_counts = freq_df.groupby('sample_type')['sample_id'].nunique().reset_index()
            sample_type_counts.columns = ['Sample Type', 'Sample Count']
            
            st.write("**By Sample Type:**")
            st.dataframe(sample_type_counts, use_container_width=True)
        
        with col2:
            # Response breakdown
            response_counts = freq_df.groupby('response')['sample_id'].nunique().reset_index()
            response_counts.columns = ['Response', 'Sample Count']
            
            st.write("**By Response:**")
            st.dataframe(response_counts, use_container_width=True)
        
        # Gender breakdown
        st.subheader("👥 Demographics")
        gender_counts = freq_df.groupby('sex')['subject_id'].nunique().reset_index()
        gender_counts.columns = ['Gender', 'Subject Count']
        st.dataframe(gender_counts, use_container_width=True)
        
        # Cross-tabulations
        st.subheader("📋 Cross-Tabulations")
        
        # Treatment x Response
        if st.checkbox("Treatment × Response", key="overview_treatment_response"):
            crosstab = pd.crosstab(
                freq_df['treatment'], 
                freq_df['response'], 
                values=freq_df['sample_id'],
                aggfunc='nunique',
                margins=True
            )
            st.write("**Sample counts by Treatment and Response:**")
            st.dataframe(crosstab)
        
        # Condition x Treatment
        if st.checkbox("Condition × Treatment", key="overview_condition_treatment"):
            crosstab2 = pd.crosstab(
                freq_df['condition'],
                freq_df['treatment'],
                values=freq_df['sample_id'],
                aggfunc='nunique',
                margins=True
            )
            st.write("**Sample counts by Condition and Treatment:**")
            st.dataframe(crosstab2)
        
        # Timepoint analysis
        st.subheader("⏰ Timepoint Analysis")
        timepoint_summary = freq_df.groupby('time_from_treatment_start')['sample_id'].nunique().reset_index()
        timepoint_summary.columns = ['Timepoint', 'Sample Count']
        
        fig = px.bar(
            timepoint_summary,
            x='Timepoint',
            y='Sample Count',
            title="Samples by Timepoint"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Age distribution
        if 'age' in freq_df.columns:
            st.subheader("📊 Age Distribution")
            
            # Get unique subjects for age analysis
            subject_df = freq_df[['subject_id', 'age', 'sex', 'condition', 'treatment']].drop_duplicates()
            
            fig = px.histogram(
                subject_df,
                x='age',
                color='sex',
                title="Age Distribution by Gender",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in data overview: {e}")

def query_and_manage_tab():
    """Interactive data exploration and sample management tab."""
    st.header("🔍 Interactive Data Explorer & Sample Management")
    
    # Two main sections
    query_section, manage_section = st.columns([2, 1])
    
    with query_section:
        st.subheader("🔍 Data Explorer")
        st.caption("Answer research questions with point-and-click queries")
        
        # Scientific question-based interface
        st.write("**Common Research Questions:**")
        
        question_type = st.selectbox(
            "What would you like to explore?",
            [
                "Show me all subjects with specific characteristics",
                "Compare treatments and their outcomes", 
                "Find samples from specific timepoints",
                "Analyze cell counts for specific populations",
                "Look at demographic patterns",
                "Custom exploration"
            ]
        )
        
        query_results = None
        
        if question_type == "Show me all subjects with specific characteristics":
            st.write("**Filter subjects by:**")
            
            col1, col2 = st.columns(2)
            with col1:
                filter_condition = st.multiselect("Disease Condition:", ["melanoma", "lung", "healthy"])
                filter_treatment = st.multiselect("Treatment:", ["tr1", "tr2", "none"])
                
            with col2:
                filter_response = st.multiselect("Response:", ["y", "n"])
                filter_sex = st.multiselect("Gender:", ["M", "F"])
            
            if st.button("🔍 Find Matching Subjects"):
                try:
                    import pandas as pd
                    from sqlalchemy import create_engine, text
                    
                    # Build query conditions
                    conditions = []
                    params = {}
                    
                    if filter_condition:
                        # Handle single vs multiple values for IN clause
                        if len(filter_condition) == 1:
                            conditions.append("condition = :condition_val")
                            params['condition_val'] = filter_condition[0]
                        else:
                            placeholders = ','.join([f':condition_{i}' for i in range(len(filter_condition))])
                            conditions.append(f"condition IN ({placeholders})")
                            for i, val in enumerate(filter_condition):
                                params[f'condition_{i}'] = val
                    if filter_treatment:
                        # Handle single vs multiple values for IN clause
                        if len(filter_treatment) == 1:
                            conditions.append("treatment = :treatment_val")
                            params['treatment_val'] = filter_treatment[0]
                        else:
                            placeholders = ','.join([f':treatment_{i}' for i in range(len(filter_treatment))])
                            conditions.append(f"treatment IN ({placeholders})")
                            for i, val in enumerate(filter_treatment):
                                params[f'treatment_{i}'] = val
                    if filter_response:
                        # Handle single vs multiple values for IN clause
                        if len(filter_response) == 1:
                            conditions.append("response = :response_val")
                            params['response_val'] = filter_response[0]
                        else:
                            placeholders = ','.join([f':response_{i}' for i in range(len(filter_response))])
                            conditions.append(f"response IN ({placeholders})")
                            for i, val in enumerate(filter_response):
                                params[f'response_{i}'] = val
                    if filter_sex:
                        # Handle single vs multiple values for IN clause
                        if len(filter_sex) == 1:
                            conditions.append("sex = :sex_val")
                            params['sex_val'] = filter_sex[0]
                        else:
                            placeholders = ','.join([f':sex_{i}' for i in range(len(filter_sex))])
                            conditions.append(f"sex IN ({placeholders})")
                            for i, val in enumerate(filter_sex):
                                params[f'sex_{i}'] = val
                    
                    where_clause = " AND ".join(conditions) if conditions else "1=1"
                    query = f"SELECT * FROM subjects WHERE {where_clause}"
                    
                    engine = create_engine(st.session_state.database_url)
                    query_results = pd.read_sql_query(text(query), engine, params=params)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif question_type == "Compare treatments and their outcomes":
            st.write("**Treatment Analysis:**")
            
            analysis_type = st.radio(
                "What comparison would you like?",
                ["Response rates by treatment", "Sample counts by treatment", "Demographics by treatment"]
            )
            
            if st.button("📊 Analyze Treatments"):
                try:
                    import pandas as pd
                    from sqlalchemy import create_engine, text
                    engine = create_engine(st.session_state.database_url)
                    
                    if analysis_type == "Response rates by treatment":
                        query = """
                        SELECT treatment, response, COUNT(*) as subject_count
                        FROM subjects 
                        WHERE treatment IS NOT NULL AND response IS NOT NULL
                        GROUP BY treatment, response
                        ORDER BY treatment, response
                        """
                        query_results = pd.read_sql_query(text(query), engine)
                        
                        # Create pivot table for better display
                        if not query_results.empty:
                            pivot_results = query_results.pivot(index='treatment', columns='response', values='subject_count').fillna(0)
                            pivot_results['Total'] = pivot_results.sum(axis=1)
                            pivot_results['Response_Rate_%'] = (pivot_results.get('y', 0) / pivot_results['Total'] * 100).round(1)
                            query_results = pivot_results.reset_index()
                    
                    elif analysis_type == "Sample counts by treatment":
                        query = """
                        SELECT s.treatment, COUNT(sa.sample_id) as sample_count, COUNT(DISTINCT s.subject_id) as subject_count
                        FROM subjects s
                        LEFT JOIN samples sa ON s.subject_id = sa.subject_id
                        GROUP BY s.treatment
                        ORDER BY s.treatment
                        """
                        query_results = pd.read_sql_query(text(query), engine)
                    
                    elif analysis_type == "Demographics by treatment":
                        query = """
                        SELECT treatment, sex, COUNT(*) as count, AVG(age) as avg_age
                        FROM subjects 
                        WHERE treatment IS NOT NULL
                        GROUP BY treatment, sex
                        ORDER BY treatment, sex
                        """
                        query_results = pd.read_sql_query(text(query), engine)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif question_type == "Find samples from specific timepoints":
            st.write("**Sample Timepoint Analysis:**")
            
            col1, col2 = st.columns(2)
            with col1:
                timepoint = st.selectbox("Timepoint:", [0, 7, 14, 21, 28, "All"])
                sample_type_filter = st.multiselect("Sample Type:", ["PBMC", "tumor"])
            
            with col2:
                condition_filter = st.multiselect("Condition:", ["melanoma", "lung", "healthy"])
                treatment_filter = st.multiselect("Treatment:", ["tr1", "tr2", "none"])
            
            if st.button("🔍 Find Samples"):
                try:
                    import pandas as pd
                    from sqlalchemy import create_engine, text
                    
                    conditions = []
                    params = {}
                    
                    if timepoint != "All":
                        conditions.append("sa.time_from_treatment_start = :timepoint")
                        params['timepoint'] = timepoint
                    if sample_type_filter:
                        # Handle single vs multiple values for IN clause
                        if len(sample_type_filter) == 1:
                            conditions.append("sa.sample_type = :sample_type_val")
                            params['sample_type_val'] = sample_type_filter[0]
                        else:
                            placeholders = ','.join([f':sample_type_{i}' for i in range(len(sample_type_filter))])
                            conditions.append(f"sa.sample_type IN ({placeholders})")
                            for i, val in enumerate(sample_type_filter):
                                params[f'sample_type_{i}'] = val
                    if condition_filter:
                        # Handle single vs multiple values for IN clause
                        if len(condition_filter) == 1:
                            conditions.append("s.condition = :condition_val")
                            params['condition_val'] = condition_filter[0]
                        else:
                            placeholders = ','.join([f':condition_{i}' for i in range(len(condition_filter))])
                            conditions.append(f"s.condition IN ({placeholders})")
                            for i, val in enumerate(condition_filter):
                                params[f'condition_{i}'] = val
                    if treatment_filter:
                        # Handle single vs multiple values for IN clause
                        if len(treatment_filter) == 1:
                            conditions.append("s.treatment = :treatment_val")
                            params['treatment_val'] = treatment_filter[0]
                        else:
                            placeholders = ','.join([f':treatment_{i}' for i in range(len(treatment_filter))])
                            conditions.append(f"s.treatment IN ({placeholders})")
                            for i, val in enumerate(treatment_filter):
                                params[f'treatment_{i}'] = val
                    
                    where_clause = " AND ".join(conditions) if conditions else "1=1"
                    query = f"""
                    SELECT sa.sample_id, sa.subject_id, s.condition, s.treatment, s.response, 
                           sa.sample_type, sa.time_from_treatment_start, s.age, s.sex
                    FROM samples sa
                    JOIN subjects s ON sa.subject_id = s.subject_id
                    WHERE {where_clause}
                    ORDER BY sa.sample_id
                    """
                    
                    engine = create_engine(st.session_state.database_url)
                    query_results = pd.read_sql_query(text(query), engine, params=params)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif question_type == "Analyze cell counts for specific populations":
            st.write("**Cell Population Analysis:**")
            
            col1, col2 = st.columns(2)
            with col1:
                population = st.selectbox("Cell Population:", ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte", "All"])
                sample_filter = st.text_input("Specific Sample ID (optional):", placeholder="e.g., s1")
            
            with col2:
                min_count = st.number_input("Minimum Cell Count:", min_value=0, value=0)
                max_count = st.number_input("Maximum Cell Count (0 = no limit):", min_value=0, value=0)
            
            if st.button("📊 Analyze Cell Counts"):
                try:
                    import pandas as pd
                    from sqlalchemy import create_engine, text
                    
                    conditions = []
                    params = {}
                    
                    if population != "All":
                        conditions.append("c.population_name = :population")
                        params['population'] = population
                    if sample_filter:
                        conditions.append("c.sample_id = :sample_id")
                        params['sample_id'] = sample_filter
                    if min_count > 0:
                        conditions.append("c.count >= :min_count")
                        params['min_count'] = min_count
                    if max_count > 0:
                        conditions.append("c.count <= :max_count")
                        params['max_count'] = max_count
                    
                    where_clause = " AND ".join(conditions) if conditions else "1=1"
                    query = f"""
                    SELECT c.sample_id, c.population_name, c.count, s.condition, s.treatment, s.response,
                           sa.sample_type, sa.time_from_treatment_start
                    FROM counts c
                    JOIN samples sa ON c.sample_id = sa.sample_id
                    JOIN subjects s ON sa.subject_id = s.subject_id
                    WHERE {where_clause}
                    ORDER BY c.sample_id, c.population_name
                    """
                    
                    engine = create_engine(st.session_state.database_url)
                    query_results = pd.read_sql_query(text(query), engine, params=params)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif question_type == "Look at demographic patterns":
            st.write("**Demographic Analysis:**")
            
            demo_analysis = st.selectbox(
                "What demographic pattern?",
                ["Age distribution by condition", "Gender breakdown by treatment", "Response rates by age group"]
            )
            
            if st.button("📊 Analyze Demographics"):
                try:
                    import pandas as pd
                    from sqlalchemy import create_engine, text
                    engine = create_engine(st.session_state.database_url)
                    
                    if demo_analysis == "Age distribution by condition":
                        query = """
                        SELECT condition, COUNT(*) as subject_count, 
                               AVG(age) as avg_age, MIN(age) as min_age, MAX(age) as max_age
                        FROM subjects 
                        WHERE age IS NOT NULL
                        GROUP BY condition
                        """
                        query_results = pd.read_sql_query(text(query), engine)
                    
                    elif demo_analysis == "Gender breakdown by treatment":
                        query = """
                        SELECT treatment, sex, COUNT(*) as count
                        FROM subjects
                        WHERE treatment IS NOT NULL AND sex IS NOT NULL
                        GROUP BY treatment, sex
                        ORDER BY treatment, sex
                        """
                        query_results = pd.read_sql_query(text(query), engine)
                    
                    elif demo_analysis == "Response rates by age group":
                        query = """
                        SELECT 
                            CASE 
                                WHEN age < 40 THEN 'Under 40'
                                WHEN age < 60 THEN '40-59'
                                ELSE '60+'
                            END as age_group,
                            response,
                            COUNT(*) as count
                        FROM subjects
                        WHERE age IS NOT NULL AND response IS NOT NULL
                        GROUP BY age_group, response
                        ORDER BY age_group, response
                        """
                        query_results = pd.read_sql_query(text(query), engine)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif question_type == "Custom exploration":
            st.write("**Advanced Options:**")
            
            with st.expander("🔧 For Advanced Users: Custom SQL Query"):
                st.caption("Only use this if you know SQL - otherwise use the options above!")
                
                # Table schema reference
                st.markdown("""
                **Database Tables:**
                - **subjects**: subject_id, project, condition, age, sex, treatment, response
                - **samples**: sample_id, subject_id, sample_type, time_from_treatment_start
                - **counts**: sample_id, population_name, count
                """)
                
                custom_query = st.text_area(
                    "SQL Query:",
                    value="SELECT * FROM subjects LIMIT 10;",
                    height=100
                )
                
                if st.button("⚡ Execute Custom Query"):
                    try:
                        import pandas as pd
                        from sqlalchemy import create_engine, text
                        engine = create_engine(st.session_state.database_url)
                        query_results = pd.read_sql_query(text(custom_query), engine)
                    except Exception as e:
                        st.error(f"SQL Error: {e}")
        
        # Display results
        if query_results is not None and not query_results.empty:
            st.success(f"✅ Found {len(query_results)} results!")
            
            st.subheader("📋 Results")
            st.dataframe(query_results, use_container_width=True)
            
            # Download option
            csv = query_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"exploration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Quick stats
            if len(query_results) > 0:
                st.info(f"📊 Quick Summary: {len(query_results)} rows, {len(query_results.columns)} columns")
        
        elif query_results is not None and query_results.empty:
            st.warning("🔍 No results found matching your criteria. Try adjusting the filters.")
    
    with manage_section:
        st.subheader("⚙️ Sample Management")
        st.caption("Add or remove individual samples")
        
        # Add Sample Section
        with st.expander("➕ Add New Sample", expanded=False):
            st.write("**Add a new sample to the database:**")
            
            with st.form("add_sample_form"):
                # Subject information
                st.write("**Subject Information:**")
                col1, col2 = st.columns(2)
                with col1:
                    new_subject_id = st.text_input("Subject ID*", placeholder="sbj14")
                    new_project = st.selectbox("Project", ["prj1", "prj2", "prj3"], index=0)
                    new_condition = st.selectbox("Condition", ["melanoma", "lung", "healthy"])
                    new_age = st.number_input("Age", min_value=18, max_value=100, value=50)
                
                with col2:
                    new_sex = st.selectbox("Sex", ["M", "F"])
                    new_treatment = st.selectbox("Treatment", ["tr1", "tr2", "none", None], index=3)
                    new_response = st.selectbox("Response", ["y", "n", None], index=2)
                
                # Sample information
                st.write("**Sample Information:**")
                col1, col2 = st.columns(2)
                with col1:
                    new_sample_id = st.text_input("Sample ID*", placeholder="s18")
                    new_sample_type = st.selectbox("Sample Type", ["PBMC", "tumor"])
                
                with col2:
                    new_timepoint = st.number_input("Time from Treatment Start", min_value=0, value=0)
                
                # Cell counts
                st.write("**Cell Counts:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_b_cell = st.number_input("B Cell", min_value=0, value=30000)
                    new_cd8 = st.number_input("CD8 T Cell", min_value=0, value=20000)
                
                with col2:
                    new_cd4 = st.number_input("CD4 T Cell", min_value=0, value=35000)
                    new_nk = st.number_input("NK Cell", min_value=0, value=5000)
                
                with col3:
                    new_monocyte = st.number_input("Monocyte", min_value=0, value=10000)
                
                submit_add = st.form_submit_button("➕ Add Sample", type="primary")
                
                if submit_add:
                    if new_subject_id and new_sample_id:
                        sample_data = {
                            'subject_id': new_subject_id,
                            'sample_id': new_sample_id,
                            'project': new_project,
                            'condition': new_condition,
                            'age': new_age,
                            'sex': new_sex,
                            'treatment': new_treatment if new_treatment != "none" else None,
                            'response': new_response,
                            'sample_type': new_sample_type,
                            'time_from_treatment_start': new_timepoint,
                            'b_cell': new_b_cell,
                            'cd8_t_cell': new_cd8,
                            'cd4_t_cell': new_cd4,
                            'nk_cell': new_nk,
                            'monocyte': new_monocyte
                        }
                        
                        try:
                            from code.loader import add_sample
                            success = add_sample(sample_data, st.session_state.database_url)
                            if success:
                                st.success(f"✅ Sample {new_sample_id} added successfully!")
                                st.info("💡 After adding samples, rerun your analyses from other tabs to see updated results!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to add sample. Check for duplicate IDs.")
                        except Exception as e:
                            st.error(f"❌ Error adding sample: {e}")
                    else:
                        st.error("⚠️ Please provide Subject ID and Sample ID")
        
        # Remove Sample Section
        with st.expander("🗑️ Remove Sample", expanded=False):
            st.write("**Remove a sample from the database:**")
            
            # Get list of available samples
            try:
                import pandas as pd
                from sqlalchemy import create_engine, text
                
                engine = create_engine(st.session_state.database_url)
                samples_df = pd.read_sql_query(
                    text("SELECT sample_id, subject_id, sample_type FROM samples ORDER BY sample_id"), 
                    engine
                )
                
                if not samples_df.empty:
                    sample_options = [f"{row['sample_id']} ({row['subject_id']}, {row['sample_type']})" 
                                    for _, row in samples_df.iterrows()]
                    
                    selected_sample = st.selectbox(
                        "Select Sample to Remove:",
                        options=sample_options,
                        help="Choose a sample to permanently remove from the database"
                    )
                    
                    if st.button("🗑️ Remove Selected Sample", type="secondary"):
                        if selected_sample:
                            sample_id = selected_sample.split(' ')[0]  # Extract sample_id
                            
                            # Confirmation
                            if st.button(f"⚠️ Confirm Removal of {sample_id}", type="secondary"):
                                try:
                                    from code.loader import remove_sample
                                    success = remove_sample(sample_id, st.session_state.database_url)
                                    if success:
                                        st.success(f"✅ Sample {sample_id} removed successfully!")
                                        st.info("💡 After removing samples, rerun your analyses from other tabs to see updated results!")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to remove sample {sample_id}")
                                except Exception as e:
                                    st.error(f"❌ Error removing sample: {e}")
                else:
                    st.info("No samples available to remove")
                    
            except Exception as e:
                st.error(f"Error loading samples: {e}")
        
        # Current Database Stats
        st.subheader("📊 Database Statistics")
        try:
            from code.loader import get_sample_summary
            summary = get_sample_summary(st.session_state.database_url)
            
            if 'error' not in summary:
                st.metric("Subjects", summary['total_subjects'])
                st.metric("Samples", summary['total_samples'])
                st.metric("Cell Count Records", summary['total_counts'])
                
                if summary.get('treatment_breakdown'):
                    st.write("**By Treatment:**")
                    for treatment, count in summary['treatment_breakdown'].items():
                        st.write(f"- {treatment}: {count} subjects")
            else:
                st.error(f"Error loading summary: {summary['error']}")
                
        except Exception as e:
            st.error(f"Error loading database summary: {e}")

if __name__ == "__main__":
    main()