"""
Analysis functions for clinical trial cell count data.

This module provides analysis functions for calculating relative frequencies,
statistical comparisons, and other analytical operations for Bob's research.
"""

import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
try:
    from .schema import Subject, Sample, Population, Count, init_db, get_session
except ImportError:
    from schema import Subject, Sample, Population, Count, init_db, get_session


def get_frequency_summary(session) -> pd.DataFrame:
    """
    Calculate relative frequencies of cell populations for each sample.
    
    Generates Bob's frequency summary table as specified in the assignment:
    "For each sample, calculate the total number of cells by summing the counts 
    across all five populations. Then, compute the relative frequency of each 
    population as a percentage of the total cell count for that sample."
    
    Args:
        session: SQLAlchemy database session
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - sample_id: Sample identifier (from 'sample' column in CSV)
            - population: Cell population name  
            - count: Absolute cell count
            - total_count: Total cells in sample (sum across all populations)
            - relative_frequency: Percentage (count/total_count * 100)
    """
    
    # Query to get all counts with sample information
    query = session.query(
        Count.sample_id,
        Count.population_name.label('population'),
        Count.count,
        Sample.subject_id,
        Sample.sample_type,
        Sample.time_from_treatment_start
    ).join(
        Sample, Count.sample_id == Sample.sample_id
    )
    
    # Convert to DataFrame for easier manipulation
    df = pd.read_sql(query.statement, session.bind)
    
    if df.empty:
        # Return empty DataFrame with correct columns if no data
        return pd.DataFrame(columns=['sample_id', 'population', 'count', 'total_count', 'relative_frequency'])
    
    # Calculate total count per sample
    total_counts = df.groupby('sample_id')['count'].sum().reset_index()
    total_counts.columns = ['sample_id', 'total_count']
    
    # Merge total counts back to main DataFrame
    df = df.merge(total_counts, on='sample_id')
    
    # Calculate relative frequency as percentage
    df['relative_frequency'] = (df['count'] / df['total_count'] * 100).round(2)
    
    # Select and order the required columns exactly as specified in assignment
    result_df = df[['sample_id', 'population', 'count', 'total_count', 'relative_frequency']].copy()
    
    # Sort by sample_id and population for consistent output
    result_df = result_df.sort_values(['sample_id', 'population']).reset_index(drop=True)
    
    return result_df


def get_frequency_summary_with_metadata(session) -> pd.DataFrame:
    """
    Get frequency summary with additional sample and subject metadata.
    
    This extended version includes subject demographics and treatment information
    which is useful for filtering and analysis.
    
    Args:
        session: SQLAlchemy database session
        
    Returns:
        pd.DataFrame: Frequency summary with additional columns:
            - subject_id, condition, treatment, response, sex, age
            - sample_type, time_from_treatment_start
    """
    
    # Query with full join to get all metadata
    query = session.query(
        Count.sample_id,
        Count.population_name.label('population'),
        Count.count,
        Sample.subject_id,
        Sample.sample_type,
        Sample.time_from_treatment_start,
        Subject.project,
        Subject.condition,
        Subject.age,
        Subject.sex,
        Subject.treatment,
        Subject.response
    ).join(
        Sample, Count.sample_id == Sample.sample_id
    ).join(
        Subject, Sample.subject_id == Subject.subject_id
    )
    
    # Convert to DataFrame
    df = pd.read_sql(query.statement, session.bind)
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate total count per sample
    total_counts = df.groupby('sample_id')['count'].sum().reset_index()
    total_counts.columns = ['sample_id', 'total_count']
    
    # Merge total counts
    df = df.merge(total_counts, on='sample_id')
    
    # Calculate relative frequency
    df['relative_frequency'] = (df['count'] / df['total_count'] * 100).round(2)
    
    # Reorder columns for better readability
    column_order = [
        'sample_id', 'subject_id', 'project', 'condition', 'treatment', 'response',
        'age', 'sex', 'sample_type', 'time_from_treatment_start',
        'population', 'count', 'total_count', 'relative_frequency'
    ]
    
    df = df[column_order].sort_values(['sample_id', 'population']).reset_index(drop=True)
    
    return df


def validate_frequency_summary(frequency_df: pd.DataFrame) -> dict:
    """
    Validate that frequency percentages sum to approximately 100% per sample.
    
    Args:
        frequency_df: DataFrame from get_frequency_summary()
        
    Returns:
        dict: Validation results with statistics and any problematic samples
    """
    
    if frequency_df.empty:
        return {'valid': True, 'message': 'No data to validate', 'samples_checked': 0}
    
    # Group by sample and sum relative frequencies
    sample_totals = frequency_df.groupby('sample_id')['relative_frequency'].sum()
    
    # Check which samples don't sum to approximately 100%
    tolerance = 0.01  # Allow for rounding errors
    valid_samples = ((sample_totals >= 100 - tolerance) & (sample_totals <= 100 + tolerance))
    
    problematic_samples = sample_totals[~valid_samples]
    
    validation_result = {
        'valid': len(problematic_samples) == 0,
        'samples_checked': len(sample_totals),
        'valid_samples': valid_samples.sum(),
        'problematic_samples': len(problematic_samples),
        'sample_totals_range': {
            'min': sample_totals.min(),
            'max': sample_totals.max(),
            'mean': sample_totals.mean()
        }
    }
    
    if len(problematic_samples) > 0:
        validation_result['problematic_sample_details'] = problematic_samples.to_dict()
        validation_result['message'] = f"Found {len(problematic_samples)} samples with incorrect totals"
    else:
        validation_result['message'] = f"All {len(sample_totals)} samples have valid frequency totals"
    
    return validation_result


def compare_tr1_response(session) -> pd.DataFrame:
    """
    Compare melanoma tr1 PBMC samples by response status.
    
    Assignment requirement: "Compare the differences in cell population relative 
    frequencies of melanoma patients receiving tr1 who respond (responders) versus 
    those who do not (non-responders), with the overarching aim of predicting 
    response to treatment tr1. Response information can be found in column response, 
    with value y for responding and value n for non-responding. Please only include 
    PBMC (blood) samples."
    
    Args:
        session: SQLAlchemy database session
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            - sample_id: Sample identifier
            - population: Cell population name
            - relative_frequency: Percentage of this population in the sample
            - response: Response status ('y' for responder, 'n' for non-responder)
            - subject_id: Subject identifier (for grouping)
            - time_from_treatment_start: Timepoint information
    """
    
    # Get frequency summary with metadata
    freq_df = get_frequency_summary_with_metadata(session)
    
    if freq_df.empty:
        return pd.DataFrame(columns=['sample_id', 'population', 'relative_frequency', 'response'])
    
    # Filter for TR1 response comparison as specified in assignment:
    # 1. melanoma patients
    # 2. receiving tr1 treatment
    # 3. PBMC samples only
    # 4. with known response status (y or n)
    tr1_df = freq_df[
        (freq_df['condition'] == 'melanoma') &
        (freq_df['treatment'] == 'tr1') &
        (freq_df['sample_type'] == 'PBMC') &
        (freq_df['response'].isin(['y', 'n']))
    ].copy()
    
    if tr1_df.empty:
        return pd.DataFrame(columns=['sample_id', 'population', 'relative_frequency', 'response'])
    
    # Return required columns for comparison
    result_columns = ['sample_id', 'population', 'relative_frequency', 'response', 'subject_id', 'time_from_treatment_start']
    result_df = tr1_df[result_columns].copy()
    
    # Sort for consistent output
    result_df = result_df.sort_values(['response', 'sample_id', 'population']).reset_index(drop=True)
    
    return result_df


def get_tr1_baseline_summary(session) -> pd.DataFrame:
    """
    Get baseline TR1 analysis summary as requested by Bob.
    
    Assignment requirement: "Identify all melanoma PBMC samples at baseline 
    (time_from_treatment_start is 0) from patients who have treatment tr1. 
    Among these samples, determine:
    - How many samples from each project
    - How many subjects were responders/non-responders
    - How many subjects were males/females"
    
    Args:
        session: SQLAlchemy database session
        
    Returns:
        pd.DataFrame: Summary of TR1 baseline samples
    """
    
    # Get frequency summary with metadata
    freq_df = get_frequency_summary_with_metadata(session)
    
    if freq_df.empty:
        return pd.DataFrame()
    
    # Filter for baseline TR1 melanoma PBMC samples
    baseline_df = freq_df[
        (freq_df['condition'] == 'melanoma') &
        (freq_df['treatment'] == 'tr1') &
        (freq_df['sample_type'] == 'PBMC') &
        (freq_df['time_from_treatment_start'] == 0)
    ].copy()
    
    if baseline_df.empty:
        return pd.DataFrame()
    
    # Get unique samples (since we have one row per population)
    unique_samples = baseline_df[['sample_id', 'subject_id', 'project', 'response', 'sex']].drop_duplicates()
    
    # Create summary
    summary_data = []
    
    # Samples by project
    project_counts = unique_samples['project'].value_counts().to_dict()
    for project, count in project_counts.items():
        summary_data.append({
            'category': 'project',
            'value': project,
            'count': count
        })
    
    # Subjects by response
    response_counts = unique_samples['response'].value_counts().to_dict()
    for response, count in response_counts.items():
        summary_data.append({
            'category': 'response',
            'value': response,
            'count': count
        })
    
    # Subjects by sex
    sex_counts = unique_samples['sex'].value_counts().to_dict()
    for sex, count in sex_counts.items():
        summary_data.append({
            'category': 'sex',
            'value': sex,
            'count': count
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def analyze_tr1_response(database_url: str = "sqlite:///cells.db") -> pd.DataFrame:
    """
    Convenience function to analyze TR1 response with database connection handling.
    
    Args:
        database_url (str): Database URL
        
    Returns:
        pd.DataFrame: TR1 response comparison data
    """
    
    engine, SessionLocal = init_db(database_url)
    session = get_session(SessionLocal)
    
    try:
        return compare_tr1_response(session)
    finally:
        session.close()


def analyze_frequencies(database_url: str = "sqlite:///cells.db", include_metadata: bool = False) -> pd.DataFrame:
    """
    Convenience function to analyze frequencies with database connection handling.
    
    Args:
        database_url (str): Database URL
        include_metadata (bool): Whether to include sample/subject metadata
        
    Returns:
        pd.DataFrame: Frequency analysis results
    """
    
    engine, SessionLocal = init_db(database_url)
    session = get_session(SessionLocal)
    
    try:
        if include_metadata:
            return get_frequency_summary_with_metadata(session)
        else:
            return get_frequency_summary(session)
    finally:
        session.close()


def analyze_tr1_baseline(database_url: str = "sqlite:///cells.db") -> pd.DataFrame:
    """
    Convenience function to analyze TR1 baseline data with database connection handling.
    
    Args:
        database_url (str): Database URL
        
    Returns:
        pd.DataFrame: TR1 baseline analysis summary
    """
    
    engine, SessionLocal = init_db(database_url)
    session = get_session(SessionLocal)
    
    try:
        return get_tr1_baseline_summary(session)
    finally:
        session.close()


if __name__ == "__main__":
    # Quick test of the frequency analysis
    print("🧪 Testing Frequency Analysis")
    print("=" * 40)
    
    try:
        # Test with demo database if it exists
        result = analyze_frequencies("sqlite:///demo_cells.db", include_metadata=True)
        
        if not result.empty:
            print(f"\n📊 Found {len(result)} frequency records")
            print(f"   Samples: {result['sample_id'].nunique()}")
            print(f"   Populations: {result['population'].nunique()}")
            
            # Show sample of results
            print(f"\n📋 Sample Results:")
            print(result.head(10))
            
            # Show frequency ranges
            print(f"\n📈 Frequency Statistics:")
            freq_stats = result['relative_frequency'].describe()
            print(freq_stats)
            
        else:
            print("No data found - make sure to load CSV first with demo_loader.py")
            
    except Exception as e:
        print(f"❌ Error testing frequency analysis: {e}") 