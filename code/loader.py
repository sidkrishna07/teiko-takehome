"""
CSV loading and data management functions for clinical trial analysis.

This module handles loading cell-count.csv into the database and provides
CRUD operations for samples and subjects.
"""

import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
try:
    from .schema import Subject, Sample, Population, Count, init_db, get_session
except ImportError:
    from schema import Subject, Sample, Population, Count, init_db, get_session


def load_csv(csv_path: str, database_url: str = "sqlite:///cells.db") -> dict:
    """
    Load cell-count.csv into the database.
    
    Args:
        csv_path (str): Path to the CSV file
        database_url (str): SQLAlchemy database URL
        
    Returns:
        dict: Statistics about the loading process
    """
    
    # Initialize database
    engine, SessionLocal = init_db(database_url)
    session = get_session(SessionLocal)
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Validate required columns are present
        required_columns = [
            'project', 'subject', 'condition', 'age', 'sex', 'treatment', 
            'response', 'sample', 'sample_type', 'time_from_treatment_start',
            'b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Statistics tracking
        stats = {
            'rows_processed': len(df),
            'subjects_added': 0,
            'samples_added': 0,
            'counts_added': 0,
            'errors': []
        }
        
        # First, seed the populations table
        populations = ['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']
        for pop_name in populations:
            existing_pop = session.query(Population).filter_by(population_name=pop_name).first()
            if not existing_pop:
                population = Population(population_name=pop_name)
                session.add(population)
        
        session.commit()
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Map CSV columns to our schema
                # Note: CSV uses 'subject', 'sample', 'condition' but assignment mentions 'subject_id', 'sample_id', 'indication'
                subject_id = str(row['subject'])
                sample_id = str(row['sample'])
                
                # Handle missing/empty values
                age = None if pd.isna(row['age']) else int(row['age'])
                response = None if pd.isna(row['response']) or row['response'] == '' else str(row['response'])
                time_from_treatment = None if pd.isna(row['time_from_treatment_start']) else int(row['time_from_treatment_start'])
                treatment = None if pd.isna(row['treatment']) or row['treatment'] == '' or row['treatment'] == 'none' else str(row['treatment'])
                
                # Upsert subject
                subject = session.query(Subject).filter_by(subject_id=subject_id).first()
                if not subject:
                    subject = Subject(
                        subject_id=subject_id,
                        project=str(row['project']),
                        condition=str(row['condition']),  # This is what assignment calls 'indication'
                        age=age,
                        sex=str(row['sex']),  # This is what assignment calls 'gender'
                        treatment=treatment,
                        response=response
                    )
                    session.add(subject)
                    stats['subjects_added'] += 1
                
                # Upsert sample
                sample = session.query(Sample).filter_by(sample_id=sample_id).first()
                if not sample:
                    sample = Sample(
                        sample_id=sample_id,
                        subject_id=subject_id,
                        sample_type=str(row['sample_type']),
                        time_from_treatment_start=time_from_treatment
                    )
                    session.add(sample)
                    stats['samples_added'] += 1
                
                # Add cell counts for each population
                for pop_name in populations:
                    count_value = int(row[pop_name])
                    
                    # Check if count already exists
                    existing_count = session.query(Count).filter_by(
                        sample_id=sample_id,
                        population_name=pop_name
                    ).first()
                    
                    if not existing_count:
                        count = Count(
                            sample_id=sample_id,
                            population_name=pop_name,
                            count=count_value
                        )
                        session.add(count)
                        stats['counts_added'] += 1
                
                # Commit after each row to handle errors gracefully
                session.commit()
                
            except Exception as e:
                session.rollback()
                error_msg = f"Row {idx + 1}: {str(e)}"
                stats['errors'].append(error_msg)
                print(f"Error processing row {idx + 1}: {e}")
        
        return stats
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def add_sample(sample_data: dict, database_url: str = "sqlite:///cells.db") -> bool:
    """
    Add a new sample to the database.
    
    Args:
        sample_data (dict): Dictionary containing sample information with keys:
            - subject_id, sample_id, sample_type, time_from_treatment_start
            - Subject info: project, condition, age, sex, treatment, response
            - Cell counts: b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte
        database_url (str): Database URL
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    engine, SessionLocal = init_db(database_url)
    session = get_session(SessionLocal)
    
    try:
        # Add or update subject
        subject = session.query(Subject).filter_by(subject_id=sample_data['subject_id']).first()
        if not subject:
            subject = Subject(
                subject_id=sample_data['subject_id'],
                project=sample_data.get('project'),
                condition=sample_data.get('condition'),
                age=sample_data.get('age'),
                sex=sample_data.get('sex'),
                treatment=sample_data.get('treatment'),
                response=sample_data.get('response')
            )
            session.add(subject)
        
        # Add sample
        sample = Sample(
            sample_id=sample_data['sample_id'],
            subject_id=sample_data['subject_id'],
            sample_type=sample_data['sample_type'],
            time_from_treatment_start=sample_data.get('time_from_treatment_start')
        )
        session.add(sample)
        
        # Add cell counts
        populations = ['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']
        for pop_name in populations:
            if pop_name in sample_data:
                count = Count(
                    sample_id=sample_data['sample_id'],
                    population_name=pop_name,
                    count=sample_data[pop_name]
                )
                session.add(count)
        
        session.commit()
        return True
        
    except Exception as e:
        session.rollback()
        print(f"Error adding sample: {e}")
        return False
    finally:
        session.close()


def remove_sample(sample_id: str, database_url: str = "sqlite:///cells.db") -> bool:
    """
    Remove a sample and its associated counts from the database.
    
    Args:
        sample_id (str): Sample identifier to remove
        database_url (str): Database URL
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    engine, SessionLocal = init_db(database_url)
    session = get_session(SessionLocal)
    
    try:
        # Remove counts first (due to foreign key constraints)
        counts_deleted = session.query(Count).filter_by(sample_id=sample_id).delete()
        
        # Remove sample
        sample = session.query(Sample).filter_by(sample_id=sample_id).first()
        if sample:
            session.delete(sample)
            session.commit()
            print(f"Removed sample {sample_id} and {counts_deleted} associated counts")
            return True
        else:
            print(f"Sample {sample_id} not found")
            return False
            
    except Exception as e:
        session.rollback()
        print(f"Error removing sample: {e}")
        return False
    finally:
        session.close()


def get_sample_summary(database_url: str = "sqlite:///cells.db") -> dict:
    """
    Get summary statistics about the database contents.
    
    Args:
        database_url (str): Database URL
        
    Returns:
        dict: Summary statistics
    """
    
    engine, SessionLocal = init_db(database_url)
    session = get_session(SessionLocal)
    
    try:
        total_subjects = session.query(Subject).count()
        total_samples = session.query(Sample).count()
        total_counts = session.query(Count).count()
        
        # Breakdown by treatment
        treatment_counts = {}
        treatments = session.query(Subject.treatment).distinct().all()
        for (treatment,) in treatments:
            if treatment:
                count = session.query(Subject).filter_by(treatment=treatment).count()
                treatment_counts[treatment] = count
        
        return {
            'total_subjects': total_subjects,
            'total_samples': total_samples,
            'total_counts': total_counts,
            'treatment_breakdown': treatment_counts
        }
        
    except Exception as e:
        return {'error': str(e)}
    finally:
        session.close()


if __name__ == "__main__":
    # Quick test of loading functionality
    import os
    csv_path = "../cell-count.csv"
    if os.path.exists(csv_path):
        print("🧪 Testing CSV loading...")
        result = load_csv(csv_path)
        print("\n📊 Database summary:")
        summary = get_sample_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
    else:
        print(f"❌ CSV file not found: {csv_path}") 