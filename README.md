# TR1 Clinical Trial Analysis Pipeline

A Python-based system for analyzing drug candidate effects on immune cell populations in clinical trials.

## Running the Code

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup and Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app/app.py
```

### Reproducing Outputs

1. **Upload Data**: Go to "Data Overview" tab and upload `cell-count.csv`
2. **Frequency Analysis**: Navigate to "Frequency Summary" tab for cell population frequencies
3. **TR1 Response Analysis**: Use "Responder Comparison" tab for statistical comparison of responders vs non-responders
4. **Baseline Analysis**: Access "Baseline Analysis" tab for TR1 melanoma PBMC baseline sample exploration

## Database Schema

### Design
The system uses a normalized SQLite database with four tables:

```sql
-- Core subject information
CREATE TABLE subjects (
    subject_id VARCHAR(50) PRIMARY KEY,
    project VARCHAR(50),
    condition VARCHAR(50),
    age INTEGER,
    sex CHAR(1),
    treatment VARCHAR(50),
    response CHAR(1)
);

-- Sample metadata
CREATE TABLE samples (
    sample_id VARCHAR(50) PRIMARY KEY,
    subject_id VARCHAR(50),
    sample_type VARCHAR(50),
    time_from_treatment_start INTEGER,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Reference table for immune cell populations
CREATE TABLE populations (
    population_name VARCHAR(50) PRIMARY KEY,
    description TEXT
);

-- Cell count measurements
CREATE TABLE counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id VARCHAR(50),
    population_name VARCHAR(50),
    count INTEGER,
    FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
    FOREIGN KEY (population_name) REFERENCES populations(population_name)
);
```

### Rationale
- **Normalization**: Prevents data duplication and ensures consistency
- **Foreign keys**: Maintain referential integrity across tables
- **Separation of concerns**: Subject demographics, sample metadata, and measurements stored separately

### Scalability

**For hundreds of projects:**
- Add `projects` table with metadata (PI, funding, dates)
- Implement project-level access controls
- Enable cross-project analytics through standardized schema

**For thousands of samples:**
- Indexes on `subject_id`, `sample_type`, `time_from_treatment_start` for fast queries
- Database partitioning by project or date ranges
- Connection pooling for concurrent access

**For various analytics:**
- **Longitudinal studies**: Time-based queries optimized by time index
- **New cell populations**: Easily added to `populations` table
- **Biomarker integration**: Additional measurement tables can reference `samples`
- **Machine learning**: Flat views created through joins for feature engineering

## Code Structure

### Architecture
```
├── app/
│   └── app.py              # Streamlit frontend application
├── code/
│   ├── schema.py           # Database schema and connection management
│   ├── loader.py           # Data loading and validation
│   ├── analysis.py         # Statistical analysis functions
│   └── viz.py              # Visualization utilities
├── cell-count.csv          # Sample data
└── requirements.txt        # Dependencies
```

### Design Philosophy

**Separation of concerns:**
- `schema.py`: Database operations and connection management
- `loader.py`: Data ingestion, validation, and transformation
- `analysis.py`: Statistical computations and scientific algorithms  
- `viz.py`: Plotting and visualization functions
- `app.py`: User interface and presentation logic

**Key decisions:**
- **SQLAlchemy ORM**: Database abstraction and SQL injection protection
- **Streamlit**: Rapid prototyping with scientist-friendly interface
- **Modular functions**: Self-contained analysis types for easy testing and reuse
- **Configuration-driven**: Database URLs passed as parameters for different environments

This design enables easy testing, clear separation of responsibilities, and straightforward extension for new analysis types. 