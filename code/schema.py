"""
Database schema for clinical trial cell count analysis.

This module defines the SQLAlchemy ORM models for storing clinical trial data
including subjects, samples, cell populations, and cell counts.
"""

from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Subject(Base):
    """Subject/patient information table."""
    __tablename__ = 'subjects'
    
    subject_id = Column(String, primary_key=True)
    project = Column(String, nullable=False, index=True)
    condition = Column(String, nullable=False, index=True)  # melanoma, lung, healthy
    age = Column(Integer)
    sex = Column(String)  # M/F
    treatment = Column(String, index=True)  # tr1, tr2, none
    response = Column(String, index=True)  # y/n or None for healthy
    
    # Relationship to samples
    samples = relationship("Sample", back_populates="subject")
    
    def __repr__(self):
        return f"<Subject(subject_id='{self.subject_id}', condition='{self.condition}', treatment='{self.treatment}')>"


class Sample(Base):
    """Sample information table."""
    __tablename__ = 'samples'
    
    sample_id = Column(String, primary_key=True)
    subject_id = Column(String, ForeignKey('subjects.subject_id'), nullable=False)
    sample_type = Column(String, nullable=False, index=True)  # PBMC, tumor
    time_from_treatment_start = Column(Integer, index=True)  # timepoint
    
    # Relationships
    subject = relationship("Subject", back_populates="samples")
    counts = relationship("Count", back_populates="sample")
    
    def __repr__(self):
        return f"<Sample(sample_id='{self.sample_id}', type='{self.sample_type}', timepoint={self.time_from_treatment_start})>"


class Population(Base):
    """Cell population reference table."""
    __tablename__ = 'populations'
    
    population_name = Column(String, primary_key=True)
    
    # Relationship to counts
    counts = relationship("Count", back_populates="population")
    
    def __repr__(self):
        return f"<Population(name='{self.population_name}')>"


class Count(Base):
    """Cell count measurements table."""
    __tablename__ = 'counts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String, ForeignKey('samples.sample_id'), nullable=False)
    population_name = Column(String, ForeignKey('populations.population_name'), nullable=False)
    count = Column(Integer, nullable=False)
    
    # Relationships
    sample = relationship("Sample", back_populates="counts")
    population = relationship("Population", back_populates="counts")
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_sample_population', 'sample_id', 'population_name'),
    )
    
    def __repr__(self):
        return f"<Count(sample='{self.sample_id}', population='{self.population_name}', count={self.count})>"


def init_db(database_url="sqlite:///cells.db"):
    """
    Initialize the database with the defined schema.
    
    Args:
        database_url (str): SQLAlchemy database URL. Defaults to SQLite.
        
    Returns:
        tuple: (engine, SessionLocal) - SQLAlchemy engine and session factory
    """
    engine = create_engine(database_url, echo=True)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return engine, SessionLocal


def get_session(SessionLocal):
    """
    Get a database session.
    
    Args:
        SessionLocal: SQLAlchemy session factory
        
    Returns:
        Session: Database session
    """
    return SessionLocal() 