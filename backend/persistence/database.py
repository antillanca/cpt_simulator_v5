from sqlalchemy import create_engine, Column, String, Float, Integer, LargeBinary, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

from backend.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Rule(Base):
    __tablename__ = "rules"

    id = Column(String, primary_key=True, index=True)
    text = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=True)
    cluster_id = Column(Integer, nullable=True)
    confidence = Column(Float, default=0.0)
    collapsed = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Snapshot(Base):
    __tablename__ = "snapshots"

    id = Column(String, primary_key=True, index=True)
    cluster_id = Column(Integer, nullable=True)
    state_json = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class SyllabusItem(Base):
    __tablename__ = "syllabus"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    objective = Column(String, nullable=False)
    target_state_json = Column(String, nullable=False) # Expected output state
    order = Column(Integer, unique=True)
    is_completed = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)

class LearningLog(Base):
    __tablename__ = "learning_logs"

    id = Column(Integer, primary_key=True, index=True)
    syllabus_item_id = Column(Integer, ForeignKey("syllabus.id"))
    rule_id = Column(String, ForeignKey("rules.id"))
    success = Column(Boolean)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
