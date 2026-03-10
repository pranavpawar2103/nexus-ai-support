"""Database models and connection management for NexusAI Support.

Defines SQLAlchemy ORM models for customers, support tickets, and billing records.
Provides connection factory and session management utilities.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""
    pass


class Customer(Base):
    """Customer account model.

    Attributes:
        id: Primary key.
        name: Full name of the customer.
        email: Unique email address.
        phone: Contact phone number.
        plan_type: Subscription plan (Basic/Pro/Enterprise).
        account_status: Account state (Active/Suspended/Cancelled).
        created_at: Account creation timestamp.
        location: Customer geographic location.
        company_name: Associated company (if applicable).
        tickets: Related support tickets.
        billing_records: Related billing records.
    """

    __tablename__ = "customers"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(String(200), nullable=False, index=True)
    email: str = Column(String(200), nullable=False, unique=True, index=True)
    phone: str = Column(String(50), nullable=True)
    plan_type: str = Column(String(50), nullable=False, default="Basic")
    account_status: str = Column(String(50), nullable=False, default="Active")
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    location: str = Column(String(200), nullable=True)
    company_name: str = Column(String(200), nullable=True)

    tickets = relationship("SupportTicket", back_populates="customer", lazy="select")
    billing_records = relationship(
        "BillingRecord", back_populates="customer", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Customer(id={self.id}, name='{self.name}', plan='{self.plan_type}')>"


class SupportTicket(Base):
    """Support ticket model.

    Attributes:
        id: Primary key.
        customer_id: Foreign key to customers table.
        title: Short summary of the issue.
        description: Detailed issue description.
        status: Current ticket status (Open/In Progress/Resolved/Closed).
        priority: Ticket urgency (Low/Medium/High/Critical).
        category: Issue category (Billing/Technical/General/Refund/Account).
        created_at: Ticket creation timestamp.
        resolved_at: Resolution timestamp (null if unresolved).
        resolution_notes: Agent notes on resolution.
        agent_name: Name of the handling agent.
        customer: Related customer object.
    """

    __tablename__ = "support_tickets"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    customer_id: int = Column(Integer, ForeignKey("customers.id"), nullable=False, index=True)
    title: str = Column(String(500), nullable=False)
    description: str = Column(Text, nullable=True)
    status: str = Column(String(50), nullable=False, default="Open", index=True)
    priority: str = Column(String(50), nullable=False, default="Medium", index=True)
    category: str = Column(String(100), nullable=False, default="General", index=True)
    created_at: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    resolved_at: Optional[datetime] = Column(DateTime, nullable=True)
    resolution_notes: Optional[str] = Column(Text, nullable=True)
    agent_name: Optional[str] = Column(String(200), nullable=True)

    customer = relationship("Customer", back_populates="tickets")

    def __repr__(self) -> str:
        return f"<SupportTicket(id={self.id}, status='{self.status}', priority='{self.priority}')>"


class BillingRecord(Base):
    """Billing record model.

    Attributes:
        id: Primary key.
        customer_id: Foreign key to customers table.
        amount: Invoice amount in USD.
        status: Payment status (Paid/Pending/Overdue).
        invoice_date: Date invoice was generated.
        due_date: Payment due date.
        description: Description of charges.
        customer: Related customer object.
    """

    __tablename__ = "billing_records"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    customer_id: int = Column(Integer, ForeignKey("customers.id"), nullable=False, index=True)
    amount: float = Column(Float, nullable=False)
    status: str = Column(String(50), nullable=False, default="Pending", index=True)
    invoice_date: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)
    due_date: datetime = Column(DateTime, nullable=False)
    description: str = Column(String(500), nullable=True)

    customer = relationship("Customer", back_populates="billing_records")

    def __repr__(self) -> str:
        return f"<BillingRecord(id={self.id}, amount={self.amount}, status='{self.status}')>"


def get_engine(db_path: Optional[str] = None) -> Engine:
    """Create and return a SQLAlchemy engine.

    Args:
        db_path: Path to SQLite database file. Uses config default if None.

    Returns:
        Engine: Configured SQLAlchemy engine with WAL mode enabled.
    """
    from core.config import get_settings

    if db_path is None:
        settings = get_settings()
        db_path = str(settings.get_db_abs_path())

    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    # Enable WAL mode for better concurrent access
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    logger.info("Database engine created: %s", db_path)
    return engine


_SessionLocal: Optional[sessionmaker] = None


def get_session_factory(db_path: Optional[str] = None) -> sessionmaker:
    """Get or create a session factory.

    Args:
        db_path: Optional database path override.

    Returns:
        sessionmaker: Configured session factory.
    """
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(db_path)
        _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return _SessionLocal


@contextmanager
def get_session(db_path: Optional[str] = None) -> Generator[Session, None, None]:
    """Context manager providing a database session with automatic cleanup.

    Args:
        db_path: Optional database path override.

    Yields:
        Session: Active database session.

    Example:
        with get_session() as session:
            customers = session.query(Customer).all()
    """
    factory = get_session_factory(db_path)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_tables(db_path: Optional[str] = None) -> None:
    """Create all database tables if they don't exist.

    Args:
        db_path: Optional database path override.
    """
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    logger.info("Database tables created/verified")


def drop_tables(db_path: Optional[str] = None) -> None:
    """Drop all database tables.

    Args:
        db_path: Optional database path override.

    Warning:
        This destroys all data. Use only for reseeding.
    """
    engine = get_engine(db_path)
    Base.metadata.drop_all(engine)
    logger.warning("All database tables dropped")