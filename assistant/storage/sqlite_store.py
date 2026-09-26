"""SQLite engine + session factory.

Lazily creates the schema on first use. Single engine per process.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from assistant.config import get_config
from assistant.storage.schema import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    cfg = get_config()
    cfg.storage.ensure_dirs()
    url = f"sqlite:///{cfg.storage.sqlite_path.as_posix()}"
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Usage:
        with session_scope() as s:
            s.add(...); ...
    """
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
