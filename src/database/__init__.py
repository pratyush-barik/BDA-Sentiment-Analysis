"""
Document Store Module (Big Data Syllabus Unit 3).
Provides unified IDocumentStore interface, MongoDocumentStore adapter,
LocalJsonDocumentStore offline engine, and get_document_store factory.
"""

from src.database.document_store import (
    IDocumentStore,
    get_document_store,
)
from src.database.local_store import LocalJsonDocumentStore
from src.database.mongo_store import MongoDocumentStore

__all__ = [
    "IDocumentStore",
    "MongoDocumentStore",
    "LocalJsonDocumentStore",
    "get_document_store",
]
