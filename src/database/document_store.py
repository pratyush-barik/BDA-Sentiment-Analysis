"""
Document Store Interface & Factory (Big Data Syllabus Unit 3).
Module: src.database.document_store

Defines the unified IDocumentStore abstract interface and the get_document_store
factory function, enabling seamless switching between live MongoDB (MongoDocumentStore)
and zero-dependency offline file storage (LocalJsonDocumentStore).
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from src.config import (
        LOCAL_DOCSTORE_PATH,
        MONGO_COLLECTION_NAME,
        MONGO_DB_NAME,
        MONGO_URI,
        is_mongo_available,
    )
except ImportError:
    from config import (
        LOCAL_DOCSTORE_PATH,
        MONGO_COLLECTION_NAME,
        MONGO_DB_NAME,
        MONGO_URI,
        is_mongo_available,
    )

logger = logging.getLogger(__name__)


class IDocumentStore(abc.ABC):
    """
    Abstract Base Class for Document Store Persistence Engines.
    Provides unified API across live MongoDB and local JSON store.
    """

    @abc.abstractmethod
    def connect(self) -> bool:
        """
        Establishes connection to the document store.
        Returns True on success, False on failure.
        """
        pass

    @abc.abstractmethod
    def insert_one(self, doc: Dict[str, Any]) -> str:
        """
        Inserts a single document into the collection.
        Returns the string identifier of the inserted document.
        """
        pass

    @abc.abstractmethod
    def insert_many(self, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Inserts multiple documents in batch.
        Returns a list of string identifiers of the inserted documents.
        """
        pass

    @abc.abstractmethod
    def find(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Queries documents matching the filter criteria.
        Supports projection (inclusion/exclusion) and limit.
        """
        pass

    @abc.abstractmethod
    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes a multi-stage aggregation pipeline sequence
        ($match, $group, $sort, $limit, $project).
        """
        pass

    @abc.abstractmethod
    def count_documents(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Returns the count of documents matching the filter criteria."""
        pass

    @abc.abstractmethod
    def create_index(self, keys: List[Tuple[str, int]]) -> str:
        """
        Creates a single or compound secondary index.
        keys: list of (field_name, direction) tuples (1 for ASC, -1 for DESC).
        Returns the generated index name.
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Flushes pending writes and closes active connections."""
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Returns True if connection is active."""
        pass

    def find_one(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Finds and returns the first matching document or None."""
        results = self.find(filter=filter, projection=projection, limit=1)
        return results[0] if results else None

    def update_one(self, filter: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """Updates a single matching document. Returns True if modified."""
        raise NotImplementedError

    def update_many(self, filter: Dict[str, Any], update: Dict[str, Any]) -> int:
        """Updates all matching documents. Returns count of modified documents."""
        raise NotImplementedError

    def delete_one(self, filter: Dict[str, Any]) -> bool:
        """Deletes a single matching document. Returns True if deleted."""
        raise NotImplementedError

    def delete_many(self, filter: Dict[str, Any]) -> int:
        """Deletes all matching documents. Returns count of deleted documents."""
        raise NotImplementedError

    def __enter__(self) -> IDocumentStore:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def get_document_store(
    prefer_mongo: bool = True,
    storage_path: Optional[Union[str, Path]] = None,
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    server_selection_timeout_ms: int = 500,
    **kwargs: Any,
) -> IDocumentStore:
    """
    Factory function producing an IDocumentStore instance.
    Auto-detects live MongoDB availability with fast timeout (500ms);
    seamlessly falls back to LocalJsonDocumentStore.
    """
    if prefer_mongo:
        try:
            from src.database.mongo_store import MongoDocumentStore

            mongo_store = MongoDocumentStore(
                mongo_uri=mongo_uri or MONGO_URI,
                db_name=db_name or MONGO_DB_NAME,
                collection_name=collection_name or MONGO_COLLECTION_NAME,
                server_selection_timeout_ms=server_selection_timeout_ms,
            )
            if mongo_store.connect():
                logger.info("Connected to live MongoDB at %s", mongo_uri or MONGO_URI)
                return mongo_store
            else:
                logger.warning(
                    "MongoDB unavailable at %s; falling back to LocalJsonDocumentStore.",
                    mongo_uri or MONGO_URI,
                )
        except Exception as ex:
            logger.warning(
                "Failed to initialize MongoDocumentStore (%s); falling back to LocalJsonDocumentStore.",
                ex,
            )

    # Offline zero-dependency local JSON fallback
    from src.database.local_store import LocalJsonDocumentStore

    target_path = storage_path or LOCAL_DOCSTORE_PATH
    return LocalJsonDocumentStore(storage_path=target_path, **kwargs)
