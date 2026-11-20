"""
MongoDB Document Store Adapter (Big Data Syllabus Unit 3).
Module: src.database.mongo_store

Provides the MongoDocumentStore adapter using PyMongo with connection pooling,
graceful timeouts, automatic BSON ObjectId stringification, and deepcopy protection.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import pymongo
    from bson import ObjectId
except ImportError:
    pymongo = None
    ObjectId = None

from src.database.document_store import IDocumentStore

try:
    from src.config import MONGO_COLLECTION_NAME, MONGO_DB_NAME, MONGO_URI
except ImportError:
    from config import MONGO_COLLECTION_NAME, MONGO_DB_NAME, MONGO_URI

logger = logging.getLogger(__name__)


class MongoDocumentStore(IDocumentStore):
    """
    Production MongoDB driver adapter implementing IDocumentStore.
    Guards PyMongo import so offline environments without PyMongo do not crash.
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        server_selection_timeout_ms: int = 1000,
    ) -> None:
        self.mongo_uri = mongo_uri or MONGO_URI
        self.db_name = db_name or MONGO_DB_NAME
        self.collection_name = collection_name or MONGO_COLLECTION_NAME
        self.server_selection_timeout_ms = server_selection_timeout_ms
        self._client: Any = None
        self._db: Any = None
        self._collection: Any = None
        self._connected: bool = False

    def connect(self) -> bool:
        """Establishes connection to MongoDB with fast timeout check."""
        if pymongo is None:
            logger.debug("PyMongo is not installed in the environment.")
            self._connected = False
            return False

        try:
            self._client = pymongo.MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=self.server_selection_timeout_ms,
            )
            # Force server ping check
            self._client.admin.command("ping")
            self._db = self._client[self.db_name]
            self._collection = self._db[self.collection_name]
            self._connected = True
            return True
        except Exception as ex:
            logger.debug("MongoDB connection failed (%s): %s", self.mongo_uri, ex)
            self._connected = False
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _ensure_connected(self) -> None:
        if not self._connected:
            if not self.connect():
                raise ConnectionError(
                    f"Unable to connect to MongoDB server at {self.mongo_uri}"
                )

    def _normalize_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Converts BSON ObjectId or non-serializable fields to standard types."""
        if not isinstance(doc, dict):
            return doc
        res = dict(doc)
        if "_id" in res and not isinstance(res["_id"], str):
            res["_id"] = str(res["_id"])
        return res

    def insert_one(self, doc: Dict[str, Any]) -> str:
        self._ensure_connected()
        doc_copy = copy.deepcopy(doc)
        result = self._collection.insert_one(doc_copy)
        return str(result.inserted_id)

    def insert_many(self, docs: List[Dict[str, Any]]) -> List[str]:
        if not docs:
            return []
        self._ensure_connected()
        docs_copy = [copy.deepcopy(d) for d in docs]
        result = self._collection.insert_many(docs_copy)
        return [str(inserted_id) for inserted_id in result.inserted_ids]

    def find(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        self._ensure_connected()
        query = filter if filter is not None else {}
        cursor = self._collection.find(query, projection)
        if limit > 0:
            cursor = cursor.limit(limit)
        return [self._normalize_doc(d) for d in cursor]

    def find_one(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        self._ensure_connected()
        query = filter if filter is not None else {}
        res = self._collection.find_one(query, projection)
        return self._normalize_doc(res) if res is not None else None

    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._ensure_connected()
        cursor = self._collection.aggregate(pipeline)
        return [self._normalize_doc(d) for d in cursor]

    def count_documents(self, filter: Optional[Dict[str, Any]] = None) -> int:
        self._ensure_connected()
        query = filter if filter is not None else {}
        return int(self._collection.count_documents(query))

    def create_index(self, keys: List[Tuple[str, int]]) -> str:
        self._ensure_connected()
        return str(self._collection.create_index(keys))

    def update_one(self, filter: Dict[str, Any], update: Dict[str, Any]) -> bool:
        self._ensure_connected()
        res = self._collection.update_one(filter, update)
        return res.modified_count > 0

    def update_many(self, filter: Dict[str, Any], update: Dict[str, Any]) -> int:
        self._ensure_connected()
        res = self._collection.update_many(filter, update)
        return int(res.modified_count)

    def delete_one(self, filter: Dict[str, Any]) -> bool:
        self._ensure_connected()
        res = self._collection.delete_one(filter)
        return res.deleted_count > 0

    def delete_many(self, filter: Dict[str, Any]) -> int:
        self._ensure_connected()
        res = self._collection.delete_many(filter)
        return int(res.deleted_count)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._connected = False
