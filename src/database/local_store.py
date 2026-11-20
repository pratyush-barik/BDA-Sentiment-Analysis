"""
Zero-Dependency Local JSON Document Store (Big Data Syllabus Unit 3).
Module: src.database.local_store

Implements a production-grade, offline, JSON-backed document store
providing high-fidelity MongoDB query filtering, aggregation pipelines,
secondary indexing simulation, thread safety, and atomic disk persistence.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.database.document_store import IDocumentStore

logger = logging.getLogger(__name__)


class LocalJsonDocumentStore(IDocumentStore):
    """
    Offline JSON Document Store providing 100% pure-Python emulation of MongoDB.
    Features:
    - Atomic JSON file persistence (temp file + os.replace).
    - Thread-safe operations via threading.RLock.
    - MongoDB query filter evaluation ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $regex, dot notation).
    - MongoDB aggregation pipeline ($match, $group, $sort, $limit, $project) with accumulators ($sum, $avg, $min, $max, $push, $first, $last, $cond).
    - Secondary indexing simulation on single and compound keys accelerating candidate lookups.
    """

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        in_memory: bool = False,
    ) -> None:
        self.storage_path = (
            Path(storage_path)
            if storage_path and str(storage_path) != ":memory:"
            else None
        )
        self.in_memory = (
            in_memory or (str(storage_path) == ":memory:") or (self.storage_path is None)
        )
        self._lock = threading.RLock()
        self._documents: List[Dict[str, Any]] = []
        self._id_map: Dict[str, Dict[str, Any]] = {}

        # Secondary indexes: spec (tuple of (field, dir)) -> index_map (key -> list of doc _ids)
        self._indexes: Dict[Tuple[Tuple[str, int], ...], Dict[Any, List[str]]] = {}
        self._index_names: Dict[str, Tuple[Tuple[str, int], ...]] = {}

        self.last_query_used_index: Optional[str] = None
        self._connected: bool = False

    def connect(self) -> bool:
        """Connects and loads documents and indexes from disk if persistent."""
        with self._lock:
            if not self.in_memory and self.storage_path is not None:
                self._load_from_disk()
            self._connected = True
            return True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        """Flushes state to disk and marks store as disconnected."""
        with self._lock:
            if not self.in_memory and self.storage_path is not None:
                self._save_to_disk()
            self._connected = False

    # ---------------------------------------------------------------------------
    # Atomic File Persistence
    # ---------------------------------------------------------------------------

    def _save_to_disk(self) -> None:
        if self.in_memory or self.storage_path is None:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.storage_path.with_suffix(
            f".tmp_{uuid.uuid4().hex[:8]}"
        )

        payload = {
            "documents": self._documents,
            "indexes": [
                {"name": name, "keys": list(spec)}
                for name, spec in self._index_names.items()
            ],
        }

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_file, self.storage_path)

    def _load_from_disk(self) -> None:
        if self.in_memory or self.storage_path is None or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                raw_docs = data
                indexes_data: List[Dict[str, Any]] = []
            elif isinstance(data, dict):
                raw_docs = data.get("documents", [])
                indexes_data = data.get("indexes", [])
            else:
                raw_docs = []
                indexes_data = []

            self._documents = raw_docs
            self._id_map = {
                str(d["_id"]): d for d in self._documents if "_id" in d
            }

            # Rebuild defined indexes
            for idx_info in indexes_data:
                keys = [tuple(k) for k in idx_info.get("keys", [])]
                if keys:
                    self.create_index(keys)  # type: ignore[arg-type]

            # Re-index all existing documents
            self._rebuild_all_indexes()
        except Exception as ex:
            logger.warning("Error loading document store from %s: %s", self.storage_path, ex)

    # ---------------------------------------------------------------------------
    # Index Management Subsystem
    # ---------------------------------------------------------------------------

    def _rebuild_all_indexes(self) -> None:
        for spec in self._indexes:
            self._indexes[spec] = defaultdict(list)
            for doc in self._documents:
                doc_id = str(doc.get("_id"))
                key = self._extract_index_key(doc, spec)
                self._indexes[spec][key].append(doc_id)

    def _extract_index_key(
        self, doc: Dict[str, Any], spec: Tuple[Tuple[str, int], ...]
    ) -> Any:
        if len(spec) == 1:
            field = spec[0][0]
            _, val = self._get_nested_field(doc, field)
            return val
        extracted = []
        for field, _ in spec:
            _, val = self._get_nested_field(doc, field)
            extracted.append(val)
        return tuple(extracted)

    def create_index(self, keys: List[Tuple[str, int]]) -> str:
        """
        Creates a secondary index on single or compound keys.
        keys: [("product_category", 1), ("rating", -1)]
        """
        with self._lock:
            spec = tuple(keys)
            index_name = "_".join(f"{field}_{direction}" for field, direction in keys)

            if spec not in self._indexes:
                self._indexes[spec] = defaultdict(list)
                self._index_names[index_name] = spec

                # Populate index with existing documents
                for doc in self._documents:
                    doc_id = str(doc.get("_id"))
                    key = self._extract_index_key(doc, spec)
                    self._indexes[spec][key].append(doc_id)

                if not self.in_memory and self.storage_path is not None:
                    self._save_to_disk()

            return index_name

    def _index_insert(self, doc: Dict[str, Any]) -> None:
        doc_id = str(doc.get("_id"))
        for spec, index_map in self._indexes.items():
            key = self._extract_index_key(doc, spec)
            index_map[key].append(doc_id)

    def _index_remove(self, doc: Dict[str, Any]) -> None:
        doc_id = str(doc.get("_id"))
        for spec, index_map in self._indexes.items():
            key = self._extract_index_key(doc, spec)
            if key in index_map and doc_id in index_map[key]:
                index_map[key].remove(doc_id)

    # ---------------------------------------------------------------------------
    # CRUD Operations
    # ---------------------------------------------------------------------------

    def insert_one(self, doc: Dict[str, Any]) -> str:
        with self._lock:
            doc_copy = copy.deepcopy(doc)
            if "_id" not in doc_copy or doc_copy["_id"] is None or doc_copy["_id"] == "":
                doc_id = f"doc_{len(self._documents) + 1}_{uuid.uuid4().hex[:8]}"
                doc_copy["_id"] = doc_id
            else:
                doc_id = str(doc_copy["_id"])
                doc_copy["_id"] = doc_id

            if doc_id in self._id_map:
                raise ValueError(f"Document with _id '{doc_id}' already exists")

            self._documents.append(doc_copy)
            self._id_map[doc_id] = doc_copy
            self._index_insert(doc_copy)

            if not self.in_memory and self.storage_path is not None:
                self._save_to_disk()

            return doc_id

    def insert_many(self, docs: List[Dict[str, Any]]) -> List[str]:
        if not docs:
            return []
        with self._lock:
            inserted_ids: List[str] = []
            for doc in docs:
                doc_copy = copy.deepcopy(doc)
                if "_id" not in doc_copy or doc_copy["_id"] is None or doc_copy["_id"] == "":
                    doc_id = f"doc_{len(self._documents) + 1}_{uuid.uuid4().hex[:8]}"
                    doc_copy["_id"] = doc_id
                else:
                    doc_id = str(doc_copy["_id"])
                    doc_copy["_id"] = doc_id

                if doc_id in self._id_map:
                    raise ValueError(f"Document with _id '{doc_id}' already exists")

                self._documents.append(doc_copy)
                self._id_map[doc_id] = doc_copy
                self._index_insert(doc_copy)
                inserted_ids.append(doc_id)

            if not self.in_memory and self.storage_path is not None:
                self._save_to_disk()

            return inserted_ids

    def count_documents(self, filter: Optional[Dict[str, Any]] = None) -> int:
        with self._lock:
            if not filter:
                return len(self._documents)
            candidates = self._find_candidates(filter)
            return len(candidates)

    def find(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            candidates = self._find_candidates(filter)
            if limit > 0:
                candidates = candidates[:limit]

            if projection:
                return [self._apply_projection(d, projection) for d in candidates]
            return [copy.deepcopy(d) for d in candidates]

    def update_one(self, filter: Dict[str, Any], update: Dict[str, Any]) -> bool:
        with self._lock:
            candidates = self._find_candidates(filter)
            if not candidates:
                return False
            target = candidates[0]
            self._index_remove(target)
            self._apply_update(target, update)
            self._index_insert(target)

            if not self.in_memory and self.storage_path is not None:
                self._save_to_disk()
            return True

    def update_many(self, filter: Dict[str, Any], update: Dict[str, Any]) -> int:
        with self._lock:
            candidates = self._find_candidates(filter)
            if not candidates:
                return 0
            for target in candidates:
                self._index_remove(target)
                self._apply_update(target, update)
                self._index_insert(target)

            if not self.in_memory and self.storage_path is not None:
                self._save_to_disk()
            return len(candidates)

    def delete_one(self, filter: Dict[str, Any]) -> bool:
        with self._lock:
            candidates = self._find_candidates(filter)
            if not candidates:
                return False
            target = candidates[0]
            self._index_remove(target)
            doc_id = str(target.get("_id"))
            self._id_map.pop(doc_id, None)
            self._documents.remove(target)

            if not self.in_memory and self.storage_path is not None:
                self._save_to_disk()
            return True

    def delete_many(self, filter: Dict[str, Any]) -> int:
        with self._lock:
            candidates = self._find_candidates(filter)
            if not candidates:
                return 0
            for target in candidates:
                self._index_remove(target)
                doc_id = str(target.get("_id"))
                self._id_map.pop(doc_id, None)
                self._documents.remove(target)

            if not self.in_memory and self.storage_path is not None:
                self._save_to_disk()
            return len(candidates)

    def _apply_update(self, doc: Dict[str, Any], update: Dict[str, Any]) -> None:
        if "$set" in update:
            for k, v in update["$set"].items():
                self._set_nested_field(doc, k, copy.deepcopy(v))
        if "$inc" in update:
            for k, v in update["$inc"].items():
                exists, current_val = self._get_nested_field(doc, k)
                new_val = (current_val if exists and isinstance(current_val, (int, float)) else 0) + v
                self._set_nested_field(doc, k, new_val)
        if "$unset" in update:
            for k in update["$unset"]:
                self._unset_nested_field(doc, k)
        # If no operators, replace or merge
        if not any(k.startswith("$") for k in update.keys()):
            for k, v in update.items():
                doc[k] = copy.deepcopy(v)

    # ---------------------------------------------------------------------------
    # Candidate Selection with Secondary Index Acceleration
    # ---------------------------------------------------------------------------

    def _find_candidates(
        self, query_filter: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        self.last_query_used_index = None
        if not query_filter:
            return list(self._documents)

        # 1. Check if direct query on _id
        if "_id" in query_filter and isinstance(query_filter["_id"], str):
            doc_id = query_filter["_id"]
            doc = self._id_map.get(doc_id)
            if doc and self._matches_document(doc, query_filter):
                return [doc]
            return []

        # 2. Check compound index match
        matched_candidate_ids: Optional[Set[str]] = None
        for spec, index_map in self._indexes.items():
            if len(spec) > 1:
                # Check if all fields in compound index are equality queries
                field_names = [f for f, _ in spec]
                if all(f in query_filter and not isinstance(query_filter[f], dict) for f in field_names):
                    key = tuple(query_filter[f] for f in field_names)
                    doc_ids = set(index_map.get(key, []))
                    matched_candidate_ids = doc_ids
                    self.last_query_used_index = "_".join(f"{f}_{d}" for f, d in spec)
                    break

        # 3. Check single-field index match if compound not used
        if matched_candidate_ids is None:
            for spec, index_map in self._indexes.items():
                if len(spec) == 1:
                    field = spec[0][0]
                    if field in query_filter:
                        filter_val = query_filter[field]
                        if not isinstance(filter_val, dict):
                            # Exact equality lookup
                            doc_ids = set(index_map.get(filter_val, []))
                            matched_candidate_ids = doc_ids
                            self.last_query_used_index = f"{field}_{spec[0][1]}"
                            break
                        elif "$eq" in filter_val:
                            doc_ids = set(index_map.get(filter_val["$eq"], []))
                            matched_candidate_ids = doc_ids
                            self.last_query_used_index = f"{field}_{spec[0][1]}"
                            break
                        elif "$in" in filter_val and isinstance(filter_val["$in"], list):
                            doc_ids = set()
                            for val in filter_val["$in"]:
                                doc_ids.update(index_map.get(val, []))
                            matched_candidate_ids = doc_ids
                            self.last_query_used_index = f"{field}_{spec[0][1]}"
                            break

        if matched_candidate_ids is not None:
            candidate_docs = [
                self._id_map[doc_id]
                for doc_id in matched_candidate_ids
                if doc_id in self._id_map
            ]
        else:
            candidate_docs = self._documents

        return [d for d in candidate_docs if self._matches_document(d, query_filter)]

    # ---------------------------------------------------------------------------
    # Query Operator Matching Engine
    # ---------------------------------------------------------------------------

    @staticmethod
    def _get_nested_field(doc: Dict[str, Any], path: str) -> Tuple[bool, Any]:
        """Safely retrieves a field via dot notation (e.g. 'metadata.token_count')."""
        if "." not in path:
            return (path in doc, doc.get(path))
        parts = path.split(".")
        curr: Any = doc
        for part in parts:
            if not isinstance(curr, dict) or part not in curr:
                return (False, None)
            curr = curr[part]
        return (True, curr)

    @staticmethod
    def _set_nested_field(doc: Dict[str, Any], path: str, val: Any) -> None:
        if "." not in path:
            doc[path] = val
            return
        parts = path.split(".")
        curr = doc
        for part in parts[:-1]:
            if part not in curr or not isinstance(curr[part], dict):
                curr[part] = {}
            curr = curr[part]
        curr[parts[-1]] = val

    @staticmethod
    def _unset_nested_field(doc: Dict[str, Any], path: str) -> None:
        if "." not in path:
            doc.pop(path, None)
            return
        parts = path.split(".")
        curr = doc
        for part in parts[:-1]:
            if not isinstance(curr, dict) or part not in curr:
                return
            curr = curr[part]
        if isinstance(curr, dict):
            curr.pop(parts[-1], None)

    def _matches_document(
        self, doc: Dict[str, Any], query_filter: Dict[str, Any]
    ) -> bool:
        if not query_filter:
            return True

        for key, cond in query_filter.items():
            if key == "$and":
                if not all(self._matches_document(doc, sub) for sub in cond):
                    return False
            elif key == "$or":
                if not any(self._matches_document(doc, sub) for sub in cond):
                    return False
            elif key == "$nor":
                if any(self._matches_document(doc, sub) for sub in cond):
                    return False
            else:
                exists, doc_val = self._get_nested_field(doc, key)
                if not self._matches_condition(doc_val, exists, cond):
                    return False

        return True

    def _matches_condition(self, doc_val: Any, exists: bool, cond: Any) -> bool:
        if not isinstance(cond, dict):
            return exists and doc_val == cond

        # If it's a dict but has no $-prefixed operator keys, treated as literal dict comparison
        if not any(k.startswith("$") for k in cond.keys()):
            return exists and doc_val == cond

        for op, target in cond.items():
            if op == "$eq":
                if target is None:
                    if exists and doc_val is not None:
                        return False
                else:
                    if not exists or doc_val != target:
                        return False
            elif op == "$ne":
                if target is None:
                    if not exists or doc_val is None:
                        return False
                else:
                    if exists and doc_val == target:
                        return False
            elif op == "$gt":
                if not exists or doc_val is None:
                    return False
                if not self._compare(doc_val, target, lambda a, b: a > b):
                    return False
            elif op == "$gte":
                if not exists or doc_val is None:
                    return False
                if not self._compare(doc_val, target, lambda a, b: a >= b):
                    return False
            elif op == "$lt":
                if not exists or doc_val is None:
                    return False
                if not self._compare(doc_val, target, lambda a, b: a < b):
                    return False
            elif op == "$lte":
                if not exists or doc_val is None:
                    return False
                if not self._compare(doc_val, target, lambda a, b: a <= b):
                    return False
            elif op == "$in":
                if not exists:
                    return False
                if not isinstance(target, (list, tuple, set)):
                    return False
                if isinstance(doc_val, list):
                    if not any(item in target for item in doc_val):
                        return False
                else:
                    if doc_val not in target:
                        return False
            elif op == "$nin":
                if not exists:
                    continue
                if isinstance(target, (list, tuple, set)):
                    if isinstance(doc_val, list):
                        if any(item in target for item in doc_val):
                            return False
                    else:
                        if doc_val in target:
                            return False
            elif op == "$exists":
                if exists != bool(target):
                    return False
            elif op == "$regex":
                if not exists or doc_val is None:
                    return False
                pattern = str(target)
                flags = 0
                if "$options" in cond and "i" in cond["$options"]:
                    flags |= re.IGNORECASE
                if not re.search(pattern, str(doc_val), flags=flags):
                    return False

        return True

    @staticmethod
    def _compare(val_a: Any, val_b: Any, op_func: Any) -> bool:
        """Type-safe comparison between numeric or homogeneous values."""
        try:
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                return bool(op_func(float(val_a), float(val_b)))
            if type(val_a) is type(val_b):
                return bool(op_func(val_a, val_b))
            return False
        except Exception:
            return False

    # ---------------------------------------------------------------------------
    # Projection Engine
    # ---------------------------------------------------------------------------

    def _apply_projection(
        self, doc: Dict[str, Any], projection: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not projection:
            return copy.deepcopy(doc)

        include_mode = any(
            bool(v) for k, v in projection.items() if k != "_id"
        )

        if include_mode:
            res: Dict[str, Any] = {}
            if projection.get("_id", 1):
                if "_id" in doc:
                    res["_id"] = copy.deepcopy(doc["_id"])
            for field, flag in projection.items():
                if field != "_id" and flag:
                    exists, val = self._get_nested_field(doc, field)
                    if exists:
                        res[field] = copy.deepcopy(val)
            return res
        else:
            # Exclusion mode
            res = copy.deepcopy(doc)
            for field, flag in projection.items():
                if not flag:
                    res.pop(field, None)
            return res

    # ---------------------------------------------------------------------------
    # Aggregation Pipeline Framework
    # ---------------------------------------------------------------------------

    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes a sequence of aggregation stages:
        $match, $group, $sort, $limit, $project.
        """
        with self._lock:
            stream = [copy.deepcopy(d) for d in self._documents]

            for stage in pipeline:
                if "$match" in stage:
                    match_filter = stage["$match"]
                    stream = [d for d in stream if self._matches_document(d, match_filter)]
                elif "$group" in stage:
                    stream = self._aggregate_group(stream, stage["$group"])
                elif "$sort" in stage:
                    stream = self._aggregate_sort(stream, stage["$sort"])
                elif "$limit" in stage:
                    limit_val = max(0, int(stage["$limit"]))
                    stream = stream[:limit_val]
                elif "$project" in stage:
                    stream = self._aggregate_project(stream, stage["$project"])

            return stream

    def _eval_expr(self, doc: Dict[str, Any], expr: Any) -> Any:
        """Evaluates an aggregation expression ($field, $$ROOT, $cond, literal)."""
        if expr == "$$ROOT":
            return doc
        if isinstance(expr, str) and expr.startswith("$"):
            field = expr[1:]
            _, val = self._get_nested_field(doc, field)
            return val
        if isinstance(expr, dict):
            if "$cond" in expr:
                cond_spec = expr["$cond"]
                if isinstance(cond_spec, list) and len(cond_spec) == 3:
                    if_expr, then_expr, else_expr = cond_spec
                elif isinstance(cond_spec, dict):
                    if_expr = cond_spec.get("if")
                    then_expr = cond_spec.get("then")
                    else_expr = cond_spec.get("else")
                else:
                    return None

                condition_met = self._eval_boolean_expr(doc, if_expr)
                return self._eval_expr(doc, then_expr if condition_met else else_expr)
        return expr

    def _eval_boolean_expr(self, doc: Dict[str, Any], if_expr: Any) -> bool:
        if isinstance(if_expr, bool):
            return if_expr
        if isinstance(if_expr, dict):
            for op, operands in if_expr.items():
                if isinstance(operands, list) and len(operands) == 2:
                    left = self._eval_expr(doc, operands[0])
                    right = self._eval_expr(doc, operands[1])
                    if op == "$eq":
                        return left == right
                    elif op == "$ne":
                        return left != right
                    elif op == "$gt":
                        return self._compare(left, right, lambda a, b: a > b)
                    elif op == "$gte":
                        return self._compare(left, right, lambda a, b: a >= b)
                    elif op == "$lt":
                        return self._compare(left, right, lambda a, b: a < b)
                    elif op == "$lte":
                        return self._compare(left, right, lambda a, b: a <= b)
            # Match condition dictionary
            return self._matches_document(doc, if_expr)
        return bool(self._eval_expr(doc, if_expr))

    def _eval_group_key(self, doc: Dict[str, Any], id_spec: Any) -> Any:
        if id_spec is None:
            return None
        if isinstance(id_spec, str) and id_spec.startswith("$"):
            field = id_spec[1:]
            _, val = self._get_nested_field(doc, field)
            return val
        if isinstance(id_spec, dict):
            extracted = []
            for k in sorted(id_spec.keys()):
                v = id_spec[k]
                val = self._eval_expr(doc, v)
                extracted.append((k, val))
            return tuple(extracted)
        return id_spec

    def _aggregate_group(
        self, stream: List[Dict[str, Any]], group_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        id_spec = group_spec.get("_id")
        groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

        for doc in stream:
            group_key = self._eval_group_key(doc, id_spec)
            groups[group_key].append(doc)

        result: List[Dict[str, Any]] = []

        for group_key, group_docs in groups.items():
            row: Dict[str, Any] = {}
            # Reconstruct _id
            if isinstance(group_key, tuple) and isinstance(id_spec, dict):
                row["_id"] = dict(group_key)
            else:
                row["_id"] = group_key

            for field, accum_spec in group_spec.items():
                if field == "_id":
                    continue

                if not isinstance(accum_spec, dict):
                    row[field] = accum_spec
                    continue

                if "$sum" in accum_spec:
                    sum_expr = accum_spec["$sum"]
                    total = 0.0
                    for d in group_docs:
                        val = self._eval_expr(d, sum_expr)
                        if isinstance(val, (int, float)) and not math.isnan(val):
                            total += float(val)
                    row[field] = int(total) if total.is_integer() else round(total, 4)

                elif "$avg" in accum_spec:
                    avg_expr = accum_spec["$avg"]
                    nums: List[float] = []
                    for d in group_docs:
                        val = self._eval_expr(d, avg_expr)
                        if isinstance(val, (int, float)) and not math.isnan(val):
                            nums.append(float(val))
                    row[field] = round(sum(nums) / len(nums), 4) if nums else None

                elif "$min" in accum_spec:
                    min_expr = accum_spec["$min"]
                    vals = [self._eval_expr(d, min_expr) for d in group_docs]
                    valid = [v for v in vals if v is not None]
                    row[field] = min(valid) if valid else None

                elif "$max" in accum_spec:
                    max_expr = accum_spec["$max"]
                    vals = [self._eval_expr(d, max_expr) for d in group_docs]
                    valid = [v for v in vals if v is not None]
                    row[field] = max(valid) if valid else None

                elif "$push" in accum_spec:
                    push_expr = accum_spec["$push"]
                    row[field] = [
                        copy.deepcopy(self._eval_expr(d, push_expr)) for d in group_docs
                    ]

                elif "$first" in accum_spec:
                    first_expr = accum_spec["$first"]
                    row[field] = (
                        copy.deepcopy(self._eval_expr(group_docs[0], first_expr))
                        if group_docs
                        else None
                    )

                elif "$last" in accum_spec:
                    last_expr = accum_spec["$last"]
                    row[field] = (
                        copy.deepcopy(self._eval_expr(group_docs[-1], last_expr))
                        if group_docs
                        else None
                    )

                elif "$count" in accum_spec:
                    row[field] = len(group_docs)

            result.append(row)

        return result

    def _aggregate_sort(
        self, stream: List[Dict[str, Any]], sort_spec: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        if not sort_spec or not stream:
            return stream

        # Python sort is stable: sorting in reverse order of fields achieves multi-key sorting
        sorted_stream = list(stream)
        for field, direction in reversed(list(sort_spec.items())):
            reverse = direction < 0

            def sort_key(d: Dict[str, Any], f: str = field) -> Tuple[int, Any]:
                exists, val = self._get_nested_field(d, f)
                if not exists or val is None:
                    return (0, None)
                if isinstance(val, (int, float, bool)):
                    try:
                        f_val = float(val)
                        if math.isnan(f_val):
                            return (0, None)
                        return (1, f_val)
                    except (ValueError, TypeError):
                        return (0, None)
                if isinstance(val, str):
                    return (2, str(val))
                return (3, str(val))

            sorted_stream.sort(key=sort_key, reverse=reverse)

        return sorted_stream

    def _aggregate_project(
        self, stream: List[Dict[str, Any]], project_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not project_spec:
            return [copy.deepcopy(d) for d in stream]

        non_id_specs = [v for k, v in project_spec.items() if k != "_id"]
        is_exclusion = (
            (len(non_id_specs) == 0 and project_spec.get("_id") in (0, False))
            or (len(non_id_specs) > 0 and all(v in (0, False) for v in non_id_specs))
        )

        result: List[Dict[str, Any]] = []
        if is_exclusion:
            for doc in stream:
                out_doc = copy.deepcopy(doc)
                for target_field, expr in project_spec.items():
                    if expr in (0, False):
                        self._unset_nested_field(out_doc, target_field)
                result.append(out_doc)
            return result
        else:
            for doc in stream:
                out_doc: Dict[str, Any] = {}
                for target_field, expr in project_spec.items():
                    if expr == 0 or expr is False:
                        continue
                    if expr == 1 or expr is True:
                        exists, val = self._get_nested_field(doc, target_field)
                        if exists:
                            out_doc[target_field] = copy.deepcopy(val)
                    else:
                        out_doc[target_field] = copy.deepcopy(self._eval_expr(doc, expr))
                result.append(out_doc)
            return result
