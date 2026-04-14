#!/usr/bin/env python3
"""SQL 字段血缘提取程序。

输入:
  - SQL 文件路径
  - 目标字段: 表名::字段名

输出:
  JSON:
  {
    "target": "SCHEMA.TABLE::COL",
    "sources": ["SRC.TABLE::COL", "CONST:xxx"]
  }
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

INDETERMINATE_MSG = "Projection requires explicit alias for deterministic lineage"


def normalize_ident(name: str) -> str:
    return name.strip().strip('"').upper()


def normalize_table(name: str) -> str:
    parts = [normalize_ident(p) for p in name.split('.') if p.strip()]
    return '.'.join(parts)


def normalize_col(name: str) -> str:
    return normalize_ident(name)


def table_to_name(table: exp.Table) -> str:
    parts = [p for p in [table.catalog, table.db, table.name] if p]
    return normalize_table(".".join(parts))


def split_target(target: str) -> Tuple[str, str]:
    if "::" not in target:
        raise ValueError("target 必须为 表名::字段名")
    t, c = target.split("::", 1)
    return normalize_table(t), normalize_col(c)


def preprocess_sql(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"\bdistribute\s+by\s+[^;]*", "", sql, flags=re.IGNORECASE)
    return sql


@dataclass
class WritePath:
    table: str
    column_exprs: Dict[str, exp.Expression] = field(default_factory=dict)
    scope_select: Optional[exp.Select] = None
    indeterminate: bool = False


class SQLLineageExtractor:
    def __init__(self, sql_text: str):
        self.sql_text = preprocess_sql(sql_text)
        self.statements = sqlglot.parse(self.sql_text, read="postgres")
        self.table_columns: Dict[str, List[str]] = {}
        self.write_paths: Dict[str, List[WritePath]] = {}
        self._index_statements()

    def _index_statements(self) -> None:
        for stmt in self.statements:
            if isinstance(stmt, exp.Create):
                self._index_create(stmt)
            elif isinstance(stmt, exp.Insert):
                self._index_insert(stmt)
            elif isinstance(stmt, exp.Update):
                self._index_update(stmt)

    def _index_create(self, stmt: exp.Create) -> None:
        table = stmt.this
        if not isinstance(table, exp.Table):
            return
        table_name = table_to_name(table)
        schema = stmt.args.get("schema")
        if isinstance(schema, exp.Schema):
            cols: List[str] = []
            for cdef in schema.expressions:
                if isinstance(cdef, exp.ColumnDef):
                    cols.append(normalize_col(cdef.this.name))
            if cols:
                self.table_columns[table_name] = cols

        as_expr = stmt.args.get("expression")
        if isinstance(as_expr, exp.Select):
            wp = self._write_path_from_select(table_name, None, as_expr)
            self.write_paths.setdefault(table_name, []).append(wp)

    def _index_insert(self, stmt: exp.Insert) -> None:
        table_expr = stmt.this
        table_name: Optional[str] = None
        target_cols: Optional[List[str]] = None

        if isinstance(table_expr, exp.Schema) and isinstance(table_expr.this, exp.Table):
            table_name = table_to_name(table_expr.this)
            target_cols = [normalize_col(c.name) for c in table_expr.expressions if isinstance(c, exp.Identifier)]
        elif isinstance(table_expr, exp.Table):
            table_name = table_to_name(table_expr)

        if not table_name:
            return

        source = stmt.expression
        if isinstance(source, exp.Select):
            wp = self._write_path_from_select(table_name, target_cols, source)
            self.write_paths.setdefault(table_name, []).append(wp)
        elif isinstance(source, exp.Union):
            for s in self._flatten_union(source):
                wp = self._write_path_from_select(table_name, target_cols, s)
                self.write_paths.setdefault(table_name, []).append(wp)

    def _index_update(self, stmt: exp.Update) -> None:
        if not isinstance(stmt.this, exp.Table):
            return
        table = table_to_name(stmt.this)
        wp = WritePath(table=table)
        for assign in stmt.expressions or []:
            if isinstance(assign, exp.EQ) and isinstance(assign.this, exp.Column):
                wp.column_exprs[normalize_col(assign.this.name)] = assign.expression
        self.write_paths.setdefault(table, []).append(wp)

    def _flatten_union(self, node: exp.Expression) -> List[exp.Select]:
        if isinstance(node, exp.Select):
            return [node]
        if isinstance(node, exp.Union):
            return self._flatten_union(node.left) + self._flatten_union(node.right)
        return []

    def _write_path_from_select(self, table: str, target_cols: Optional[List[str]], select: exp.Select) -> WritePath:
        wp = WritePath(table=table, scope_select=select)
        projections = select.expressions or []

        if target_cols is None:
            target_cols = self.table_columns.get(table)
            if not target_cols:
                wp.indeterminate = True
                return wp

        if len(projections) == 1 and isinstance(projections[0], exp.Star):
            for c in target_cols:
                wp.column_exprs[c] = exp.column(c)
            return wp

        if len(target_cols) != len(projections):
            for i, proj in enumerate(projections):
                col_name = self._projection_name(proj, i)
                if col_name:
                    wp.column_exprs[col_name] = proj
            return wp

        for tcol, proj in zip(target_cols, projections):
            wp.column_exprs[tcol] = proj
        return wp

    def _projection_name(self, expr_: exp.Expression, idx: int) -> Optional[str]:
        if isinstance(expr_, exp.Alias):
            return normalize_col(expr_.alias)
        if isinstance(expr_, exp.Column):
            return normalize_col(expr_.name)
        return normalize_col(f"COL_{idx+1}")

    def extract(self, target: str) -> Dict[str, object]:
        target_table, target_col = split_target(target)
        seen: Set[Tuple[str, str]] = set()
        sources, indeterminate = self._extract_from_table_col(target_table, target_col, seen)
        if indeterminate:
            payload: object = INDETERMINATE_MSG
        else:
            payload = sorted(sources)
        return {"target": f"{target_table}::{target_col}", "sources": payload}

    def _extract_from_table_col(
        self, table: str, col: str, seen: Set[Tuple[str, str]]
    ) -> Tuple[Set[str], bool]:
        key = (table, col)
        if key in seen:
            return set(), False
        seen.add(key)

        paths = self.write_paths.get(table, [])
        if not paths:
            return set(), False

        out: Set[str] = set()
        indeterminate = False

        for path in paths:
            expr_ = path.column_exprs.get(col)
            if expr_ is None:
                continue
            refs, inde = self._sources_from_expr(expr_, path.scope_select)
            if inde:
                indeterminate = True
                continue
            for ref_table, ref_col in refs:
                if ref_table == "CONST":
                    out.add(f"CONST:{ref_col}")
                elif ref_table in self.write_paths:
                    nested, nde = self._extract_from_table_col(ref_table, ref_col, seen)
                    if nde:
                        indeterminate = True
                    out.update(nested)
                else:
                    out.add(f"{ref_table}::{ref_col}")
        return out, indeterminate

    def _sources_from_expr(
        self, expr_: exp.Expression, scope_select: Optional[exp.Select]
    ) -> Tuple[Set[Tuple[str, str]], bool]:
        refs: Set[Tuple[str, str]] = set()

        # CASE: 仅提取 THEN / ELSE
        if isinstance(expr_, exp.Case):
            for if_ in expr_.args.get("ifs") or []:
                true_expr = if_.args.get("true")
                if true_expr is not None:
                    r, _ = self._sources_from_expr(true_expr, scope_select)
                    refs.update(r)
            default = expr_.args.get("default")
            if default is not None:
                r, _ = self._sources_from_expr(default, scope_select)
                refs.update(r)
            else:
                refs.add(("CONST", "NULL"))
            return refs, False

        # 常量
        if isinstance(expr_, exp.Literal):
            refs.add(("CONST", expr_.this))
            return refs, False

        # alias
        if isinstance(expr_, exp.Alias):
            return self._sources_from_expr(expr_.this, scope_select)

        for col in expr_.find_all(exp.Column):
            table, inde = self._resolve_column(col, scope_select)
            if inde:
                return set(), True
            if table is None:
                return set(), True
            refs.add((table, normalize_col(col.name)))

        for lit in expr_.find_all(exp.Literal):
            if lit.is_string or lit.is_int or lit.is_number:
                refs.add(("CONST", lit.this))

        if not refs and isinstance(expr_, exp.Null):
            refs.add(("CONST", "NULL"))

        return refs, False

    def _resolve_column(self, col: exp.Column, scope_select: Optional[exp.Select]) -> Tuple[Optional[str], bool]:
        if scope_select is None:
            if col.table:
                return normalize_table(col.table), False
            return None, True

        if col.table:
            alias = normalize_ident(col.table)
            amap = self._alias_map(scope_select)
            src = amap.get(alias)
            if src:
                return src, False
            return normalize_table(alias), False

        amap = self._alias_map(scope_select)
        unique_tables = set(amap.values())
        if len(unique_tables) == 1:
            return next(iter(unique_tables)), False
        return None, True

    def _alias_map(self, select: exp.Select) -> Dict[str, str]:
        amap: Dict[str, str] = {}
        from_ = select.args.get("from_")
        if from_:
            for t in from_.find_all(exp.Table):
                tname = table_to_name(t)
                alias = normalize_ident(t.alias_or_name)
                amap[alias] = tname
        for j in select.args.get("joins") or []:
            for t in j.find_all(exp.Table):
                tname = table_to_name(t)
                alias = normalize_ident(t.alias_or_name)
                amap[alias] = tname
        return amap


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL 字段血缘提取")
    parser.add_argument("sql_file", help="SQL 文件路径")
    parser.add_argument("target", help="目标字段，格式: 表名::字段名")
    args = parser.parse_args()

    sql_text = Path(args.sql_file).read_text(encoding="utf-8")
    extractor = SQLLineageExtractor(sql_text)
    print(json.dumps(extractor.extract(args.target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
