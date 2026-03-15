"""
Candidate Generation V3 - LLM Schema Matcher
==============================================
Uses a fine-tuned small LLM (Phi-3-mini + LoRA) to directly generate
source column mappings and transform types.

The LLM's pre-trained world knowledge handles:
  - Synonym understanding (product = item, employee = worker)
  - Compositional reasoning (lead_name = first_name + last_name)
  - Arithmetic inference (unit_cost = total_cost / quantity)
  - Cross-table FK lookups with minimal columns

Architecture:
  1. Serialize source schema + joins as structured prompt
  2. For each target column, construct the mapping query
  3. LLM generates: source_columns, transform_type, reasoning
  4. Parse & validate the output (ensure columns exist)
  5. Return results in the same format as V2 for seamless integration
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

from join_graph_builder_v2 import Column, ColumnType, JoinEdge, Table


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a schema mapping expert. Given a source database schema and a "
    "target column, identify which source column(s) should be mapped to the "
    "target and what transformation is needed.\n"
    "\n"
    "COLUMN SELECTION RULES:\n"
    "- Use ONLY columns that exist in the source schema\n"
    "- Choose the MINIMUM number of source columns needed\n"
    "- ALWAYS output the VALUE column(s) that hold the actual data, NOT the "
    "FK/join key columns used to traverse between tables. Example: to get a "
    "department name via a join, output departments.dept_name, NOT the "
    "employees.department_id FK column\n"
    "- When a mapping requires columns from MULTIPLE tables (e.g. date_diff "
    "between dates in different tables), select the actual data columns from "
    "each table, not the FK columns that link them\n"
    "- For surrogate keys, use the primary key of the matching entity table\n"
    "\n"
    "TRANSFORM DECISION RULES:\n"
    "- rename: ONLY when the source column is in the SAME table as the "
    "target's primary entity and no computation is needed. If a join/FK is "
    "required to reach the column, it is NOT a rename\n"
    "- direct_copy: identical to rename but emphasizes no name change\n"
    "- fk_lookup: when the source column is in a DIFFERENT table reached "
    "via a single FK join. The key distinction from rename is that a join "
    "is required\n"
    "- lookup_join: when the source column requires traversing 2+ joins "
    "(multi-hop). Use this instead of fk_lookup for multi-table chains\n"
    "- concat: combining 2+ columns into one string value\n"
    "- date_part: extracting year, month, day, etc. from a date/datetime\n"
    "- date_diff: computing the difference between two dates (duration, age)\n"
    "- date_format: converting a date to a display string format\n"
    "- date_parse: parsing a string into a date/datetime type\n"
    "- arithmetic: math operations (add, subtract, multiply, divide) between "
    "columns. Do NOT assume arithmetic from column names alone -- if salary "
    "maps to annual_salary with no formula specified, use rename\n"
    "- conditional: deriving a BOOLEAN true/false flag from a status/code "
    "column. The target must be boolean/flag type\n"
    "- code_to_label: mapping a code/status column to a human-readable LABEL "
    "string (e.g. gender_code -> gender_label, status_code -> status_name). "
    "This is NOT the same as conditional -- code_to_label produces a label "
    "string, conditional produces a boolean\n"
    "- bucketing: grouping a numeric/continuous value into range categories\n"
    "- type_cast: converting data type without changing the value\n"
    "- template: assembling columns into a formatted display string pattern\n"
    "\n"
    "Valid transform types: rename, direct_copy, concat, fk_lookup, date_part, "
    "date_diff, date_format, date_parse, arithmetic, conditional, bucketing, "
    "code_to_label, type_cast, lookup_join, template\n"
    "\n"
    "SUB-OPERATION: For each transform, specify the fine-grained operation:\n"
    "- rename: rename_only\n"
    "- direct_copy: direct_copy\n"
    "- concat: concat_two, concat_multi (3+ columns)\n"
    "- fk_lookup: fk_dimension_lookup\n"
    "- lookup_join: multi_hop_lookup\n"
    "- date_part: extract_year, extract_month, extract_day, extract_quarter, "
    "extract_hour\n"
    "- date_diff: date_difference, age_calculation, duration_days, "
    "duration_hours\n"
    "- date_format: format_date\n"
    "- date_parse: parse_date\n"
    "- arithmetic: add, subtract, multiply, divide, ratio_percentage, "
    "scaling_unit_conversion\n"
    "- conditional: threshold_flag, status_flag, equality_check, "
    "null_presence_flag\n"
    "- code_to_label: code_to_label, category_harmonization\n"
    "- bucketing: bucketing_binning, range_classification\n"
    "- type_cast: type_cast_numeric, type_cast_string, type_cast_date\n"
    "- template: string_template\n"
    "\n"
    "OUTPUT RULES:\n"
    "- Respond with EXACTLY ONE mapping for the requested target column\n"
    "- Do NOT output mappings for other columns\n"
    "\n"
    "Respond in EXACTLY this format:\n"
    "source_columns: <table.column>, <table.column>, ...\n"
    "transform_type: <transform>\n"
    "sub_operation: <fine-grained operation>\n"
    "reasoning: <brief explanation>"
)

VALID_TRANSFORMS = {
    "rename", "direct_copy", "concat", "fk_lookup", "date_part",
    "date_diff", "date_format", "date_parse", "arithmetic",
    "conditional", "bucketing", "code_to_label", "type_cast",
    "lookup_join", "template",
}

VALID_SUB_OPERATIONS = {
    # rename / direct_copy
    "rename_only", "direct_copy",
    # concat
    "concat_two", "concat_multi",
    # fk_lookup / lookup_join
    "fk_dimension_lookup", "multi_hop_lookup",
    # date_part
    "extract_year", "extract_month", "extract_day", "extract_quarter", "extract_hour",
    # date_diff
    "date_difference", "age_calculation", "duration_days", "duration_hours",
    # date_format / date_parse
    "format_date", "parse_date",
    # arithmetic
    "add", "subtract", "multiply", "divide", "ratio_percentage", "scaling_unit_conversion",
    # conditional
    "threshold_flag", "status_flag", "equality_check", "null_presence_flag",
    # code_to_label
    "code_to_label", "category_harmonization",
    # bucketing
    "bucketing_binning", "range_classification",
    # type_cast
    "type_cast_numeric", "type_cast_string", "type_cast_date",
    # template
    "string_template",
}


# ---------------------------------------------------------------
# Schema serialization (mirrors training data generation)
# ---------------------------------------------------------------

def serialize_source_schema(source_tables: Dict[str, Table]) -> str:
    """Serialize source tables into the prompt format."""
    lines = []
    for tname, table in source_tables.items():
        col_parts = []
        for cname, col in table.columns.items():
            flags = []
            if col.constraints and col.constraints.is_primary_key:
                flags.append("PK")
            if col.constraints and col.constraints.is_foreign_key:
                flags.append("FK")
            base_type = getattr(col.col_type, "base_type", str(col.col_type)).lower()
            flag_str = " ".join(flags)
            if flag_str:
                col_parts.append(f"{cname} {flag_str} {base_type}")
            else:
                col_parts.append(f"{cname} {base_type}")
        lines.append(f"  {tname}({', '.join(col_parts)})")
    return "\n".join(lines)


def serialize_join_edges(join_edges: List[JoinEdge]) -> str:
    """Serialize join edges into the prompt format."""
    if not join_edges:
        return "  (none)"
    lines = set()
    for e in join_edges:
        for lc, rc in zip(e.left_cols, e.right_cols):
            lines.add(f"  {e.left_table}.{lc} = {e.right_table}.{rc}")
    return "\n".join(sorted(lines)) if lines else "  (none)"


def build_mapping_prompt(
    source_tables: Dict[str, Table],
    join_edges: List[JoinEdge],
    target_table: str,
    target_column: str,
    target_type: str,
    target_desc: str = "",
) -> List[Dict[str, str]]:
    """Build the chat messages for a single mapping query."""
    schema_text = serialize_source_schema(source_tables)
    joins_text = serialize_join_edges(join_edges)

    desc_part = f' - "{target_desc}"' if target_desc else ""
    target_line = f"Map target: {target_table}.{target_column} ({target_type}){desc_part}"

    user_msg = (
        f"Source Schema:\n{schema_text}\n\n"
        f"Joins:\n{joins_text}\n\n"
        f"{target_line}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------

_SOURCE_RE = re.compile(r"source_columns?\s*:\s*(.+)", re.IGNORECASE)
_TRANSFORM_RE = re.compile(r"transform_type\s*:\s*(\S+)", re.IGNORECASE)
_SUB_OP_RE = re.compile(r"sub_operation\s*:\s*(\S+)", re.IGNORECASE)
_REASONING_RE = re.compile(r"reasoning\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_llm_output(text: str) -> Dict[str, Any]:
    """Parse the structured LLM output into source columns, transform, and sub_operation."""
    result = {
        "source_columns": [],
        "transform_type": "unmapped",
        "sub_operation": "",
        "reasoning": "",
        "parse_success": False,
    }

    # Extract source columns
    m = _SOURCE_RE.search(text)
    if m:
        cols_str = m.group(1).strip()
        # Split by comma, clean each
        cols = []
        for part in cols_str.split(","):
            part = part.strip().rstrip(".")
            # Remove any trailing text after the column ref
            part = part.split("\n")[0].strip()
            if "." in part:
                # Validate format: table.column
                pieces = part.split(".")
                if len(pieces) == 2:
                    cols.append(part)
        result["source_columns"] = cols

    # Extract transform type
    m = _TRANSFORM_RE.search(text)
    if m:
        transform = m.group(1).strip().lower()
        if transform in VALID_TRANSFORMS:
            result["transform_type"] = transform
        else:
            # Try fuzzy match
            for vt in VALID_TRANSFORMS:
                if transform.startswith(vt) or vt.startswith(transform):
                    result["transform_type"] = vt
                    break

    # Extract sub_operation
    m = _SUB_OP_RE.search(text)
    if m:
        sub_op = m.group(1).strip().lower()
        if sub_op in VALID_SUB_OPERATIONS:
            result["sub_operation"] = sub_op
        else:
            # Try fuzzy match
            for vs in VALID_SUB_OPERATIONS:
                if sub_op.startswith(vs) or vs.startswith(sub_op):
                    result["sub_operation"] = vs
                    break
            if not result["sub_operation"]:
                result["sub_operation"] = sub_op  # keep raw value

    # Extract reasoning
    m = _REASONING_RE.search(text)
    if m:
        result["reasoning"] = m.group(1).strip()

    # Mark success if we got at least source columns
    if result["source_columns"]:
        result["parse_success"] = True

    return result


def validate_columns(
    parsed: Dict[str, Any],
    source_tables: Dict[str, Table],
) -> Dict[str, Any]:
    """Validate that parsed source columns actually exist in the schema."""
    valid_cols = []
    for col_ref in parsed["source_columns"]:
        parts = col_ref.split(".", 1)
        if len(parts) != 2:
            continue
        tname, cname = parts
        if tname in source_tables and cname in source_tables[tname].columns:
            valid_cols.append(col_ref)
    parsed["source_columns"] = valid_cols
    if not valid_cols:
        parsed["parse_success"] = False
    return parsed


# ---------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------

class SchemaMatcherLLM:
    """
    LLM-based schema matcher that generates source column mappings.
    Compatible with CandidateGeneratorV2's output interface.
    """

    def __init__(
        self,
        source_tables: Dict[str, Table],
        join_edges: List[JoinEdge],
        model_dir: str = "schema_matcher_llm1",
        base_model: str = "microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens: int = 150,
        temperature: float = 0.0,
        device: str = "cpu",
    ):
        if not HAS_LLM:
            raise ImportError(
                "LLM dependencies required: pip install torch transformers peft"
            )

        self.tables = source_tables
        self.edges = join_edges
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device.lower() if device else "cpu"

        # Determine model paths
        adapter_path = os.path.join(model_dir, "adapter")
        metadata_path = os.path.join(model_dir, "training_metadata.json")

        # Load base model name from metadata if available
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            base_model = metadata.get("base_model", base_model)
            print(f"[V3] Base model from metadata: {base_model}")

        has_adapter = os.path.exists(adapter_path) and any(
            os.path.exists(os.path.join(adapter_path, f))
            for f in ["adapter_model.safetensors", "adapter_model.bin",
                       "adapter_config.json"]
        )

        # Load tokenizer
        tokenizer_path = adapter_path if has_adapter else base_model
        print(f"[V3] Loading tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        print(f"[V3] Loading base model: {base_model} (device={self.device})")
        if self.device == "cpu":
            device_map = None
            dtype = torch.float32
        else:
            device_map = "auto"
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        # Fix Phi-3 rope_scaling compatibility with newer transformers
        from transformers import AutoConfig
        _cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        if hasattr(_cfg, "rope_scaling") and _cfg.rope_scaling is not None:
            rs = _cfg.rope_scaling
            scaling_type = rs.get("type", rs.get("rope_type", ""))
            if scaling_type in ("default", ""):
                _cfg.rope_scaling = None
            else:
                if "type" not in rs and "rope_type" in rs:
                    rs["type"] = rs["rope_type"]
                elif "rope_type" not in rs and "type" in rs:
                    rs["rope_type"] = rs["type"]
                _cfg.rope_scaling = rs

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=_cfg,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="eager",  # Reliable on CPU; matches train/eval scripts
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        if has_adapter:
            print(f"[V3] Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
            print("[V3] Adapter merged into base model")
        else:
            print("[V3] No adapter found, using base model only")

        self.model.eval()

        # Pre-compute schema text (shared across all queries)
        self._schema_text = serialize_source_schema(source_tables)
        self._joins_text = serialize_join_edges(join_edges)

        print(f"[V3] Ready. Schema: {len(source_tables)} tables, {len(join_edges)} joins")

    @torch.no_grad()
    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from the LLM."""
        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback formatting
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"<|system|>\n{content}<|end|>")
                elif role == "user":
                    parts.append(f"<|user|>\n{content}<|end|>")
            parts.append("<|assistant|>\n")
            prompt = "\n".join(parts)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - self.max_new_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.9

        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def rank(
        self,
        target_table: str,
        target_column: str,
        target_type: str,
        target_desc: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Map a single target column using the LLM.
        Returns result in the same format as V2's rank() method.
        """
        t0 = time.time()

        # Build prompt
        messages = build_mapping_prompt(
            self.tables, self.edges,
            target_table, target_column, target_type, target_desc,
        )

        # Generate
        response = self._generate(messages)

        # Parse and validate
        parsed = parse_llm_output(response)
        parsed = validate_columns(parsed, self.tables)

        # Determine confidence based on parse quality
        if parsed["parse_success"]:
            confidence = 0.90  # High confidence for successful LLM parse
        else:
            confidence = 0.10  # Low confidence if parsing failed

        # Build result in V2-compatible format
        elapsed = time.time() - t0

        top_candidate = {
            "rank": 1,
            "source_columns": parsed["source_columns"],
            "source_columns_typed": [],
            "join_path": [],
            "join_edges_raw": [],
            "transform_type": parsed["transform_type"],
            "sub_operation": parsed.get("sub_operation", ""),
            "bi_score": 0.0,
            "cross_score": 0.0,
            "combined_score": confidence,
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", ""),
        }

        # Fill in typed columns
        for col_ref in parsed["source_columns"]:
            parts = col_ref.split(".", 1)
            if len(parts) == 2:
                tname, cname = parts
                if tname in self.tables and cname in self.tables[tname].columns:
                    col_obj = self.tables[tname].columns[cname]
                    base_type = getattr(col_obj.col_type, "base_type", str(col_obj.col_type))
                    top_candidate["source_columns_typed"].append((tname, cname, base_type))

        # Detect join path from source columns
        src_tables = {col_ref.split(".")[0] for col_ref in parsed["source_columns"] if "." in col_ref}
        if len(src_tables) > 1:
            for e in self.edges:
                if e.left_table in src_tables and e.right_table in src_tables:
                    top_candidate["join_path"].append(
                        f"{e.left_table}.{e.left_cols[0]} = {e.right_table}.{e.right_cols[0]}"
                    )
                    top_candidate["join_edges_raw"].append({
                        "from": e.left_table, "to": e.right_table,
                        "left_cols": list(e.left_cols), "right_cols": list(e.right_cols),
                        "confidence": round(float(e.confidence), 4),
                    })

        return {
            "target": {
                "table": target_table,
                "column": target_column,
                "type": target_type,
            },
            "top_candidates": [top_candidate] if parsed["parse_success"] else [],
            "retrieval_count": 1,
            "total_candidates": 1,
            "time_seconds": round(elapsed, 3),
            "llm_raw_response": response,
        }

    def rank_all_targets(
        self,
        target_tables: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Map all target columns using the LLM.
        Returns results in V2-compatible format for pipeline_bridge.
        """
        all_results = {"tables": [], "stats": {}}
        total = 0
        mapped = 0
        t_start = time.time()

        for ttable in target_tables:
            tname = ttable["name"]
            table_result = {"name": tname, "columns": []}

            for tcol in ttable.get("columns", []):
                total += 1
                result = self.rank(
                    target_table=tname,
                    target_column=tcol["name"],
                    target_type=tcol.get("type", "string"),
                    target_desc=tcol.get("description", ""),
                )

                top = result["top_candidates"]
                best = top[0] if top else None
                col_result = {
                    "name": tcol["name"],
                    "type": tcol.get("type", "string"),
                    "description": tcol.get("description", ""),
                }

                if best:
                    col_result["final_source"] = best["source_columns"]
                    col_result["final_transform"] = best["transform_type"]
                    col_result["final_sub_operation"] = best.get("sub_operation", "")
                    col_result["final_confidence"] = best["confidence"]
                    col_result["stage_a"] = {
                        "source_columns": best["source_columns"],
                        "join_path": best.get("join_edges_raw", []),
                        "transform_family": best["transform_type"],
                        "sub_operation": best.get("sub_operation", ""),
                        "confidence": best["confidence"],
                        "time": result["time_seconds"],
                        "abstain": best["confidence"] < 0.3,
                    }
                    col_result["alternatives"] = []
                    col_result["reasoning"] = best.get("reasoning", "")
                    mapped += 1
                else:
                    col_result["final_source"] = []
                    col_result["final_transform"] = "unmapped"
                    col_result["final_sub_operation"] = ""
                    col_result["final_confidence"] = 0
                    col_result["stage_a"] = {
                        "source_columns": [], "transform_family": "unmapped",
                        "sub_operation": "",
                        "confidence": 0, "time": 0, "abstain": True,
                    }
                    col_result["alternatives"] = []

                table_result["columns"].append(col_result)
                print(f"  [V3] {tname}.{tcol['name']} -> "
                      f"{', '.join(best['source_columns']) if best else 'unmapped'} "
                      f"({best['transform_type'] if best else 'N/A'}) "
                      f"[{result['time_seconds']:.1f}s]")

            all_results["tables"].append(table_result)

        all_results["stats"] = {
            "total_columns": total,
            "mapped_columns": mapped,
            "mapping_rate": round(mapped / max(1, total) * 100, 1),
            "total_time": round(time.time() - t_start, 2),
            "total_candidates_enumerated": total,
            "has_stage_b": False,
            "engine": "llm_v3",
        }
        return all_results


# ---------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------

def _run_test():
    """Quick test with a small HR schema."""
    from pipeline_bridge import json_to_tables, discover_joins

    source_json = {
        "tables": [
            {"name": "employees", "columns": [
                {"name": "emp_id", "type": "int", "is_pk": True},
                {"name": "first_name", "type": "string"},
                {"name": "last_name", "type": "string"},
                {"name": "email", "type": "string"},
                {"name": "hire_date", "type": "date"},
                {"name": "salary", "type": "decimal"},
                {"name": "department_id", "type": "int", "is_fk": True},
            ]},
            {"name": "departments", "columns": [
                {"name": "dept_id", "type": "int", "is_pk": True},
                {"name": "dept_name", "type": "string"},
                {"name": "location", "type": "string"},
            ]},
        ]
    }

    targets = [
        {"name": "dim_employee", "columns": [
            {"name": "employee_key", "type": "int", "description": "surrogate key"},
            {"name": "full_name", "type": "string", "description": "employee full name"},
            {"name": "email_address", "type": "string", "description": "employee email"},
            {"name": "hire_year", "type": "int", "description": "year hired"},
            {"name": "department_name", "type": "string", "description": "department name"},
            {"name": "annual_salary", "type": "decimal", "description": "yearly salary"},
        ]},
    ]

    tables = json_to_tables(source_json["tables"])
    edges, _ = discover_joins(tables)

    engine = SchemaMatcherLLM(
        source_tables=tables,
        join_edges=edges,
    )

    result = engine.rank_all_targets(targets, top_k=1)

    print("\n" + "="*60)
    print("  TEST RESULTS")
    print("="*60)
    for tbl in result["tables"]:
        for col in tbl["columns"]:
            src = ", ".join(col.get("final_source", []))
            xform = col.get("final_transform", "?")
            conf = col.get("final_confidence", 0)
            print(f"  {tbl['name']}.{col['name']}: {src} -> {xform} ({conf:.0%})")

    print(f"\n  Stats: {result['stats']}")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_test()
    else:
        print("Usage: python candidate_generation_v3.py --test")
        print("       (Run the test to verify LLM schema matcher)")
