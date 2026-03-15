"""
Segmentation Model  -  Training Data Generator
================================================
Produces ~50 000+ labelled examples for per-character boundary detection.

Each example is a JSON line:
  {"text": "custacct", "labels": [0,0,0,1,0,0,0,1], "parts": ["cust","acct"]}

Label convention:  1 = last character of a sub-token (boundary position)
                   0 = not a boundary

Data sources
------------
1. ABBREV_DICT combo pairs/triples/quads
2. Full-word concatenations (schema / DB vocabulary)
3. Mixed abbreviation + full-word
4. Negative examples  (single words, no split)
5. Domain-specific realistic column names
6. Augmentation  (prefix noise, casing noise, numeric suffixes)

Run:
  python generate_seg_training_data.py
  -> seg_training_data/seg_train.jsonl   (85 %)
  -> seg_training_data/seg_val.jsonl     (15 %)
"""
from __future__ import annotations

import itertools
import json
import os
import random
from typing import List, Tuple

SEED = 42
random.seed(SEED)

OUTPUT_DIR = "seg_training_data"

# =====================================================================
# 1. ABBREVIATION DICTIONARY  (mirrors seg_classf_abbrev_test.py)
# =====================================================================
ABBREV_DICT = {
    "id": "identifier", "pk": "primary", "fk": "foreign", "ref": "reference",
    "key": "key", "code": "code", "cust": "customer", "usr": "user",
    "acct": "account", "acc": "account", "dept": "department",
    "org": "organization", "emp": "employee", "mgr": "manager",
    "mngr": "manager", "adm": "admin", "cfg": "configuration",
    "conf": "configuration", "param": "parameter", "attr": "attribute",
    "meta": "metadata", "db": "database", "tbl": "table", "col": "column",
    "idx": "index", "seq": "sequence", "repo": "repository",
    "amt": "amount", "bal": "balance", "qty": "quantity",
    "txn": "transaction", "inv": "invoice", "ord": "order",
    "pay": "payment", "ship": "shipment", "addr": "address",
    "dest": "destination", "src": "source", "ts": "timestamp",
    "dt": "date", "tm": "time", "dob": "birthdate",
    "svc": "service", "api": "interface", "ctrl": "controller",
    "proc": "process", "exec": "execution",
    "stat": "status", "sts": "status", "grp": "group",
    "map": "mapping", "rel": "relation",
    "desc": "description", "msg": "message", "err": "error",
    "num": "number", "val": "value", "cnt": "count",
    "min": "minimum", "max": "maximum",
    "cat": "category", "prd": "product", "prj": "project",
    "loc": "location", "rgn": "region", "cty": "city",
    "phn": "phone", "tel": "telephone", "eml": "email",
    "pwd": "password", "auth": "authentication", "tok": "token",
    "sess": "session", "req": "request", "resp": "response",
    "hdr": "header", "bdy": "body", "len": "length",
    "sz": "size", "wt": "weight", "ht": "height",
    "wid": "width", "dep": "depth", "vol": "volume",
    "pct": "percent", "avg": "average", "tot": "total",
    "sub": "subtotal", "disc": "discount", "tax": "tax",
    "cur": "currency", "rt": "rate", "prc": "price",
    "rev": "revenue", "exp": "expense", "pnl": "profit",
    "stk": "stock", "sku": "sku", "wrh": "warehouse",
    "dlv": "delivery", "trk": "tracking", "rtn": "return",
    "cncl": "cancel", "appr": "approval", "rej": "reject",
    "lvl": "level", "pri": "priority", "sev": "severity",
    "ver": "version", "env": "environment", "inst": "instance",
    "cls": "class", "obj": "object", "fn": "function",
    "mth": "method", "lib": "library", "pkg": "package",
    "mod": "module", "cmp": "component", "wgt": "widget",
    "btn": "button", "lbl": "label", "img": "image",
    "txt": "text", "fmt": "format", "tmpl": "template",
    "rpt": "report", "ntf": "notification", "alrt": "alert",
    "schd": "schedule", "cal": "calendar", "evt": "event",
}

ABBREV_KEYS = list(ABBREV_DICT.keys())

# =====================================================================
# 2. FULL-WORD VOCABULARY  (schema / DB terms - no split needed)
# =====================================================================
FULL_WORDS = sorted(set([
    # --- common column name words ---
    "first", "last", "name", "date", "time", "price", "order", "value",
    "type", "code", "status", "state", "start", "end", "count", "total",
    "amount", "number", "index", "level", "group", "class", "title",
    "email", "phone", "image", "table", "field", "label", "color",
    "width", "height", "depth", "length", "weight", "point", "score",
    "month", "year", "week", "hour", "second", "minute", "quarter",
    "street", "city", "country", "region", "postal",
    "active", "create", "update", "delete", "insert", "remove",
    "credit", "debit", "balance", "budget", "salary", "profit",
    "source", "target", "parent", "child", "owner", "admin",
    "buyer", "seller", "sender", "receiver", "manager", "employee",
    "customer", "account", "address", "payment", "invoice", "product",
    "service", "project", "department", "description", "transaction",
    "configuration", "organization", "identifier", "reference",
    "minimum", "maximum", "average", "percent", "version", "comment",
    "message", "error", "flag", "mapping", "relation", "sequence",
    "process", "controller", "interface", "execution", "repository",
    "shipment", "destination", "timestamp", "birthdate", "metadata",
    # --- extended DB vocabulary ---
    "rate", "fee", "tax", "cost", "unit", "note", "batch",
    "line", "item", "plan", "role", "user", "mode", "size",
    "age", "gender", "birth", "death", "hire", "fire", "join",
    "leave", "open", "close", "begin", "finish", "complete",
    "approve", "reject", "submit", "cancel", "refund", "return",
    "ship", "deliver", "receive", "send", "post", "publish",
    "draft", "review", "archive", "restore", "merge", "split",
    "copy", "move", "link", "share", "lock", "unlock",
    "enable", "disable", "allow", "deny", "grant", "revoke",
    "stock", "store", "warehouse", "shelf", "bin", "rack",
    "vendor", "supplier", "partner", "client", "contact", "lead",
    "prospect", "campaign", "channel", "segment", "tier",
    "ticket", "issue", "task", "sprint", "release", "milestone",
    "domain", "schema", "column", "row", "record", "entry",
    "event", "action", "trigger", "alert", "log", "audit",
    "token", "session", "cookie", "cache", "queue", "topic",
    "header", "body", "footer", "page", "view", "form",
    "button", "icon", "badge", "tag", "category", "folder",
    "file", "path", "route", "endpoint", "method", "scope",
    "policy", "rule", "filter", "sort", "limit", "offset",
    "margin", "padding", "border", "radius", "opacity",
    "latitude", "longitude", "altitude", "distance",
    "temperature", "pressure", "humidity", "voltage", "current",
    "frequency", "duration", "interval", "timeout", "retry",
    "threshold", "capacity", "utilization", "throughput",
    "revenue", "expense", "income", "loss", "equity", "asset",
    "liability", "dividend", "interest", "principal", "premium",
    "discount", "coupon", "voucher", "reward", "bonus",
    "penalty", "fine", "surcharge", "markup", "markdown",
    "diagnosis", "symptom", "treatment", "prescription", "dosage",
    "patient", "doctor", "nurse", "clinic", "hospital", "ward",
    "course", "lesson", "module", "grade", "score", "exam",
    "student", "teacher", "professor", "dean", "campus",
    "flight", "route", "terminal", "gate", "seat", "cabin",
    "passenger", "crew", "pilot", "cargo", "freight",
]))

# =====================================================================
# 3. DOMAIN-SPECIFIC COLUMN PATTERNS
# =====================================================================
DOMAIN_COLUMNS = {
    "finance": [
        ("txn", "amount"), ("acct", "balance"), ("inv", "total"),
        ("pay", "date"), ("ord", "status"), ("cust", "name"),
        ("credit", "limit"), ("debit", "amount"), ("rev", "month"),
        ("exp", "category"), ("pnl", "quarter"), ("tax", "rate"),
        ("disc", "percent"), ("cur", "code"), ("fee", "type"),
    ],
    "hr": [
        ("emp", "name"), ("emp", "id"), ("dept", "code"),
        ("mgr", "id"), ("hire", "date"), ("sal", "amount"),
        ("leave", "balance"), ("perf", "score"), ("role", "level"),
        ("org", "unit"), ("pos", "title"), ("ben", "type"),
    ],
    "healthcare": [
        ("pat", "id"), ("doc", "name"), ("diag", "code"),
        ("presc", "date"), ("treat", "type"), ("ward", "num"),
        ("nurse", "id"), ("lab", "result"), ("vitals", "time"),
        ("allergy", "type"), ("insur", "plan"), ("claim", "status"),
    ],
    "ecommerce": [
        ("prd", "name"), ("prd", "price"), ("sku", "code"),
        ("cart", "total"), ("ship", "addr"), ("ship", "date"),
        ("dlv", "status"), ("rtn", "reason"), ("rev", "rating"),
        ("cat", "name"), ("brand", "id"), ("stock", "qty"),
    ],
    "logistics": [
        ("wrh", "code"), ("ship", "id"), ("trk", "number"),
        ("dest", "addr"), ("src", "addr"), ("dlv", "date"),
        ("freight", "cost"), ("cargo", "weight"), ("route", "id"),
        ("carrier", "name"), ("dock", "num"), ("load", "type"),
    ],
    "it_ops": [
        ("svc", "name"), ("svc", "status"), ("inst", "id"),
        ("env", "type"), ("cfg", "param"), ("log", "level"),
        ("alert", "sev"), ("evt", "time"), ("proc", "id"),
        ("sess", "token"), ("req", "count"), ("resp", "time"),
    ],
    "education": [
        ("student", "id"), ("course", "code"), ("grade", "point"),
        ("exam", "date"), ("enroll", "status"), ("prof", "name"),
        ("dept", "name"), ("sem", "year"), ("class", "size"),
        ("assign", "due"), ("attend", "count"), ("gpa", "val"),
    ],
    "real_estate": [
        ("prop", "id"), ("prop", "type"), ("addr", "line"),
        ("price", "list"), ("sqft", "total"), ("bed", "count"),
        ("bath", "count"), ("lot", "size"), ("build", "year"),
        ("agent", "name"), ("mls", "num"), ("tax", "assess"),
    ],
    "telecom": [
        ("sub", "id"), ("plan", "type"), ("data", "usage"),
        ("call", "duration"), ("sms", "count"), ("bill", "amount"),
        ("sim", "num"), ("net", "type"), ("roam", "status"),
        ("tower", "id"), ("signal", "strength"), ("band", "freq"),
    ],
    "insurance": [
        ("pol", "id"), ("pol", "type"), ("prem", "amount"),
        ("claim", "date"), ("claim", "status"), ("ben", "name"),
        ("cov", "type"), ("ded", "amount"), ("agent", "code"),
        ("risk", "score"), ("und", "decision"), ("renew", "date"),
    ],
}

# =====================================================================
# HELPERS
# =====================================================================

def make_labels(parts: List[str]) -> List[int]:
    """
    Given sub-token parts, produce per-character boundary labels.
    Label 1 at the last character of each sub-token.
    For single-word (no split), all labels are 0.
    """
    if len(parts) <= 1:
        text = parts[0] if parts else ""
        return [0] * len(text)

    labels = []
    for i, part in enumerate(parts):
        for j, _ in enumerate(part):
            if j == len(part) - 1 and i < len(parts) - 1:
                labels.append(1)  # boundary
            else:
                labels.append(0)
    return labels


def make_example(parts: List[str]) -> dict:
    text = "".join(parts).lower()
    parts_lower = [p.lower() for p in parts]
    labels = make_labels(parts_lower)
    assert len(text) == len(labels), f"Length mismatch: {text!r} vs {labels}"
    return {"text": text, "labels": labels, "parts": parts_lower}


# =====================================================================
# SOURCE 1: Abbreviation dictionary combos
# =====================================================================

def generate_abbrev_combos(target: int = 12000) -> List[dict]:
    """Combine 2-4 abbreviations from the dictionary."""
    examples = []

    keys = ABBREV_KEYS.copy()

    # 2-part combos: sample from all pairs
    pairs = list(itertools.permutations(keys, 2))
    random.shuffle(pairs)
    for a, b in pairs[:target // 2]:
        ex = make_example([a, b])
        if len(ex["text"]) <= 40:
            examples.append(ex)

    # 3-part combos
    for _ in range(target // 3):
        combo = random.sample(keys, 3)
        ex = make_example(combo)
        if len(ex["text"]) <= 40:
            examples.append(ex)

    # 4-part combos (rarer)
    for _ in range(target // 6):
        combo = random.sample(keys, 4)
        ex = make_example(combo)
        if len(ex["text"]) <= 40:
            examples.append(ex)

    print(f"  [Source 1] Abbreviation combos: {len(examples)}")
    return examples


# =====================================================================
# SOURCE 2: Full-word concatenations
# =====================================================================

def generate_fullword_combos(target: int = 8000) -> List[dict]:
    """Concatenate 2-3 full English words."""
    examples = []
    words = FULL_WORDS.copy()

    # 2-word
    for _ in range(target * 2 // 3):
        a, b = random.sample(words, 2)
        ex = make_example([a, b])
        if len(ex["text"]) <= 40:
            examples.append(ex)

    # 3-word
    for _ in range(target // 3):
        combo = random.sample(words, 3)
        ex = make_example(combo)
        if len(ex["text"]) <= 40:
            examples.append(ex)

    print(f"  [Source 2] Full-word combos: {len(examples)}")
    return examples


# =====================================================================
# SOURCE 3: Mixed abbreviation + full-word
# =====================================================================

def generate_mixed_combos(target: int = 10000) -> List[dict]:
    """Mix abbreviations with full words."""
    examples = []
    keys = ABBREV_KEYS.copy()
    words = FULL_WORDS.copy()

    # abbrev + full
    for _ in range(target // 3):
        a = random.choice(keys)
        b = random.choice(words)
        ex = make_example([a, b])
        if len(ex["text"]) <= 40:
            examples.append(ex)

    # full + abbrev
    for _ in range(target // 3):
        a = random.choice(words)
        b = random.choice(keys)
        ex = make_example([a, b])
        if len(ex["text"]) <= 40:
            examples.append(ex)

    # abbrev + full + abbrev
    for _ in range(target // 6):
        a = random.choice(keys)
        b = random.choice(words)
        c = random.choice(keys)
        ex = make_example([a, b, c])
        if len(ex["text"]) <= 40:
            examples.append(ex)

    # full + abbrev + full
    for _ in range(target // 6):
        a = random.choice(words)
        b = random.choice(keys)
        c = random.choice(words)
        ex = make_example([a, b, c])
        if len(ex["text"]) <= 40:
            examples.append(ex)

    print(f"  [Source 3] Mixed combos: {len(examples)}")
    return examples


# =====================================================================
# SOURCE 4: Negative examples (single tokens, no split)
# =====================================================================

def generate_negatives(target: int = 20000) -> List[dict]:
    """
    Single words that must NOT be split.
    Critical: ~40% of all data should be negatives.
    We generate many variants to survive deduplication.
    """
    examples = []

    # Full words from vocabulary
    for w in FULL_WORDS:
        examples.append(make_example([w]))

    # Expanded forms of abbreviations (these are full words too)
    for v in set(ABBREV_DICT.values()):
        for word in v.split():
            if len(word) >= 3:
                examples.append(make_example([word]))

    # The abbreviations themselves as standalone (no split)
    for k in ABBREV_KEYS:
        examples.append(make_example([k]))

    # Extended common English words that appear in DB schemas
    _extra_words = [
        "primary", "foreign", "unique", "constraint", "nullable", "default",
        "cascade", "restrict", "trigger", "procedure", "function", "view",
        "materialized", "partition", "cluster", "shard", "replica", "backup",
        "snapshot", "migration", "rollback", "commit", "transaction",
        "isolation", "deadlock", "concurrent", "parallel", "sequential",
        "aggregate", "calculate", "compute", "transform", "normalize",
        "validate", "sanitize", "encrypt", "decrypt", "compress", "decompress",
        "serialize", "deserialize", "marshal", "unmarshal", "encode", "decode",
        "authenticate", "authorize", "permission", "privilege", "credential",
        "certificate", "signature", "checksum", "digest", "algorithm",
        "protocol", "handshake", "negotiate", "establish", "terminate",
        "initialize", "configure", "provision", "deploy", "monitor",
        "dashboard", "analytics", "telemetry", "benchmark", "performance",
        "latency", "bandwidth", "throughput", "availability", "reliability",
        "scalability", "elasticity", "redundancy", "failover", "resilience",
        "warehouse", "inventory", "logistics", "procurement", "fulfillment",
        "customer", "supplier", "manufacturer", "distributor", "retailer",
        "wholesale", "marketplace", "storefront", "checkout", "shipping",
        "tracking", "delivery", "dispatch", "manifest", "consignment",
        "physician", "pharmacist", "therapist", "counselor", "specialist",
        "outpatient", "inpatient", "emergency", "intensive", "surgical",
        "diagnostic", "radiology", "pathology", "neurology", "cardiology",
        "enrollment", "registration", "attendance", "curriculum", "syllabus",
        "assignment", "submission", "evaluation", "assessment", "transcript",
        "scholarship", "fellowship", "internship", "apprentice", "graduate",
        "undergraduate", "doctoral", "faculty", "semester", "academic",
        "portfolio", "investment", "securities", "commodity", "derivative",
        "exchange", "settlement", "clearance", "custodian", "fiduciary",
        "compliance", "regulatory", "statutory", "jurisdiction", "arbitration",
        "subscriber", "membership", "subscription", "renewal", "cancellation",
        "activation", "deactivation", "suspension", "termination", "migration",
        "bandwidth", "throughput", "congestion", "allocation", "provisioning",
        "temperature", "humidity", "pressure", "velocity", "acceleration",
        "coordinates", "elevation", "trajectory", "navigation", "satellite",
        "scheduled", "estimated", "actual", "planned", "projected",
        "approved", "pending", "rejected", "completed", "cancelled",
        "standard", "premium", "enterprise", "professional", "starter",
    ]
    for w in _extra_words:
        examples.append(make_example([w.lower()]))

    # Generate numeric-suffixed variants: word + 1-2 digit number (still single token)
    base_pool = FULL_WORDS + _extra_words
    for _ in range(target // 4):
        w = random.choice(base_pool).lower()
        suffix = str(random.randint(1, 99))
        examples.append(make_example([w + suffix]))

    # Generate prefix-variants: x + word, v + word (still single token)
    for _ in range(target // 4):
        w = random.choice(base_pool).lower()
        prefix = random.choice(["x", "v", "i", "n", "p", "s", "t"])
        examples.append(make_example([prefix + w]))

    # Generate length-varied substrings (teach model that partial words aren't split)
    for _ in range(target // 5):
        w = random.choice(base_pool).lower()
        if len(w) >= 5:
            start = random.randint(0, len(w) - 4)
            end = random.randint(start + 3, len(w))
            examples.append(make_example([w[start:end]]))

    # Deduplicate
    seen = set()
    unique = []
    for ex in examples:
        key = ex["text"]
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    print(f"  [Source 4] Negatives (no-split): {len(unique)}")
    return unique


# =====================================================================
# SOURCE 5: Domain-specific column names
# =====================================================================

def generate_domain_examples(target: int = 5000) -> List[dict]:
    """Generate realistic column-name concatenations from domain patterns."""
    examples = []

    all_patterns = []
    for domain, patterns in DOMAIN_COLUMNS.items():
        all_patterns.extend(patterns)

    # Direct patterns
    for parts in all_patterns:
        examples.append(make_example(list(parts)))

    # Combine patterns: pattern + extra abbrev
    for _ in range(target // 3):
        parts = list(random.choice(all_patterns))
        extra = random.choice(ABBREV_KEYS)
        if random.random() < 0.5:
            parts.append(extra)
        else:
            parts.insert(0, extra)
        ex = make_example(parts)
        if len(ex["text"]) <= 40:
            examples.append(ex)

    # Combine two domain patterns
    for _ in range(target // 3):
        p1 = random.choice(all_patterns)
        p2 = random.choice(all_patterns)
        # Take first token from p1, second from p2
        parts = [p1[0], p2[1]]
        ex = make_example(list(parts))
        if len(ex["text"]) <= 40:
            examples.append(ex)

    # Combine pattern with full word
    for _ in range(target // 3):
        p = random.choice(all_patterns)
        w = random.choice(FULL_WORDS)
        parts = list(p) + [w]
        ex = make_example(parts)
        if len(ex["text"]) <= 40:
            examples.append(ex)

    print(f"  [Source 5] Domain-specific: {len(examples)}")
    return examples


# =====================================================================
# SOURCE 6: Augmentation / perturbation
# =====================================================================

def augment_examples(base_examples: List[dict], target: int = 5000) -> List[dict]:
    """Apply perturbations to existing examples."""
    augmented = []

    for _ in range(target):
        ex = random.choice(base_examples)
        parts = list(ex["parts"])
        text = ex["text"]

        aug_type = random.choice([
            "numeric_suffix", "prefix_noise", "single_char_prefix",
            "duplicate", "reverse_parts",
        ])

        if aug_type == "numeric_suffix":
            # Add numeric suffix: custid1, acctbal02
            suffix = random.choice(["1", "2", "01", "02", "99", "00"])
            new_parts = parts[:-1] + [parts[-1] + suffix]
            new_ex = make_example(new_parts)
            if len(new_ex["text"]) <= 40:
                augmented.append(new_ex)

        elif aug_type == "prefix_noise":
            # Add common prefix: x_custid, src_acctbal
            prefix = random.choice(["x", "src", "tgt", "old", "new", "tmp", "raw"])
            new_parts = [prefix] + parts
            new_ex = make_example(new_parts)
            if len(new_ex["text"]) <= 40:
                augmented.append(new_ex)

        elif aug_type == "single_char_prefix":
            # Single char prefix: acustid, bcustname
            ch = random.choice("abcdefghijklmnopqrstuvwxyz")
            # This becomes a negative -- don't split the prefix char
            new_parts = [ch + parts[0]] + parts[1:] if len(parts) > 1 else [ch + parts[0]]
            # Actually, the prefix merges with first token for negative behavior
            # Just prepend as its own token
            new_parts = [ch] + parts
            new_ex = make_example(new_parts)
            if len(new_ex["text"]) <= 40:
                augmented.append(new_ex)

        elif aug_type == "duplicate":
            # Repeat a part: custcustid
            if parts:
                dup = random.choice(parts)
                pos = random.randint(0, len(parts))
                new_parts = parts[:pos] + [dup] + parts[pos:]
                new_ex = make_example(new_parts)
                if len(new_ex["text"]) <= 40:
                    augmented.append(new_ex)

        elif aug_type == "reverse_parts":
            # Reverse order: idcust instead of custid
            new_parts = list(reversed(parts))
            new_ex = make_example(new_parts)
            if len(new_ex["text"]) <= 40:
                augmented.append(new_ex)

    print(f"  [Source 6] Augmented: {len(augmented)}")
    return augmented


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 60)
    print("  Segmentation Training Data Generator")
    print("=" * 60)

    print("\nGenerating data from 6 sources...")

    src1 = generate_abbrev_combos(12000)
    src2 = generate_fullword_combos(8000)
    src3 = generate_mixed_combos(10000)
    src4 = generate_negatives(20000)
    src5 = generate_domain_examples(5000)

    # Augment a sample of src1-3 and src5
    base_for_aug = src1 + src2[:2000] + src3[:2000] + src5[:1000]
    src6 = augment_examples(base_for_aug, 5000)

    all_data = src1 + src2 + src3 + src4 + src5 + src6

    # Deduplicate by text
    seen = set()
    unique = []
    for ex in all_data:
        if ex["text"] not in seen:
            seen.add(ex["text"])
            unique.append(ex)

    random.shuffle(unique)

    total = len(unique)
    split_idx = int(total * 0.85)
    train_data = unique[:split_idx]
    val_data = unique[split_idx:]

    # Compute stats
    neg_count = sum(1 for ex in unique if sum(ex["labels"]) == 0)
    two_part = sum(1 for ex in unique if sum(ex["labels"]) == 1)
    three_part = sum(1 for ex in unique if sum(ex["labels"]) == 2)
    four_plus = sum(1 for ex in unique if sum(ex["labels"]) >= 3)

    print(f"\n--- Dataset Statistics ---")
    print(f"  Total unique examples: {total}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"  Negatives (no split): {neg_count}  ({neg_count/total*100:.1f}%)")
    print(f"  2-part splits:        {two_part}  ({two_part/total*100:.1f}%)")
    print(f"  3-part splits:        {three_part}  ({three_part/total*100:.1f}%)")
    print(f"  4+ part splits:       {four_plus}  ({four_plus/total*100:.1f}%)")

    # Write
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "seg_train.jsonl")
    val_path = os.path.join(OUTPUT_DIR, "seg_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved:")
    print(f"  {train_path}  ({len(train_data)} examples)")
    print(f"  {val_path}  ({len(val_data)} examples)")
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
