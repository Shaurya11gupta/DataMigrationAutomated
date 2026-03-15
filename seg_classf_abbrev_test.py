import re
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from functools import lru_cache

# =====================================================
# DEVICE
# =====================================================
# =====================================================
# ABBREVIATION DICTIONARY (HIGH CONFIDENCE FIRST)
# =====================================================

ABBREV_DICT = {

    # ---- identity / keys ----
    "id": "identifier",
    "pk": "primary",
    "fk": "foreign",
    "ref": "reference",

    # ---- entities ----
    "cust": "customer",
    "usr": "user",
    "acct": "account",
    "acc": "account",
    "dept": "department",
    "org": "organization",
    "emp": "employee",
    "mgr": "manager",
    "mngr": "manager",
    "adm": "admin",

    # ---- technical ----
    "cfg": "configuration",
    "conf": "configuration",
    "param": "parameter",
    "attr": "attribute",
    "meta": "metadata",

    # ---- data / storage ----
    "db": "database",
    "tbl": "table",
    "col": "column",
    "idx": "index",
    "seq": "sequence",
    "repo": "repository",

    # ---- finance ----
    "amt": "amount",
    "bal": "balance",
    "qty": "quantity",
    "txn": "transaction",
    "inv": "invoice",
    "ord": "order",
    "pay": "payment",
    "prc": "price",
    "sal": "salary",

    # ---- logistics ----
    "ship": "shipment",
    "addr": "address",
    "dest": "destination",
    "src": "source",

    # ---- time ----
    "ts": "timestamp",
    "dt": "date",
    "tm": "time",
    "dob": "birthdate",

    # ---- system ----
    "svc": "service",
    "api": "interface",
    "ctrl": "controller",
    "proc": "process",
    "exec": "execution",
    "cfgmgr": "configuration manager",

    # ---- status / flags ----
    "stat": "status",
    "sts": "status",
    "flg": "flag",
    "typ": "type",

    # ---- grouping ----
    "grp": "group",
    "grpmap": "group mapping",
    "map": "mapping",
    "rel": "relation",

    # ---- misc ----
    "desc": "description",
    "msg": "message",
    "err": "error",
    "num": "number",
    "val": "value",
    "cnt": "count",
    "min":"minimum",
    "max":"maximum",
    #--currency--
    "usd": "us dollar",
    "us$": "us dollar",
    "$": "us dollar",

    "eur": "euro",
    "€": "euro",

    "inr": "indian rupee",
    "₹": "indian rupee",
    "rs": "indian rupee",

    "gbp": "british pound sterling",
    "£": "british pound sterling",

    "jpy": "japanese yen",
    "¥": "japanese yen",

    "cny": "chinese yuan",
    "rmb": "chinese yuan",

    "aud": "australian dollar",
    "a$": "australian dollar",

    "cad": "canadian dollar",
    "c$": "canadian dollar",

    "chf": "swiss franc",

    "sgd": "singapore dollar",
    "s$": "singapore dollar",

    "hkd": "hong kong dollar",
    "hk$": "hong kong dollar",

    "nzd": "new zealand dollar",

    "sek": "swedish krona",
    "nok": "norwegian krone",
    "dkk": "danish krone",

    "zar": "south african rand",

    "rub": "russian ruble",

    "brl": "brazilian real",

    "mxn": "mexican peso",

    "krw": "south korean won",
    "₩": "south korean won",

    "try": "turkish lira",

    "aed": "uae dirham",

    "sar": "saudi riyal",

    "qar": "qatari riyal",

    "kwd": "kuwaiti dinar",

    "bhd": "bahraini dinar",

    "omr": "omani rial",

    "pkr": "pakistani rupee",

    "bdt": "bangladeshi taka",

    "lkr": "sri lankan rupee",

    "thb": "thai baht",
    "฿": "thai baht",

    "idr": "indonesian rupiah",

    "myr": "malaysian ringgit",

    "php": "philippine peso",

    "vnd": "vietnamese dong",

    "pln": "polish zloty",

    "czk": "czech koruna",

    "huf": "hungarian forint",

    "ils": "israeli shekel",
    "₪": "israeli shekel"
}

# =====================================================
# NEGATIVE / ANTONYM TOKEN ENGINE
# =====================================================

# =====================================================
# ADVANCED NAME CONFLICT DETECTORS
# =====================================================

# ---------------------------
# 1️⃣ Directional pairs
# ---------------------------

DIRECTIONAL_PAIRS = [
    ("min","max"),
    ("minimum","maximum"),
    ("start","end"),
    ("begin","end"),
    ("from","to"),
    ("low","high"),
    ("lower","upper"),
    ("first","last"),
    ("left","right"),
    ("top","bottom"),
    ("front","back"),
    ("before","after"),
    ("previous","next"),
    ("prev","next"),
    ("above","below"),
    ("ascending","descending"),
    ("asc","desc"),
    ("forward","backward"),
    ("inner","outer"),
    ("internal","external"),
    ("north","south"),
    ("east","west"),
    ("head","tail"),
    ("prefix","suffix"),
    ("earliest","latest"),
    ("newest","oldest"),
    ("youngest","eldest"),
    ("smallest","largest"),
    ("shortest","longest"),
    ("initial","final"),
]

# ---------------------------
# 2️⃣ Units
# ---------------------------

UNIT_GROUPS = [
    {"usd","inr","eur","gbp","jpy","aud","cad","chf","sgd","hkd","nzd",
     "sek","nok","dkk","zar","rub","brl","mxn","krw","try","aed","sar",
     "us dollar","indian rupee","euro","british pound sterling","japanese yen",
     "australian dollar","canadian dollar","swiss franc","singapore dollar",
     "hong kong dollar","new zealand dollar","dollar","pound","rupee","yen","yuan"},
    {"kg","kilogram","gram","g","lb","pound weight","oz","ounce","ton","tonne"},
    {"km","kilometer","mile","mi","meter","m","ft","foot","inch","in","yard","yd","cm","mm"},
    {"sec","second","ms","millisecond","minute","min","hour","hr","day","week","month","year"},
    {"c","celsius","f","fahrenheit","kelvin","k"},
    {"byte","kb","mb","gb","tb","pb","kilobyte","megabyte","gigabyte","terabyte"},
    {"watt","kw","kilowatt","mw","megawatt","hp","horsepower"},
    {"liter","litre","gallon","gal","ml","milliliter","pint","quart","cup"},
]

# ---------------------------
# 3️⃣ Temporal state words
# ---------------------------

TEMPORAL_STATE = [
    ("created","deleted"),
    ("created","updated"),
    ("inserted","removed"),
    ("active","inactive"),
    ("open","closed"),
    ("enabled","disabled"),
    ("started","stopped"),
    ("started","finished"),
    ("started","completed"),
    ("pending","completed"),
    ("pending","cancelled"),
    ("approved","rejected"),
    ("locked","unlocked"),
    ("subscribed","unsubscribed"),
    ("enrolled","withdrawn"),
    ("hired","terminated"),
    ("born","deceased"),
    ("birth","death"),
    ("login","logout"),
    ("checkin","checkout"),
    ("entry","exit"),
]

# ---------------------------
# 4️⃣ Role pairs
# ---------------------------

ROLE_PAIRS = [
    ("buyer","seller"),
    ("sender","receiver"),
    ("source","target"),
    ("parent","child"),
    ("input","output"),
    ("credit","debit"),
    ("employer","employee"),
    ("lender","borrower"),
    ("landlord","tenant"),
    ("vendor","customer"),
    ("supplier","consumer"),
    ("author","reader"),
    ("teacher","student"),
    ("doctor","patient"),
    ("server","client"),
    ("host","guest"),
    ("master","slave"),
    ("primary","secondary"),
    ("principal","agent"),
    ("domestic","international"),
    ("wholesale","retail"),
    ("inbound","outbound"),
]

# =====================================================
# Build lookup maps
# =====================================================

def build_pair_map(pairs):
    m={}
    for a,b in pairs:
        m.setdefault(a,set()).add(b)
        m.setdefault(b,set()).add(a)
    return m

DIR_MAP = build_pair_map(DIRECTIONAL_PAIRS)
ROLE_MAP = build_pair_map(ROLE_PAIRS)
TEMP_MAP = build_pair_map(TEMPORAL_STATE)

UNIT_LOOKUP = {}
for grp in UNIT_GROUPS:
    for u in grp:
        UNIT_LOOKUP[u] = grp


# =====================================================
# Conflict detectors
# =====================================================

def _find_pair_conflicts(A, B, pair_map):
    """Find all conflicting token pairs between A and B using a pair map.
    Returns list of (token_a, token_b) pairs that conflict."""
    sA = set(A); sB = set(B)
    found = []
    for t in sA:
        if t in pair_map:
            for opp in pair_map[t] & sB:
                found.append((t, opp))
    return found


def directional_conflict(A, B):
    pairs = _find_pair_conflicts(A, B, DIR_MAP)
    return (1.0, pairs) if pairs else (0.0, [])


def role_conflict(A, B):
    pairs = _find_pair_conflicts(A, B, ROLE_MAP)
    return (1.0, pairs) if pairs else (0.0, [])


def temporal_conflict(A, B):
    pairs = _find_pair_conflicts(A, B, TEMP_MAP)
    return (1.0, pairs) if pairs else (0.0, [])


def unit_conflict(A, B):
    sA = set(A); sB = set(B)
    uA = [u for u in sA if u in UNIT_LOOKUP]
    uB = [u for u in sB if u in UNIT_LOOKUP]

    if not uA or not uB:
        return 0.0, []

    found = []
    for a in uA:
        for b in uB:
            if a != b and UNIT_LOOKUP[a] == UNIT_LOOKUP[b]:
                # Same group but different units -> conflict
                found.append((a, b))
            elif UNIT_LOOKUP.get(a) != UNIT_LOOKUP.get(b):
                # Different groups entirely -> conflict
                found.append((a, b))

    return (1.0, found) if found else (0.0, [])


# =====================================================
# Combined penalty score
# =====================================================

def semantic_conflict_penalty(tokensA, tokensB):
    """Compute total penalty and return (penalty_score, conflict_details)."""

    dir_score, dir_pairs = directional_conflict(tokensA, tokensB)
    role_score, role_pairs = role_conflict(tokensA, tokensB)
    temp_score, temp_pairs = temporal_conflict(tokensA, tokensB)
    unit_score, unit_pairs = unit_conflict(tokensA, tokensB)

    penalties = [
        dir_score  * 0.45,
        role_score * 0.40,
        temp_score * 0.35,
        unit_score * 0.50,
    ]

    total_penalty = min(1.0, sum(penalties))

    details = []
    seen_keys = set()

    def _add(ctype, pairs, weight):
        for tok_a, tok_b in pairs:
            key = (ctype, tok_a, tok_b)
            if key not in seen_keys:
                seen_keys.add(key)
                details.append({"type": ctype, "token_a": tok_a, "token_b": tok_b, "weight": weight})

    _add("directional", dir_pairs, 0.45)
    _add("role", role_pairs, 0.40)
    _add("temporal", temp_pairs, 0.35)
    _add("unit", unit_pairs, 0.50)

    return total_penalty, details


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# ---------------- SEGMENTATION MODEL -----------------
# (exact structure from your file)
# =====================================================

SEG_MODEL_PATH = "segmentation_model_final.pt"
seg_ckpt = torch.load(SEG_MODEL_PATH, map_location=DEVICE)

seg_c2i = seg_ckpt["c2i"]
seg_i2c = {v:k for k,v in seg_c2i.items()}
SEG_VOCAB = len(seg_c2i)
SEG_MAX_LEN = seg_ckpt.get("max_len", 40)

class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(SEG_VOCAB,128,padding_idx=0)
        self.conv = nn.Conv1d(128,128,3,padding=1)

        enc = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.15,
            batch_first=True
        )

        self.tr = nn.TransformerEncoder(enc,3)
        self.lstm = nn.LSTM(128,128,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(256,2)

    def forward(self,x):
        e=self.emb(x).permute(0,2,1)
        h=F.relu(self.conv(e)).permute(0,2,1)
        h=self.tr(h)
        h,_=self.lstm(h)
        return self.fc(h)

seg_model = SegModel().to(DEVICE)
seg_model.load_state_dict(seg_ckpt["model"])
seg_model.eval()

def seg_encode(s):
    s = s.lower()[:SEG_MAX_LEN]
    ids = [seg_c2i.get(c,1) for c in s]
    return torch.tensor(ids).unsqueeze(0).to(DEVICE), s

# ── Known full words that should NEVER be segmented ──
_FULL_WORD_WHITELIST = {
    "first", "last", "name", "date", "time", "price", "order", "value",
    "type", "code", "status", "state", "start", "end", "count", "total",
    "amount", "number", "index", "level", "group", "class", "title",
    "email", "phone", "image", "table", "field", "label", "color",
    "width", "height", "depth", "length", "weight", "point", "score",
    "month", "year", "week", "hour", "second", "minute", "quarter",
    "street", "city", "country", "region", "state", "postal",
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
    "schema", "sync", "async", "batch", "cache", "proxy", "queue",
    "token", "event", "alert", "debug", "trace", "model", "scope",
    "store", "trade", "share", "grant", "shift", "leave", "skill",
    "grade", "badge", "bonus", "extra", "gross", "range", "delta",
    "ratio", "scale", "stock", "claim", "lease", "node", "link",
    "role", "tier", "slot", "span", "unit", "zone", "area", "file",
    "line", "page", "view", "rule", "test", "plan", "task", "note",
    "list", "item", "send", "read", "load", "push", "pull", "dump",
    "host", "port", "path", "mode", "step", "lock", "seed", "root",
    "hash", "mask", "salt", "text", "null", "true", "void", "user",
    "rate", "cost", "age", "key", "tax", "fee", "day", "tag", "url",
    "row", "log", "pin", "bin", "bit", "set", "map", "job", "hub",
    "run", "mobile", "office", "branch", "vendor", "client", "member",
    "worker", "doctor", "period", "domain", "object", "record",
    "format", "output", "report", "metric", "signal", "thread",
    "cursor", "socket", "stream", "buffer", "filter", "layout",
    "database", "discount", "document", "download", "exchange",
    "feedback", "filename", "function", "hardware", "identity",
    "instance", "interval", "language", "location", "material",
    "operator", "ordering", "overview", "password", "platform",
    "position", "previous", "priority", "progress", "property",
    "protocol", "provider", "purchase", "quantity", "question",
    "recovery", "register", "relation", "required", "resource",
    "response", "schedule", "security", "software", "standard",
    "strategy", "supplier", "template", "terminal", "timeline",
    "tracking", "transfer", "variable", "warranty", "workflow",
}

# Minimum sub-token length: if any part is shorter, reject the split
_MIN_SUBTOKEN_LEN = 2

# Boundary confidence threshold (higher = more conservative)
_SEG_THRESHOLD = 0.65


@lru_cache(maxsize=50000)
def segment_token(token):

    # Fast path: known full words should never be split
    if token.lower() in _FULL_WORD_WHITELIST:
        return (token.lower(),)

    x, raw = seg_encode(token)

    with torch.no_grad():
        out = seg_model(x)[0]
        probs = torch.softmax(out, dim=1)[:,1].cpu().tolist()

    cuts = [i for i,p in enumerate(probs) if p > _SEG_THRESHOLD]

    parts=[]
    start=0
    for c in cuts:
        parts.append(raw[start:c+1])
        start=c+1

    if start < len(raw):
        parts.append(raw[start:])

    # Guard: reject split if any sub-token is too short
    if len(parts) > 1 and any(len(p) < _MIN_SUBTOKEN_LEN for p in parts):
        return (token.lower(),)

    return tuple(parts) if len(parts)>1 else (token.lower(),)


print("[OK] segmentation model ready")

# =====================================================
# ---------------- CLASSIFIER MODEL -------------------
# =====================================================

CLF_MODEL_PATH = "gatekeeper_universal_enhanced.pth"
VOCAB_PATH = "gatekeeper_vocab.json"

char_to_int = json.load(open(VOCAB_PATH))
PAD = char_to_int["<pad>"]
CLF_VOCAB = len(char_to_int)

class UniversalGatekeeper(nn.Module):

    def __init__(self, vocab, hidden=128):
        super().__init__()

        self.emb = nn.Embedding(vocab, hidden, padding_idx=PAD)

        self.conv2 = nn.Conv1d(hidden, hidden, 2, padding=1)
        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=1)

        self.gru = nn.GRU(hidden*3, hidden, batch_first=True, bidirectional=True)

        # ✅ EXACT MATCH TO TRAINING
        self.head = nn.Sequential(
            nn.Linear(hidden*2, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64,1)
        )

    def forward(self,x,lengths):

        e = self.emb(x)
        c = e.permute(0,2,1)

        f2 = F.relu(self.conv2(c))[:,:,:x.size(1)]
        f3 = F.relu(self.conv3(c))[:,:,:x.size(1)]

        feat = torch.cat([
            e,
            f2.permute(0,2,1),
            f3.permute(0,2,1)
        ], dim=2)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            feat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _,h = self.gru(packed)

        h = torch.cat([h[-2],h[-1]],dim=1)

        return self.head(h)


clf_model = UniversalGatekeeper(CLF_VOCAB).to(DEVICE)
clf_model.load_state_dict(torch.load(CLF_MODEL_PATH, map_location=DEVICE))
clf_model.eval()

def clf_encode(word):
    return torch.tensor(
        [char_to_int.get(c.lower(), PAD) for c in word],
        dtype=torch.long
    )

@lru_cache(maxsize=50000)
def should_expand(word):

    seq = clf_encode(word)
    lengths = torch.tensor([len(seq)])
    padded = seq.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = clf_model(padded, lengths)
        prob = torch.sigmoid(logit)[0].item()

    return prob > 0.4, prob


print("[OK] classifier model ready")

# =====================================================
# ---------------- EXPANSION MODEL --------------------
# =====================================================

EXP_MODEL_PATH = "abbrev_expander_finetuned.pt"
HAS_EXPANDER = False

class PosEnc(nn.Module):
    def __init__(self, d, maxlen=64):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000) / d))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self,x):
        return x + self.pe[:,:x.size(1)]

class Expander(nn.Module):
    def __init__(self, vocab_size, pad_idx=0):
        super().__init__()
        D=256
        self.emb = nn.Embedding(vocab_size,D,padding_idx=pad_idx)
        self.pos = PosEnc(D)
        self.conv = nn.Conv1d(D,D,3,padding=1)

        self.tr = nn.Transformer(
            d_model=D,nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=512,
            batch_first=True
        )

        self.fc = nn.Linear(D,vocab_size)

    def causal_mask(self,n,dev):
        return torch.triu(torch.ones(n,n,device=dev),1).bool()

    def forward(self,src,tgt):
        se=self.pos(self.emb(src))
        se=self.conv(se.permute(0,2,1)).permute(0,2,1)
        te=self.pos(self.emb(tgt))

        out=self.tr(
            se,te,
            tgt_mask=self.causal_mask(tgt.size(1),src.device),
            src_key_padding_mask=(src==PAD_T),
            tgt_key_padding_mask=(tgt==PAD_T),
            memory_key_padding_mask=(src==PAD_T)
        )
        return self.fc(out)


# Defaults used when expander model is not available
PAD_T, SOS, EOS, UNK = range(4)
stoi = {}
itos = {}
exp_model = None

import os as _os
if _os.path.exists(EXP_MODEL_PATH):
    try:
        exp_ckpt = torch.load(EXP_MODEL_PATH, map_location=DEVICE)
        stoi = exp_ckpt["stoi"]
        itos = exp_ckpt["itos"]
        EXP_VOCAB = len(itos)

        exp_model = Expander(EXP_VOCAB, pad_idx=PAD_T).to(DEVICE)
        exp_model.load_state_dict(exp_ckpt["model"])
        exp_model.eval()
        HAS_EXPANDER = True
        print("[OK] expansion model ready")
    except Exception as _e:
        print(f"[WARN] expansion model failed to load: {_e}")
        exp_model = None
else:
    print("[WARN] expansion model not found (abbrev_expander_finetuned.pt) - using dictionary only")


def exp_encode(text,max_len=16):
    ids=[stoi.get(c,UNK) for c in text.lower()][:max_len]
    return torch.tensor(ids).unsqueeze(0).to(DEVICE)

@lru_cache(maxsize=50000)
@torch.no_grad()
def expand_word(word):
    if exp_model is None:
        return word

    src = exp_encode(word)
    tgt = torch.tensor([[SOS]],device=DEVICE)

    for _ in range(32):
        out = exp_model(src,tgt)
        nxt = out[:,-1].argmax(dim=1,keepdim=True)
        tgt = torch.cat([tgt,nxt],dim=1)
        if nxt.item()==EOS:
            break

    out=[]
    for i in tgt[0,1:].tolist():
        if i==EOS: break
        ch=itos[i]
        if ch not in ["<pad>","<sos>","<eos>"]:
            out.append(ch)

    return "".join(out) or word

# =====================================================
# REGEX SPLIT
# =====================================================

CAMEL = re.compile(r'[A-Z]?[a-z]+|[A-Z]+|\d+')

def regex_split(name):
    name = name.replace("_"," ").replace("-"," ")
    out=[]
    for part in name.split():
        out.extend(CAMEL.findall(part))
    return out

# =====================================================
# FULL PIPELINE
# =====================================================
def expand_with_dictionary_then_model(token):

    t = token.lower()

    # dictionary
    if t in ABBREV_DICT:
        return ABBREV_DICT[t], "dict", 1.0

    # Skip classifier for known full English words -- they never need expansion
    if t in _FULL_WORD_WHITELIST:
        return t, "none", 0.0

    flag, prob = should_expand(t)

    if flag:
        out = expand_word(t)
        if len(out) >= 3 and out != t:
            return out, "model", prob

    return t, "none", prob

@lru_cache(maxsize=20000)
def process_column(name):

    tokens = regex_split(name)

    seg_tokens=[]
    for t in tokens:
        seg_tokens.extend(segment_token(t))

    final=[]
    sources={}
    probs={}

    for t in seg_tokens:
        out,src,p = expand_with_dictionary_then_model(t)
        final.append(out)
        sources[t]=src
        probs[t]=p

    return tuple(final), sources, probs



# =====================================================
# CONTEXT SIMILARITY
# =====================================================

from sentence_transformers import SentenceTransformer
embed = SentenceTransformer("all-MiniLM-L6-v2")

@lru_cache(maxsize=50000)
def embed_cached(text):
    return embed.encode(text, normalize_embeddings=True)


def name_similarity(a,b):

    t0 = time.time()

    tokensA, srcA, probA = process_column(a)
    tokensB, srcB, probB = process_column(b)

    phraseA = " ".join(tokensA)
    phraseB = " ".join(tokensB)

    va = embed_cached(phraseA)
    vb = embed_cached(phraseB)

    base_sim = float(va @ vb)

    # Build combined token sets: expanded tokens + raw pre-expansion tokens
    # This ensures we catch conflicts at both abbreviation and expanded levels
    raw_a = list(regex_split(a))
    raw_b = list(regex_split(b))
    seg_a = []
    seg_b = []
    for t in raw_a:
        seg_a.extend(segment_token(t))
    for t in raw_b:
        seg_b.extend(segment_token(t))

    # Merge expanded + raw + segmented tokens for comprehensive conflict check
    all_tokens_a = list(tokensA) + [t.lower() for t in raw_a] + list(seg_a)
    all_tokens_b = list(tokensB) + [t.lower() for t in raw_b] + list(seg_b)

    penalty, conflict_details = semantic_conflict_penalty(all_tokens_a, all_tokens_b)
    final = base_sim * (1 - penalty)

    # Build conflict category summary
    conflict_types = sorted(set(d["type"] for d in conflict_details))

    # Per-token conflict annotation
    token_conflicts_a = {}
    token_conflicts_b = {}
    for d in conflict_details:
        ta, tb = d["token_a"], d["token_b"]
        tag = f"{d['type']}: '{ta}' vs '{tb}'"
        token_conflicts_a.setdefault(ta, []).append(tag)
        token_conflicts_b.setdefault(tb, []).append(tag)

    # confidence label
    if final > 0.85:
        confidence="HIGH"
    elif final > 0.65:
        confidence="MEDIUM"
    else:
        confidence="LOW"

    return {
        "similarity": round(final,4),
        "confidence": confidence,
        "base_similarity": round(base_sim,4),
        "penalty": round(penalty, 4),
        "tokens_A": list(tokensA),
        "tokens_B": list(tokensB),
        "conflicts": conflict_types,
        "conflict_details": conflict_details,
        "token_conflicts_a": token_conflicts_a,
        "token_conflicts_b": token_conflicts_b,
        "expansion_sources": {**srcA, **srcB},
        "runtime_ms": round((time.time()-t0)*1000,2)
    }




# =====================================================
# TEST RUN
# =====================================================

if __name__ == "__main__":

    tests=[
        ("custId","customer_id"),
        ("acctbal","balance_account"),
        ("cfgmgr","configuration_manager"),
        ("shipaddr","shipment_address"),
        ("usrgrpmap","user_group_mapping"),
        ("minprice","maxprice")
    ]

    for a,b in tests:
        print("\n---------------------")
        print(a,"->",process_column(a))
        print(b,"->",process_column(b))
        print(name_similarity(a,b))
