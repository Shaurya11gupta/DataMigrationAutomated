#!/usr/bin/env python3
"""
Generate high-quality training data for the expansion classifier (UniversalGatekeeper).

The classifier's job: given a single token, decide whether it is an ABBREVIATION
that needs expansion (label=1) or a FULL WORD that should be left alone (label=0).

Key insight: the current model fails on SHORT full words (3-4 chars) like 'last',
'time', 'type', 'code', 'flag', 'age' because training data lacked enough of them.
This generator produces a balanced dataset rich in hard negatives.

Output: clf_training_data/clf_train.jsonl and clf_val.jsonl
Each line: {"token": "...", "label": 0 or 1}
"""

import json
import os
import random
from pathlib import Path
from collections import Counter

random.seed(42)

# =============================================================================
# 1. ABBREVIATION POSITIVES (label=1) -- tokens that SHOULD be expanded
# =============================================================================

# Core abbreviation dictionary (must-have positives)
ABBREV_TOKENS = {
    # identity / keys
    "id", "pk", "fk", "ref",
    # entities
    "cust", "usr", "acct", "acc", "dept", "org", "emp", "mgr", "mngr", "adm",
    # technical
    "cfg", "conf", "param", "attr", "meta",
    # data/storage
    "db", "tbl", "col", "idx", "seq", "repo",
    # finance
    "amt", "bal", "qty", "txn", "inv", "ord", "pay", "prc",
    # logistics
    "ship", "addr", "dest", "src",
    # time
    "ts", "dt", "tm", "dob", "yr", "mo", "wk",
    # system
    "svc", "api", "ctrl", "proc", "exec",
    # status
    "stat", "sts", "flg", "typ",
    # grouping
    "grp", "grpmap", "rel",
    # misc
    "desc", "msg", "err", "num", "val", "cnt", "min", "max",
    # currency codes
    "usd", "eur", "inr", "gbp", "jpy", "aud", "cad", "chf", "sgd", "hkd",
    "nzd", "sek", "nok", "dkk", "zar", "rub", "brl", "mxn", "krw", "aed", "sar",
    # common DB abbreviations
    "nm", "nbr", "no", "cd", "tp", "cr", "dr", "dsc", "trk",
    "shp", "wt", "ht", "ln", "fn", "mn", "lbl", "phn", "tel",
    "ctry", "cty", "st", "prov", "rgn", "zip", "pstl",
    "tgt", "tgt", "loc", "pos", "lvl", "cat", "sub", "grp",
    "img", "pic", "doc", "rpt", "tmpl", "schd", "freq",
    "pct", "pnt", "scr", "avg", "tot", "cum", "prev", "nxt",
    "init", "upd", "del", "ins", "sel", "crt", "mod", "gen",
    "auth", "perm", "privl", "sess", "tok",
    "req", "resp", "hdr", "bdy", "endpt",
    "sz", "len", "cnt", "dur", "intv", "lim", "cap",
    "ver", "rev", "bld", "rls",
    "env", "dev", "stg", "prd", "qa",
    "fg", "bg", "pri", "sec",
    "trf", "bw", "lat", "thpt",
    # More abbreviations commonly used in schemas
    "assn", "assoc", "benef", "recip", "distr",
    "alloc", "apprvl", "cancl", "compl", "pndg",
    "rjct", "vrfy", "cert", "lic",
    "sched", "calib", "maint", "notif",
    "enrl", "regn", "xfer", "conv",
    "agg", "calc", "est", "proj",
    "qual", "spec", "std", "perf",
    "dpnd", "assmbly", "compnt",
}

# Additional abbreviation patterns (generated)
EXTRA_ABBREVS = set()

# Common abbreviation-forming patterns: drop vowels, truncate
ABBREV_ROOTS = [
    "actn", "admn", "agnt", "appl", "bnft", "brch", "bkng", "clnt",
    "cmnt", "cmpn", "cntct", "cntry", "crdt", "dlvr", "dpst",
    "dvlp", "drct", "dscnt", "dstrb", "dvce", "empl", "engnr",
    "evnt", "expns", "fclty", "fnce", "gndr", "hlth", "hstry",
    "incm", "insur", "intrnl", "itmzd", "lctn", "lgcy", "mktg",
    "mnfctr", "mtrl", "ntwrk", "oprtn", "prdct", "prfl", "prgm",
    "prjct", "prprt", "prvdr", "rcrd", "rfrn", "rgnl", "rqst",
    "rsrc", "rvnu", "sched", "schl", "scrty", "sgntr", "slry",
    "spcfn", "spprt", "stfng", "stmnt", "sttlmnt", "supvr",
    "systm", "trmnl", "trnsctn", "wrhs",
]
EXTRA_ABBREVS.update(ABBREV_ROOTS)

# 2-char abbreviations (very common in DB schemas)
TWO_CHAR_ABBREVS = [
    "fn", "ln", "mn", "nm", "dt", "tm", "ts", "db", "pk", "fk",
    "id", "cd", "tp", "st", "cr", "dr", "wt", "ht", "sz", "bg",
    "fg", "nr", "yr", "mo", "wk", "hr", "sc", "ct", "qt",
]
EXTRA_ABBREVS.update(TWO_CHAR_ABBREVS)

ALL_ABBREVS = ABBREV_TOKENS | EXTRA_ABBREVS


# =============================================================================
# 2. FULL WORD NEGATIVES (label=0) -- tokens that should NOT be expanded
# =============================================================================

# CRITICAL: Emphasize short words (3-4 chars) since the model struggles with these

SHORT_WORDS_2_3 = [
    # 2-char full words
    "an", "am", "as", "at", "be", "by", "do", "go", "he", "if",
    "in", "is", "it", "me", "my", "of", "on", "or", "so", "up",
    "us", "we", "to", "ok",
    # 3-char full words (this is where the model fails most)
    "ace", "act", "add", "age", "ago", "aid", "aim", "air", "all",
    "and", "any", "ape", "arc", "are", "ark", "arm", "art", "ash",
    "ask", "ate", "awe", "axe", "bad", "bag", "ban", "bar", "bat",
    "bay", "bed", "bet", "bid", "big", "bin", "bit", "bob", "bog",
    "bow", "box", "boy", "bud", "bug", "bun", "bus", "but", "buy",
    "cab", "cam", "can", "cap", "car", "cop", "cow", "cry", "cub",
    "cup", "cur", "cut", "dab", "dam", "dew", "did", "dig", "dim",
    "dip", "dog", "dot", "dry", "dub", "dud", "due", "dug", "dun",
    "duo", "dye", "ear", "eat", "eel", "egg", "ego", "elm", "end",
    "era", "eve", "ewe", "eye", "fad", "fan", "far", "fat", "fax",
    "fed", "fee", "few", "fig", "fin", "fir", "fit", "fix", "fly",
    "foe", "fog", "for", "fox", "fry", "fun", "fur", "gag", "gal",
    "gap", "gas", "gel", "gem", "get", "gig", "gin", "god", "got",
    "gum", "gun", "gut", "guy", "gym", "had", "ham", "has", "hat",
    "hay", "hen", "her", "hid", "him", "hip", "his", "hit", "hog",
    "hop", "hot", "how", "hub", "hue", "hug", "hut", "ice", "icy",
    "ill", "imp", "ink", "inn", "ion", "ire", "irk", "its", "ivy",
    "jab", "jag", "jam", "jar", "jaw", "jay", "jet", "jig", "job",
    "jog", "joy", "jug", "jut", "keg", "ken", "key", "kid", "kin",
    "kit", "lab", "lad", "lag", "lap", "law", "lay", "lea", "led",
    "leg", "let", "lid", "lie", "lip", "lit", "log", "lot", "low",
    "lug", "mad", "man", "map", "mar", "mat", "maw", "may", "men",
    "met", "mid", "mix", "mob", "mod", "mop", "mow", "mud", "mug",
    "nab", "nag", "nap", "net", "new", "nil", "nip", "nit", "nod",
    "nor", "not", "now", "nun", "nut", "oak", "oar", "oat", "odd",
    "ode", "off", "oft", "ohm", "oil", "old", "one", "opt", "orb",
    "ore", "our", "out", "owe", "owl", "own", "pad", "pal", "pan",
    "pat", "paw", "pea", "peg", "pen", "per", "pet", "pie", "pig",
    "pin", "pit", "ply", "pod", "pop", "pot", "pow", "pub", "pug",
    "pun", "pup", "pus", "put", "rag", "ram", "ran", "rap", "rat",
    "raw", "ray", "red", "rib", "rid", "rig", "rim", "rip", "rob",
    "rod", "roe", "rot", "row", "rub", "rug", "rum", "run", "rut",
    "rye", "sac", "sad", "sag", "sap", "sat", "saw", "say", "sea",
    "set", "sew", "shy", "sin", "sip", "sir", "sis", "sit", "six",
    "ski", "sky", "sly", "sob", "sod", "son", "sop", "sot", "sow",
    "soy", "spa", "spy", "sty", "sub", "sue", "sum", "sun", "sup",
    "tab", "tad", "tag", "tan", "tap", "tar", "tax", "tea", "ten",
    "the", "thy", "tic", "tie", "tin", "tip", "toe", "ton", "too",
    "top", "tot", "tow", "toy", "try", "tub", "tug", "two", "urn",
    "use", "van", "vat", "vet", "vex", "via", "vie", "vim", "vow",
    "wad", "wag", "war", "was", "wax", "way", "web", "wed", "wet",
    "who", "why", "wig", "win", "wit", "woe", "wok", "won", "woo",
    "wow", "yak", "yam", "yap", "yaw", "yea", "yes", "yet", "yew",
    "you", "zap", "zed", "zen", "zig", "zip", "zoo",
]

SHORT_WORDS_4 = [
    # 4-char full words -- the hardest category
    "able", "ache", "acid", "acre", "aged", "also", "arch", "area",
    "army", "away", "baby", "back", "bait", "bake", "bald", "ball",
    "band", "bane", "bang", "bank", "bare", "bark", "barn", "base",
    "bath", "bead", "beak", "beam", "bean", "bear", "beat", "been",
    "beef", "beer", "bell", "belt", "bend", "bent", "best", "bias",
    "bike", "bill", "bind", "bird", "bite", "blew", "blow", "blue",
    "blur", "boar", "boat", "body", "bold", "bolt", "bomb", "bond",
    "bone", "book", "boom", "boot", "bore", "born", "boss", "both",
    "bout", "bowl", "bred", "brew", "buck", "bulk", "bull", "bump",
    "burn", "burr", "busy", "buzz", "cage", "cake", "calf", "call",
    "calm", "came", "camp", "cane", "cape", "card", "care", "cart",
    "case", "cash", "cast", "cave", "cell", "chat", "chin", "chip",
    "chop", "cite", "city", "clad", "clam", "clap", "claw", "clay",
    "clip", "clot", "club", "clue", "coal", "coat", "coil", "coin",
    "cold", "colt", "comb", "come", "cook", "cool", "cope", "copy",
    "cord", "core", "cork", "corn", "cost", "coup", "cove", "crab",
    "crew", "crop", "crow", "cult", "curb", "cure", "curl", "cute",
    "dale", "dame", "damn", "damp", "dare", "dark", "dart", "dash",
    "data", "date", "dawn", "days", "dead", "deaf", "deal", "dean",
    "dear", "debt", "deck", "deed", "deem", "deep", "deer", "demo",
    "deny", "desk", "dial", "dice", "diet", "dirt", "disc", "dish",
    "disk", "dock", "does", "dome", "done", "doom", "door", "dose",
    "dove", "down", "drag", "draw", "drew", "drip", "drop", "drug",
    "drum", "dual", "duck", "duel", "dull", "dumb", "dump", "dune",
    "dusk", "dust", "duty", "dyed",
    "each", "ease", "east", "easy", "edge", "edit", "else", "emit",
    "ends", "epic", "euro", "even", "ever", "evil", "exam", "exit",
    "eyed", "face", "fact", "fade", "fail", "fair", "fake", "fall",
    "fame", "fare", "farm", "fast", "fate", "fawn", "fear", "feat",
    "feed", "feel", "feet", "fell", "felt", "file", "fill", "film",
    "find", "fine", "fire", "firm", "fish", "fist", "five", "flag",
    "flat", "flaw", "flea", "fled", "flew", "flip", "flog", "flow",
    "foam", "foil", "fold", "folk", "fond", "food", "fool", "foot",
    "ford", "fore", "fork", "form", "fort", "foul", "four", "free",
    "frog", "from", "fuel", "full", "fume", "fund", "fuse", "fury",
    "fuss", "gait", "gale", "game", "gang", "gape", "garb", "gate",
    "gave", "gaze", "gear", "gift", "gild", "gist", "give", "glad",
    "glow", "glue", "gnat", "gnaw", "goal", "goat", "goes", "gold",
    "golf", "gone", "good", "gore", "grab", "gram", "gray", "grew",
    "grey", "grid", "grim", "grin", "grip", "grit", "grow", "gulf",
    "gust", "guts",
    "hack", "hail", "hair", "hale", "half", "hall", "halt", "hand",
    "hang", "hare", "harm", "harp", "hash", "hast", "hate", "haul",
    "have", "haze", "hazy", "head", "heal", "heap", "hear", "heat",
    "heed", "heel", "held", "help", "here", "hero", "hide", "high",
    "hike", "hill", "hilt", "hind", "hint", "hire", "hiss", "hive",
    "hoard","hold", "hole", "home", "hone", "hood", "hook", "hoop",
    "hope", "horn", "hose", "host", "hour", "howl", "huge", "hull",
    "hump", "hung", "hunt", "hurl", "hurt", "hush", "hymn",
    "icon", "idea", "idle", "inch", "into", "iron", "isle", "item",
    "jack", "jade", "jail", "jazz", "jest", "join", "joke", "jolt",
    "jump", "june", "jury", "just",
    "keen", "keep", "kelp", "kept", "kick", "kill", "kind", "king",
    "kite", "knee", "knew", "knit", "knob", "knot", "know",
    "lace", "lack", "lacy", "laid", "lair", "lake", "lamb", "lame",
    "lamp", "land", "lane", "lard", "lark", "lash", "lass", "last",
    "late", "lawn", "lazy", "lead", "leaf", "leak", "lean", "leap",
    "left", "lend", "lens", "less", "lest", "levy", "liar", "lick",
    "lied", "lieu", "life", "lift", "like", "lily", "limb", "lime",
    "limp", "line", "link", "lint", "lion", "list", "live", "load",
    "loaf", "loan", "lock", "loft", "logo", "lone", "long", "look",
    "loom", "loop", "lord", "lore", "lose", "loss", "lost", "lots",
    "loud", "love", "luck", "lull", "lump", "lung", "lure", "lurk",
    "lush", "lust",
    "mace", "made", "maid", "mail", "maim", "main", "make", "male",
    "mall", "malt", "mane", "many", "mare", "mark", "mars", "mart",
    "mash", "mask", "mass", "mast", "mate", "maze", "mead", "meal",
    "mean", "meat", "meet", "meld", "melt", "memo", "mend", "menu",
    "mere", "mesh", "mess", "mild", "mile", "milk", "mill", "mime",
    "mind", "mine", "mint", "mire", "miss", "mist", "mite", "mitt",
    "moan", "moat", "mock", "mode", "mold", "mole", "mood", "moon",
    "moor", "more", "moss", "most", "moth", "move", "much", "mule",
    "mull", "murk", "muse", "must", "mute", "myth",
    "nail", "name", "nape", "navy", "near", "neat", "neck", "need",
    "nest", "news", "next", "nice", "nine", "node", "none", "noon",
    "norm", "nose", "note", "noun", "null",
    "oafs", "oath", "obey", "odds", "ogre", "okay", "once", "only",
    "onto", "ooze", "opal", "open", "opus", "oral", "ours", "oust",
    "oval", "oven", "over", "owed",
    "pace", "pack", "pact", "page", "paid", "pail", "pain", "pair",
    "pale", "palm", "pane", "pang", "pant", "pare", "park", "part",
    "pass", "past", "path", "pave", "pawn", "peak", "peal", "pear",
    "peat", "peck", "peel", "peer", "pelt", "pend", "perk", "pest",
    "pick", "pier", "pike", "pile", "pine", "pink", "pipe", "pith",
    "plan", "play", "plea", "plot", "plod", "plow", "ploy", "plug",
    "plum", "plus", "poem", "poet", "poll", "polo", "pond", "pool",
    "poor", "pope", "pore", "pork", "port", "pose", "post", "pour",
    "pray", "prey", "prod", "prop", "prow", "pull", "pulp", "pump",
    "pure", "purr", "push",
    "quad", "quit", "quiz",
    "race", "rack", "raft", "rage", "raid", "rail", "rain", "rake",
    "ramp", "rang", "rank", "rant", "rash", "rasp", "rate", "rave",
    "read", "real", "ream", "reap", "rear", "reef", "reel", "rely",
    "rend", "rent", "rest", "rice", "rich", "ride", "rift", "rind",
    "ring", "riot", "ripe", "rise", "risk", "road", "roam", "roar",
    "robe", "rock", "rode", "role", "roll", "roof", "room", "root",
    "rope", "rose", "rove", "ruby", "rude", "ruin", "rule", "rump",
    "rung", "rush", "rust",
    "sack", "safe", "sage", "said", "sail", "sake", "sale", "salt",
    "same", "sand", "sane", "sang", "sank", "save", "says", "scan",
    "scar", "seal", "seam", "seat", "sect", "seed", "seek", "seem",
    "seen", "self", "sell", "send", "sent", "shed", "shin", "ship",
    "shoe", "shoo", "shop", "shot", "show", "shut", "sick", "side",
    "sigh", "sign", "silk", "sill", "silt", "sing", "sink", "sire",
    "site", "size", "slab", "slag", "slam", "slap", "slaw", "sled",
    "slew", "slid", "slim", "slip", "slit", "slot", "slow", "slug",
    "slum", "slur", "smog", "snap", "snip", "snow", "snub", "snug",
    "soak", "soap", "soar", "sock", "soda", "soft", "soil", "sold",
    "sole", "some", "song", "soon", "soot", "sore", "sort", "soul",
    "sour", "span", "spar", "sped", "spin", "spit", "spot", "spur",
    "stab", "stag", "star", "stay", "stem", "step", "stew", "stir",
    "stop", "stow", "stub", "stud", "stun", "such", "suck", "suit",
    "sulk", "sung", "sunk", "sure", "surf", "swam", "swan", "swap",
    "sway", "swim",
    "tack", "tact", "tail", "take", "tale", "talk", "tall", "tame",
    "tang", "tank", "tape", "tart", "task", "team", "tear", "teem",
    "tell", "tend", "tent", "term", "test", "text", "than", "that",
    "them", "then", "they", "thin", "this", "thorn","thus", "tick",
    "tide", "tidy", "tier", "tile", "till", "tilt", "time", "tine",
    "tiny", "tire", "toad", "toil", "told", "toll", "tomb", "tone",
    "took", "tool", "tops", "tore", "torn", "toss", "tour", "town",
    "trap", "tray", "tree", "trek", "trim", "trio", "trip", "trod",
    "true", "tube", "tuck", "tuft", "tuna", "tune", "turf", "turn",
    "twig", "twin", "type",
    "ugly", "undo", "unit", "unto", "upon", "urge", "used", "user",
    "vain", "vale", "vane", "vary", "vase", "vast", "veil", "vein",
    "vent", "verb", "very", "vest", "veto", "vice", "vied", "view",
    "vine", "void", "vole", "volt", "vote", "vows",
    "wade", "wage", "wail", "wait", "wake", "walk", "wall", "wand",
    "want", "ward", "warm", "warn", "warp", "wart", "wary", "wash",
    "wave", "wavy", "waxy", "ways", "weak", "wean", "wear", "weed",
    "week", "weld", "well", "went", "wept", "were", "west", "what",
    "when", "whom", "wick", "wide", "wife", "wild", "will", "wilt",
    "wily", "wind", "wine", "wing", "wink", "wipe", "wire", "wise",
    "wish", "wisp", "with", "woke", "wolf", "womb", "wood", "wool",
    "word", "wore", "work", "worm", "worn", "wove", "wrap", "writ",
    "yard", "yarn", "yawn", "year", "yell", "yoga", "yoke", "your",
    "zeal", "zero", "zest", "zinc", "zone", "zoom",
]

LONGER_FULL_WORDS = [
    # 5-char
    "about", "above", "abuse", "adapt", "admit", "adopt", "adult", "after",
    "again", "agree", "ahead", "alarm", "alive", "allow", "alone", "along",
    "alter", "among", "angel", "anger", "angle", "angry", "ankle", "annex",
    "apart", "apple", "apply", "arena", "argue", "arise", "aside", "asset",
    "audio", "audit", "avoid", "award", "aware", "badge", "baron", "basic",
    "basis", "beach", "begin", "being", "below", "bench", "berry", "birth",
    "black", "blade", "blame", "bland", "blank", "blast", "blaze", "bleed",
    "blend", "bless", "blind", "block", "blood", "bloom", "blown", "board",
    "boost", "bound", "brain", "brand", "brave", "bread", "break", "breed",
    "brick", "brief", "bring", "broad", "broke", "brook", "brown", "brush",
    "build", "bunch", "burst", "buyer",
    "cabin", "cargo", "carry", "catch", "cause", "chain", "chair", "chalk",
    "charm", "chase", "cheap", "check", "cheek", "chess", "chest", "chief",
    "child", "chunk", "civil", "claim", "class", "clean", "clear", "clerk",
    "climb", "cling", "clock", "clone", "close", "cloth", "cloud", "coach",
    "coast", "color", "comet", "comma", "coral", "count", "court", "cover",
    "crack", "craft", "crash", "crazy", "cream", "crest", "crime", "crisp",
    "cross", "crowd", "crown", "crush", "curve", "cycle",
    "daily", "dance", "datum", "death", "debug", "decay", "decoy", "delay",
    "delta", "dense", "depot", "depth", "digit", "dirty", "doubt", "draft",
    "drain", "drama", "drank", "drape", "drawn", "dream", "dress", "dried",
    "drift", "drill", "drink", "drive", "drove", "drown", "drunk",
    "eager", "early", "earth", "eight", "elder", "elect", "elite", "embed",
    "empty", "enemy", "enjoy", "enter", "entry", "equal", "equip", "error",
    "essay", "event", "every", "exact", "exert", "extra",
    "faint", "faith", "fancy", "fault", "feast", "fence", "ferry", "fetch",
    "fever", "fewer", "fiber", "field", "fight", "final", "first", "fixed",
    "flame", "flash", "flask", "flesh", "float", "flock", "flood", "floor",
    "flour", "fluid", "flush", "flute", "focus", "force", "forge", "forth",
    "forum", "found", "frame", "frank", "fraud", "fresh", "front", "frost",
    "froze", "fruit", "fully", "funny",
    "gauge", "genre", "ghost", "giant", "given", "glass", "glide", "globe",
    "glory", "gloss", "glove", "going", "grace", "grade", "grain", "grand",
    "grant", "graph", "grasp", "grass", "grave", "great", "green", "greet",
    "grief", "grill", "grind", "gross", "group", "grove", "grown", "guard",
    "guess", "guest", "guide", "guild",
    "habit", "happy", "harsh", "haven", "heart", "hence", "honey", "horse",
    "hotel", "house", "human", "humor",
    "ideal", "image", "imply", "index", "indie", "inner", "input", "issue",
    "ivory",
    "jewel", "joint", "judge", "juice",
    "knife", "knock", "known",
    "label", "labor", "large", "laser", "later", "laugh", "layer", "learn",
    "lease", "least", "leave", "legal", "lemon", "level", "light", "limit",
    "linen", "liver", "local", "logic", "loose", "lorry", "lover", "lower",
    "loyal", "lucky", "lunch",
    "magic", "major", "maker", "maple", "march", "match", "mayor", "media",
    "mercy", "merge", "merit", "metal", "meter", "might", "minor", "minus",
    "mixed", "model", "month", "moral", "motor", "mount", "mouse", "mouth",
    "movie", "muddy", "music", "myths",
    "naive", "nerve", "never", "night", "noble", "noise", "north", "noted",
    "novel", "nurse",
    "occur", "ocean", "offer", "often", "olive", "onset", "opera", "orbit",
    "order", "organ", "other", "outer", "owned", "owner", "oxide",
    "paint", "panel", "panic", "paper", "party", "pasta", "paste", "patch",
    "pause", "peace", "pearl", "penny", "phase", "phone", "photo", "piano",
    "piece", "pilot", "pitch", "pixel", "pizza", "place", "plain", "plane",
    "plant", "plate", "plaza", "plead", "plumb", "plume", "plump", "point",
    "polar", "porch", "poser", "pound", "power", "press", "price", "pride",
    "prime", "print", "prior", "prize", "probe", "proof", "proud", "prove",
    "psalm", "pulse", "punch", "pupil", "purse",
    "queen", "query", "quest", "queue", "quick", "quiet", "quite", "quota",
    "quote",
    "radar", "radio", "raise", "rally", "ranch", "range", "rapid", "ratio",
    "reach", "react", "ready", "realm", "rebel", "refer", "reign", "relax",
    "renew", "reply", "reset", "rider", "ridge", "right", "rigid", "rival",
    "river", "roast", "robin", "robot", "rocky", "rough", "round", "route",
    "rover", "royal", "ruler", "rumor", "rural",
    "saint", "salad", "scale", "scare", "scene", "scope", "score", "scout",
    "sense", "serve", "setup", "seven", "shade", "shake", "shall", "shame",
    "shape", "share", "shark", "sharp", "shave", "sheep", "sheer", "sheet",
    "shelf", "shell", "shift", "shire", "shock", "shore", "short", "shout",
    "sight", "since", "sixth", "sixty", "skill", "skull", "slate", "slave",
    "sleep", "slice", "slide", "slope", "small", "smart", "smell", "smile",
    "smith", "smoke", "snake", "solid", "solve", "sorry", "sound", "south",
    "space", "spare", "spark", "speak", "speed", "spend", "spent", "spice",
    "spill", "spine", "spite", "split", "spoke", "sport", "spray", "squad",
    "stack", "staff", "stage", "stain", "stair", "stake", "stale", "stall",
    "stamp", "stand", "stark", "start", "state", "stays", "steak", "steam",
    "steel", "steep", "steer", "stern", "stick", "stiff", "still", "sting",
    "stock", "stole", "stone", "stood", "storm", "story", "stove", "strap",
    "straw", "strip", "stuck", "stuff", "style", "sugar", "suite", "sunny",
    "super", "surge", "swamp", "swear", "sweet", "swept", "swift", "swing",
    "sword",
    "table", "taste", "teach", "teeth", "thank", "theme", "there", "thick",
    "thing", "think", "third", "those", "three", "threw", "throw", "thumb",
    "tiger", "tight", "timer", "title", "today", "token", "topic", "torch",
    "total", "touch", "tough", "tower", "toxic", "trace", "track", "trade",
    "trail", "train", "trait", "trash", "treat", "trend", "trial", "tribe",
    "trick", "truck", "truly", "trump", "trunk", "trust", "truth", "tumor",
    "twelve", "twice", "twist",
    "uncle", "under", "union", "unity", "until", "upper", "upset", "urban",
    "usage", "usual", "utter",
    "valid", "value", "vault", "venue", "verse", "video", "vigor", "viral",
    "visit", "vital", "vivid", "vocal", "voice", "voter",
    "waste", "watch", "water", "weave", "weigh", "weird", "wheat", "wheel",
    "where", "which", "while", "white", "whole", "whose", "width", "witch",
    "woman", "women", "works", "world", "worry", "worse", "worst", "worth",
    "would", "wound", "wrath", "write", "wrong", "wrote",
    "yield", "young", "youth",

    # 6+ char common words
    "accept", "access", "action", "active", "actual", "adjust", "advice",
    "affect", "afford", "agency", "agenda", "amount", "annual", "appeal",
    "assign", "assist", "assume", "attack", "attend", "august",
    "backup", "battle", "beauty", "became", "become", "before", "behalf",
    "behind", "belong", "beside", "better", "beyond", "bigger", "border",
    "borrow", "bottom", "bounce", "branch", "bridge", "broken", "browse",
    "bucket", "budget", "buffer", "bundle", "burden", "button",
    "cancel", "carbon", "career", "castle", "center", "chance", "change",
    "charge", "choice", "choose", "church", "circle", "client", "closed",
    "closer", "coffee", "column", "combat", "coming", "commit", "common",
    "comply", "config", "cookie", "corner", "cotton", "county", "couple",
    "course", "create", "credit", "crisis", "custom",
    "damage", "danger", "dealer", "debate", "decade", "defeat", "defend",
    "define", "degree", "delete", "demand", "denial", "deploy", "desert",
    "design", "desire", "detail", "detect", "device", "devote", "dialog",
    "differ", "digest", "dinner", "direct", "docker", "doctor", "domain",
    "donate", "double", "driver", "during",
    "earned", "easily", "eating", "editor", "effect", "effort", "eighth",
    "either", "eleven", "emerge", "empire", "employ", "enable", "endure",
    "energy", "engine", "enough", "ensure", "entire", "entity", "equity",
    "escape", "estate", "ethnic", "evolve", "exceed", "except", "excuse",
    "exempt", "exist", "expand", "expect", "expert", "export", "expose",
    "extend", "extent",
    "fabric", "facial", "factor", "failed", "fairly", "family", "famous",
    "farmer", "faster", "father", "faucet", "female", "fierce", "figure",
    "filter", "finale", "finger", "finish", "fiscal", "flavor", "flight",
    "flower", "follow", "forbid", "forced", "forest", "forget", "formal",
    "former", "fossil", "foster", "fourth", "freeze", "friend", "frozen",
    "future",
    "gained", "galaxy", "garden", "gather", "gender", "gentle", "global",
    "golden", "govern", "growth",
    "handle", "happen", "harbor", "hardly", "hazard", "header", "health",
    "height", "helped", "hidden", "hiking", "holder", "honest", "hunger",
    "hybrid",
    "ignore", "impact", "import", "impose", "income", "indeed", "inform",
    "inject", "injury", "inline", "insert", "inside", "insist", "intact",
    "intend", "intent", "invest", "inward", "island", "itself",
    "jacket", "jersey", "jungle", "junior", "justice",
    "kernel", "kidney", "killer", "knight", "knight",
    "ladder", "landed", "laptop", "launch", "leader", "league", "legacy",
    "lender", "length", "lesson", "letter", "liable", "linear", "liquid",
    "listen", "little", "living", "locate", "lonely", "lookup", "losing",
    "lowest", "luxury",
    "maiden", "mainly", "manage", "manner", "margin", "marked", "market",
    "master", "matrix", "matter", "medium", "member", "memory", "mental",
    "mentor", "method", "middle", "mighty", "minute", "mirror", "mobile",
    "modern", "modest", "modify", "module", "moment", "monkey", "mother",
    "motion", "murder", "muscle", "mutual",
    "namely", "narrow", "nation", "native", "nature", "nearby", "nearly",
    "nested", "neural", "newest", "nickel", "nobody", "normal", "notice",
    "notion", "number", "nurses",
    "object", "obtain", "occupy", "offend", "office", "online", "opener",
    "option", "orange", "origin", "orphan", "outfit", "outlet", "output",
    "outset",
    "packet", "palace", "parent", "partly", "patron", "patter", "paused",
    "people", "period", "permit", "person", "phrase", "pickup", "pillar",
    "planet", "played", "player", "please", "pledge", "plenty", "plunge",
    "pocket", "poetry", "police", "policy", "polish", "polite", "poorly",
    "portal", "posted", "potato", "powder", "prayer", "prefer", "prince",
    "prison", "profit", "prompt", "proper", "proven", "public", "pulled",
    "punish", "puppet", "purple", "pursue", "puzzle",
    "rabbit", "racial", "random", "rarely", "rating", "reader", "reason",
    "recall", "recent", "recipe", "record", "reduce", "reform", "refuge",
    "regard", "regime", "region", "regret", "reject", "relate", "relief",
    "remain", "remedy", "remind", "remote", "remove", "render", "rental",
    "repair", "repeat", "report", "rescue", "resign", "resist", "resort",
    "result", "retail", "retain", "retire", "return", "reveal", "review",
    "revolt", "reward", "ribbon", "robust", "rocket", "roster", "ruling",
    "runner",
    "sacred", "safely", "safety", "salary", "sample", "scared", "schema",
    "school", "screen", "script", "scroll", "search", "season", "second",
    "secret", "sector", "secure", "select", "seller", "senior", "serial",
    "server", "settle", "severe", "shadow", "shield", "signal", "silent",
    "silver", "simple", "simply", "single", "sister", "sketch", "slight",
    "smooth", "socket", "solely", "solemn", "sought", "source", "speech",
    "sphere", "spirit", "spoken", "spread", "spring", "square", "stable",
    "statue", "status", "steady", "stolen", "stored", "strain", "strand",
    "stream", "street", "stress", "strict", "strike", "string", "stroke",
    "strong", "struck", "studio", "submit", "subtle", "suburb", "sudden",
    "suffer", "summit", "Sunday", "supper", "supply", "surely", "survey",
    "switch", "symbol", "syntax", "system",
    "tablet", "tackle", "tactic", "talent", "target", "temple", "tenant",
    "tender", "tenure", "terror", "thanks", "throat", "throne", "thrown",
    "ticket", "timber", "toggle", "tongue", "toward", "treaty", "tribal",
    "triple", "trophy", "tunnel", "turnip",
    "unique", "united", "unlike", "unpack", "update", "upload", "upside",
    "useful",
    "vacant", "valley", "valued", "varied", "vector", "vendor", "vessel",
    "victim", "viewer", "violet", "virtue", "vision", "volume",
    "walker", "wallet", "wander", "warmth", "wealth", "weapon", "weekly",
    "weight", "window", "winner", "winter", "wisdom", "worker",
    "yearly",
    "zodiac",

    # DB/IT domain full words that should NOT be expanded
    "account", "address", "balance", "boolean", "catalog", "channel",
    "cluster", "comment", "contact", "content", "context", "control",
    "counter", "country", "current", "decimal", "default", "deposit",
    "digital", "display", "element", "enabled", "encrypt", "execute",
    "expense", "feature", "freight", "gateway", "general", "handler",
    "history", "integer", "invalid", "invoice", "journal", "keyword",
    "library", "license", "limited", "listing", "loading", "logical",
    "machine", "message", "monitor", "natural", "network", "numeric",
    "package", "partner", "passage", "patient", "payload", "payment",
    "pending", "percent", "picture", "pointer", "premium", "primary",
    "printer", "privacy", "private", "process", "product", "profile",
    "program", "project", "promise", "protect", "provide", "publish",
    "quality", "quarter", "receipt", "recover", "release", "replace",
    "request", "require", "reserve", "resolve", "restore", "revenue",
    "routine", "runtime", "savings", "segment", "service", "session",
    "setting", "sitemap", "snippet", "storage", "summary", "support",
    "suspend", "teacher", "template","texture", "timeout", "tracker",
    "trigger", "utility", "version", "virtual", "webhook", "website",
    "welcome", "workflow",
    "customer", "database", "discount", "document", "download", "employee",
    "exchange", "feedback", "filename", "firewall", "function", "generate",
    "hardware", "hospital", "hostname", "identity", "instance", "interval",
    "language", "location", "material", "metadata", "midnight", "mortgage",
    "neighbor", "notebook", "operator", "optional", "ordering", "overview",
    "package", "password", "platform", "position", "practice", "previous",
    "priority", "progress", "property", "protocol", "provider", "purchase",
    "quantity", "question", "reaction", "received", "recovery", "redirect",
    "register", "relation", "remember", "required", "resource", "response",
    "restrict", "reviewer", "rotation", "schedule", "security", "selected",
    "sequence", "shipping", "software", "standard", "strategy", "strength",
    "supplier", "surround", "template", "terminal", "thinking", "thousand",
    "timeline", "together", "tracking", "transfer", "umbrella", "uncommon",
    "validate", "variable", "warranty", "whatever", "wireless",
    "abandoned", "algorithm", "alignment", "allowance", "apartment",
    "automatic", "awareness", "benchmark", "blueprint", "bootstrap",
    "broadcast", "calculate", "cancelled", "catalogue", "challenge",
    "character", "classroom", "clearance", "collision", "committee",
    "community", "complaint", "component", "condition", "configure",
    "connected", "container", "converted", "copyright", "corporate",
    "corrected", "criterion", "dashboard", "deduction", "delivered",
    "dependent", "described", "detection", "determine", "developed",
    "developer", "dimension", "direction", "directory", "discovery",
    "education", "effective", "efficient", "emergency", "encourage",
    "equipment", "essential", "establish", "estimated", "evaluated",
    "exception", "execution", "expansion", "expensive", "explained",
    "extension", "financial", "following", "formation", "framework",
    "frequency", "generally", "generator", "guarantee", "guideline",
    "happiness", "highlight", "identical", "implement", "important",
    "improving", "increased", "indicator", "inflation", "influence",
    "inspector", "insurance", "integrate", "intention", "interface",
    "introduce", "inventory", "investing", "invisible", "landscape",
    "lifecycle", "logistics", "magnitude", "marketing", "mechanism",
    "migration", "milestone", "namespace", "narrative", "navigator",
    "necessary", "negotiate", "newspaper", "normalize", "nurturing",
    "objective", "occupancy", "operation", "organizer", "otherwise",
    "overwrite", "parameter", "partially", "partition", "passenger",
    "permalink", "permanent", "permitted", "placement", "plaintiff",
    "populated", "portfolio", "potential", "preceding", "precision",
    "predicted", "preferred", "preparing", "presented", "preserved",
    "prevented", "primarily", "primitive", "principal", "procedure",
    "processed", "processor", "producing", "profiling", "programme",
    "promising", "promoting", "protected", "provision", "publisher",
    "purchased", "qualifier", "quarterly", "receiving", "recommend",
    "recording", "recurring", "reduction", "reference", "reflected",
    "regarding", "registrar", "regulated", "remaining", "rendering",
    "replacing", "reporting", "represent", "requested", "requiring",
    "reserving", "resolving", "restoring", "resulting", "retention",
    "retrieved", "reversing", "reviewing", "revolving", "selection",
    "sensitive", "separator", "signature", "situation", "something",
    "somewhere", "specified", "sponsored", "statement", "strategic",
    "structure", "submitted", "succeeded", "suffering", "suggested",
    "supported", "suspended", "targeting", "temporary", "thousands",
    "threshold", "timestamp", "tolerance", "touchable", "transform",
    "transport", "triggered", "undefined", "universal", "upgrading",
    "utilities", "validated", "violation", "warehouse", "wholesale",
    "withdrawn", "workplace",
    "absolutely", "accessible", "accomplish", "accounting", "adjustment",
    "allocation", "alphabetic", "annotation", "appearance", "applicable",
    "attendance", "authorized", "background", "bankruptcy", "blockchain",
    "calculated", "capability", "compliance", "configured", "connection",
    "constraint", "controller", "convention", "credential", "department",
    "deprecated", "deployment", "descriptor", "determined", "dictionary",
    "difficulty", "dispatcher", "distribute", "documented", "efficiency",
    "employment", "enterprise", "evaluating", "everything", "experiment",
    "expression", "government", "graduation", "healthcare", "horizontal",
    "identifier", "impression", "indication", "industrial", "initialize",
    "instrument", "laboratory", "limitation", "management", "membership",
    "middleware", "monitoring", "multiplier", "newsletter", "obligation",
    "occurrence", "opposition", "optimizing", "ordinarily", "originally",
    "permission", "persistent", "playground", "population", "preference",
    "processing", "production", "profession", "programmer", "prohibited",
    "properties", "protection", "qualifying", "reasonable", "reflection",
    "regulation", "repository", "reputation", "resolution", "responsive",
    "restaurant", "retirement", "scheduling", "settlement", "simulation",
    "specialist", "strengthen", "subscriber", "subsequent", "substitute",
    "supervisor", "supporting", "tournament", "trajectory", "transition",
    "underlying", "validation", "vocabulary", "vulnerable",
]

# Database/schema-specific full words (commonly seen in column names)
DB_FULL_WORDS = [
    "key", "code", "flag", "type", "age", "tax", "fee", "day", "end",
    "tag", "url", "row", "log", "pin", "bin", "bit", "set", "map",
    "job", "hub", "bus", "run", "max", "min", "sum", "avg",
    "date", "time", "name", "last", "first", "cost", "rate", "size",
    "rank", "sort", "hash", "path", "port", "host", "root", "seed",
    "mode", "step", "lock", "role", "tier", "slot", "span", "unit",
    "zone", "area", "file", "line", "page", "view", "rule", "test",
    "plan", "task", "memo", "note", "list", "item", "node", "link",
    "send", "read", "load", "dump", "sync", "ping", "push", "pull",
    "void", "null", "true", "byte", "char", "long", "enum", "blob",
    "text", "guid", "uuid", "mask", "salt",
    "email", "phone", "image", "video", "audio", "label", "title",
    "field", "table", "query", "index", "count", "limit", "batch",
    "queue", "cache", "proxy", "model", "class", "scope", "token",
    "event", "alert", "error", "debug", "trace", "level", "state",
    "owner", "admin", "guest", "group", "agent", "store", "price",
    "stock", "trade", "share", "order", "claim", "lease", "grant",
    "shift", "leave", "skill", "score", "grade", "badge", "point",
    "bonus", "extra", "total", "gross", "debit", "credit", "value",
    "range", "delta", "ratio", "scale",
    "source", "target", "origin", "parent", "master", "backup",
    "active", "status", "weight", "height", "length", "margin",
    "salary", "amount", "gender", "number", "street", "region",
    "mobile", "office", "branch", "vendor", "client", "member",
    "worker", "doctor", "period", "domain", "object", "record",
    "schema", "format", "output", "report", "metric", "signal",
    "thread", "cursor", "socket", "stream", "buffer", "filter",
    "layout", "medium", "bitmap", "vector",
    "address", "account", "balance", "channel", "cluster", "company",
    "contact", "content", "control", "country", "counter", "current",
    "default", "deposit", "display", "element", "enabled", "freight",
    "gateway", "handler", "history", "integer", "invoice", "journal",
    "keyword", "library", "manager", "message", "network", "package",
    "partner", "patient", "payment", "pending", "picture", "primary",
    "printer", "privacy", "process", "product", "profile", "program",
    "project", "quality", "quarter", "receipt", "release", "request",
    "reserve", "restore", "routine", "service", "session", "setting",
    "storage", "summary", "support", "teacher", "timeout", "tracker",
    "trigger", "utility", "version", "website",
    "customer", "database", "discount", "document", "download", "employee",
    "exchange", "feedback", "filename", "function", "hardware", "identity",
    "instance", "interval", "language", "location", "material", "metadata",
    "mortgage", "operator", "ordering", "overview", "password", "platform",
    "position", "previous", "priority", "progress", "property", "protocol",
    "provider", "purchase", "quantity", "question", "recovery", "register",
    "relation", "required", "resource", "response", "reviewer", "schedule",
    "security", "sequence", "shipping", "software", "standard", "strategy",
    "supplier", "template", "terminal", "timeline", "tracking", "transfer",
    "variable", "warranty", "workflow",
    "algorithm", "component", "configure", "container", "dashboard",
    "developer", "directory", "education", "equipment", "exception",
    "execution", "expansion", "extension", "framework", "frequency",
    "generator", "guarantee", "guideline", "highlight", "implement",
    "indicator", "insurance", "interface", "inventory", "logistics",
    "marketing", "mechanism", "migration", "namespace", "operation",
    "parameter", "partition", "permalink", "procedure", "processor",
    "reference", "reporting", "represent", "selection", "separator",
    "signature", "statement", "structure", "timestamp", "transform",
    "transport", "undefined", "utilities", "validated", "warehouse",
    "allocation", "annotation", "compliance", "connection", "constraint",
    "controller", "convention", "credential", "department", "deployment",
    "descriptor", "dictionary", "enterprise", "expression", "identifier",
    "management", "membership", "middleware", "monitoring", "occurrence",
    "permission", "persistent", "population", "preference", "processing",
    "production", "programmer", "properties", "protection", "repository",
    "resolution", "retirement", "settlement", "simulation", "specialist",
    "subscriber", "supervisor", "tournament", "transition", "validation",
    "transaction", "description",
]


# =============================================================================
# 3. GENERATE THE DATASET
# =============================================================================

def generate_dataset():
    """Generate balanced training dataset."""
    examples = []

    # --- POSITIVES (label=1): abbreviation tokens ---
    for tok in ALL_ABBREVS:
        # Add each abbreviation multiple times with slight weight
        examples.append({"token": tok.lower(), "label": 1})
        # Also add some case variants
        if len(tok) >= 3:
            examples.append({"token": tok.lower(), "label": 1})

    # Generate additional abbreviation patterns
    # Pattern 1: first N chars of common words (truncation)
    trunc_sources = [
        "customer", "account", "address", "transaction", "department",
        "employee", "manager", "balance", "quantity", "description",
        "payment", "invoice", "configuration", "controller", "process",
        "service", "interface", "execution", "repository", "parameter",
        "attribute", "organization", "administration", "identifier",
        "reference", "sequence", "destination", "statistics", "tracking",
        "shipment", "certificate", "notification", "allocation",
        "frequency", "maintenance", "registration", "verification",
        "authorization", "subscription", "environment", "production",
        "development", "integration", "application", "infrastructure",
    ]
    for word in trunc_sources:
        for n in [2, 3, 4]:
            if n < len(word) - 1:
                abbr = word[:n]
                if abbr not in DB_FULL_WORDS and abbr.lower() not in {w.lower() for w in SHORT_WORDS_2_3 + SHORT_WORDS_4}:
                    examples.append({"token": abbr, "label": 1})

    # Pattern 2: consonant-only abbreviations (drop vowels)
    for word in trunc_sources:
        consonants = "".join(c for c in word.lower() if c not in "aeiou")
        for n in [2, 3, 4, 5]:
            if n <= len(consonants):
                abbr = consonants[:n]
                if len(abbr) >= 2:
                    examples.append({"token": abbr, "label": 1})

    # --- NEGATIVES (label=0): full word tokens ---
    all_full_words = set()
    for w in SHORT_WORDS_2_3:
        all_full_words.add(w.lower())
    for w in SHORT_WORDS_4:
        all_full_words.add(w.lower())
    for w in LONGER_FULL_WORDS:
        all_full_words.add(w.lower())
    for w in DB_FULL_WORDS:
        all_full_words.add(w.lower())

    # Remove any words that are in our abbreviation set
    all_full_words -= ALL_ABBREVS

    for word in all_full_words:
        examples.append({"token": word, "label": 0})
        # Extra weight for short words (3-4 chars) -- these are the hard cases
        if len(word) <= 4:
            examples.append({"token": word, "label": 0})
        if len(word) <= 3:
            examples.append({"token": word, "label": 0})

    return examples


def main():
    examples = generate_dataset()

    # Deduplicate (keep all instances for weighting)
    # Count distribution
    pos = [e for e in examples if e["label"] == 1]
    neg = [e for e in examples if e["label"] == 0]

    print(f"Raw examples: {len(examples)}")
    print(f"  Positives (abbreviations): {len(pos)}")
    print(f"  Negatives (full words):    {len(neg)}")
    print(f"  Ratio: {len(pos)/len(neg):.2f}")

    # Keep ALL data -- handle imbalance via class weights during training
    # Oversample the minority class (positives) to improve representation
    if len(pos) < len(neg):
        oversample_factor = max(1, round(len(neg) / len(pos)))
        pos_oversampled = pos * oversample_factor
        random.shuffle(pos_oversampled)
        pos_oversampled = pos_oversampled[:len(neg)]  # cap at neg count
        examples = pos_oversampled + neg
    else:
        examples = pos + neg

    random.shuffle(examples)

    print(f"\nAfter oversampling: {len(examples)}")
    print(f"  Positives: {sum(1 for e in examples if e['label']==1)}")
    print(f"  Negatives: {sum(1 for e in examples if e['label']==0)}")
    print(f"  Unique positive tokens: {len(set(e['token'] for e in examples if e['label']==1))}")
    print(f"  Unique negative tokens: {len(set(e['token'] for e in examples if e['label']==0))}")

    # Length distribution of negatives
    neg_lens = Counter(len(e["token"]) for e in examples if e["label"] == 0)
    print(f"\nNegative token length distribution:")
    for length in sorted(neg_lens):
        print(f"  {length} chars: {neg_lens[length]}")

    # Split: 85% train, 15% val
    split = int(len(examples) * 0.85)
    train_data = examples[:split]
    val_data = examples[split:]

    # Save
    out_dir = Path("clf_training_data")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "clf_train.jsonl", "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    with open(out_dir / "clf_val.jsonl", "w") as f:
        for ex in val_data:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved to {out_dir}/:")
    print(f"  clf_train.jsonl: {len(train_data)} examples")
    print(f"  clf_val.jsonl:   {len(val_data)} examples")

    # Show some samples from hard cases
    print(f"\nSample hard negatives (3-4 char full words):")
    hard_negs = [e for e in val_data if e["label"] == 0 and len(e["token"]) <= 4]
    for e in hard_negs[:20]:
        print(f"  '{e['token']}' -> label={e['label']}")

    print(f"\nSample positives (abbreviations):")
    sample_pos = [e for e in val_data if e["label"] == 1]
    for e in sample_pos[:20]:
        print(f"  '{e['token']}' -> label={e['label']}")


if __name__ == "__main__":
    main()
