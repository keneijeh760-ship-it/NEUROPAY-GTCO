import re
from dataclasses import dataclass


@dataclass
class PreprocessedMessage:
    original: str
    cleaned: str
    language: str  # pidgin | english | yoruba | igbo | hausa | mixed
    noise_level: str  # clean | mild | heavy


class Preprocessor:
    # ── Abbreviations ─────────────────────────────────────────────────────────
    # Order matters — longer/more specific patterns first
    ABBREVIATIONS = [
        # Greetings
        ("gud mrn", "good morning"),
        ("gd mrng", "good morning"),
        ("gud aftn", "good afternoon"),
        ("gd evn", "good evening"),
        ("gm", "good morning"),
        ("wlcm", "welcome"),
        ("tnx", "thanks"),
        ("thnks", "thanks"),
        ("thx", "thanks"),
        ("ty", "thank you"),
        ("pls", "please"),
        ("plz", "please"),
        ("abeg", "please"),

        # Question words (longer first)
        ("hw mch", "how much"),
        ("hw much", "how much"),
        ("how mch", "how much"),
        ("hw mny", "how many"),
        ("how mny", "how many"),
        ("wetin be d price", "what is the price"),
        ("wetin b d price", "what is the price"),
        ("wetin be price", "what is the price"),
        ("weytin", "what is"),
        ("wetin", "what is"),
        ("wats", "what is"),
        ("wat is", "what is"),
        ("whr", "where"),
        ("wen", "when"),
        ("pric", "price"),
        ("hw", "how"),

        # Pidgin phrases (longer first)
        ("how e dey go", "how much does it sell for"),
        ("dem dey sell", "they sell"),
        ("dem sell am", "they sell it"),
        ("e dey go 4", "it sells for"),
        ("dey go 4", "sells for"),
        ("dey go for", "sells for"),
        ("dey sell 4", "sells for"),
        ("dey sell for", "sells for"),
        ("e dey go", "it sells for"),
        ("dey sell", "sells for"),
        ("e don reach", "it has gotten to"),
        ("e don change", "it has changed"),
        ("e still dey", "it is still"),
        ("no be", "is not"),
        ("nor be", "is not"),
        ("e no reach", "it is less than"),
        ("sabi", "know"),
        ("sha", "anyway"),
        ("oya", "okay"),
        ("wahala", "problem"),
        ("palava", "problem"),
        ("abi", "or"),
        ("sef", "even"),

        # Tense / time markers
        ("2day", "today"),
        ("tday", "today"),
        ("ystrdy", "yesterday"),
        ("yest", "yesterday"),
        ("ystd", "yesterday"),
        ("tmrw", "tomorrow"),
        ("lst wk", "last week"),
        ("lst week", "last week"),
        ("b4", "before"),
        ("afta", "after"),
        ("nw", "now"),
        ("na", "is"),

        # Quantity shorthand
        ("hf doz", "half dozen"),
        ("hlf doz", "half dozen"),
        ("half doz", "half dozen"),
        ("qtr bag", "quarter bag"),
        ("hf bag", "half bag"),
        ("hlf bag", "half bag"),
        ("doz", "dozen"),
        ("dzn", "dozen"),
        ("pcs", "pieces"),
        ("pkt", "packet"),
        ("pkts", "packets"),
        ("kilo", "kilogram"),
        ("ltr", "litre"),
        ("ltrs", "litres"),
        ("hf", "half"),
        ("hlf", "half"),
        ("qtr", "quarter"),

        # Chat abbreviations
        ("4rm", "from"),
        ("frm", "from"),
        ("bcus", "because"),
        ("bcos", "because"),
        ("cos", "because"),
        ("cuz", "because"),
        ("wit", "with"),
        ("wt", "with"),
        ("abt", "about"),
        ("smtin", "something"),
        ("nid", "need"),
        ("shud", "should"),
        ("wud", "would"),
        ("cud", "could"),
        ("mk", "make"),
        ("gt", "got"),
        ("hv", "have"),
        ("cn", "can"),
        ("nd", "and"),

        # Noise / filler — strip these
        ("lol", ""),
        ("haha", ""),
        ("hehe", ""),
        ("chai", ""),
        ("choi", ""),
    ]

    NUMBER_SUBS = [
        (r'\b4\b', "for"),
        (r'\b2\b', "to"),
        (r'\bu\b', "you"),
        (r'\br\b', "are"),
        (r'\bbt\b', "but"),
        (r'\bd\b', "the"),
        (r'\b8\b', "ate"),
        (r'\bm8\b', "mate"),
    ]

    # ── Product aliases ───────────────────────────────────────────────────────
    PRODUCT_ALIASES = {
        # Tomato
        "tomatoe": "tomato",
        "tomatos": "tomato",
        "tomatoes": "tomato",
        "atarodo": "tomato",
        "tomatillo": "tomato",
        "tomato pepper": "tomato",

        # Pepper
        "tatashe": "red bell pepper",
        "tatase": "red bell pepper",
        "ata tatashe": "red bell pepper",
        "shombo": "cayenne pepper",
        "rodo": "scotch bonnet pepper",
        "ata rodo": "scotch bonnet pepper",
        "bawa": "dry pepper",
        "green pepper": "green bell pepper",

        # Onion
        "onions": "onion",
        "albasa": "onion",
        "yabasi": "onion",

        # Fish
        "titus": "mackerel",
        "panla": "dried stockfish",
        "eja osan": "smoked fish",
        "eja kika": "dried catfish",
        "eja nla": "big fish",
        "bonga fish": "bonga fish",
        "bonga": "bonga fish",
        "cat fish": "catfish",
        "stock fish": "stockfish",
        "dry fish": "dried fish",
        "fresh fish": "fresh fish",
        "crafish": "crayfish",
        "kifi": "crayfish",
        "ede": "crayfish",

        # Meat
        "ponmo": "cow skin",
        "kanda": "cow skin",
        "bokoto": "cow leg",
        "cow leg": "cow leg",
        "isi ewu": "goat head",
        "goat": "goat meat",
        "shaki": "tripe",
        "saki": "tripe",
        "abodi": "intestine",
        "ngolo": "snail",
        "bush meat": "bushmeat",

        # Grains & staples
        "gari": "garri",
        "eba": "garri",
        "garri ijebu": "ijebu garri",
        "ijebu gari": "ijebu garri",
        "ogi": "corn pap",
        "akamu": "corn pap",
        "pap": "corn pap",
        "fufu": "pounded yam flour",
        "semo": "semolina",
        "semovita": "semovita",

        # Rice
        "abakaliki rice": "local rice",
        "ofada": "ofada rice",
        "foreign rice": "foreign rice",
        "imported rice": "foreign rice",
        "thai rice": "foreign rice",
        "basmati": "basmati rice",
        "mama gold rice": "mama gold rice",

        # Beans & legumes
        "black eye beans": "black-eyed peas",
        "honey beans": "honey beans",
        "oloyin": "honey beans",
        "ewa": "beans",
        "groundnut": "groundnut",
        "peanut": "groundnut",
        "egusi": "melon seeds",
        "egushi": "melon seeds",
        "ogbono": "ogbono seeds",
        "ukwa": "breadfruit",
        "gyada": "groundnut",

        # Yam & tubers
        "isu": "yam",
        "cocoyam": "cocoyam",
        "sweet potato": "sweet potato",
        "irish potato": "irish potato",
        "unripe plantain": "unripe plantain",
        "ripe plantain": "ripe plantain",

        # Oils
        "red oil": "palm oil",
        "veg oil": "vegetable oil",
        "man shanu": "palm oil",

        # Vegetables
        "efo": "leafy vegetable",
        "ugwu": "fluted pumpkin leaf",
        "ugu": "fluted pumpkin leaf",
        "pumpkin leaf": "fluted pumpkin leaf",
        "waterleaf": "water leaf",
        "water leaf": "water leaf",
        "bitterleaf": "bitter leaf",
        "bitter leaf": "bitter leaf",
        "efirin": "basil leaf",
        "scent leaf": "scent leaf",
        "okro": "okra",
        "garden egg": "garden egg",
        "garden eggs": "garden egg",
        "igba": "garden egg",
        "corn": "maize",

        # Condiments
        "maggi": "seasoning cube",
        "knorr": "seasoning cube",
        "royco": "seasoning powder",
        "tin tomato": "canned tomato",
        "tin tomatoe": "canned tomato",
        "tomato paste": "tomato paste",
        "tomato puree": "tomato paste",
        "dawadawa": "locust beans",

        # Dairy & drinks
        "peak milk": "peak milk",
        "nono": "fresh milk",
        "wara": "local cheese",
        "zobo": "zobo drink",
        "kunu": "kunu drink",

        # Processed
        "indomie": "instant noodles",
        "agege bread": "agege bread",
        "golden morn": "golden morn cereal",
    }

    # ── Unit aliases ──────────────────────────────────────────────────────────
    UNIT_ALIASES = {
        # Containers
        "paint bucket": "5L container",
        "paint rubber": "5L container",
        "paint tin": "5L container",
        "painter": "5L container",
        "5l": "5L container",
        "small rubber": "small container",
        "big rubber": "large container",
        "jerry can": "jerry can",
        "jerrican": "jerry can",
        "kerosene keg": "kerosene keg",
        "big bottle": "large bottle",
        "small bottle": "small bottle",

        # Dry measures
        "mudu": "mudu (dry measure)",
        "olodo": "olodo measure",
        "congo": "congo measure",
        "milk tin": "milk tin measure",
        "tomato tin": "tomato tin measure",
        "big cup": "large cup",
        "small cup": "small cup",
        "big bowl": "large bowl",
        "small bowl": "small bowl",
        "big plate": "large plate",

        # Baskets & bags
        "big basket": "large basket",
        "small basket": "small basket",
        "crate": "crate (30 units)",
        "egg crate": "crate (30 units)",
        "tray": "tray (30 units)",
        "half bag": "25kg bag",
        "quarter bag": "12.5kg bag",
        "50kg": "50kg bag",
        "25kg": "25kg bag",
        "10kg": "10kg bag",
        "5kg": "5kg bag",

        # Weight
        "kilo": "kilogram",
        "kg": "kilogram",

        # Count
        "pcs": "pieces",
        "dozen": "dozen",
        "doz": "dozen",
        "half dozen": "half dozen",
        "hf doz": "half dozen",
        "hand": "hand (bunch)",

        # Volume
        "ltr": "litre",
        "ltrs": "litres",
    }

    # ── Location aliases ──────────────────────────────────────────────────────
    # Keys are lowercase. Values are properly cased canonical names.
    # Applied AFTER lowercasing — output is canonical cased string.
    LOCATION_ALIASES = {
        # Lagos
        "mile12 market": "Mile 12",
        "mile 12 market": "Mile 12",
        "mile12 mkt": "Mile 12",
        "mile 12 mkt": "Mile 12",
        "mile12": "Mile 12",
        "mile 12": "Mile 12",
        "oyingbo market": "Oyingbo Market",
        "oyingbo mkt": "Oyingbo Market",
        "oyingbo": "Oyingbo Market",
        "oshodi market": "Oshodi Market",
        "oshodi mkt": "Oshodi Market",
        "oshodi": "Oshodi Market",
        "mushin market": "Mushin Market",
        "mushin mkt": "Mushin Market",
        "mushin": "Mushin Market",
        "tejuosho market": "Tejuosho Market",
        "tejuosho": "Tejuosho Market",
        "tejo": "Tejuosho Market",
        "agege market": "Agege Market",
        "agege mkt": "Agege Market",
        "agege": "Agege Market",
        "daleko market": "Daleko Market",
        "daleko": "Daleko Market",
        "idumota market": "Idumota Market",
        "idumota": "Idumota Market",
        "balogun market": "Balogun Market",
        "balogun mkt": "Balogun Market",
        "balogun": "Balogun Market",
        "computer village ikeja": "Computer Village Ikeja",
        "computer village": "Computer Village Ikeja",
        "alaba international market": "Alaba International Market",
        "alaba intl market": "Alaba International Market",
        "alaba intl": "Alaba International Market",
        "alaba": "Alaba International Market",
        "trade fair": "Lagos Trade Fair",
        "tradefair": "Lagos Trade Fair",
        "ketu market": "Ketu Market",
        "ketu": "Ketu Market",
        "surulere": "Surulere Market",
        "ikorodu": "Ikorodu Market",
        "lekki market": "Lekki Market",
        "lekki mkt": "Lekki Market",
        "lekki": "Lekki Market",
        "ajah": "Ajah Market",
        "festac": "Festac Market",
        "iyana ipaja": "Iyana Ipaja Market",
        "abule egba": "Abule Egba Market",
        "otto": "Otto Market Lagos",

        # Abuja
        "wuse 2 market": "Wuse 2 Market",
        "wuse 2": "Wuse 2 Market",
        "wuse market": "Wuse Market",
        "wuse mkt": "Wuse Market",
        "wuse": "Wuse Market",
        "garki market": "Garki Market",
        "garki mkt": "Garki Market",
        "garki": "Garki Market",
        "utako market": "Utako Market",
        "utako": "Utako Market",
        "nyanya market": "Nyanya Market",
        "nyanya": "Nyanya Market",
        "kubwa": "Kubwa Market",
        "lugbe": "Lugbe Market",
        "gwagwalada": "Gwagwalada Market",
        "mararaba": "Mararaba Market",
        "jabi": "Jabi Market",
        "gwarimpa": "Gwarimpa Market",
        "area 1": "Area 1 Market Abuja",
        "area 3": "Area 3 Market Abuja",

        # Kano
        "kurmi market": "Kurmi Market",
        "kurmi mkt": "Kurmi Market",
        "kurmi": "Kurmi Market",
        "kasuwar barci": "Kasuwar Barci",
        "yankura": "Yankura Market Kano",
        "sabon gari market kano": "Sabon Gari Market Kano",
        "sabon gari": "Sabon Gari Market Kano",
        "rimi market": "Rimi Market Kano",
        "rimi": "Rimi Market Kano",
        "kantin kwari": "Kantin Kwari Market",
        "dawanau": "Dawanau Grain Market",
        "kasuwa": "Central Market Kano",

        # Port Harcourt
        "mile 3 market": "Mile 3 Market PH",
        "mile 3 mkt": "Mile 3 Market PH",
        "mile3 mkt": "Mile 3 Market PH",
        "mile3": "Mile 3 Market PH",
        "mile 3": "Mile 3 Market PH",
        "rumuola market": "Rumuola Market",
        "rumuola mkt": "Rumuola Market",
        "rumuola": "Rumuola Market",
        "waterlines": "Waterlines Market PH",
        "creek road": "Creek Road Market PH",
        "oil mill": "Oil Mill Market PH",
        "choba": "Choba Market",
        "rumuokoro": "Rumuokoro Market",

        # Ibadan
        "bodija market": "Bodija Market",
        "bodija mkt": "Bodija Market",
        "bodija": "Bodija Market",
        "dugbe market": "Dugbe Market",
        "dugbe mkt": "Dugbe Market",
        "dugbe": "Dugbe Market",
        "gbagi": "Gbagi Market Ibadan",
        "oja oba ibadan": "Oja Oba Market Ibadan",
        "oja oba": "Oja Oba Market Ibadan",
        "agbeni": "Agbeni Market Ibadan",
        "molete": "Molete Market Ibadan",
        "ojoo": "Ojoo Market Ibadan",

        # Onitsha / Aba
        "onitsha main market": "Onitsha Main Market",
        "onitsha mkt": "Onitsha Main Market",
        "onitsha": "Onitsha Main Market",
        "ariaria market aba": "Ariaria Market Aba",
        "ariaria mkt": "Ariaria Market Aba",
        "ariaria": "Ariaria Market Aba",
        "ngwa road": "Ngwa Road Market Aba",
        "aba": "Aba Main Market",

        # Enugu
        "ogbete market": "Ogbete Market Enugu",
        "ogbete mkt": "Ogbete Market Enugu",
        "ogbete": "Ogbete Market Enugu",
        "abakpa": "Abakpa Market Enugu",

        # Kaduna
        "kawo": "Kawo Market Kaduna",
        "tuesday market kaduna": "Tuesday Market Kaduna",
        "tuesday market": "Tuesday Market Kaduna",
        "kaduna central": "Central Market Kaduna",
        "bantamba": "Bantamba Market Kaduna",

        # Benin City
        "uselu": "Uselu Market Benin",
        "oba market": "Oba Market Benin",
        "new benin": "New Benin Market",

        # Warri
        "igbudu": "Igbudu Market Warri",
        "ekpan": "Ekpan Market Warri",

        # Jos
        "terminus market": "Terminus Market Jos",
        "jos terminus": "Terminus Market Jos",
        "terminus": "Terminus Market Jos",

        # Maiduguri
        "monday market maiduguri": "Monday Market Maiduguri",
        "monday market": "Monday Market Maiduguri",

        # Sokoto
        "sokoto central": "Central Market Sokoto",
        "emir market": "Emir Market Sokoto",
    }

    # ── Language markers ──────────────────────────────────────────────────────
    PIDGIN_MARKERS = {
        "dey", "na", "abeg", "wetin", "weytin", "abi", "sef",
        "dem", "una", "wahala", "chop", "ginger", "sabi",
        "sha", "palava", "kpele", "ehn", "ehen",
        "walahi", "kai", "chai", "choi", "haba",
        "nah", "nor", "e don", "e dey", "waka",
        "correct", "razz", "famz", "kolo", "mumu", "olodo",
        "joor", "jare", "jo", "nau", "now now",
        "siddon", "comot", "carry", "give am", "take am",
        "buy am", "sell am", "how e", "no be", "nor be",
        "e no", "na im", "na so", "no wahala",
        "no worry", "e easy", "e hard", "enter",
    }

    YORUBA_MARKERS = {
        "eja", "efo", "ogi", "ponmo", "isu", "ewa", "asun",
        "ewedu", "gbegiri", "moin", "ekuru", "gbodo",
        "eja osan", "eja kika", "ata", "tatashe",
        "atarodo", "efirin", "ugwu", "ugu", "igba",
        "eku", "bawo", "pele", "kaabo", "odabo",
        "jowo", "dupe", "ese", "ope", "beeni", "rara",
        "eyin", "emi", "awa", "won", "ibo",
        "tabi", "sugbon", "nitori", "nibo", "nkan",
        "ogun", "egbe", "owo", "ile", "oko",
        "onje", "omi", "egan", "oja", "oja oba",
        "iru", "ogiri",
    }

    IGBO_MARKERS = {
        "ugwu", "ofe", "ukwa", "abacha", "oha", "egusi",
        "ogiri", "utazi", "uziza", "ede",
        "nkwobi", "akpu", "ji",
        "biko", "daalu", "ezigbo", "nna",
        "nne", "nwanne", "mba", "ee",
        "bia", "kedu", "o di mma",
        "oku", "mmiri", "nri", "ahia",
        "onye", "ihe", "ulo", "ogige",
        "obodo", "madu", "ora",
        "ofe onugbu", "ofe akwu", "ofe oha",
        "oka", "nkwobi",
    }

    HAUSA_MARKERS = {
        "tuwo", "fura", "kilishi", "kunu", "miyan",
        "dawa", "gero", "masara", "gyada", "tsamiya",
        "albasa", "karas", "wake", "alkama", "shinkafa",
        "naman", "kifi", "kwai", "mai", "tattasai",
        "dawadawa", "barkono", "karago",
        "sannu", "yawwa", "nagode", "lafiya",
        "ina kwana", "ina wuni", "sai anjima",
        "to", "ai", "da", "ba", "ko",
        "amma", "don", "abin", "wane", "yaya",
        "abinci", "kasuwa", "gida", "ruwa",
        "mutum", "yaro", "mace", "dan",
        "rana", "dare", "safe", "yau",
        "gobe", "jiya",
    }

    def __init__(self):
        self._number_patterns = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.NUMBER_SUBS
        ]

    # ── Public interface ──────────────────────────────────────────────────────
    def process(self, raw_message: str) -> PreprocessedMessage:
        noise_level = self._measure_noise(raw_message)
        language = self._detect_language(raw_message)  # detect on original
        cleaned = self._clean(raw_message)

        return PreprocessedMessage(
            original=raw_message,
            cleaned=cleaned,
            language=language,
            noise_level=noise_level,
        )

    # ── Cleaning pipeline ─────────────────────────────────────────────────────
    def _clean(self, text: str) -> str:
        text = text.lower().strip()
        text = self._strip_punctuation(text)
        text = self._apply_abbreviations(text)
        text = self._apply_number_subs(text)
        text = self._normalize_products(text)
        text = self._normalize_units(text)
        text = self._normalize_locations(text)  # last — preserves canonical case
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _strip_punctuation(self, text: str) -> str:
        text = re.sub(r'[!?]{2,}', '', text)
        text = re.sub(r'[^\w\s₦.,\'-]', ' ', text)
        return text

    def _apply_abbreviations(self, text: str) -> str:
        for abbrev, full in self.ABBREVIATIONS:
            # Use word-boundary replacement for single-word abbrevs
            if ' ' in abbrev:
                text = text.replace(abbrev, full)
            else:
                text = re.sub(rf'\b{re.escape(abbrev)}\b', full, text)
        # Clean up any double spaces or dangling "is be" artifacts
        text = re.sub(r'\bis\s+be\b', 'is', text)
        text = re.sub(r'\bwhat is\s+is\b', 'what is', text)
        return text

    def _apply_number_subs(self, text: str) -> str:
        for pattern, replacement in self._number_patterns:
            text = pattern.sub(replacement, text)
        return text

    def _normalize_products(self, text: str) -> str:
        for raw, canonical in sorted(self.PRODUCT_ALIASES.items(),
                                     key=lambda x: len(x[0]), reverse=True):
            text = re.sub(rf'\b{re.escape(raw)}\b', canonical, text)
        return text

    def _normalize_units(self, text: str) -> str:
        for raw, canonical in self.UNIT_ALIASES.items():
            # Multi-word unit aliases first via plain replace, single-word via boundary
            if ' ' in raw:
                text = text.replace(raw, canonical)
            else:
                text = re.sub(rf'\b{re.escape(raw)}\b', canonical, text)
        return text

    def _normalize_locations(self, text: str) -> str:
        """
        Applied on lowercased text. Replaces keys with properly-cased
        canonical location names. Longer keys checked first to avoid
        partial matches (e.g. 'onitsha mkt' before 'onitsha').
        Once a location is replaced we stop checking to avoid re-matching
        inside the canonical output.
        """
        for raw, canonical in sorted(self.LOCATION_ALIASES.items(),
                                     key=lambda x: len(x[0]), reverse=True):
            if raw in text:
                text = text.replace(raw, canonical)
                break  # stop after first match to prevent re-matching canonical
        return text

    # ── Language detection ────────────────────────────────────────────────────
    def _detect_language(self, text: str) -> str:
        words = set(text.lower().split())
        scores = {
            "pidgin": len(words & self.PIDGIN_MARKERS),
            "yoruba": len(words & self.YORUBA_MARKERS),
            "igbo": len(words & self.IGBO_MARKERS),
            "hausa": len(words & self.HAUSA_MARKERS),
        }
        top = max(scores, key=scores.get)
        top_val = scores[top]

        if top_val == 0:
            return "english"

        non_zero = [k for k, v in scores.items() if v > 0]
        if len(non_zero) > 1:
            return "mixed"

        return top

    # ── Noise measurement ─────────────────────────────────────────────────────
    def _measure_noise(self, text: str) -> str:
        score = 0
        words = text.lower().split()
        for word in words:
            if len(word) <= 2 and not word.isdigit():
                score += 1
            if re.search(r'(.)\1{2,}', word):
                score += 1
            if re.search(r'\d', word) and re.search(r'[a-z]', word):
                score += 1
        ratio = score / max(len(words), 1)
        if ratio >= 0.4:
            return "heavy"
        elif ratio >= 0.15:
            return "mild"
        else:
            return "clean"