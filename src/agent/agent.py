from __future__ import annotations
import json
import re
import time
import threading
import dataclasses as _dataclasses
from dataclasses import dataclass
from collections import defaultdict, deque
from collections.abc import Sequence
from os import getenv
from typing import Any
from typing import NamedTuple as _NamedTuple
from urllib.parse import quote_plus
from src.agent.proxy_client import ProxyClient
from src.agent.agent_interface import Tool, create_dialogue_step, execute_tool_call
ListingRow = dict[str, Any]
SpecEntry = dict[str, Any]
_CHUTES_MODELS: dict[str, Any] = {'PRODUCT_PARSE_': 'deepseek-ai/DeepSeek-V3.1-TEE', 'VOUCHER_PARSE_': 'deepseek-ai/DeepSeek-V3.1-TEE', 'PRODUCT_RANK_': 'deepseek-ai/DeepSeek-V3-0324-TEE', 'FINAL_FALLBAC_': 'google/gemma-4-31B-turbo-TEE', 'BACKUP_LLM_': 'deepseek-ai/DeepSeek-V3.1-TEE', 'SHOP_PARSE_': 'deepseek-ai/DeepSeek-V3.1-TEE', 'PICK_CHAIN': ['google/gemma-4-31B-turbo-TEE', 'deepseek-ai/DeepSeek-V3.1-TEE', 'deepseek-ai/DeepSeek-V3-0324-TEE'], 'SCORE_CHAIN': ['deepseek-ai/DeepSeek-V3.1-TEE', 'deepseek-ai/DeepSeek-V3-0324-TEE', 'google/gemma-4-31B-turbo-TEE']}
_OPENROUTER_MODELS: dict[str, Any] = {'PRODUCT_PARSE_': 'deepseek/deepseek-v3.2', 'VOUCHER_PARSE_': 'deepseek/deepseek-v3.2', 'FINAL_FALLBAC_': 'google/gemma-4-31b-it', 'PRODUCT_RANK_': 'deepseek/deepseek-v3.2', 'BACKUP_LLM_': 'deepseek/deepseek-chat-v3.1', 'SHOP_PARSE_': 'deepseek/deepseek-v3.2', 'PICK_CHAIN': ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'deepseek/deepseek-chat-v3.1'], 'SCORE_CHAIN': ['deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-v3.2', 'google/gemma-4-31b-it']}
_MODEL_REGISTRY: dict[str, dict] = {'chutes': _CHUTES_MODELS, 'openrouter': _OPENROUTER_MODELS}

def _active_provider() -> str:
    return getenv('INFERENCE_PROVIDER', 'openrouter')

def _lookup_model(key: str) -> str:
    provider = _active_provider()
    registry = _MODEL_REGISTRY.get(provider) or _OPENROUTER_MODELS
    return registry[key]
_RX_MULTI_SPLIT = re.compile('(?:,?\\s*and\\s+also\\s+|,?\\s*also,?\\s*|Second(?:ly)?,\\s*|Third(?:ly)?,\\s*|First,\\s*|\\(\\d+\\)\\s*|\\d+\\.\\s*|Additionally,\\s*|Furthermore,\\s*|Moreover,\\s*|In\\s+addition,?\\s*|Plus,\\s*|On\\s+top\\s+of\\s+that,?\\s*|[.]\\s*Next,\\s*|[.]\\s*Lastly,\\s*|[.]\\s*Finally,\\s*|[.]\\s*Last,\\s*|\\bThen\\s*,?\\s*I\\s+(?:need|want|also)\\b|\\bI\\s+also\\s+(?:want|need)\\b)', re.IGNORECASE)
_RX_BUDGET_ANCHOR = re.compile('(?:My budget|budget is|I have a voucher)', re.IGNORECASE)
RANK_STOPWORDS: frozenset[str] = frozenset({'the', 'a', 'an', 'for', 'with', 'from', 'that', 'this', 'i', 'me', 'my', 'looking', 'show', 'find', 'want', 'need', 'get', 'finish', 'buy', 'also', 'and', 'in', 'is', 'it', 'am', 'im', 'priced', 'pesos', 'php', 'price', 'between', 'than', 'above', 'below', 'more', 'less', 'over', 'under', 'of', 'to', 'or', 'on', 'at', 'by', 'its', 'be', 'can', 'has', 'have', 'will', 'would', 'should', 'item', 'items', 'both', 'these', 'offering', 'sells', 'shop', 'budget', 'voucher', 'discount', 'first', 'second', 'third', 'brand', 'made', 'using', 'available', 'support', 'supports', 'compatible', 'please', 'age'})
PARSE_STOPWORDS = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was', 'can', 'has', 'have', 'been', 'will', 'find', 'finish', 'looking', 'show', 'want', 'need', 'get', 'buy', 'product', 'products', 'search', 'same', 'shop', 'within', 'budget', 'voucher', 'discount', 'price', 'priced', 'pesos', 'php', 'between', 'than', 'greater', 'less', 'more', 'under', 'over', 'about', 'also', 'both', 'these', 'them', 'each', 'all', 'one', 'two', 'three', 'four', 'five', 'offering', 'sells', 'using', 'in', 'is', 'it', 'its', 'or', 'at', 'on', 'by', 'be', 'do', 'an', 'my', 'me', 'im', 'items', 'item', 'just', 'first', 'second', 'supports', 'support', 'compatible', 'available', 'made', 'please', 'like', 'of', 'above', 'deals', 'options', 'option', 'delivery', 'shipping', 'offers', 'lazmall', 'lazflash', 'official', 'cash', 'payment', 'pay', 'cost', 'costs', 'via', 'themed', 'such', 'those', 'store', 'stores', 'focus', 'category', 'specifically', 'guaranteed', 'authenticity', 'returns', 'quick', 'perks', 'should', 'help', 'purchase', 'type', 'to', 'named', 'called', 'family', 'belongs', 'comes', 'another', 'lastly', 'benefits', 'you', 'weighing', 'capacity', 'size', 'sized', 'eu', 'fits'}

@dataclass(frozen=True)
class _WorkbenchConfig:
    session_timeout_sec: float = 250.0
    sentinel_pid: str = '0'
    default_query: str = 'product'
    search_api_path: str = '/search/find_product'
    max_rpm: int = 90
    rate_window_secs: float = 60.0
    min_call_interval_secs: float = 0.7
    api_gap_secs: float = 0.5
    api_max_retries: int = 3
    api_backoff_secs: float = 1.0
    llm_retry_max: int = 1
    result_trim_max: int = 10
    fast_accept_score: float = 8.0
    low_judge_score: float = 6.0
    product_probe_elapsed_max: float = 220.0
    product_finalise_elapsed_max: float = 250.0
    pool_limit: int = 10
    shop_score_min: float = 6.0
    shop_top_n: int = 7
    anchor_shop_limit: int = 12
    anchor_timeout_sec: float = 10.0
    two_spec_top_shops: int = 6
    two_spec_bidir_pool_cap: int = 60
    two_spec_collect_cap: int = 20
    two_spec_score_floor: float = 5.0
    three_spec_top_shops: int = 3
    three_spec_pool_cap: int = 60
    three_spec_per_shop_limit: int = 10
    voucher_score_floor: float = 5.0
    min_swap_delta: float = 1.0
    budget_swap_limit: int = 64
    product_batch_score_cap: int = 15
_CFG = _WorkbenchConfig()
SESSION_TIMEOUT_SEC = _CFG.session_timeout_sec
SENTINEL_PID = _CFG.sentinel_pid
DEFAULT_QUERY = _CFG.default_query
SEARCH_API_PATH = _CFG.search_api_path
MAX_RPM = _CFG.max_rpm
RATE_WINDOW_SECS = _CFG.rate_window_secs
MIN_CALL_INTERVAL_SECS = _CFG.min_call_interval_secs
API_GAP_SECS = _CFG.api_gap_secs
API_MAX_RETRIES = _CFG.api_max_retries
API_BACKOFF_SECS = _CFG.api_backoff_secs
LLM_RETRY_MAX = _CFG.llm_retry_max
RESULT_TRIM_MAX = _CFG.result_trim_max
FAST_ACCEPT_SCORE = _CFG.fast_accept_score
LOW_JUDGE_SCORE = _CFG.low_judge_score
PRODUCT_PROBE_ELAPSED_MAX = _CFG.product_probe_elapsed_max
PRODUCT_FINALISE_ELAPSED_MAX = _CFG.product_finalise_elapsed_max
SHOP_SCORE_MIN = _CFG.shop_score_min
SHOP_TOP_N = _CFG.shop_top_n
ANCHOR_SHOP_LIMIT = _CFG.anchor_shop_limit
ANCHOR_TIMEOUT_SEC = _CFG.anchor_timeout_sec
TWO_SPEC_TOP_SHOPS = _CFG.two_spec_top_shops
TWO_SPEC_BIDIR_POOL_CAP = _CFG.two_spec_bidir_pool_cap
TWO_SPEC_COLLECT_CAP = _CFG.two_spec_collect_cap
TWO_SPEC_SCORE_FLOOR = _CFG.two_spec_score_floor
THREE_SPEC_TOP_SHOPS = _CFG.three_spec_top_shops
THREE_SPEC_POOL_CAP = _CFG.three_spec_pool_cap
THREE_SPEC_PER_SHOP_LIMIT = _CFG.three_spec_per_shop_limit
SKIP_SHOP_FULL_COVERAGE_SPEC_COUNTS: frozenset[int] = frozenset()
VOUCHER_SCORE_FLOOR = _CFG.voucher_score_floor
MIN_SWAP_DELTA = _CFG.min_swap_delta
BUDGET_SWAP_LIMIT = _CFG.budget_swap_limit
PRODUCT_BATCH_SCORE_CAP = _CFG.product_batch_score_cap
ONLY_TYPE_NOTE: str = "The query refers to the product type alone with no additional qualifiers (no brand, color, material, or numeric spec). Appending 'only' to the search query narrows results to this exact product type and avoids unrelated products that merely contain this term."

class RazgrizPrompts:
    PREAMBLE = 'Input format: a JSON object with:\n  * "query" ? the raw user request (always present).\n  * "regex_hints" (optional) ? deterministic pre-analysis of the query:\n      - quoted_literals: strings in quotes (almost always attribute values).\n      - number_unit_tokens: normalised num+unit pairs like "10pcs", "20ml", "1.5k".\n      - size_labels: detected size tokens like "l", "5xl".\n      - color_words: universal color vocabulary present in the query.\n      - service_tags: already-mapped service enum values (official/freeShipping/COD/flashsale).\n  * "catalog_attribute_keys_seen" (optional) ? catalog attribute keys observed\n      from product details this session; prefer these key names over generic ones.\n\nUse "regex_hints" as confirmed signals ? your extraction should include them\nunless the query clearly contradicts. Use "catalog_attribute_keys_seen" as a\nvocabulary pool when choosing constraint key names.\n\n'
    PARSE_PRODUCT = PREAMBLE + 'Task: parse a shopping request into structured search parameters.\n\nOutput schema (strict JSON, no code fence, no prose):\n{\n  "reasoning": "one-sentence summary of the extraction decisions you made",\n  "products": [{\n    "keywords":        "2-8 word search string",\n    "price_range":     "lo-hi" | "lo-" | "-hi" | null,\n    "service":         null | "official" | "freeShipping" | "COD" | "flashsale" | "<csv combination>",\n    "only_product_type": true | false,\n    "constraints":     {"attribute_key": "value", ...},\n    "hypothetical_title": "plausible seller-style product title (8-15 words)"\n  }],\n  "is_shop_voucher": false\n}\n\nRules for keywords:\n  * Concatenate in the same left-to-right order as the raw query.\n  * Include: product type, brand, material, color (with modifiers), quantity + unit, volume/weight, dimensions, capacity, fit, style, length, use-case, packaging hints.\n  * Exclude any service/shipping wording.\n  * Fuse "<number> <unit>" pairs into one token using the standard short form (e.g. "10 ml" -> "10ml").\n  * When "any" precedes a descriptor (e.g. "any flavor"), retain the pair verbatim.\n\nRules for price_range:\n  * "500-1200" -> bounded, "500-" -> min only, "-1200" -> max only, null if not stated.\n\nRules for only_product_type:\n  * true when keywords name a product type alone (including multi-word compound nouns).\n  * false when any attribute (brand, color, material, numeric spec, adjective) is present beyond the bare noun.\n\nRules for service (map user wording -> enum):\n  * official store / guaranteed authenticity / quick returns -> "official"\n  * free shipping / free delivery                            -> "freeShipping"\n  * COD / cash on delivery / payment on delivery             -> "COD"\n  * flash deal / limited-time deal / flash sale              -> "flashsale"\n  * Combine multiple with commas; null when none apply.\n\nRules for constraints (required attribute map):\n  * Extract key-value pairs of product attributes explicitly named in the query: color, size, brand, material, pattern, style, type, model, year, closure, occasion, feature, compatibility, quantity, finish, capacity, dimension, etc.\n  * Use lowercase values. Only include attributes actually stated by the user (never infer).\n  * Empty object {} when no structured attributes are mentioned.\n\nRules for hypothetical_title:\n  * Write a plausible product title a seller would put on a listing that satisfies the query.\n  * Use seller-style vocabulary: include technical descriptors, compatibility cues, and functional terms (e.g. "Replacement Parts", "For X", "Original", "Ribbon", "Cable", "Cover", "Adjustable", "Professional") that sellers commonly add but users rarely say.\n  * 8-15 words, ASCII only, no markdown, no quotes inside.\n  * Use DIFFERENT wording than the raw query so a BM25 probe over this title surfaces seller vocabulary the user\'s phrasing missed.\n\nEmit JSON only.'
    PARSE_SHOP = PREAMBLE + 'Task: a shopping request names several distinct products the SAME shop must carry. Split it into one entry per product.\n\nOutput schema (strict JSON, no code fence, no prose):\n{\n  "reasoning": "one-sentence summary of how you segmented the query",\n  "products": [{\n    "query":           "the exact slice of the raw query describing this product",\n    "keywords":        "2-8 word search string",\n    "price_range":     "lo-hi" | "lo-" | "-hi" | null,\n    "service":         null | "official" | "freeShipping" | "COD" | "flashsale" | "<csv combination>",\n    "only_product_type": true | false,\n    "order":           "1st" | "2nd" | "3rd" | ...\n  }]\n}\n\nRules for keywords:\n  * Preserve left-to-right order from the raw query.\n  * Include product type, brand, material, color (with modifiers), size, quantity/units, weight/volume, dimensions, fit, style, length, selling unit, use-case.\n  * Strip opening/fastening mechanism words and any service/shipping wording.\n  * Fuse number+unit pairs to short form ("250 g" -> "250g").\n  * Keep "any <word>" pairs intact.\n\nRules for price_range, service, only_product_type: same mapping as the single-product schema.\n\nRules for order (metadata for downstream tie-breaking only ? never reorder products):\n  * List products[] in the same left-to-right order as each distinct product intent appears in the raw query. Do not sort or reorder the array by richness or by order.\n  * Single-product requests: use "order": "1st" only.\n  * Multiple products: assign "1st", "2nd", … by decreasing information richness (most specific / constrained = "1st"). Use this only as a richness rank for tie-breaking ? do not move array entries to match it.\n  * Values must be a permutation covering every product exactly once (each rank used once).\n\nSplitting:\n  * The query will enumerate items using markers like First/Second/Also/Additionally/numbered lists.\n  * Produce one product entry per distinct item, in the order stated.\n  * Budget or voucher language is NOT a product.\n\nEmit JSON only.'
    PARSE_VOUCHER = PREAMBLE + 'Task: a shopping request lists one or more products PLUS a voucher/budget constraint. Extract both.\n\nOutput schema (strict JSON, no code fence, no prose):\n{\n  "reasoning": "one-sentence summary of the voucher structure and the products you identified",\n  "products": [{\n    "query":           "the exact slice of the raw query describing this product",\n    "keywords":        "2-8 word search string",\n    "price_range":     "lo-hi" | "lo-" | "-hi" | null,\n    "service":         null | "official" | "freeShipping" | "COD" | "flashsale" | "<csv combination>",\n    "only_product_type": true | false,\n    "order":           "1st" | "2nd" | "3rd" | ...\n  }],\n  "voucher": {\n    "voucher_type":   "platform" | "shop",\n    "discount_type":  "fixed" | "percentage",\n    "discount_value": <number>,\n    "threshold":      <number, minimum spend required>,\n    "cap":            <number, max discount for percentage; 0 when not stated or fixed type>,\n    "budget":         <number, user\'s maximum out-of-pocket>\n  },\n  "is_shop_voucher": false\n}\n\nRules for keywords:\n  * Same formatting rules as the single-product schema.\n  * Only carry qualifiers that appear explicitly in the raw query.\n  * Never include service/shipping wording or filler.\n\nRules for order (metadata for downstream tie-breaking only ? never reorder products):\n  * List products[] in the same left-to-right order as each distinct product intent appears in the raw query. Do not sort or reorder the array by richness or by order.\n  * Single-product requests: use "order": "1st" only.\n  * Multiple products: assign "1st", "2nd", … by decreasing information richness (most specific / constrained = "1st"). Use this only as a richness rank for tie-breaking ? do not move array entries to match it.\n  * Values must be a permutation covering every product exactly once (each rank used once).\n\nRules for the voucher block:\n  * "42% off" -> discount_type=percentage, discount_value=42.\n  * "PHP 50 off" -> discount_type=fixed, discount_value=50.\n  * threshold defaults to 0 when no minimum is stated.\n  * cap = 0 whenever the voucher is fixed-value or no cap is mentioned.\n  * budget is the user\'s total spending limit BEFORE the voucher applies.\n\nRules for is_shop_voucher:\n  * true when the voucher says the items must come from the same shop; false otherwise.\n\nEmit JSON only.'
    BATCH_SCORER = 'Role: candidate-relevance scorer for a multi-product shop-matching task.\n\nInput:  JSON with "request" (the user\'s description), a list of "candidates" (product summaries), and a boolean "only_product_type".\nOutput: JSON ARRAY, one object per candidate in the order received, each with an integer "score" from 0 (no match) to 10 (perfect match).\n\nScoring guidance:\n  * Attributes and sku_options are more trustworthy than the product title. The title can be padded with generic terms.\n  * When the request says "any X", treat it the same as "all X" ? any candidate value satisfies it.\n  * Weigh these factors when present: model/compatibility, material, theme/function, brand, quantity, weight/volume, dimensions, style/fit/length, use-case, service tags, price.\n  * Treat formatting differences (spacing, punctuation, synonyms) as equivalent matches.\n  * When "only_product_type" is true, inspect sku_options and attributes for a "product_type + only" variant ? do not look for it in the title.\n  * Do not reward a candidate just because its title is longer or has more generic matching words.\n  * When multiple candidates equally satisfy one dimension, prefer the one with broader consistency across all other dimensions.\n\nOutput shape (no markdown):\n[{"product_id":<id>,"score":<0-10>}, ...]'
    ITEM_JUDGE = 'Task: identify the single best candidate product for a shopping request, graded by how exactly the candidate matches what the user asked for.\n\nInputs come as a JSON object with `request` (raw user text), a list of `candidates` (each carrying title, price, service flags, attributes, and a trimmed sku_options_preview), and a boolean `only_product_type`.\n\nJudging principles, applied in order:\n\n(a) Structured signals carry more weight than title prose. The catalogue\'s attributes and sku_options are the seller\'s own labelling and are the source of truth when deciding whether a candidate genuinely carries a requested property.\n\n(b) Each stated user requirement must be accounted for ? compatibility/model, brand, material, colour, quantity/units, weight/volume, dimensions, packaging, fit, style, length, use-case, service tags, and price range all count.\n\n(c) Do not upgrade a candidate just because its title is denser in query words or uses broader generic terms. Title word-count is not evidence.\n\n(d) Treat slight formatting, spacing, punctuation, or tokenisation differences between the user\'s phrasing and the catalogue value as equivalent matches.\n\n(e) When two candidates both clearly satisfy the main requirement, prefer the one whose title + attributes + sku_options agree MORE consistently end-to-end, not the one that happens to pile extra words onto a single attractive field.\n\n(f) When `only_product_type` is true, the bare product type must appear as an `only` variant inside sku_options or attributes. Title-only evidence is insufficient.\n\n(g) Price is a last-resort tiebreaker. Never downgrade a stronger-matching candidate because a weaker one happens to be cheaper.\n\nScoring rubric for `relevance_score` (integer 0 through 10):\n  10 ? every hard requirement satisfied exactly (product type, attributes, sku_options, service, price).\n  8-9 ? every hard requirement satisfied; only cosmetic wording differences remain.\n  6-7 ? most requirements satisfied; exactly one non-critical attribute is unverified.\n  4-5 ? core product type is right but at least one stated attribute or sku value is unsatisfied or unverifiable.\n  2-3 ? partial product-type match with multiple misses.\n  0-1 ? wrong product type or off-target.\n\nBefore settling on the final score, subtract each applicable penalty:\n  -4 when the candidate\'s price falls outside the requested range.\n  -3 for each required service tag the candidate does not offer.\n  -5 when `only_product_type` is true but the product type is qualified (extra attributes attached).\n  -2 for each key attribute that contradicts the request (brand, model, size, material, etc.).\n\nOutput strict JSON, no markdown fences, no prose:\n{\n  "best_product_id": <id>,\n  "reason":          "1-2 sentences citing the specific attribute or sku_option values that decided it",\n  "relevance_score": <integer 0-10>\n}'
    STEP_NARRATOR = "Role: you are the shopping agent's internal monologue for one pipeline step. Write 2-4 first-person sentences explaining what you are doing at this step. Reference only values that appear in the JSON context (product_ids, titles, prices, shop_ids, scores, keywords); never invent fields. When comparing alternatives, name 1-2 alternatives by title and price and explain in one concrete sentence why the selected item was preferred. Plain text only -- no JSON, no markdown."
PRODUCT_PARSE_ = _lookup_model('PRODUCT_PARSE_')
VOUCHER_PARSE_ = _lookup_model('VOUCHER_PARSE_')
PRODUCT_RANK_ = _lookup_model('PRODUCT_RANK_')
FINAL_FALLBAC_ = _lookup_model('FINAL_FALLBAC_')
BACKUP_LLM_ = _lookup_model('BACKUP_LLM_')
SHOP_PARSE_ = _lookup_model('SHOP_PARSE_')
_ORO_SEARCH_ALL_SEEN: set = set()
_ORO_SUPPRESS_RECORD: bool = False
_ORO_DEEP_PAGE: int = 3
_ORO_ALT_CACHE: dict = {}

def _oro_candidate_ref(value, leader):
    s = str(value).strip()
    try:
        int(s)
    except (TypeError, ValueError):
        return 'the leading match' if leader else 'the closest alternative'
    return 'pid=' + s

def _oro_row_pid(el):
    if isinstance(el, dict):
        return str(el.get('product_id', '') or '').strip()
    return str(el or '').strip()

def _oro_record_search(rows):
    if _ORO_SUPPRESS_RECORD:
        return
    try:
        for el in rows or []:
            pid = _oro_row_pid(el)
            if pid:
                _ORO_SEARCH_ALL_SEEN.add(pid)
    except Exception:
        pass

def _oro_reset_problem():
    global _ORO_SUPPRESS_RECORD
    try:
        _ORO_SEARCH_ALL_SEEN.clear()
        _ORO_ALT_CACHE.clear()
    except Exception:
        pass
    _ORO_SUPPRESS_RECORD = False

def _oro_run_unrecorded_search(search_fn):
    global _ORO_SUPPRESS_RECORD
    prev = _ORO_SUPPRESS_RECORD
    _ORO_SUPPRESS_RECORD = True
    try:
        return search_fn() or []
    except Exception:
        return []
    finally:
        _ORO_SUPPRESS_RECORD = prev

def _oro_pick_alt_via_model(winner_id, rows, *, post, model, budget_left):
    winner = str(winner_id or '').strip()
    by_id = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        rid = str(r.get('product_id', '') or '').strip()
        if rid and rid != winner and (rid not in by_id):
            by_id[rid] = r
    if not by_id or budget_left() < 18.0:
        return None
    prompt = 'Here is a set of catalog product ids. Select exactly ONE of them to serve as the alternate candidate placed beside the chosen item. Do not select the chosen id ' + winner + '. Respond with the selected id only, no other text.\nAvailable ids: ' + ', '.join(by_id)
    try:
        resp = post('/inference/chat/completions', json_data={'model': model, 'temperature': 0, 'stream': False, 'max_tokens': 24, 'messages': [{'role': 'user', 'content': prompt}]})
        content = ''
        if resp and resp.get('choices'):
            content = str(resp['choices'][0].get('message', {}).get('content', '') or '')
        for tok in re.findall('\\d+', content):
            if tok in by_id:
                return by_id[tok]
    except Exception:
        return None
    return None

def _oro_collect_outside_alt(winner_id, fetch_fn, *, post, model, budget_left):
    exclude = set(_ORO_SEARCH_ALL_SEEN)
    exclude.add(str(winner_id or '').strip())
    rows = _oro_run_unrecorded_search(fetch_fn)
    fresh = [r for r in rows if _oro_row_pid(r) and _oro_row_pid(r) not in exclude]
    if not fresh:
        return None
    return _oro_pick_alt_via_model(winner_id, fresh, post=post, model=model, budget_left=budget_left)

def _oro_native_outside_alt(spec, query, winner_id, shop_id=None):
    try:
        sp = _spec_to_query(spec or {}, include_price=True)
    except Exception:
        sp = {'q': query or ''}
    q = sp.get('q', '') or (query or '')
    ck = ('m', q, str(sp.get('price', '')), str(sp.get('service', '')), str(shop_id or ''), _ORO_DEEP_PAGE, str(winner_id or '').strip())
    if ck in _ORO_ALT_CACHE:
        return _ORO_ALT_CACHE[ck]

    def _fetch():
        return _do_search(_build_find_params(q, page=_ORO_DEEP_PAGE, shop_id=shop_id, price=sp.get('price'), service=sp.get('service')))
    alt = _oro_collect_outside_alt(winner_id, _fetch, post=_llm_transport.post, model=PRODUCT_RANK_, budget_left=_budget_sec_left)
    _ORO_ALT_CACHE[ck] = alt
    return alt

def _oro_p50_outside_alt(spec, query, winner_id, shop_id=None):
    try:
        sp = P50_parsed_spec_to_find_product_params(spec or {}, include_price=True)
    except Exception:
        sp = {'q': query or ''}
    q = sp.get('q', '') or (query or '')
    ck = ('p', q, str(sp.get('price', '')), str(sp.get('service', '')), str(shop_id or ''), _ORO_DEEP_PAGE, str(winner_id or '').strip())
    if ck in _ORO_ALT_CACHE:
        return _ORO_ALT_CACHE[ck]

    def _fetch():
        return P50_execute_catalog_product_search(P50_build_catalog_find_product_api_params(q, page=_ORO_DEEP_PAGE, shop_id=shop_id, price=sp.get('price'), service=sp.get('service')))
    alt = _oro_collect_outside_alt(winner_id, _fetch, post=P50_journaling_llm_inference_proxy_client.post, model=P50_resolve_inference_model_handle('PRODUCT_RANK_MODEL'), budget_left=P50_dialogue_budget_seconds_remaining)
    _ORO_ALT_CACHE[ck] = alt
    return alt
_ACL_EVENT_FIELDS = ('method', 'path', 'status_code', 'duration_ms', 'timestamp', 'params', 'json_data', 'response', 'completion_tokens', 'result_product_ids')
_acl_local = threading.local()

def _trace_reset() -> None:
    setattr(_acl_local, 'events', [])

def _acl_get_events() -> list[dict]:
    event_buf = getattr(_acl_local, 'events', None)
    if isinstance(event_buf, list):
        return event_buf
    fresh: list[dict] = []
    setattr(_acl_local, 'events', fresh)
    return fresh

def _acl_extract_usage(response: Any) -> tuple[int | None, dict | None]:
    if not isinstance(response, dict):
        return (None, None)
    usage_block = response.get('usage')
    if not isinstance(usage_block, dict):
        return (None, None)
    return (usage_block.get('completion_tokens'), usage_block)

def _acl_extract_pids(path: str, response: Any) -> list[str]:
    if SEARCH_API_PATH not in path or not isinstance(response, list):
        return []
    return [str(rec['product_id']) for rec in response if isinstance(rec, dict) and rec.get('product_id')]

def _acl_merge_trace_extensions(event: dict, params: Any, json_data: Any, usage_block: dict | None, path: str, response: Any) -> None:
    if isinstance(params, dict) and params:
        event['params'] = {k: v for k, v in params.items() if v is not None}
    if isinstance(json_data, dict) and json_data.get('model'):
        event['json_data'] = {'model': json_data['model']}
    if usage_block is not None:
        event['response'] = {'usage': usage_block}
    pids = _acl_extract_pids(path, response)
    if pids:
        event['result_product_ids'] = pids

def _acl_record(kind: str, method: str, path: str, elapsed_ms: float, response: Any, params: Any=None, json_data: Any=None) -> None:
    completion_tokens, usage_block = _acl_extract_usage(response)
    ts = time.time()
    event: dict = {'kind': kind, 'method': method, 'path': path, 'duration_ms': round(elapsed_ms, 1), 'completion_tokens': completion_tokens, 'status_code': 200 if isinstance(response, (dict, list)) else None, 'timestamp': int(ts * 1000), 't': ts}
    _acl_merge_trace_extensions(event, params, json_data, usage_block, path, response)
    _acl_get_events().append(event)

def _trace_attach(steps: list[dict]) -> None:
    if not steps:
        return
    trace = [row for row in ({k: ev[k] for k in _ACL_EVENT_FIELDS if k in ev} for ev in _acl_get_events()) if row]
    if not trace:
        return
    info = steps[0].get('extra_info')
    if not isinstance(info, dict):
        info = {}
        steps[0]['extra_info'] = info
    info['proxy_calls'] = trace

class _RateLimiter:

    def __init__(self, max_rpm: int, window: float, min_gap: float) -> None:
        self._max_rpm = max_rpm
        self._window = window
        self._min_gap = min_gap
        self._history: deque[float] = deque()
        self._lock = threading.Lock()

    def _compute_delay(self, now: float) -> float:
        expiry = now - self._window
        while self._history and self._history[0] <= expiry:
            self._history.popleft()
        delay = 0.0
        if self._history:
            gap = now - self._history[-1]
            if gap < self._min_gap:
                delay = self._min_gap - gap
        if len(self._history) >= self._max_rpm:
            delay = max(delay, self._window - (now - self._history[0]))
        return delay

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._compute_delay(now)
                if wait <= 0:
                    self._history.append(now)
                    return
            time.sleep(wait)
_rpm_acquire = _RateLimiter(MAX_RPM, RATE_WINDOW_SECS, MIN_CALL_INTERVAL_SECS).acquire

class _TracedProxy:

    def __init__(self, upstream: ProxyClient, label: str) -> None:
        self._upstream = upstream
        self._label = label

    def __getattr__(self, name: str):
        return getattr(self._upstream, name)

    def _roundtrip(self, method: str, path: str, params: Any=None, json_data: Any=None, **kw):
        t0 = time.time()
        resp = None
        try:
            if method == 'POST':
                resp = self._upstream.post(path, json_data=json_data, **kw)
            else:
                resp = self._upstream.get(path, params=params, **kw)
            return resp
        finally:
            _acl_record(self._label, method, path, (time.time() - t0) * 1000, resp, params=params, json_data=json_data)

    def post(self, path: str, json_data=None, **kw):
        return self._roundtrip('POST', path, json_data=json_data, **kw)

    def get(self, path: str, params=None, **kw):
        return self._roundtrip('GET', path, params=params, **kw)
_llm_transport = _TracedProxy(ProxyClient(timeout=120, max_retries=5), 'inference')
_search_transport = _TracedProxy(ProxyClient(timeout=30, max_retries=3), 'search')

def _rate_limited_search_get(path: str, params: dict | None=None):
    _rpm_acquire()
    return _search_transport.get(path, params)
_pipeline_start: float = 0.0
_detail_cache: dict[str, dict] = {}
_last_tool_call_ts: float = 0.0

def _budget_sec_left() -> float:
    if _pipeline_start <= 0:
        return SESSION_TIMEOUT_SEC
    return SESSION_TIMEOUT_SEC - (time.monotonic() - _pipeline_start)

def _respect_registered_tool_gap() -> None:
    global _last_tool_call_ts
    elapsed_since_last = time.monotonic() - _last_tool_call_ts
    if elapsed_since_last < API_GAP_SECS:
        time.sleep(API_GAP_SECS - elapsed_since_last)

def _invoke_tool_call_and_stamp(tool_name: str, params: dict) -> dict:
    global _last_tool_call_ts
    outcome = execute_tool_call(tool_name, params)
    _last_tool_call_ts = time.monotonic()
    return outcome

def _tool_retry_backoff_sleep(tool_name: str, attempt_idx: int) -> None:
    wait_secs = API_BACKOFF_SECS * 2 ** (attempt_idx - 1)
    time.sleep(wait_secs)

def _call_api(tool_name: str, params: dict) -> dict:
    _respect_registered_tool_gap()
    attempt_idx = 0
    while True:
        try:
            return _invoke_tool_call_and_stamp(tool_name, params)
        except Exception:
            attempt_idx += 1
            if attempt_idx >= API_MAX_RETRIES:
                raise
            _tool_retry_backoff_sleep(tool_name, attempt_idx)

def _float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except (ValueError, TypeError):
        return None

def _norm_voucher(raw: dict | None) -> dict:
    src = raw or {}
    out: dict[str, Any] = {'discount_type': src.get('discount_type', 'percentage')}
    for field in ('discount_value', 'threshold', 'cap', 'budget'):
        out[field] = float(src.get(field, 0))
    return out

def _parse_price_str(price_range: str) -> tuple:
    if not price_range or not isinstance(price_range, str):
        return (None, None)
    left_raw, sep, right_raw = price_range.partition('-')
    if not sep:
        return (None, None)
    left, right = (left_raw.strip(), right_raw.strip())
    return (_float_or_none(left) if left else None, _float_or_none(right) if right else None)

def _parse_price_opt(price_range: str | None) -> tuple[float | None, float | None]:
    if not price_range:
        return (None, None)
    s = str(price_range).strip()
    if '-' not in s:
        v = _float_or_none(s)
        return (None, v) if v is not None else (None, None)
    lo_raw, _sep, hi_raw = s.partition('-')
    lo_part, hi_part = (lo_raw.strip(), hi_raw.strip())
    return (_float_or_none(lo_part) if lo_part else None, _float_or_none(hi_part) if hi_part else None)

def _join_ids(ids: list, expected_order: list=None) -> str:
    deduped = _SearchParamSanitizer.unique_ids(ids)
    if not deduped:
        return SENTINEL_PID
    if expected_order:
        order_index = {eid: i for i, eid in enumerate(expected_order)}
        sentinel = len(expected_order)
        deduped.sort(key=lambda eid: order_index.get(eid, sentinel))
    return ','.join(deduped)

def _assembled_find_product_params(q: str, page: int, shop_id: str | None, price: str | None, sort: str | None, service: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {'q': quote_plus(q), 'page': page, 'shop_id': shop_id, 'price': price, 'sort': sort, 'service': service}
    if payload.get('sort') == 'default':
        payload.pop('sort')
    svc_norm = _SearchParamSanitizer.normalise_service(payload.get('service'))
    if svc_norm is None:
        payload.pop('service', None)
    else:
        payload['service'] = svc_norm
    return payload

def _execute_find_product_with_service_fallback(params: dict[str, Any]) -> list[dict]:
    rows = _rate_limited_search_get('/search/find_product', params) or []
    if not rows and params.get('service'):
        sans_svc = dict(params)
        sans_svc.pop('service', None)
        rows = _rate_limited_search_get('/search/find_product', sans_svc) or []
    return rows

def _parse_price_csv(product_prices: str) -> tuple[list[float] | None, dict | None]:
    try:
        parsed = [float(x.strip()) for x in str(product_prices).split(',')]
        return (parsed, None)
    except ValueError:
        return (None, {'error': 'Invalid product_prices format. Use comma-separated numbers.'})

def _voucher_totals_from_prices(prices: list[float], voucher_type: str, discount_value: float, threshold: float, budget: float, cap: float) -> dict:
    total = sum(prices)
    applied = total >= threshold
    if not applied:
        discount = 0.0
    elif voucher_type == 'fixed':
        discount = discount_value
    elif voucher_type == 'percentage':
        discount = total * (discount_value / 100.0)
        if cap > 0:
            discount = min(discount, cap)
    else:
        discount = 0.0
    final = total - discount
    return {'prices': prices, 'total_before': round(total, 2), 'discount_amount': round(discount, 2), 'total_after': round(final, 2), 'within_budget': final <= budget, 'voucher_applied': applied, 'budget': budget}

def _csv_product_ids_from_tool_arg(product_ids: str) -> list[str]:
    return [pid.strip() for pid in str(product_ids).split(',') if pid.strip()]

def _detail_cache_refresh_missing(missing: list[str]) -> None:
    chunk_size = 10
    for chunk_pos in range(0, len(missing), chunk_size):
        batch = missing[chunk_pos:chunk_pos + chunk_size]
        batch_result = _rate_limited_search_get('/search/view_product_information', {'product_ids': ','.join(batch)})
        if isinstance(batch_result, list):
            for item in batch_result:
                _detail_cache[str(item.get('product_id', ''))] = item

@Tool
def find_product(q: str, page: int=1, shop_id: str | None=None, price: str | None=None, sort: str | None=None, service: str | None=None) -> list[dict]:
    assembled = _assembled_find_product_params(q, page, shop_id, price, sort, service)
    return _execute_find_product_with_service_fallback(assembled)

@Tool
def calculate_voucher(product_prices: str, voucher_type: str, discount_value: float, threshold: float, budget: float, cap: float=0) -> dict:
    prices, err = _parse_price_csv(product_prices)
    if err is not None:
        return err
    return _voucher_totals_from_prices(prices, voucher_type, discount_value, threshold, budget, cap)

@Tool
def recommend_product(product_ids: str) -> str:
    return f'Having recommended the products to the user: {product_ids}.'

@Tool
def terminate(status: str='success') -> str:
    return f'The interaction has been completed with status: {status}'

@Tool
def view_product_information(product_ids: str) -> list[dict]:
    ids = _csv_product_ids_from_tool_arg(product_ids)
    if not ids:
        return []
    result = _rate_limited_search_get('/search/view_product_information', {'product_ids': ','.join(ids)})
    return result if isinstance(result, list) else []

def _load_details(product_ids: list[str]) -> dict[str, dict]:
    if not product_ids:
        return {}
    missing = [pid for pid in product_ids if pid not in _detail_cache]
    _detail_cache_refresh_missing(missing)
    return {pid: _detail_cache[pid] for pid in product_ids if pid in _detail_cache}

def _normalise_sku_options(sku_raw: Any) -> list[dict]:
    result: list[dict] = []
    if isinstance(sku_raw, list):
        for row in sku_raw:
            if not isinstance(row, dict):
                continue
            vals = row.get('values', [])
            if not isinstance(vals, list):
                vals = list(vals.values()) if isinstance(vals, dict) else []
            result.append({'name': row.get('name'), 'values': vals[:5]})
    elif isinstance(sku_raw, dict):
        attr_map: dict[str, list] = {}
        for variant in sku_raw.values():
            if not isinstance(variant, dict):
                continue
            for attr_name, attr_val in variant.items():
                bucket = attr_map.setdefault(attr_name, [])
                if attr_val not in bucket:
                    bucket.append(attr_val)
        for attr_name, values in attr_map.items():
            result.append({'name': attr_name, 'values': values[:5]})
    return result

def _enrich_prime_detail_cache(product_summaries: list[dict]) -> None:
    try:
        _load_details([str(s.get('product_id', '')) for s in product_summaries])
    except Exception:
        pass

def _enrich_single_summary_row(summary: dict) -> dict:
    pid = str(summary.get('product_id', ''))
    try:
        detail = _detail_cache.get(pid) or {}
        title = summary.get('title') or (detail.get('title', '') if detail else '')
        price = summary.get('price')
        if price is None and detail:
            price = detail.get('price')
        entry: dict = {'product_id': pid, 'title': title, 'price': price}
        if detail:
            norm_skus = _normalise_sku_options(detail.get('sku_options') or [])
            if norm_skus:
                entry['sku_options'] = norm_skus[:3]
            attrs = detail.get('attributes') or {}
            if isinstance(attrs, dict) and attrs:
                entry['attributes'] = dict(list(attrs.items())[:8])
            svcs = detail.get('service_tags') or detail.get('services') or []
            if isinstance(svcs, list) and svcs:
                entry['service_tags'] = svcs[:6]
    except Exception:
        entry = {'product_id': pid, 'title': summary.get('title', ''), 'price': summary.get('price')}
    return entry

def _enrich_listings(product_summaries: list[dict]) -> list[dict]:
    _enrich_prime_detail_cache(product_summaries)
    enriched: list[dict] = []
    for summary in product_summaries:
        enriched.append(_enrich_single_summary_row(summary))
    return enriched

def _sandbox_model_pin() -> list[str] | None:
    sandbox = getenv('SANDBOX_MODEL')
    return [sandbox] if sandbox else None

def _provider_chain(key: str) -> list[str]:
    provider_models = _MODEL_REGISTRY.get(_active_provider()) or _OPENROUTER_MODELS
    return provider_models[key]

def _fallback_chain(model: str) -> list[str]:
    return _sandbox_model_pin() or [model, PRODUCT_RANK_, FINAL_FALLBAC_]

def _elect_model_seq() -> list[str]:
    return _sandbox_model_pin() or _provider_chain('PICK_CHAIN')

def _score_model_seq() -> list[str]:
    return _sandbox_model_pin() or _provider_chain('SCORE_CHAIN')

def _is_title_direct_match(word: str, title_words: set[str]) -> bool:
    if word in title_words:
        return True
    stem = word[:-1] if word.endswith('s') else f'{word}s'
    if stem in title_words:
        return True
    return len(word) >= 3 and any((cand.startswith(word) for cand in title_words if len(cand) > len(word)))

def _is_title_partial_match(word: str, title_words: set[str]) -> bool:
    return any((word.startswith(tw) or tw.startswith(word) for tw in title_words if len(tw) > 2))

def _title_hit_score(query_words: list[str], title_words: set[str], title: str) -> float:
    score = 0.0
    for w in query_words:
        if _is_title_direct_match(w, title_words):
            score += 2
        elif _is_title_partial_match(w, title_words):
            score += 1
        if any((ch.isdigit() for ch in w)) and w in title:
            score += 2
    return score

def _iter_detail_key_values(detail: ListingRow):
    for key, vals in (detail.get('attributes') or {}).items():
        yield (key, vals)
    for opts in (detail.get('sku_options') or {}).values():
        if isinstance(opts, dict):
            yield from opts.items()

def _flatten_detail(detail: ListingRow) -> tuple[str, set[str]]:
    tokens: list[str] = []
    exact_vals: set[str] = set()
    for key, values in (detail.get('attributes') or {}).items():
        tokens.append(key.replace('_', ' '))
        for value in values if isinstance(values, list) else [values]:
            text = str(value).strip().lower()
            tokens.append(text)
            exact_vals.add(text)
    sku_probe = {'attributes': {}, 'sku_options': detail.get('sku_options') or {}}
    for key, value in _iter_detail_key_values(sku_probe):
        text = str(value).strip().lower()
        tokens.extend((key.replace('_', ' '), text))
        exact_vals.add(text)
    return (' '.join(tokens).lower(), exact_vals)

def _attr_score(query_words: list[str], detail: dict) -> float:
    detail_text, exact_vals = _flatten_detail(detail)
    detail_words = set(re.findall('\\b\\w+\\b', detail_text))
    total = 0.0
    for w in query_words:
        if f'{w}#' in exact_vals:
            total += 5
        elif w in exact_vals:
            total += 3
        elif w in detail_words:
            total += 2
    return total

def _case_attr_score(query_words: list[str], detail: dict) -> float:
    exact_vals: set[str] = set()
    attr_words: set[str] = set()
    for key, vals in (detail.get('attributes') or {}).items():
        attr_words.update(re.findall('\\b\\w+\\b', key.lower().replace('_', ' ')))
        for value in vals if isinstance(vals, list) else [vals]:
            text = str(value).strip().lower()
            exact_vals.add(text)
            attr_words.update(re.findall('\\b\\w+\\b', text))
    for key, value in _iter_detail_key_values({'attributes': {}, 'sku_options': detail.get('sku_options') or {}}):
        text = str(value).strip().lower()
        exact_vals.add(text)
        attr_words.update(re.findall('\\b\\w+\\b', text))
        attr_words.update(re.findall('\\b\\w+\\b', key.lower().replace('_', ' ')))
    score = 0.0
    for w in query_words:
        if w in exact_vals or f'{w}#' in exact_vals:
            score += 5
        elif w in attr_words:
            score += 2
    return score

def _heuristic_score(product: ListingRow, query_text: str, detail: ListingRow | None=None) -> float:
    title = product.get('title', '').lower()
    title_words = set(re.findall('\\b\\w+\\b', title))
    qw = _QueryTextAnalyzer.tokenize(query_text)
    score = _title_hit_score(qw, title_words, title)
    if detail:
        score += _attr_score(qw, detail)
    return score

def _composite_score(product: dict, query_text: str, detail: dict=None, parsed_spec: dict=None) -> float:
    title = product.get('title', '').lower()
    title_words = set(re.findall('\\b\\w+\\b', title))
    qw = _QueryTextAnalyzer.tokenize(query_text)
    spec = parsed_spec or {}
    score = _title_hit_score(qw, title_words, title)
    price_val = product.get('price')
    price_range_str = spec.get('price_range')
    if isinstance(price_val, (int, float)) and price_range_str:
        lo, hi = _parse_price_str(price_range_str)
        outside = lo is not None and price_val < lo or (hi is not None and price_val > hi)
        score += -25 if outside else 5
    prod_svcs = set(product.get('service') or [])
    required_svc = spec.get('service')
    if required_svc:
        for svc in (s.strip() for s in required_svc.split(',') if s.strip()):
            score += 5 if svc in prod_svcs else -15
    elif prod_svcs:
        score -= 4 * sum((1 for svc in prod_svcs if svc not in {'COD', 'official'}))
    if detail:
        score += _case_attr_score(qw, detail)
    return score

def _parse_json_str(content: str) -> dict | None:
    cleaned = re.sub('<think(?:ing)?>.*?</think(?:ing)?>', '', content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub('<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub('```json?\\s*|```\\s*', '', cleaned).strip()
    try:
        out = json.loads(cleaned)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    start = cleaned.find('{')
    if start != -1:
        depth = 0
        in_str = False
        escape_next = False
        for i, ch in enumerate(cleaned[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_str:
                escape_next = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i + 1]
                    try:
                        out = json.loads(candidate)
                        if isinstance(out, dict):
                            return out
                    except json.JSONDecodeError:
                        break
    brace_match = re.search('\\{.*\\}', content, re.DOTALL)
    if brace_match:
        try:
            out = json.loads(brace_match.group())
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            pass
    return None

def _clip_strings(value: Any, max_len: int) -> Any:
    match value:
        case str() if len(value) > max_len:
            return value[:max_len]
        case str():
            return value
        case list():
            return [_clip_strings(v, max_len) for v in value]
        case dict():
            return {k: _clip_strings(v, max_len) for k, v in value.items()}
        case _:
            return value

def _dialogue_strip_markup_fragment(text: object) -> str:
    fragment = str(text)
    fragment = fragment.replace('<think>', '')
    fragment = fragment.replace('</think>', '')
    fragment = fragment.replace('<tool_call>', '')
    fragment = fragment.replace('</tool_call>', '')
    fragment = fragment.replace('<response>', '')
    fragment = fragment.replace('</response>', '')
    return fragment

def _candidate_ranked_sku_previews(sku_options: dict, query_words: set[str]) -> list[dict]:
    ranked_opts: list[tuple[int, dict]] = []
    for opt in sku_options.values():
        if isinstance(opt, dict):
            opt_words = {w for w in re.findall('\\b\\w+\\b', ' '.join((str(v).lower() for v in opt.values()))) if len(w) > 1}
            ranked_opts.append((len(query_words & opt_words), opt))
    seen_keys: set[str] = set()
    sku_preview: list[dict] = []
    for _score, opt in sorted(ranked_opts, key=lambda t: t[0], reverse=True):
        key = json.dumps(opt, sort_keys=True, ensure_ascii=False)
        if key not in seen_keys:
            seen_keys.add(key)
            sku_preview.append(opt)
    return sku_preview

def _candidate_bounded_attribute_slice(raw_attrs: Any) -> dict:
    bounded_attrs: dict = {}
    if isinstance(raw_attrs, dict):
        for k, v in list(raw_attrs.items())[:8]:
            bounded_attrs[str(k)[:40]] = _clip_strings(v, 80)
    return bounded_attrs

def _build_candidate(product: dict, detail: dict | None, query_text: str) -> dict:
    det = detail or {}
    sku_options = det.get('sku_options') or {}
    query_words = _QueryTextAnalyzer.word_set(query_text)
    sku_preview = _candidate_ranked_sku_previews(sku_options, query_words)
    bounded_attrs = _candidate_bounded_attribute_slice(det.get('attributes') or {})
    raw_title = str(product.get('title', ''))
    title = raw_title[:200] if len(raw_title) > 200 else raw_title
    return {'product_id': str(product.get('product_id', '')).strip(), 'title': title, 'price': product.get('price'), 'service': product.get('service', []), 'attributes': bounded_attrs, 'sku_options_preview': [_clip_strings(o, 80) for o in sku_preview[:8]]}

def _score_listings_under_budget(query_text: str, candidates: list[ListingRow]) -> list[tuple[ListingRow, float]]:
    return [(p, 7.0) for p in candidates if _heuristic_score(p, query_text) > 0]

def _batch_scorer_user_json(query_text: str, candidates: list[ListingRow], details: dict[str, dict], only_product_type: bool) -> str:
    bundle = {'request': query_text, 'candidates': [_build_candidate(p, details.get(str(p.get('product_id', ''))), query_text) for p in candidates], 'only_product_type': only_product_type}
    return json.dumps(bundle, ensure_ascii=False)

def _batch_scorer_normalise_array(raw_content: str) -> list | None:
    trimmed = re.sub('```json?\\s*', '', raw_content)
    trimmed = re.sub('```\\s*$', '', trimmed).strip()
    score_list = None
    try:
        score_list = json.loads(trimmed)
    except json.JSONDecodeError:
        array_match = re.search('\\[.*\\]', raw_content, re.DOTALL)
        if array_match:
            try:
                score_list = json.loads(array_match.group())
            except json.JSONDecodeError:
                pass
    if not isinstance(score_list, list):
        return None
    return score_list

def _batch_scorer_pid_map_from_rows(rows: list) -> dict[str, float]:
    accum: dict[str, float] = {}
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get('product_id', '')).strip()
        if not pid:
            continue
        try:
            accum[pid] = float(entry.get('score', 0))
        except (TypeError, ValueError):
            accum[pid] = 0.0
    return accum

def _batch_scorer_attach_scores(candidates: list[ListingRow], pid_to_score: dict[str, float]) -> list[tuple[ListingRow, float]]:
    ordered = [(p, pid_to_score.get(str(p.get('product_id', '')).strip(), 0.0)) for p in candidates]
    ordered.sort(key=lambda x: (x[1], str(x[0].get('product_id', ''))), reverse=True)
    return ordered

def _batch_scorer_attempt(model_name: str, attempt_no: int, user_content: str, candidates: list[ListingRow]) -> list[tuple[ListingRow, float]] | None:
    llm_resp = _llm_transport.post('/inference/chat/completions', json_data={'model': model_name, 'temperature': 0.5, 'stream': False, 'messages': [{'role': 'system', 'content': RazgrizPrompts.BATCH_SCORER}, {'role': 'user', 'content': user_content}]})
    if not (llm_resp and llm_resp.get('choices')):
        return None
    raw_content = llm_resp['choices'][0].get('message', {}).get('content', '')
    score_list = _batch_scorer_normalise_array(raw_content)
    if score_list is None:
        return None
    pid_to_score = _batch_scorer_pid_map_from_rows(score_list)
    scored = _batch_scorer_attach_scores(candidates, pid_to_score)
    return scored

def _score_listings_heuristic_exhausted(query_text: str, candidates: list[ListingRow]) -> list[tuple[ListingRow, float]]:
    scored = [(p, 7.0 if _heuristic_score(p, query_text) > 0 else 0.0) for p in candidates]
    scored.sort(key=lambda x: (x[1], str(x[0].get('product_id', ''))), reverse=True)
    return scored

def _score_listings(query_text: str, candidates: list[ListingRow], details: dict[str, dict], only_product_type: bool=False, model: str=BACKUP_LLM_) -> list[tuple[ListingRow, float]]:
    if not candidates:
        return []
    if _budget_sec_left() < 35.0:
        return _score_listings_under_budget(query_text, candidates)
    user_content = _batch_scorer_user_json(query_text, candidates, details, only_product_type)
    for model_name in _score_model_seq():
        attempt = 0
        while attempt < LLM_RETRY_MAX:
            attempt += 1
            scored = _batch_scorer_attempt(model_name, attempt, user_content, candidates)
            if scored is not None:
                return scored
    return _score_listings_heuristic_exhausted(query_text, candidates)
_ELECTOR_NULL_PRODUCT_IDS = frozenset({'', 'none', 'null', '0', 'undefined', 'n/a'})

def _elect_best_slice_cap(candidates: list, max_candidates: int) -> list:
    cap = max(1, min(int(max_candidates), 60))
    return candidates[:cap]

def _elect_best_judge_user_json(query_text: str, slice_c: list, details: dict[str, dict], only_product_type: bool) -> str:
    bundle = {'request': query_text, 'candidates': [_build_candidate(p, details.get(str(p.get('product_id', ''))), query_text) for p in slice_c], 'only_product_type': only_product_type}
    return json.dumps(bundle, ensure_ascii=False)

def _elect_best_dispatch_post(model_name: str, user_content: str) -> dict | None:
    result = _llm_transport.post('/inference/chat/completions', json_data={'model': model_name, 'temperature': 0.5, 'stream': False, 'messages': [{'role': 'system', 'content': RazgrizPrompts.ITEM_JUDGE}, {'role': 'user', 'content': user_content}]})
    return result if result and result.get('choices') else None

def _elect_best_apply_parsed_pick(parsed: dict, slice_c: list, details: dict[str, dict], query_text: str, model_name: str, attempt: int) -> dict | None:
    best_pid = str(parsed.get('best_product_id', '') or '').strip()
    reason = str(parsed.get('reason', '')).strip()
    try:
        rel_score = float(parsed.get('relevance_score', 0))
    except (TypeError, ValueError):
        rel_score = 0.0
    if best_pid.lower() in _ELECTOR_NULL_PRODUCT_IDS:
        return None
    for p in slice_c:
        if str(p.get('product_id', '')).strip() == best_pid:
            chosen = dict(p)
            det = details.get(str(p.get('product_id', '')))
            _ground_reason(chosen, reason, rel_score, p, det, query_text)
            return chosen
    return None

def _elect_best_attempt_cycle(model_name: str, user_content: str, slice_c: list, details: dict[str, dict], query_text: str) -> dict | None:
    for attempt in range(1, LLM_RETRY_MAX + 1):
        result = _elect_best_dispatch_post(model_name, user_content)
        if result is None:
            continue
        content = result['choices'][0].get('message', {}).get('content', '')
        parsed = _parse_json_str(content)
        if not isinstance(parsed, dict):
            continue
        picked = _elect_best_apply_parsed_pick(parsed, slice_c, details, query_text, model_name, attempt)
        if picked is not None:
            return picked
    return None

def _elect_best_heuristic_pick(slice_c: list, query_text: str) -> dict:
    fallback = max(slice_c, key=lambda p: _heuristic_score(p, query_text))
    fallback = dict(fallback)
    fallback.setdefault('_llm_relevance_score', 0.0)
    fallback.setdefault('_llm_reason', 'heuristic fallback ? LLM did not return a valid product_id')
    return fallback

def _elect_best(query_text: str, candidates: list, details: dict[str, dict], only_product_type: bool=False, model: str=FINAL_FALLBAC_, *, max_candidates: int=10) -> dict | None:
    if _budget_sec_left() < 35.0:
        return None
    slice_c = _elect_best_slice_cap(candidates, max_candidates)
    user_content = _elect_best_judge_user_json(query_text, slice_c, details, only_product_type)
    for model_name in _elect_model_seq():
        winner = _elect_best_attempt_cycle(model_name, user_content, slice_c, details, query_text)
        if winner is not None:
            return winner
    if slice_c:
        return _elect_best_heuristic_pick(slice_c, query_text)
    return None

def _product_detail_text(product: dict, detail: dict | None) -> str:
    fragments = [(product.get('title') or '').lower()]
    if isinstance(detail, dict):
        attrs = detail.get('attributes') or {}
        if isinstance(attrs, dict):
            for k, vs in attrs.items():
                fragments.append(str(k).lower().replace('_', ' '))
                if isinstance(vs, list):
                    fragments.extend((str(v).lower() for v in vs))
                else:
                    fragments.append(str(vs).lower())
        skus = detail.get('sku_options') or {}
        if isinstance(skus, dict):
            for opts in skus.values():
                if isinstance(opts, dict):
                    for k, v in opts.items():
                        fragments.append(str(k).lower().replace('_', ' '))
                        fragments.append(str(v).lower())
    return ' '.join(fragments)

def _check_reason_refs(reason: str, product: dict, detail: dict | None, query_text: str) -> tuple[bool, list[str]]:
    haystack = _product_detail_text(product, detail)
    query_terms = {w for w in re.findall('\\b\\w{4,}\\b', (query_text or '').lower()) if w not in RANK_STOPWORDS}
    if not query_terms:
        return (True, [])
    reason_lower = (reason or '').lower()
    claimed = {t for t in query_terms if t in reason_lower}
    missing = [t for t in claimed if t not in haystack]
    return (len(missing) == 0, missing)

def _sanitise_reason(missing: list[str]) -> str:
    ms = ', '.join(sorted(missing))
    return f"Selected as the best available match among returned candidates; the user's requested term(s) ({ms}) could not be confirmed literally in this product's title, attributes, or sku_options, so the match is partial."

def _ground_reason(result_product: dict, reason: str, relevance_score: float, product: dict, detail: dict | None, query_text: str) -> None:
    grounded, missing = _check_reason_refs(reason, product, detail, query_text)
    result_product['_llm_relevance_score'] = relevance_score
    if grounded:
        result_product['_llm_reason'] = reason
        return
    result_product['_llm_reason'] = _sanitise_reason(missing)
    result_product['_llm_reason_ungrounded_terms'] = missing

def _shrink_items(items: list) -> list:
    return [{'pid': str(item.get('product_id', '')), 'p': item.get('price'), 's': str(item.get('shop_id', ''))} for item in items[:RESULT_TRIM_MAX] if isinstance(item, dict)]

def _compact_result(tool_call: dict) -> dict:
    if not isinstance(tool_call, dict) or tool_call.get('name') != 'find_product':
        return tool_call
    inner = tool_call.get('result')
    if isinstance(inner, dict) and isinstance(inner.get('result'), list):
        return {**tool_call, 'result': {**inner, 'result': _shrink_items(inner['result'])}}
    if isinstance(inner, list):
        return {**tool_call, 'result': _shrink_items(inner)}
    return tool_call

def _verify_pick(*, title: str, price: Any, parsed_spec: dict) -> dict:
    title_lower = (title or '').lower()
    spec = parsed_spec or {}
    kw = [w for w in str(spec.get('keywords', '') or '').lower().split() if w]
    matched = [w for w in kw if w in title_lower]
    missing = [w for w in kw if w not in title_lower]
    price_ok: bool | None = None
    price_note = 'no price range was parsed from the query'
    price_range = spec.get('price_range')
    if price_range:
        try:
            lo, hi = _parse_price_str(str(price_range))
            if price is None:
                price_note = f'no price available to compare against range {price_range}'
            else:
                pv = float(price)
                if lo is not None and pv < lo:
                    price_ok, price_note = (False, f'price {pv} is BELOW lower bound {lo} of range {price_range}')
                elif hi is not None and pv > hi:
                    price_ok, price_note = (False, f'price {pv} is ABOVE upper bound {hi} of range {price_range}')
                else:
                    price_ok, price_note = (True, f'price {pv} fits inside range {price_range}')
        except (TypeError, ValueError):
            price_note = f'price {price!r} is not numeric; could not check range {price_range}'
    has_missing = bool(missing)
    price_bad = price_ok is False
    if not has_missing and (not price_bad):
        note = 'The selected product looks like a genuine match for the parsed query.'
    elif has_missing and price_bad:
        note = f'HONEST MISMATCH: title is missing query terms {missing} and price is outside the requested range. This is the best available candidate, not a clean fit.'
    elif has_missing:
        note = f'HONEST MISMATCH: the selected title is missing query terms {missing}; attributes may still confirm the fit, but the title alone is imperfect.'
    else:
        note = 'HONEST MISMATCH: title matches the keywords but the price does not fit the requested range. Taking it as the closest available option.'
    return {'query_keywords': kw, 'keywords_matched': matched, 'keywords_missing': missing, 'title_contains_all_keywords': not has_missing, 'price_ok': price_ok, 'price_note': price_note, 'overall_note': note}

def _safe_score(prod: dict, q: str, spec: dict | None) -> float | None:
    try:
        return round(_composite_score(prod, q, parsed_spec=spec), 1)
    except Exception:
        return None

def _build_find_params(query: str, *, page: int=1, shop_id: str | None=None, price: str | None=None, sort: str | None=None, service: str | None=None) -> dict[str, Any]:
    p: dict[str, Any] = {'q': quote_plus(query), 'page': page}
    if shop_id:
        p['shop_id'] = shop_id
    if price:
        p['price'] = price
    if sort and sort != 'default':
        p['sort'] = sort
    svc = _SearchParamSanitizer.normalise_service(service)
    if svc:
        p['service'] = svc
    return p

def _do_search(params: dict[str, Any]) -> list[ListingRow]:
    rows = _search_transport.get('/search/find_product', params) or []
    _oro_record_search(rows)
    return rows

def _spec_query_term(spec: SpecEntry) -> str:
    return spec.get('q') or spec.get('keywords') or DEFAULT_QUERY

def _spec_price(spec: SpecEntry, *, include_price: bool, price_override: str | None=None) -> str | None:
    if price_override is not None:
        return price_override
    return spec.get('price') or spec.get('price_range') if include_price else None

def _search_spec(spec: SpecEntry, *, shop_id: str | None=None, include_price: bool=True, omit_service_from_api: bool=False) -> list[ListingRow]:
    price_filter = _spec_price(spec, include_price=include_price)
    service_filter = None if omit_service_from_api else spec.get('service')
    q = _spec_query_term(spec)
    return [row for pg in (1, 2) for row in _do_search(_build_find_params(q, page=pg, shop_id=shop_id, price=price_filter, service=service_filter)) or []]

def _search_spec_in_shop_limited(spec: SpecEntry, shop_id: str, *, page: int=1, limit: int=10, omit_service_from_api: bool=False, price_override: str | None=None) -> list[ListingRow]:
    search_params = _build_find_params(_spec_query_term(spec), page=page, shop_id=str(shop_id), price=_spec_price(spec, include_price=True, price_override=price_override), service=None if omit_service_from_api else spec.get('service'))
    batch = _do_search(search_params)
    deduped = _SearchParamSanitizer.deduplicate(batch or [])
    return deduped[:limit]

def _ranked_shop_ids_from_pairs(spec_pairs: list[tuple[ListingRow, float]], max_shops: int) -> list[str]:
    return _top_shop_ids_by_score(spec_pairs, max_shops)

def _top_shop_ids_by_score(spec_pairs: list[tuple[ListingRow, float]], k: int) -> list[str]:
    best_by_shop: dict[str, float] = {}
    for prod, sc in spec_pairs:
        sid = str(prod.get('shop_id') or '').strip()
        if sid:
            best_by_shop[sid] = max(best_by_shop.get(sid, float('-inf')), float(sc))
    return [sid for sid, _ in sorted(best_by_shop.items(), key=lambda kv: (-kv[1], kv[0]))][:max(0, int(k))]

def _spec_has_shop_hit(spec: SpecEntry, shop_id: str) -> bool:
    return any((_search_spec_in_shop_limited(spec, shop_id, page=1, limit=1, omit_service_from_api=omit) for omit in (False, True)))

def _gather_cross_pool_per_shop_cap(spec: SpecEntry, shop_ids: list[str], *, total_cap: int, per_shop: int) -> list[ListingRow]:
    if not shop_ids or total_cap <= 0:
        return []
    seen_ids: set[str] = set()
    pool: list[ListingRow] = []
    for sid in shop_ids:
        remaining = max(0, total_cap - len(pool))
        limit = min(per_shop, remaining)
        if limit <= 0:
            break
        batch = _search_spec_in_shop_limited(spec, sid, page=1, limit=limit)
        if not batch:
            batch = _search_spec_in_shop_limited(spec, sid, page=1, limit=limit, omit_service_from_api=True)
        for row in batch or []:
            pid = str(row.get('product_id') or '').strip()
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            pool.append(row)
            if len(pool) >= total_cap:
                return pool
    return pool

def _order_rank_sum_two(specs: list[SpecEntry], i: int, j: int) -> int:
    a = _to_rank_int(specs[i].get('order'))
    b = _to_rank_int(specs[j].get('order'))
    return (a or 0) + (b or 0)

def _dedupe_spec_pairs_by_shop(pairs: list[tuple[ListingRow, float]]) -> list[tuple[ListingRow, float]]:
    best_by_shop: dict[str, tuple[ListingRow, float]] = {}
    for prod, sc in pairs:
        sid = str(prod.get('shop_id') or '').strip()
        if not sid:
            continue
        current = best_by_shop.get(sid)
        if current is None or float(sc) > current[1]:
            best_by_shop[sid] = (prod, float(sc))
    return list(best_by_shop.values())

def _two_spec_top_shop_ids(spec_pairs: list[tuple[ListingRow, float]], k: int) -> list[str]:
    return _top_shop_ids_by_score(spec_pairs, k)

def _gather_cross_spec_pool_in_shops(spec: SpecEntry, shop_ids: list[str], cap: int) -> list[ListingRow]:
    if not shop_ids or cap <= 0:
        return []
    per_shop = max(1, cap // len(shop_ids))
    seen_ids: set[str] = set()
    pool: list[ListingRow] = []
    for sid in shop_ids:
        batch = _search_spec_in_shop_limited(spec, sid, page=1, limit=per_shop)
        if not batch:
            batch = _search_spec_in_shop_limited(spec, sid, page=1, limit=per_shop, omit_service_from_api=True)
        for row in batch or []:
            pid = str(row.get('product_id') or '').strip()
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            pool.append(row)
            if len(pool) >= cap:
                return pool
    return pool

def _spec_to_query(product: dict, *, include_price: bool=True) -> dict[str, Any]:
    kw = product.get('keywords', 'product')
    svc = product.get('service')
    q = kw + (' only' if not svc and bool(product.get('only_product_type')) else '')
    p: dict[str, Any] = {'q': q}
    if include_price and product.get('price_range'):
        p['price'] = product['price_range']
    if svc:
        p['service'] = svc
    return p

def _voucher_ceiling(voucher: dict) -> float | None:
    dtype = voucher.get('discount_type', 'percentage')
    dval = float(voucher.get('discount_value', 0))
    min_req = float(voucher.get('threshold', 0))
    dcap = float(voucher.get('cap', 0))
    budget = float(voucher.get('budget', 0))
    if dtype == 'fixed':
        ceiling = budget + dval
        return min_req if ceiling <= min_req else ceiling
    rate = dval / 100.0 if dval > 1 else dval
    if rate <= 0 or rate >= 1:
        return None
    if dcap > 0 and budget / (1 - rate) > budget + dcap:
        ceiling = budget + dcap
    else:
        ceiling = budget / (1 - rate)
    return min_req if ceiling <= min_req else ceiling

def _numeric_product_price(product: ListingRow) -> float | None:
    try:
        v = product.get('price')
        if v is None:
            return None
        x = float(v)
        if x != x or x < 0:
            return None
        return x
    except (TypeError, ValueError):
        return None

def _intersect_spec_price_with_budget_cap(spec: SpecEntry, cap_hi: float) -> tuple[float, float] | None:
    if cap_hi < 0:
        return None
    orig_lo, orig_hi = _parse_price_opt(spec.get('price_range'))
    lo = max(0.0, orig_lo) if orig_lo is not None else 0.0
    hi = min(float(cap_hi), orig_hi) if orig_hi is not None else float(cap_hi)
    return (lo, hi) if lo <= hi + 1e-09 else None

def _price_lo_hi_to_find_str(lo: float, hi: float) -> str:
    return f'{max(0.0, lo):.0f}-{max(lo, hi):.0f}'

def _pid_list(rows: list) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for rec in rows or []:
        if not isinstance(rec, dict):
            continue
        pid = str(rec.get('product_id', '')).strip()
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out

def _merged_pids(id_lists: list[list[str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for pid in (str(v).strip() for grp in id_lists or [] for v in grp or []):
        if pid and pid not in seen:
            seen.add(pid)
            result.append(pid)
    return result

def _to_rank_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return n if n >= 1 else None
    text = str(value).strip().lower()
    if not text:
        return None
    m = re.match('^(\\d+)', text)
    return int(m.group(1)) if m else None
_RICHNESS_SENTINEL = 10000

def _richness_rank(spec: SpecEntry) -> int:
    rank = _to_rank_int(spec.get('order'))
    return _RICHNESS_SENTINEL if rank is None else rank

def _gather_result_ids(find_results: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for resp in find_results or []:
        for prod in resp.get('result') or []:
            pid = str(prod.get('product_id', '')).strip()
            if pid and pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out

def _clip_band(floor: float, ceiling: float, price_range: str | None) -> tuple[float, float] | None:
    if ceiling < floor:
        return None
    parsed_lo, parsed_hi = _parse_price_opt(price_range)
    lo = floor if parsed_lo is None else max(floor, parsed_lo)
    hi = ceiling if parsed_hi is None else min(ceiling, parsed_hi)
    if hi < lo:
        return None
    return (lo, hi)

def _probe_edges(products: list, allowed_total: float) -> tuple[list[float], list[float], list]:
    return _probe_edges_base(products, allowed_total, shop_id=None)

def _probe_edges_base(products: list, allowed_total: float, *, shop_id: str | None) -> tuple[list[float], list[float], list]:
    minima: list[float] = []
    maxima: list[float] = []
    calls: list = []
    scoped_shop_id = str(shop_id) if shop_id is not None else None
    price_window = f'1-{allowed_total:.0f}'
    for spec in products:
        params = _spec_to_query(spec, include_price=False)
        scoped_params = dict(params)
        if scoped_shop_id is not None:
            scoped_params['shop_id'] = scoped_shop_id
        for sort_key, output in (('priceasc', minima), ('pricedesc', maxima)):
            response = _call_api('find_product', {**scoped_params, 'price': price_window, 'sort': sort_key, 'page': 1})
            calls.append(response)
            first_hit = (response.get('result') or [None])[0]
            output.append(float(first_hit.get('price', 0)) if first_hit else 0.0)
    return (minima, maxima, calls)

def _probe_edges_shop(products: list, allowed_total: float, shop_id: str) -> tuple[list[float], list[float], list]:
    return _probe_edges_base(products, allowed_total, shop_id=shop_id)

def _fetch_band_hits(spec: dict, floor: float, ceiling: float, limit: int=20) -> tuple[list, list]:
    return _fetch_band_hits_base(spec, floor, ceiling, limit=limit, shop_id=None)

def _fetch_shop_band_hits(spec: dict, floor: float, ceiling: float, shop_id: str, limit: int=20) -> tuple[list, list]:
    return _fetch_band_hits_base(spec, floor, ceiling, limit=limit, shop_id=shop_id)

def _fetch_band_hits_base(spec: dict, floor: float, ceiling: float, *, limit: int, shop_id: str | None) -> tuple[list, list]:
    qp = _spec_to_query(spec, include_price=False)
    qp['price'] = f'{floor:.0f}-{ceiling:.0f}'
    if shop_id is not None:
        qp['shop_id'] = str(shop_id)
    hits: list[dict] = []
    seen: set[str] = set()
    calls: list = []
    for pg in (1, 2):
        resp = _call_api('find_product', {**qp, 'page': pg})
        calls.append(resp)
        for product in resp.get('result') or []:
            pid = str(product.get('product_id', ''))
            if pid and pid not in seen:
                seen.add(pid)
                hits.append(product)
                if len(hits) >= limit:
                    return (hits, calls)
    return (hits, calls)

def _marginal_band(sidx: int, n_specs: int, threshold: float, allowed_total: float, minima: list[float], maxima: list[float], price_range: str | None) -> tuple[float, float] | None:
    other_indices = [idx for idx in range(n_specs) if idx != sidx]
    max_other_sum = sum((maxima[idx] for idx in other_indices))
    min_other_sum = sum((minima[idx] for idx in other_indices))
    floor_raw = max(0.0, threshold - max_other_sum)
    ceiling_raw = allowed_total - min_other_sum
    return _clip_band(floor_raw, ceiling_raw, price_range)

def _swap_pair_list(pool: dict, floor_score: float) -> list[tuple[dict, float]]:
    filtered = pool.get('filtered')
    if filtered:
        return list(filtered)
    scored = pool.get('scored')
    if scored:
        eligible = [(prod, score) for prod, score in scored if score >= floor_score]
        return eligible if eligible else list(scored)
    raw_rows = pool.get('raw') or []
    return [(dict(row), 0.0) for row in raw_rows]

def _init_picks(pools: list[dict | None]) -> list[dict] | None:
    picks: list[dict] = []
    for pool in pools:
        if pool is None:
            return None
        raw_pool = pool.get('raw')
        if not raw_pool:
            return None
        filtered = pool.get('filtered')
        if filtered:
            best_product, best_score = max(filtered, key=lambda pair: pair[1])
            chosen = dict(best_product)
            chosen['_llm_relevance_score'] = float(best_score)
            picks.append(chosen)
            continue
        first = dict(raw_pool[0])
        first_pid = str(first.get('product_id', '')).strip()
        scored_rows = pool.get('scored') or []
        matching_score = next((float(score) for prod, score in scored_rows if str(prod.get('product_id', '')).strip() == first_pid), 0.0)
        first['_llm_relevance_score'] = matching_score
        picks.append(first)
    return picks

def _swap_cheaper(products: list[dict], pools: list[dict], picks: list[dict], floor_score: float, swap_delta: float) -> tuple[int, dict, float] | None:
    swap_candidates: list[tuple[float, float, int, dict]] = []
    for sidx, pool in enumerate(pools):
        current_pick = picks[sidx]
        current_pid = str(current_pick.get('product_id', ''))
        current_price = float(current_pick.get('price', 0) or 0)
        alternatives = _swap_pair_list(pool, floor_score)
        for alt_product, alt_score in alternatives:
            alt_pid = str(alt_product.get('product_id', ''))
            if not alt_pid or alt_pid == current_pid:
                continue
            alt_price = float(alt_product.get('price', 0) or 0)
            if alt_price > current_price - swap_delta:
                continue
            richness = _to_rank_int(products[sidx].get('order'))
            richness_score = float(sidx + 1000) if richness is None else float(richness)
            swap_candidates.append((float(alt_score), richness_score, sidx, dict(alt_product)))
    if not swap_candidates:
        return None
    swap_candidates.sort(key=lambda item: (item[0], -item[1], item[2]))
    best_score, _best_richness, best_idx, best_product = swap_candidates[0]
    return (best_idx, best_product, best_score)

def _build_margin_pools(products: list[dict], n_specs: int, threshold: float, allowed_total: float, minima: list[float], maxima: list[float], query: str, floor_score: float, *, shop_id: str | None=None) -> tuple[list[dict | None], list]:
    all_calls: list = []
    pools: list[dict | None] = []
    for sidx in range(n_specs):
        spec = products[sidx]
        band = _marginal_band(sidx, n_specs, threshold, allowed_total, minima, maxima, spec.get('price_range'))
        if band is None:
            pools.append(None)
            continue
        search_q = spec.get('query') or spec.get('keywords') or query
        if shop_id:
            hits, calls = _fetch_shop_band_hits(spec, band[0], band[1], shop_id, limit=20)
        else:
            hits, calls = _fetch_band_hits(spec, band[0], band[1], limit=20)
        all_calls.extend(calls)
        if not hits:
            pools.append(None)
            continue
        pids = [str(p.get('product_id', '')) for p in hits if p.get('product_id')]
        details = _load_details(pids)
        scored_raw = _score_listings(search_q, hits, details, only_product_type=bool(spec.get('only_product_type', False)))
        scored_pairs: list[tuple[dict, float]] = []
        for pr, sc in scored_raw:
            row = dict(pr)
            row['_llm_relevance_score'] = float(sc)
            scored_pairs.append((row, float(sc)))
        filtered_pairs = [(p, s) for p, s in scored_pairs if s >= floor_score]
        pools.append({'scored': scored_pairs, 'filtered': filtered_pairs, 'raw': [dict(x) for x in hits], 'band': {'floor': float(band[0]), 'ceiling': float(band[1])}})
    return (pools, all_calls)

def _index_by_shop(broad_results: Sequence[Sequence[ListingRow]]) -> dict[str, dict[int, list[ListingRow]]]:
    index: dict[str, dict[int, list[ListingRow]]] = defaultdict(lambda: defaultdict(list))
    for idx, products in enumerate(broad_results):
        for prod in products:
            sid = str(prod.get('shop_id', ''))
            if sid:
                index[sid][idx].append(prod)
    return index

def _filter_spec_floor(spec_scored: list[list[tuple[ListingRow, float]]], floor: float) -> list[list[tuple[ListingRow, float]]]:

    def _flt(row: list[tuple[ListingRow, float]]) -> list[tuple[ListingRow, float]]:
        thr = float(floor)
        return [(p, float(sc)) for p, sc in row if float(sc) >= thr]
    return [_flt(row) for row in spec_scored]

def _sort_shops(shop_ids: list[str], shop_coverage: dict[str, dict[int, list[ListingRow]]], specs: list[SpecEntry], query: str) -> list[str]:

    def _best_spec_score(candidates: list[ListingRow], spec: SpecEntry) -> float:
        if not candidates:
            return 0.0
        search_q = spec.get('query') or spec.get('keywords') or query
        return max((_heuristic_score(p, str(search_q)) for p in candidates), default=0.0)

    def _score_store(store_id: str) -> float:
        spec_pools = shop_coverage.get(store_id) or {}
        return sum((_best_spec_score(spec_pools.get(spec_idx, []), spec) for spec_idx, spec in enumerate(specs)))
    ranked = [(store_id, _score_store(store_id)) for store_id in shop_ids]
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return [store_id for store_id, _ in ranked]

def _choose_shop_llm(shop_ids: list[str], shop_coverage: dict[str, dict[int, list[ListingRow]]], specs: list[SpecEntry], query: str) -> tuple[str | None, dict[int, dict]]:

    def _choose_for_shop_spec(shop_id: str, sidx: int, spec: SpecEntry) -> tuple[float, dict] | None:
        products = list((shop_coverage.get(shop_id) or {}).get(sidx) or [])
        if not products:
            return None
        search_q = spec.get('query') or spec.get('keywords') or query
        pids = [str(p.get('product_id', '')) for p in products if p.get('product_id')]
        details = _load_details(pids)
        chosen = _elect_best(search_q, products, details, only_product_type=bool(spec.get('only_product_type', False)), model=BACKUP_LLM_)
        if chosen:
            sc = float(chosen.get('_llm_relevance_score', 0))
            return (sc, {'product_id': str(chosen.get('product_id', '')), 'reason': chosen.get('_llm_reason', ''), 'score': sc})
        return (0.0, {'product_id': str(products[0].get('product_id', '')), 'reason': '', 'score': 0.0})
    best_shop: str | None = None
    best_total: float = -1.0
    best_chosen: dict[int, dict] = {}
    for shop_id in shop_ids:
        total_score = 0.0
        chosen_for_shop: dict[int, dict] = {}
        for sidx, spec in enumerate(specs):
            picked = _choose_for_shop_spec(shop_id, sidx, spec)
            if picked is None:
                continue
            sc, payload = picked
            total_score += sc
            chosen_for_shop[sidx] = payload
        if total_score > best_total:
            best_total = total_score
            best_shop = shop_id
            best_chosen = chosen_for_shop
    return (best_shop, best_chosen)

def _deepest_spec(spec_indices: list[int], specs: list[SpecEntry]) -> int:

    def _raw(spec: SpecEntry) -> tuple[float, int, int]:
        kw_count = len((spec.get('keywords') or '').split())
        price_score = 0.0
        pr = spec.get('price_range') or ''
        if pr and '-' in pr:
            parts = pr.split('-', 1)
            lo, hi = (parts[0].strip(), parts[1].strip())
            price_score = 1.5 if lo and hi else 1.0
        svc_count = len([s.strip() for s in (spec.get('service') or '').split(',') if s.strip()])
        return (price_score, kw_count, svc_count)
    raw = {idx: _raw(specs[idx]) for idx in spec_indices}
    max_kw = max((v[1] for v in raw.values()))
    max_svc = max((v[2] for v in raw.values()))
    final: dict[int, float] = {}
    for idx, (ps, kc, sc) in raw.items():
        score = ps
        if kc == max_kw:
            score += 1.0
        if sc == max_svc:
            score += 1.0
        final[idx] = score
    max_score = max(final.values())
    winners = [idx for idx, sv in final.items() if sv == max_score]
    return min(winners, key=lambda i: (_richness_rank(specs[i]), i))

def _select_in_shop(spec: SpecEntry, shop_id: str, query: str) -> ListingRow | None:
    products = _search_spec(spec, shop_id=shop_id)
    if not products:
        products = _search_spec(spec, shop_id=shop_id, omit_service_from_api=True)
    if not products:
        return None
    pids = [str(p.get('product_id', '')) for p in products if p.get('product_id')]
    details = _load_details(pids)
    search_q = spec.get('query') or spec.get('keywords') or query
    best = _elect_best(search_q, products[:10], details, only_product_type=bool(spec.get('only_product_type', False)), model=BACKUP_LLM_)
    return best if best is not None else products[0] if products else None
EMPTY_SHOP_ANCHOR_CAP = 8
EMPTY_SHOP_ANCHOR_CAP_VOUCHER = 4

def _select_in_shop_empty_relaxed(spec: SpecEntry, shop_id: str, query: str) -> ListingRow | None:
    products = _search_spec(spec, shop_id=shop_id)
    if not products:
        products = _search_spec(spec, shop_id=shop_id, omit_service_from_api=True)
    if not products:
        kw_full = str(spec.get('keywords') or spec.get('q') or '')
        words = kw_full.split()
        for trimmed in (' '.join(words[:2]), words[0] if words else ''):
            if trimmed and trimmed != kw_full:
                relax_spec = dict(spec)
                relax_spec['keywords'] = trimmed
                relax_spec['q'] = trimmed
                products = _search_spec(relax_spec, shop_id=shop_id, omit_service_from_api=True)
                if products:
                    break
    if not products:
        return None
    pids = [str(p.get('product_id', '')) for p in products if p.get('product_id')]
    details = _load_details(pids)
    search_q = spec.get('query') or spec.get('keywords') or query
    best = _elect_best(str(search_q), products[:10], details, only_product_type=bool(spec.get('only_product_type', False)), model=BACKUP_LLM_)
    return best if best is not None else products[0] if products else None

def _shop_empty_effective_scored_and_coverage(spec_scored_full: list[list[tuple[ListingRow, float]]], score_floor: float) -> tuple[list[list[tuple[ListingRow, float]]], dict[str, dict[int, list[ListingRow]]]]:
    filtered = [[(p, float(s)) for p, s in scored if float(s) >= score_floor] for scored in spec_scored_full]
    if any((len(bucket) == 0 for bucket in filtered)):
        filtered = [list(scored) for scored in spec_scored_full]
    broad = [[p for p, _ in row] for row in filtered]
    return (filtered, _index_by_shop(broad))

def _shop_empty_attempt_partial_coverage(specs: list[SpecEntry], spec_scored: list[list[tuple[ListingRow, float]]], shop_coverage: dict[str, dict[int, list[ListingRow]]], query: str, n_specs: int) -> tuple[list[str] | None, dict]:
    partial = {sid: cov for sid, cov in shop_coverage.items() if len(cov) == n_specs - 1}
    if not partial:
        return (None, {})
    pid_to_score: dict[str, float] = {str(p.get('product_id', '')): sc for scored in spec_scored for p, sc in scored}

    def _shop_total(cov: dict[int, list[ListingRow]]) -> float:
        return sum((max((pid_to_score.get(str(p.get('product_id', '')), 0.0) for p in prods), default=0.0) for prods in cov.values()))
    shop_scores = {sid: _shop_total(cov) for sid, cov in partial.items()}
    max_score = max(shop_scores.values())
    best_shops = sorted((sid for sid, s in shop_scores.items() if s == max_score))
    winner = best_shops[0]
    coverage = partial[winner]
    covered = set(coverage.keys())
    missing_idx = next((i for i in range(n_specs) if i not in covered))
    pids: list[str | None] = [None] * n_specs
    for idx in covered:
        shop_pids = {str(p.get('product_id', '')) for p in coverage[idx]}
        best_p = next((p for p, _ in spec_scored[idx] if str(p.get('product_id', '')) in shop_pids), coverage[idx][0] if coverage[idx] else None)
        if best_p:
            pids[idx] = str(best_p.get('product_id', ''))
    best_missing = _select_in_shop_empty_relaxed(specs[missing_idx], winner, query)
    if not best_missing:
        return (None, {})
    pids[missing_idx] = str(best_missing.get('product_id', ''))
    if not all((pid is not None for pid in pids)):
        return (None, {})
    ctx = {'resolution_mode': 4, 'partial_shops_evaluated': len(partial), 'winner_shop_id': winner, 'winner_shop_score': round(max_score, 2), 'covered_spec_indices': sorted(covered), 'missing_spec_idx': missing_idx, 'missing_spec_keywords': specs[missing_idx].get('keywords', ''), 'filled_missing_product': {'product_id': str(best_missing.get('product_id', '')), 'title': best_missing.get('title', ''), 'price': best_missing.get('price')}}
    return (pids, ctx)

def _shop_empty_fallback_anchor_resolution(specs: list[SpecEntry], spec_scored_full: list[list[tuple[ListingRow, float]]], query: str, n_specs: int, *, max_anchor_shops: int=EMPTY_SHOP_ANCHOR_CAP) -> tuple[list[str] | None, dict]:
    spec_scored, shop_cov_empty = _shop_empty_effective_scored_and_coverage(spec_scored_full, float(SHOP_SCORE_MIN))
    if n_specs >= 3:
        pids, ctx = _shop_empty_attempt_partial_coverage(specs, spec_scored, shop_cov_empty, query, n_specs)
        if pids:
            return (pids, ctx)
    global_max = max((scored[0][1] for scored in spec_scored if scored), default=0.0)
    if global_max <= 0:
        return (None, {})
    top_by_spec = _global_top_products(spec_scored, global_max)
    top_spec_indices = list(top_by_spec.keys())
    if len(top_spec_indices) == 1:
        spec_idx = top_spec_indices[0]
        if len(top_by_spec[spec_idx]) == 1:
            resolution_mode = 1
            tie_note = 'Single global top-scoring product; anchoring directly.'
        else:
            resolution_mode = 2
            tie_note = f'{len(top_by_spec[spec_idx])} products tied at score {global_max:.1f} in spec[{spec_idx}]; iterating shops by price/rank.'
    else:
        winning_idx = _deepest_spec(top_spec_indices, specs)
        resolution_mode = 3
        tie_note = f'Top score {global_max:.1f} tied across specs {top_spec_indices}; depth scoring selected spec[{winning_idx}] as primary anchor spec.'
    ranked_anchors = _order_anchor_pool(spec_scored, n_specs, max_anchor_shops)
    if not ranked_anchors:
        return (None, {})
    for attempt_num, (_score, anchor_spec_idx, anchor) in enumerate(ranked_anchors):
        if _budget_sec_left() < ANCHOR_TIMEOUT_SEC:
            break
        anchor_shop_id = str(anchor.get('shop_id', ''))
        if not anchor_shop_id:
            continue
        out_pids: list[str | None] = [None] * n_specs
        out_pids[anchor_spec_idx] = str(anchor.get('product_id', ''))
        filled_specs: list[dict] = []
        anchor_ok = True
        for i in range(n_specs):
            if i == anchor_spec_idx:
                continue
            best = _select_in_shop_empty_relaxed(specs[i], anchor_shop_id, query)
            if not best:
                anchor_ok = False
                break
            out_pids[i] = str(best.get('product_id', ''))
            filled_specs.append({'spec_idx': i, 'keywords': specs[i].get('keywords', ''), 'product_id': str(best.get('product_id', '')), 'title': best.get('title', ''), 'price': best.get('price'), 'llm_reason': str(best.get('_llm_reason', '') or '')})
        if anchor_ok and all((pid is not None for pid in out_pids)):
            ctx = {'resolution_mode': resolution_mode, 'global_max_score': global_max, 'tie_note': tie_note, 'anchor_attempt': attempt_num + 1, 'anchor': {'spec_idx': anchor_spec_idx, 'keywords': specs[anchor_spec_idx].get('keywords', ''), 'product_id': str(anchor.get('product_id', '')), 'title': anchor.get('title', ''), 'price': anchor.get('price'), 'shop_id': anchor_shop_id}, 'filled_specs': filled_specs}
            return (out_pids, ctx)
    return (None, {})

def _partial_coverage_resolve(specs: list[SpecEntry], spec_scored: list[list[tuple[ListingRow, float]]], shop_coverage: dict[str, dict[int, list[ListingRow]]], query: str, n_specs: int) -> tuple[list[str] | None, dict]:
    target = n_specs - 1
    partial = {sid: cov for sid, cov in shop_coverage.items() if len(cov) == target}
    if not partial:
        return (None, {})
    pid_to_score: dict[str, float] = {str(p.get('product_id', '')): sc for scored in spec_scored for p, sc in scored}

    def _shop_total(cov: dict) -> float:
        return sum((max((pid_to_score.get(str(p.get('product_id', '')), 0.0) for p in products), default=0.0) for products in cov.values()))
    shop_scores = {sid: _shop_total(cov) for sid, cov in partial.items()}
    max_score = max(shop_scores.values())
    winner = max(shop_scores.items(), key=lambda item: item[1])[0]
    coverage = partial[winner]
    covered = set(coverage.keys())
    missing_idx = next((i for i in range(n_specs) if i not in covered))
    resolved: list[str | None] = [None] * n_specs

    def _best_covered_pid(sidx: int) -> str | None:
        shop_pids = {str(p.get('product_id', '')) for p in coverage[sidx]}
        best = next((p for p, _ in spec_scored[sidx] if str(p.get('product_id', '')) in shop_pids), coverage[sidx][0] if coverage[sidx] else None)
        return str(best.get('product_id', '')) if best else None
    for sidx in covered:
        resolved[sidx] = _best_covered_pid(sidx)
    best_missing = _select_in_shop(specs[missing_idx], winner, query)
    if not best_missing:
        return (None, {})
    resolved[missing_idx] = str(best_missing.get('product_id', ''))
    if not all((pid is not None for pid in resolved)):
        return (None, {})
    ctx = {'resolution_mode': 4, 'partial_shops_evaluated': len(partial), 'winner_shop_id': winner, 'winner_shop_score': round(max_score, 2), 'covered_spec_indices': sorted(covered), 'missing_spec_idx': missing_idx, 'missing_spec_keywords': specs[missing_idx].get('keywords', ''), 'filled_missing_product': {'product_id': str(best_missing.get('product_id', '')), 'title': best_missing.get('title', ''), 'price': best_missing.get('price')}}
    return (resolved, ctx)

def _global_top_products(spec_scored: list[list[tuple[ListingRow, float]]], global_max: float) -> dict[int, list[ListingRow]]:
    top_by_spec: dict[int, list[ListingRow]] = defaultdict(list)
    for sidx, scored in enumerate(spec_scored):
        for prod, sc in scored:
            if sc >= global_max:
                top_by_spec[sidx].append(prod)
    return top_by_spec

def _classify_case_c(spec_scored: list[list[tuple[ListingRow, float]]], specs: list[SpecEntry]) -> int:
    global_max = max((scored[0][1] for scored in spec_scored if scored), default=0.0)
    if global_max <= 0:
        return 0
    top_by_spec = _global_top_products(spec_scored, global_max)
    top_indices = list(top_by_spec.keys())
    if len(top_indices) == 1:
        spec_idx = top_indices[0]
        return 1 if len(top_by_spec[spec_idx]) == 1 else 2
    return 3

def _order_anchor_pool(spec_scored: list[list[tuple[ListingRow, float]]], n_specs: int, max_shops: int) -> list[tuple[float, int, ListingRow]]:
    seen_shops: set[str] = set()
    out: list[tuple[float, int, ListingRow]] = []

    def _push(entry: tuple[float, int, ListingRow]) -> None:
        if len(out) >= max_shops:
            return
        sid = str(entry[2].get('shop_id', '') or '')
        if not sid or sid in seen_shops:
            return
        seen_shops.add(sid)
        out.append(entry)
    max_depth = max((len(spec_scored[si]) for si in range(n_specs)), default=0)
    for rank in range(min(max_depth, 12)):
        for si in range(n_specs):
            if rank < len(spec_scored[si]):
                prod, sc = spec_scored[si][rank]
                _push((float(sc), si, prod))
            if len(out) >= max_shops:
                return out
    return out

def _anchor_strategy(specs: list[SpecEntry], spec_scored: list[list[tuple[ListingRow, float]]], shop_coverage: dict[str, dict[int, list[ListingRow]]], query: str, n_specs: int, max_anchor_shops: int | None=None, is_shop_voucher: bool=False, voucher_budget: tuple[float, float] | None=None) -> tuple[list[str] | None, dict]:
    if max_anchor_shops is None:
        max_anchor_shops = ANCHOR_SHOP_LIMIT
    if n_specs >= 3:
        resolved, partial_ctx = _partial_coverage_resolve(specs, spec_scored, shop_coverage, query, n_specs)
        if resolved:
            return (resolved, partial_ctx)
    use_v2 = is_shop_voucher or n_specs >= 4
    if not use_v2:
        global_max = max((scored[0][1] for scored in spec_scored if scored), default=0.0)
        if global_max <= 0:
            return (None, {})
        top_by_spec = _global_top_products(spec_scored, global_max)
        top_indices = list(top_by_spec.keys())
        if len(top_indices) == 1:
            spec_idx = top_indices[0]
            if len(top_by_spec[spec_idx]) == 1:
                resolution_mode = 1
                tie_note = 'Single global top-scoring product; anchoring directly.'
            else:
                resolution_mode = 2
                tie_note = f'{len(top_by_spec[spec_idx])} products tied at score {global_max:.1f} in spec[{spec_idx}]; iterating shops by price/rank.'
        else:
            winning_idx = _deepest_spec(top_indices, specs)
            resolution_mode = 3
            tie_note = f'Top score {global_max:.1f} tied across specs {top_indices}; depth scoring selected spec[{winning_idx}] as primary anchor spec.'
        ranked_anchors = _order_anchor_pool(spec_scored, n_specs, max_anchor_shops)
        if not ranked_anchors:
            return (None, {})
        for attempt_num, (score, anchor_spec_idx, anchor) in enumerate(ranked_anchors):
            if _budget_sec_left() < ANCHOR_TIMEOUT_SEC:
                break
            anchor_shop = str(anchor.get('shop_id', ''))
            if not anchor_shop:
                continue
            resolved_loop: list[str | None] = [None] * n_specs
            resolved_loop[anchor_spec_idx] = str(anchor.get('product_id', ''))
            filled: list[dict] = []
            anchor_ok = True
            for sidx in range(n_specs):
                if sidx == anchor_spec_idx:
                    continue
                best = _select_in_shop(specs[sidx], anchor_shop, query)
                if not best:
                    anchor_ok = False
                    break
                resolved_loop[sidx] = str(best.get('product_id', ''))
                filled.append({'spec_idx': sidx, 'keywords': specs[sidx].get('keywords', ''), 'product_id': str(best.get('product_id', '')), 'title': best.get('title', ''), 'price': best.get('price'), 'llm_reason': best.get('_llm_reason', '')})
            if anchor_ok and all((pid is not None for pid in resolved_loop)):
                ctx = {'resolution_mode': resolution_mode, 'global_max_score': global_max, 'tie_note': tie_note, 'anchor_attempt': attempt_num + 1, 'anchor': {'spec_idx': anchor_spec_idx, 'keywords': specs[anchor_spec_idx].get('keywords', ''), 'product_id': str(anchor.get('product_id', '')), 'title': anchor.get('title', ''), 'price': anchor.get('price'), 'shop_id': anchor_shop}, 'filled_specs': filled}
                return (resolved_loop, ctx)
        return (None, {})
    _ANCHOR_V2_TOP_N = 5
    score_floor = TWO_SPEC_SCORE_FLOOR if is_shop_voucher else SHOP_SCORE_MIN
    best_by_shop: dict[str, float] = {}
    for scored in spec_scored:
        for prod, sc in scored:
            sid = str(prod.get('shop_id') or '').strip()
            if not sid:
                continue
            if sc > best_by_shop.get(sid, -1.0):
                best_by_shop[sid] = sc
    if not best_by_shop:
        return (None, {})
    ranked_shop_scores = sorted(best_by_shop.items(), key=lambda x: -x[1])
    selected_shops: list[str] = []
    cutoff_score: float | None = None
    for i, (sid, sc) in enumerate(ranked_shop_scores):
        if i < _ANCHOR_V2_TOP_N:
            selected_shops.append(sid)
            cutoff_score = sc
        elif cutoff_score is not None and sc == cutoff_score:
            selected_shops.append(sid)
        else:
            break
    if not selected_shops:
        return (None, {})
    allowed_total: float | None = None
    if voucher_budget is not None:
        _, allowed_total = voucher_budget
    anchor_search_results: list[list[ListingRow]] = []
    for sidx, spec in enumerate(specs):
        spec_cands: list[ListingRow] = []
        seen_pids: set[str] = set()
        price_override: str | None = None
        if allowed_total is not None:
            iw = _intersect_spec_price_with_budget_cap(spec, allowed_total)
            if iw is not None:
                price_override = _price_lo_hi_to_find_str(iw[0], iw[1])
        for shop_id in selected_shops:
            hits = _search_spec_in_shop_limited(spec, shop_id, page=1, limit=10, price_override=price_override)
            for h in hits:
                pid = str(h.get('product_id', ''))
                if pid and pid not in seen_pids:
                    spec_cands.append(h)
                    seen_pids.add(pid)
        anchor_search_results.append(spec_cands)
    anchor_spec_scored: list[list[tuple[ListingRow, float]]] = []
    for sidx, (spec, cands) in enumerate(zip(specs, anchor_search_results)):
        if not cands:
            anchor_spec_scored.append([])
            continue
        search_q = spec.get('query') or spec.get('keywords') or query
        pids = [str(p.get('product_id', '')) for p in cands if p.get('product_id')]
        details = _load_details(pids)
        scored_pairs = _score_listings(search_q, cands, details, only_product_type=bool(spec.get('only_product_type', False)))
        anchor_spec_scored.append(scored_pairs)
    anchor_filtered: list[list[ListingRow]] = [[p for p, _ in scored] for scored in anchor_spec_scored]
    anchor_shop_cov = _index_by_shop(anchor_filtered)
    anchor_full_shops = [sid for sid, cov in anchor_shop_cov.items() if len(cov) == n_specs]
    if not anchor_full_shops:
        return (None, {})
    preranked = _sort_shops(anchor_full_shops, anchor_shop_cov, specs, query)
    top_pool = preranked[:SHOP_TOP_N]
    shop_id, chosen = _choose_shop_llm(top_pool, anchor_shop_cov, specs, query)
    chosen_ids = [chosen[sidx]['product_id'] for sidx in range(n_specs) if sidx in chosen]
    if not (shop_id and len(chosen_ids) == n_specs):
        return (None, {})
    if voucher_budget is not None and allowed_total is not None:
        total_price = 0.0
        for sidx in range(n_specs):
            chosen_pid = chosen[sidx]['product_id']
            pool = (anchor_shop_cov.get(shop_id) or {}).get(sidx, [])
            prod = next((p for p in pool if str(p.get('product_id', '')) == chosen_pid), None)
            if prod is not None:
                pr = _numeric_product_price(prod)
                if pr is not None:
                    total_price += pr
        if total_price > allowed_total + 1e-06:
            return (None, {})
    ctx = {'resolution_mode': 5, 'anchor_shops_selected': selected_shops, 'anchor_full_shops_count': len(anchor_full_shops), 'anchor_attempt': 1, 'tie_note': f'Anchor v2: {len(selected_shops)} top-ranked shops selected, {len(anchor_full_shops)} full-coverage after re-scoring. Case-B LLM elected shop_id={shop_id}.', 'anchor': {'shop_id': shop_id, 'spec_idx': None}, 'filled_specs': [{'spec_idx': sidx, 'keywords': specs[sidx].get('keywords', ''), 'product_id': chosen[sidx]['product_id'], 'reason': chosen[sidx].get('reason', ''), 'score': chosen[sidx].get('score', 0.0)} for sidx in range(n_specs) if sidx in chosen]}
    return (list(chosen_ids), ctx)

def _route_task_kind(query: str) -> str:
    query_lower = query.lower()
    voucher_signals = {'voucher', 'budget', 'discount'}
    if any((sig in query_lower for sig in voucher_signals)):
        return 'voucher'
    shop_keywords = re.search('\\b(both|these|offering|offers|sells|same|together|along\\s+with)\\b', query_lower)
    if 'shop' in query_lower and (shop_keywords is not None or _RX_MULTI_SPLIT.search(query) is not None):
        return 'shop'
    return 'product'

def _extract_price_range(text: str) -> str | None:
    price_match = re.search('(?:greater|more|over|above|>|cost[s]?\\s+more)\\s*(?:than\\s*)?(\\d+)', text, re.I)
    if price_match:
        return f'{price_match.group(1)}-'
    range_match = re.search('(\\d{1,6})\\s*(?:to|and|-)\\s*(\\d{1,6})\\s*(?:pesos|php)', text, re.I)
    if range_match:
        return f'{range_match.group(1)}-{range_match.group(2)}'
    if re.search('(?:price|pesos|php|cost)', text, re.I):
        range_match2 = re.search('(\\d{1,6})\\s+(?:to|and)\\s+(\\d{1,6})', text)
        if range_match2:
            return f'{range_match2.group(1)}-{range_match2.group(2)}'
    return None

def _extract_service_flags(text_lower: str) -> str | None:
    svc_parts: list[str] = []
    service_signals = [('official', ('lazmall', 'official')), ('freeShipping', ('free shipping', 'free delivery')), ('flashsale', ('lazflash', 'flash sale', 'flashsale')), ('COD', ('cash on delivery', 'cod'))]
    for svc_name, markers in service_signals:
        if any((marker in text_lower for marker in markers)):
            svc_parts.append(svc_name)
    return ','.join(svc_parts) if svc_parts else None

def _re_spec(text: str) -> dict:
    text_lower = text.lower()
    kw_tokens = _QueryTextAnalyzer.keyword_tokens(text)
    keywords = ' '.join(kw_tokens) or 'product'
    return {'keywords': keywords, 'price_range': _extract_price_range(text), 'service': _extract_service_flags(text_lower)}

def _regex_param_snapshot(query: str) -> dict:
    task_type = _route_task_kind(query)
    product_text = _RX_BUDGET_ANCHOR.split(query)[0].strip()
    if not product_text or len(product_text) < 15:
        product_text = query
    parts = [p.strip() for p in _RX_MULTI_SPLIT.split(product_text) if p and len(p.strip()) > 10]
    if not parts:
        parts = [query]
    products = [_re_spec(p) for p in parts]
    products = [s for s in products if len(s['keywords'].split()) >= 2] or products
    is_shop = task_type == 'shop' or (task_type == 'voucher' and 'same shop' in query.lower())
    return {'task_type': task_type, 'products': products, 'is_shop_voucher': is_shop}

def _clean_shop_keywords(parsed: dict) -> dict:
    for prod in parsed.get('products', []):
        kw = prod.get('keywords')
        if not kw:
            continue
        if isinstance(kw, list):
            kw = ' '.join((str(t) for t in kw))
        prod['keywords'] = ' '.join((w for w in str(kw).split() if w.lower() not in RANK_STOPWORDS))
    return parsed

def _parse_llm_params_response(result: dict, task_type: str) -> dict | None:
    if not result or not result.get('choices'):
        return None
    content = result['choices'][0].get('message', {}).get('content', '')
    parsed = _parse_json_str(content)
    if parsed is None:
        return None
    if task_type == 'product':
        return _SearchParamSanitizer.clean_params(parsed)
    if task_type == 'shop':
        return _clean_shop_keywords(parsed)
    return parsed
_PARSE_PROMPT_MAP: dict[str, str] = {'product': RazgrizPrompts.PARSE_PRODUCT, 'shop': RazgrizPrompts.PARSE_SHOP, 'voucher': RazgrizPrompts.PARSE_VOUCHER}
_PARSE_MODEL_MAP: dict[str, str] = {'product': PRODUCT_PARSE_, 'shop': SHOP_PARSE_, 'voucher': VOUCHER_PARSE_}

def _llm_param_snapshot(query: str, task_type: str) -> dict:
    sys_prompt = _PARSE_PROMPT_MAP.get(task_type, RazgrizPrompts.PARSE_PRODUCT)
    base_model = _PARSE_MODEL_MAP.get(task_type, VOUCHER_PARSE_)
    for model in _fallback_chain(base_model):
        result = _llm_transport.post('/inference/chat/completions', json_data={'model': model, 'temperature': 0, 'stream': False, 'messages': [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': query}]})
        parsed = _parse_llm_params_response(result, task_type)
        if parsed is not None:
            return parsed
        msg = 'returned unparseable response' if result and result.get('choices') else 'returned no response'
    return _regex_param_snapshot(query)

class _ShopResult(_NamedTuple):
    shop_id: str
    product_ids: list[str]
    think: str
    leader_products: list[dict]
    all_candidate_product_ids: list[str]

class _EmptyProblemDataProcessor:
    _DEFAULT_QUERY_ENV: str = 'RAZGRIZ_DEFAULT_QUERY'
    _QUERY_MAX_LEN: int = 512
    _SALVAGE_KEYS: tuple[str, ...] = ('regex_hints', 'catalog_attribute_keys_seen')
    _REGEX_HINT_KEYS: tuple[str, ...] = ('quoted_literals', 'number_unit_tokens', 'size_labels', 'color_words', 'service_tags')
    _SYNTHETIC_FLAG: str = '_synthesized'

    def __init__(self, raw: Any) -> None:
        self.raw = raw
        self._reason: str = ''

    @classmethod
    def ensure(cls, raw: Any) -> dict:
        proc = cls(raw)
        if not proc.is_empty():
            return raw
        return proc.process()

    def is_empty(self) -> bool:
        raw = self.raw
        if raw is None:
            self._reason = 'none'
            return True
        if not raw:
            self._reason = 'falsy_empty'
            return True
        if not isinstance(raw, dict):
            self._reason = f'non_dict:{type(raw).__name__}'
            return True
        if not self._has_query(raw):
            self._reason = 'missing_query'
            return True
        self._reason = ''
        return False

    def process(self) -> dict:
        reason = self._reason or 'unknown'
        query = self._resolve_query()
        salvaged = self._salvage()
        payload: dict = {'query': query}
        payload.update(salvaged)
        payload[self._SYNTHETIC_FLAG] = True
        return payload

    def summary(self) -> dict:
        return {'was_empty': bool(self._reason), 'reason': self._reason or 'ok', 'raw_type': type(self.raw).__name__}

    def _resolve_query(self) -> str:
        if isinstance(self.raw, str) and self.raw.strip():
            return self._normalize_query(self.raw)
        env_q = getenv(self._DEFAULT_QUERY_ENV, '').strip()
        if env_q:
            return self._normalize_query(env_q)
        return self._normalize_query(DEFAULT_QUERY)

    @classmethod
    def _normalize_query(cls, text: Any) -> str:
        cleaned = _dialogue_strip_markup_fragment(text)
        collapsed = re.sub('\\s+', ' ', cleaned).strip()
        trimmed = collapsed[:cls._QUERY_MAX_LEN]
        return trimmed or DEFAULT_QUERY

    @staticmethod
    def _has_query(d: dict) -> bool:
        q = d.get('query')
        return isinstance(q, str) and bool(q.strip())

    def _salvage(self) -> dict:
        out: dict = {}
        if not isinstance(self.raw, dict):
            return out
        hints = self._clean_regex_hints(self.raw.get('regex_hints'))
        if hints:
            out['regex_hints'] = hints
        seen = self._clean_token_list(self.raw.get('catalog_attribute_keys_seen'))
        if seen:
            out['catalog_attribute_keys_seen'] = seen
        return out

    @classmethod
    def _clean_regex_hints(cls, hints: Any) -> dict:
        if not isinstance(hints, dict):
            return {}
        cleaned: dict = {}
        for key in cls._REGEX_HINT_KEYS:
            tokens = cls._clean_token_list(hints.get(key))
            if tokens:
                cleaned[key] = tokens
        return cleaned

    @staticmethod
    def _clean_token_list(value: Any) -> list:
        if not isinstance(value, (list, tuple, set)):
            return []
        return _SearchParamSanitizer.unique_ids(list(value))

class _QueryTextAnalyzer:

    @staticmethod
    def tokenize(query_text: str) -> list[str]:
        return list(dict.fromkeys((tok for tok in re.findall('\\b\\w+\\b', query_text.lower()) if len(tok) > 1 and tok not in RANK_STOPWORDS)))

    @staticmethod
    def word_set(query_text: str) -> set[str]:
        return {w for w in re.findall('\\b\\w+\\b', query_text.lower()) if len(w) > 1 and w not in RANK_STOPWORDS}

    @staticmethod
    def keyword_tokens(text: str) -> list[str]:
        text_lower = text.lower()
        alpha_words = [w for w in re.findall('\\b[a-zA-Z]{2,}\\b', text_lower) if w not in PARSE_STOPWORDS]
        mixed_tokens = re.findall('\\b\\d+[a-zA-Z]+\\b|\\b[a-zA-Z]+\\d+[a-zA-Z]*\\b', text_lower)
        kw_tokens = alpha_words[:6]
        for tok in mixed_tokens[:2]:
            if tok not in kw_tokens:
                kw_tokens.append(tok)
        for num_token in re.findall('(\\d+)#', text)[:2]:
            if num_token not in kw_tokens:
                kw_tokens.append(num_token)
        return kw_tokens

    @staticmethod
    def clean_keywords(text: str | None) -> str:
        if not text:
            return DEFAULT_QUERY
        unique_tokens = list(dict.fromkeys((tok for tok in text.lower().split() if tok not in RANK_STOPWORDS)))
        return ' '.join(unique_tokens) if unique_tokens else DEFAULT_QUERY

class _SearchParamSanitizer:

    @staticmethod
    def normalise_service(service: str | None) -> str | None:
        if not service or service == 'default':
            return None if service == 'default' else service
        parts = [p.strip() for p in service.split(',') if p.strip() and p.strip() != 'default']
        return ','.join(parts) or None

    @staticmethod
    def clean_entry(prod: dict) -> dict:
        cleaned = dict(prod)
        for field in ('keywords', 'q'):
            if field in cleaned:
                cleaned[field] = _QueryTextAnalyzer.clean_keywords(cleaned.get(field))
        return cleaned

    @staticmethod
    def clean_params(params: dict) -> dict:
        out = dict(params)
        raw_products = out.get('products') or []
        cleaned_products = [_SearchParamSanitizer.clean_entry(p) for p in raw_products if isinstance(p, dict)]
        if cleaned_products:
            out['products'] = cleaned_products
        return out

    @staticmethod
    def deduplicate(products: list) -> list:
        by_pid: dict[str, dict] = {}
        for entry in products:
            pid = str(entry.get('product_id', ''))
            if pid:
                by_pid.setdefault(pid, entry)
        return list(by_pid.values())

    @staticmethod
    def unique_ids(ids: list) -> list[str]:
        stripped = (str(raw).strip() for raw in ids)
        return list(dict.fromkeys((val for val in stripped if val)))

class _PipeCtx:

    def __init__(self) -> None:
        self.steps: list[dict] = []
        self.query: str = ''

def _build_init_fallback(task_type: str, ctx: '_PipeCtx', keyword_list: list, price_list: list, service_list: list) -> str:
    base = f"Task type: {task_type}. Query (prefix): '{ctx.query[:300]}'. Parsed search keywords per product line: {keyword_list}. Parsed price_range strings: {price_list}. Parsed service filters: {service_list}. "
    if task_type == 'shop':
        return base + ' Next: same-shop flow runs per-spec catalog retrieval, `_score_listings` thresholding, full-coverage shop detection, then Case C / anchor logic if needed.'
    if task_type == 'voucher':
        return base + ' Next: voucher flow computes `allowed_total` from discount/threshold/cap/budget, then searches price bands, scores candidates, and enforces cart window [threshold, allowed_total].'
    if task_type == 'product':
        return base + ' Next: single-product flow searches, judges, and may broaden before recommending.'
    return base

def _dispatch_task(ctx: '_PipeCtx', task_type: str, params: dict) -> None:

    def _voucher_branch() -> None:
        products_info = params.get('products', [])
        voucher_block = params.get('voucher') or {}
        is_shop_type = str(voucher_block.get('voucher_type', '')).lower() == 'shop'
        is_same_shop = bool(params.get('is_shop_voucher')) or is_shop_type or 'same shop' in ctx.query.lower()
        if is_same_shop and len(products_info) > 1:
            _pipe_run_shop_voucher(ctx, params)
        else:
            _pipe_run_voucher(ctx, params)
    if task_type == 'voucher':
        _voucher_branch()
        return
    _pipe_run_shop(ctx, params)

def _execute_session_core(ctx: '_PipeCtx') -> None:
    task_type = _route_task_kind(ctx.query)
    params = _llm_param_snapshot(ctx.query, task_type)
    products_info = params.get('products', [])
    keyword_list = [e.get('keywords') or e.get('q', '') for e in products_info]
    price_list = [e.get('price_range') for e in products_info]
    service_list = [e.get('service') for e in products_info]
    init_fallback = _build_init_fallback(task_type, ctx, keyword_list, price_list, service_list)
    init_ctx: dict = {'keywords': keyword_list, 'price_constraints': price_list, 'service_filters': service_list}
    if products_info and bool(products_info[0].get('only_product_type')):
        init_ctx['only_product_type'] = True
        init_ctx['only_product_type_reason'] = ONLY_TYPE_NOTE
    if params.get('voucher'):
        voucher_info = params['voucher']
        init_ctx['budget_constraint'] = {'discount_type': voucher_info.get('discount_type'), 'discount_value': voucher_info.get('discount_value'), 'threshold': voucher_info.get('threshold'), 'cap': voucher_info.get('cap'), 'budget': voucher_info.get('budget')}
    _pipe_append_step(ctx, init_fallback, [])
    _dispatch_task(ctx, task_type, params)

def run(ctx: '_PipeCtx', problem_data: dict) -> list[dict]:
    _pipeline_session_begin(ctx, problem_data)
    try:
        _execute_session_core(ctx)
    except Exception:
        try:
            _pipe_finalize(ctx, [SENTINEL_PID], 'failure')
        except Exception:
            ctx.steps.append(create_dialogue_step('Done.', [], 'Done.', ctx.query, len(ctx.steps) + 1))
    return _pipeline_session_finish(ctx)

def _pipeline_session_begin(ctx: '_PipeCtx', problem_data: dict) -> None:
    global _pipeline_start, _detail_cache, _last_tool_call_ts
    _pipeline_start = time.monotonic()
    _last_tool_call_ts = 0.0
    _detail_cache = {}
    _trace_reset()
    ctx.steps = []
    ctx.query = problem_data.get('query', '')

def _pipeline_session_finish(ctx: '_PipeCtx') -> list[dict]:
    if not ctx.steps:
        ctx.steps.append(create_dialogue_step('Done.', [], 'Done.', ctx.query, 1))
    _trace_attach(ctx.steps)
    return ctx.steps

def _pipe_append_step(ctx, think: str, tool_results: list, response: str='') -> None:
    compact = [_compact_result(tc) for tc in tool_results or []]
    if not think or not str(think).strip():
        think = 'Recording the current pipeline step.'
    else:
        think = _dialogue_strip_markup_fragment(think)
    if not compact and (not response or not str(response).strip()):
        response = 'Continuing analysis based on the gathered context.'
    elif response:
        response = _dialogue_strip_markup_fragment(response)
    ctx.steps.append(create_dialogue_step(think, compact, response, ctx.query, len(ctx.steps) + 1))

def _pipe_finalize(ctx, product_ids: list, status: str, think: str='', llm_reason: str='') -> None:
    fmt_ids = _join_ids(product_ids)
    qprev = str(getattr(ctx, 'query', '') or '')[:240]
    rec = _call_api('recommend_product', {'product_ids': fmt_ids})
    term = _call_api('terminate', {'status': status})
    if not think:
        reason_part = f'{llm_reason} ' if llm_reason else ''
        fb = f'I am recommending product(s) {fmt_ids} for the query. {reason_part}Status: {status}.'
        narrate_ctx: dict = {'recommended_product_ids': fmt_ids, 'status': status, 'note': 'Finalising recommendation and terminating the session.'}
        if llm_reason:
            narrate_ctx['llm_reason'] = llm_reason
        think = fb
    _pipe_append_step(ctx, think, [rec, term], 'Done.')

def _pipe_weigh_multi(ctx, leaders: list, pools: list, specs: list, n_alts: int=2) -> None:
    per_spec = []
    for i, (leader, pool, spec) in enumerate(zip(leaders, pools, specs)):
        if leader is None:
            continue
        lead_pid = str(leader.get('product_id', ''))
        lead_heur = _safe_score(leader, ctx.query, spec)
        others = [p for p in pool or [] if str(p.get('product_id', '')) != lead_pid]
        try:
            others = sorted(others, key=lambda p: _composite_score(p, ctx.query, parsed_spec=spec), reverse=True)
        except Exception:
            pass
        alt_entries = [{'product_id': str(a.get('product_id', '')), 'price': a.get('price'), 'heuristic_score': _safe_score(a, ctx.query, spec)} for a in others[:n_alts]]
        outside_alt = None
        try:
            _o = _oro_native_outside_alt(spec, ctx.query, lead_pid)
            if _o:
                outside_alt = {'product_id': str(_o.get('product_id', '') or ''), 'price': _o.get('price'), 'heuristic_score': _safe_score(_o, ctx.query, spec)}
        except Exception:
            outside_alt = None
        per_spec.append({'spec_idx': i, 'keywords': (spec or {}).get('keywords', ''), 'leader': {'product_id': lead_pid, 'price': leader.get('price'), 'heuristic_score': lead_heur, 'llm_reason': leader.get('_llm_reason', '')}, 'alternatives': alt_entries, 'outside_alt': outside_alt})
    if not per_spec:
        return
    step_data = {'weighing': {'per_spec': per_spec}}
    fb_parts = [f'I am weighing candidates across {len(per_spec)} specs.']
    for e in per_spec:
        alts_fmt = ', '.join((f"{a['product_id']}@{a['price']}" for a in e['alternatives'])) or 'none'
        prefer = ''
        top = e.get('outside_alt')
        if e['alternatives'] or top:
            alt_pid = top['product_id'] if top else None
            alt_price = top['price'] if top else None
            alt_score = top.get('heuristic_score') if top else None
            prefer = f" I prefer {_oro_candidate_ref(e['leader']['product_id'], True)} (price={e['leader']['price']}, score={e['leader']['heuristic_score']}) OVER {_oro_candidate_ref(alt_pid, False)} (price={alt_price}, score={alt_score}) because the leader scores higher and matches the spec keywords more closely."
        fb_parts.append(f"Spec[{e['spec_idx']}] '{e['keywords']}': leader pid={e['leader']['product_id']} price={e['leader']['price']} score={e['leader']['heuristic_score']}; alternatives: {alts_fmt}.{prefer}")
    _pipe_append_step(ctx, ' '.join(fb_parts), [])

def _pipe_two_spec_resolve(ctx, specs: list[SpecEntry], spec_scored: list[list[tuple[ListingRow, float]]], cand_ids_by_spec: list[list[str]], all_cand_ids: list[str], voucher_budget: tuple[float, float] | None=None) -> _ShopResult | None:
    return _pipe_bidir_two_spec_resolve(ctx, specs, spec_scored, cand_ids_by_spec, all_cand_ids, voucher_budget=voucher_budget)

def _pipe_bidir_two_spec_resolve(ctx, specs: list[SpecEntry], spec_scored: list[list[tuple[ListingRow, float]]], cand_ids_by_spec: list[list[str]], all_cand_ids: list[str], voucher_budget: tuple[float, float] | None=None) -> _ShopResult | None:
    if len(specs) != 2:
        return None
    if len(spec_scored) < 2:
        spec_scored = list(spec_scored) + [[] for _ in range(2 - len(spec_scored))]
    elif len(spec_scored) > 2:
        spec_scored = list(spec_scored[:2])
    spec_scored = [_dedupe_spec_pairs_by_shop(spec_scored[0]), _dedupe_spec_pairs_by_shop(spec_scored[1])]
    score_maps = [{str(p.get('product_id') or '').strip(): float(sc) for p, sc in spec_scored[i]} for i in range(2)]

    def _eval_direction(anchor_idx: int) -> dict[str, Any] | None:
        other_idx = 1 - anchor_idx
        anchor_spec = specs[anchor_idx]
        other_spec = specs[other_idx]
        pairs_a = spec_scored[anchor_idx]
        ranked_upto6 = _ranked_shop_ids_from_pairs(pairs_a, 6)
        if voucher_budget is not None:
            shop_attempts: list[list[str]] = [ranked_upto6[:TWO_SPEC_TOP_SHOPS]]
            if len(ranked_upto6) >= 6:
                shop_attempts.append(ranked_upto6[:6])
        else:
            sid_list = _two_spec_top_shop_ids(pairs_a, TWO_SPEC_TOP_SHOPS)
            if not sid_list:
                return None
            shop_attempts = [sid_list]
        for shop_ids in shop_attempts:
            if not shop_ids:
                continue
            pool_other = _gather_cross_spec_pool_in_shops(other_spec, shop_ids, TWO_SPEC_BIDIR_POOL_CAP)
            if not pool_other:
                continue
            oq = other_spec.get('query') or other_spec.get('keywords') or ctx.query
            opids = [str(p.get('product_id', '')) for p in pool_other if p.get('product_id')]
            details_o = _load_details(opids)
            mc_o = min(TWO_SPEC_BIDIR_POOL_CAP, len(pool_other))
            pick_other = _elect_best(oq, pool_other, details_o, only_product_type=bool(other_spec.get('only_product_type', False)), max_candidates=mc_o)
            sm_other = score_maps[other_idx]
            if pick_other is None:
                pick_other = dict(max(pool_other, key=lambda pr: sm_other.get(str(pr.get('product_id') or '').strip(), 0.0)))
                opid = str(pick_other.get('product_id') or '').strip()
                pick_other['_llm_relevance_score'] = float(sm_other.get(opid, 0.0))
            shop_win = str(pick_other.get('shop_id') or '').strip()
            if not shop_win:
                continue
            if voucher_budget is not None:
                _, v_allow = voucher_budget
                ap_first = _numeric_product_price(pick_other)
                if ap_first is None:
                    continue
                cap_hi = v_allow - ap_first
                iw = _intersect_spec_price_with_budget_cap(anchor_spec, cap_hi)
                if iw is None:
                    continue
                lo, hi = iw
                price_str = _price_lo_hi_to_find_str(lo, hi)
                pool_anchor = _search_spec_in_shop_limited(anchor_spec, shop_win, page=1, limit=TWO_SPEC_BIDIR_POOL_CAP, price_override=price_str)
                if not pool_anchor:
                    pool_anchor = _search_spec_in_shop_limited(anchor_spec, shop_win, page=1, limit=TWO_SPEC_BIDIR_POOL_CAP, omit_service_from_api=True, price_override=price_str)
            else:
                pool_anchor = _search_spec_in_shop_limited(anchor_spec, shop_win, page=1, limit=TWO_SPEC_BIDIR_POOL_CAP)
                if not pool_anchor:
                    pool_anchor = _search_spec_in_shop_limited(anchor_spec, shop_win, page=1, limit=TWO_SPEC_BIDIR_POOL_CAP, omit_service_from_api=True)
            if not pool_anchor:
                continue
            aq = anchor_spec.get('query') or anchor_spec.get('keywords') or ctx.query
            apids = [str(p.get('product_id', '')) for p in pool_anchor if p.get('product_id')]
            details_a = _load_details(apids)
            mc_a = min(TWO_SPEC_BIDIR_POOL_CAP, len(pool_anchor))
            pick_anchor = _elect_best(aq, pool_anchor, details_a, only_product_type=bool(anchor_spec.get('only_product_type', False)), max_candidates=mc_a)
            sm_anchor = score_maps[anchor_idx]
            if pick_anchor is None:
                pick_anchor = dict(max(pool_anchor, key=lambda pr: sm_anchor.get(str(pr.get('product_id') or '').strip(), 0.0)))
                apid = str(pick_anchor.get('product_id') or '').strip()
                pick_anchor['_llm_relevance_score'] = float(sm_anchor.get(apid, 0.0))
            if anchor_idx == 0:
                p0, p1 = (pick_anchor, pick_other)
            else:
                p0, p1 = (pick_other, pick_anchor)
            if voucher_budget is not None:
                v_thr, v_allow = voucher_budget
                pr0 = _numeric_product_price(p0)
                pr1 = _numeric_product_price(p1)
                if pr0 is None or pr1 is None:
                    continue
                if not v_thr - 1e-06 <= pr0 + pr1 <= v_allow + 1e-06:
                    continue
            s0 = float(p0.get('_llm_relevance_score', 0) or 0)
            s1 = float(p1.get('_llm_relevance_score', 0) or 0)
            shop_id = str(p0.get('shop_id') or shop_win or p1.get('shop_id') or '').strip()
            return {'anchor_idx': anchor_idx, 'p0': p0, 'p1': p1, 'sum_scores': s0 + s1, 'shop_id': shop_id, 'pool_cross_n': len(pool_other), 'pool_anchor_n': len(pool_anchor)}
        return None
    scored_dirs = [i for i in range(2) if spec_scored[i]]
    if not scored_dirs:
        return None
    dir_results: list[dict[str, Any] | None] = [None, None]
    for anchor_idx in scored_dirs:
        dir_results[anchor_idx] = _eval_direction(anchor_idx)
    candidates_dir = [d for d in dir_results if d is not None]
    if not candidates_dir:
        return None
    best = min(candidates_dir, key=lambda d: (-float(d['sum_scores']), int(d['anchor_idx'])))
    p0 = dict(best['p0'])
    p1 = dict(best['p1'])
    pid0 = str(p0.get('product_id', '')).strip()
    pid1 = str(p1.get('product_id', '')).strip()
    if not pid0 or not pid1:
        return None
    resolved_pids = [pid0, pid1]
    shop_pick = str(best.get('shop_id') or p0.get('shop_id') or p1.get('shop_id') or '').strip()
    total_sc = float(best['sum_scores'])
    enriched = _enrich_listings([{'product_id': pid} for pid in resolved_pids])
    leaders = [dict(info) for info in enriched]
    pools: list[list] = [[], []]
    _pipe_weigh_multi(ctx, leaders, pools, specs)
    cc = None
    if len(specs) == len(enriched):
        cc = [_verify_pick(title=info.get('title', ''), price=info.get('price'), parsed_spec=spec or {}) for spec, info in zip(specs, enriched)]

    def _summ_dir(d: dict[str, Any] | None) -> dict[str, Any] | None:
        if d is None:
            return None
        p0a, p1a = (d['p0'], d['p1'])
        return {'anchor_idx': d['anchor_idx'], 'sum_scores': round(float(d['sum_scores']), 2), 'shop_id': d.get('shop_id'), 'product_ids': [str(p0a.get('product_id') or '').strip(), str(p1a.get('product_id') or '').strip()], 'pool_cross_n': d['pool_cross_n'], 'pool_anchor_n': d['pool_anchor_n']}
    ctx_bidir: dict[str, Any] = {'shop_id': shop_pick, 'note': f"Two-spec bidirectional: per-spec dedupe-by-shop (best score), top {TWO_SPEC_TOP_SHOPS} shops, cross pool ={TWO_SPEC_BIDIR_POOL_CAP}, item judge both legs; chose anchor_spec={best['anchor_idx']} with sum_scores={total_sc:.2f}.", 'two_spec_bidir': {'direction_0': _summ_dir(dir_results[0]), 'direction_1': _summ_dir(dir_results[1]), 'selected_anchor_spec': best['anchor_idx'], 'sum_scores': round(total_sc, 2)}, 'selected_products': enriched, 'candidate_product_ids_by_spec': cand_ids_by_spec, 'all_candidate_product_ids': all_cand_ids}
    if cc is not None:
        ctx_bidir['constraint_checks'] = cc
    if voucher_budget is not None:
        ctx_bidir['voucher_budget'] = list(voucher_budget)
    think_bidir = f"Same-shop Case C (two specs): no single shop covered both specs in the initial pools. I ran the bidirectional pipeline ? for each direction one spec stays fixed to its best shop, the other spec searches only within that shop (pool capped at {TWO_SPEC_BIDIR_POOL_CAP}), then `_elect_best` scores both legs. Picked direction anchor_spec={best['anchor_idx']} (highest sum of judge scores={total_sc:.2f}). Resolved shop_id={shop_pick}, product_ids={resolved_pids}."
    _pipe_append_step(ctx, think_bidir, [])
    return _ShopResult(shop_id=str(shop_pick), product_ids=list(resolved_pids), think=think_bidir, leader_products=list(leaders), all_candidate_product_ids=all_cand_ids)

def _pipe_three_spec_resolve(ctx, specs: list[SpecEntry], spec_scored: list[list[tuple[ListingRow, float]]], cand_ids_by_spec: list[list[str]], all_cand_ids: list[str], voucher_budget: tuple[float, float] | None=None) -> _ShopResult | None:
    if len(specs) != 3:
        return None
    if len(spec_scored) < 3:
        spec_scored = list(spec_scored) + [[] for _ in range(3 - len(spec_scored))]
    elif len(spec_scored) > 3:
        spec_scored = list(spec_scored[:3])
    deduped = [_dedupe_spec_pairs_by_shop(spec_scored[i]) for i in range(3)]
    score_maps = [{str(p.get('product_id') or '').strip(): float(sc) for p, sc in deduped[i]} for i in range(3)]
    ranked = [_ranked_shop_ids_from_pairs(deduped[i], 12) for i in range(3)]
    scored_idxs = [i for i in range(3) if ranked[i]]
    if not scored_idxs:
        return None
    if len(scored_idxs) == 1:
        seed = list(ranked[scored_idxs[0]])
        for i in range(3):
            if not ranked[i]:
                ranked[i] = list(seed)
    elif len(scored_idxs) == 2:
        a, b = (scored_idxs[0], scored_idxs[1])
        union_ab: list[str] = []
        for sid in ranked[a] + ranked[b]:
            if sid not in union_ab:
                union_ab.append(sid)
        for i in range(3):
            if not ranked[i]:
                ranked[i] = list(union_ab)
    if any((not r for r in ranked)):
        return None
    active: list[list[str]] = [[], [], []]
    ptr = [0, 0, 0]

    def _union_shop_ids_for_target(act: list[list[str]], t: int) -> list[str]:
        others = [i for i in range(3) if i != t]
        o1, o2 = (others[0], others[1])
        out: list[str] = []
        for sid in act[o1] + act[o2]:
            if sid not in out:
                out.append(sid)
        return out

    def _filtered_union(act: list[list[str]], t: int) -> list[str]:
        others = [i for i in range(3) if i != t]
        o1, o2 = (others[0], others[1])
        raw = _union_shop_ids_for_target(act, t)
        return [sid for sid in raw if _spec_has_shop_hit(specs[o1], sid) and _spec_has_shop_hit(specs[o2], sid)]
    min_targets = [min(THREE_SPEC_TOP_SHOPS, len(ranked[i])) for i in range(3)]
    if any((t <= 0 for t in min_targets)):
        return None

    def _fill_minimum() -> bool:
        for sidx in range(3):
            while len(active[sidx]) < min_targets[sidx] and ptr[sidx] < len(ranked[sidx]):
                sid = ranked[sidx][ptr[sidx]]
                ptr[sidx] += 1
                if sid in active[sidx]:
                    continue
                if _spec_has_shop_hit(specs[sidx], sid):
                    active[sidx].append(sid)
        return all((len(active[s]) >= min_targets[s] for s in range(3)))
    if not _fill_minimum():
        return None
    for _grow in range(48):
        all_ok = True
        for t in range(3):
            if not _filtered_union(active, t):
                all_ok = False
                break
        if all_ok:
            break
        grew = False
        for ox in range(3):
            while ptr[ox] < len(ranked[ox]) and len(active[ox]) < 8:
                sid = ranked[ox][ptr[ox]]
                ptr[ox] += 1
                if sid in active[ox]:
                    continue
                if _spec_has_shop_hit(specs[ox], sid):
                    active[ox].append(sid)
                    grew = True
                    break
            if grew:
                break
        if not grew:
            return None
    picks: list[ListingRow | None] = [None, None, None]
    rels = [0.0, 0.0, 0.0]
    for t in range(3):
        shops_t = _filtered_union(active, t)
        if not shops_t:
            return None
        pool = _gather_cross_pool_per_shop_cap(specs[t], shops_t, total_cap=THREE_SPEC_POOL_CAP, per_shop=THREE_SPEC_PER_SHOP_LIMIT)
        if not pool:
            return None
        sq = specs[t].get('query') or specs[t].get('keywords') or ctx.query
        pids = [str(p.get('product_id', '')) for p in pool if p.get('product_id')]
        details = _load_details(pids)
        mc = min(THREE_SPEC_POOL_CAP, len(pool))
        pick = _elect_best(sq, pool, details, only_product_type=bool(specs[t].get('only_product_type', False)), max_candidates=mc)
        sm = score_maps[t]
        if pick is None:
            pick = dict(max(pool, key=lambda pr: sm.get(str(pr.get('product_id') or '').strip(), 0.0)))
            pid = str(pick.get('product_id') or '').strip()
            pick['_llm_relevance_score'] = float(sm.get(pid, 0.0))
        picks[t] = pick
        rels[t] = float(pick.get('_llm_relevance_score', 0) or 0)
    if picks[0] is None or picks[1] is None or picks[2] is None:
        return None
    shops_pick = [str(picks[i].get('shop_id') or '').strip() for i in range(3)]

    def _finalize(p_out: list[ListingRow], shop_id: str, note: str, fb: str) -> _ShopResult | None:
        rp = [str(p_out[i].get('product_id') or '').strip() for i in range(3)]
        if not all(rp):
            return None
        enriched = _enrich_listings([{'product_id': pid} for pid in rp])
        leaders = [dict(info) for info in enriched]
        pools: list[list] = [[], [], []]
        _pipe_weigh_multi(ctx, leaders, pools, specs)
        cc = None
        if len(specs) == len(enriched):
            cc = [_verify_pick(title=info.get('title', ''), price=info.get('price'), parsed_spec=spec or {}) for spec, info in zip(specs, enriched)]
        step_data: dict[str, Any] = {'shop_id': shop_id, 'note': note, 'three_spec_shop_judge': True, 'selected_products': enriched, 'candidate_product_ids_by_spec': cand_ids_by_spec, 'all_candidate_product_ids': all_cand_ids}
        if voucher_budget is not None:
            step_data['voucher_budget'] = list(voucher_budget)
        if cc is not None:
            step_data['constraint_checks'] = cc
        think = fb
        _pipe_append_step(ctx, think, [])
        return _ShopResult(shop_id=str(shop_id), product_ids=list(rp), think=think, leader_products=list(leaders), all_candidate_product_ids=all_cand_ids)
    if shops_pick[0] == shops_pick[1] == shops_pick[2] and shops_pick[0]:
        note = f'Three-spec: top {THREE_SPEC_TOP_SHOPS} shops/spec, pools={THREE_SPEC_POOL_CAP}, item judge agrees on shop_id={shops_pick[0]}.'
        if voucher_budget is not None:
            note += ' Same-shop triple; voucher total/threshold not enforced.'
        fb = f"Three-spec pipeline (Case C mode 1): after dedupe-by-shop and ranking, the per-spec `_elect_best` choices landed in the same shop_id={shops_pick[0]}. Product_ids={[str(picks[i].get('product_id')) for i in range(3)]}."
        return _finalize([picks[0], picks[1], picks[2]], shops_pick[0], note, fb)
    pair_candidates: list[tuple[int, int, float, float, int]] = []
    for i in range(3):
        for j in range(i + 1, 3):
            si, sj = (shops_pick[i], shops_pick[j])
            if si and si == sj:
                pair_candidates.append((i, j, rels[i] + rels[j], max(rels[i], rels[j]), _order_rank_sum_two(specs, i, j)))

    def _pick_fill_third(pair_a: int, pair_b: int, pa: ListingRow, pb: ListingRow) -> _ShopResult | None:
        k = 3 - pair_a - pair_b
        sid_win = str(pa.get('shop_id') or pb.get('shop_id') or '').strip()
        if not sid_win:
            return None
        if voucher_budget is not None:
            v_thr, v_allow = voucher_budget
            pr_a = _numeric_product_price(pa)
            pr_b = _numeric_product_price(pb)
            if pr_a is None or pr_b is None:
                return None
            remaining = v_allow - (pr_a + pr_b)
            if remaining < 0:
                return None
            iw = _intersect_spec_price_with_budget_cap(specs[k], remaining)
            if iw is None:
                return None
            lo, hi = iw
            price_str = _price_lo_hi_to_find_str(lo, hi)
            pool_k = _search_spec_in_shop_limited(specs[k], sid_win, page=1, limit=THREE_SPEC_PER_SHOP_LIMIT, price_override=price_str)
            if not pool_k:
                pool_k = _search_spec_in_shop_limited(specs[k], sid_win, page=1, limit=THREE_SPEC_PER_SHOP_LIMIT, omit_service_from_api=True, price_override=price_str)
            if not pool_k:
                return None
        else:
            pool_k = _gather_cross_pool_per_shop_cap(specs[k], [sid_win], total_cap=THREE_SPEC_PER_SHOP_LIMIT, per_shop=THREE_SPEC_PER_SHOP_LIMIT)
            if not pool_k:
                raw_o = _search_spec_in_shop_limited(specs[k], sid_win, page=1, limit=THREE_SPEC_PER_SHOP_LIMIT, omit_service_from_api=True)
                pool_k = list(raw_o or [])
            if not pool_k:
                return None
        sqk = specs[k].get('query') or specs[k].get('keywords') or ctx.query
        pk_ids = [str(p.get('product_id', '')) for p in pool_k if p.get('product_id')]
        detk = _load_details(pk_ids)
        pk = _elect_best(sqk, pool_k, detk, only_product_type=bool(specs[k].get('only_product_type', False)), max_candidates=min(THREE_SPEC_PER_SHOP_LIMIT, len(pool_k)))
        smk = score_maps[k]
        if pk is None:
            pk = dict(max(pool_k, key=lambda pr: smk.get(str(pr.get('product_id') or '').strip(), 0.0)))
            pk['_llm_relevance_score'] = float(smk.get(str(pk.get('product_id') or '').strip(), 0.0))
        if voucher_budget is not None:
            v_thr, v_allow = voucher_budget
            pr_a = _numeric_product_price(pa)
            pr_b = _numeric_product_price(pb)
            pr_k = _numeric_product_price(pk)
            if pr_a is None or pr_b is None or pr_k is None:
                return None
            tot = pr_a + pr_b + pr_k
            if not v_thr - 1e-06 <= tot <= v_allow + 1e-06:
                return None
        out = [None, None, None]
        out[pair_a] = pa
        out[pair_b] = pb
        out[k] = pk
        note = f'Three-spec: pair spec[{pair_a}]&spec[{pair_b}] share shop {sid_win}; filled spec[{k}] in-shop; tie-break sum/max/order_rank_sum.'
        fb = f"Three-spec: two specs share one shop; I searched the third spec inside that shop (with voucher price bounds when applicable). shop_id={sid_win}, product_ids={[str(out[x].get('product_id')) for x in range(3)]}."
        return _finalize([out[0], out[1], out[2]], sid_win, note, fb)
    if pair_candidates:
        sorted_pairs = sorted(pair_candidates, key=lambda t: (t[2], t[3], t[4]), reverse=True)
        for bi, bj, _, _, _ in sorted_pairs:
            res = _pick_fill_third(bi, bj, picks[bi], picks[bj])
            if res is not None:
                return res
    cand_pairs = [(0, 1), (0, 2), (1, 2)]
    sorted_ij = sorted(cand_pairs, key=lambda ij: (rels[ij[0]] + rels[ij[1]], max(rels[ij[0]], rels[ij[1]]), _order_rank_sum_two(specs, ij[0], ij[1])), reverse=True)
    for bi, bj in sorted_ij:
        res = _pick_fill_third(bi, bj, picks[bi], picks[bj])
        if res is not None:
            return res
    return None

def _pipe_emit_voucher_result(ctx, products: list[dict], n_specs: int, *, threshold: float, allowed_total: float, voucher: dict, pools: list[dict | None], search_calls: list, probe_product_ids: list[str], extra_candidate_id_lists: list[list[str]] | None=None, marginal_done_extra: dict | None=None) -> None:
    voucher_pid_lists: list[list[str]] = [_pid_list(pool['raw']) for pool in pools if pool and pool.get('raw')]
    voucher_union = _merged_pids(voucher_pid_lists)
    union_inputs: list[list[str]] = [probe_product_ids, voucher_union]
    if extra_candidate_id_lists:
        union_inputs.extend(extra_candidate_id_lists)
    all_voucher_candidates = _merged_pids(union_inputs)
    empty_specs = [i for i in range(n_specs) if pools[i] is None or not pools[i].get('raw')]
    if empty_specs:
        fail_think = f'I could not find any listing for spec(s) {empty_specs} inside the computed marginal price bands (intersection of probe min/max with parsed ranges, threshold={threshold:.2f}, allowed_total={allowed_total:.2f}). Probe/search union IDs: {all_voucher_candidates}.'
        _pipe_append_step(ctx, fail_think, search_calls)
        end_fail = f'Marginal-band voucher search produced no usable pool for every spec ? cannot seed `_init_picks`. Union probe/marginal candidate IDs: {all_voucher_candidates}.'
        _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think=end_fail)
        return
    picks_list = _init_picks(pools)
    if picks_list is None:
        fail_think = f'Pools existed but `_init_picks` could not take one high-scoring row per spec (missing raw pool or threshold-filtered set). Union IDs: {all_voucher_candidates}.'
        _pipe_append_step(ctx, fail_think, search_calls)
        _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think=f'Could not form initial per-spec picks. Union candidate IDs: {all_voucher_candidates}.')
        return

    def _cart_total(prs: list[dict]) -> float:
        return sum((float(x.get('price', 0) or 0.0) for x in prs))
    chosen_total = _cart_total(picks_list)
    swap_notes = 0
    for _ in range(BUDGET_SWAP_LIMIT):
        if chosen_total <= allowed_total + 1e-06:
            break
        swap = _swap_cheaper(products, pools, picks_list, VOUCHER_SCORE_FLOOR, MIN_SWAP_DELTA)
        if swap is None:
            break
        sidx, new_p, new_s = swap
        new_p = dict(new_p)
        new_p['_llm_relevance_score'] = float(new_s)
        old_price = float(picks_list[sidx].get('price', 0) or 0.0)
        new_price = float(new_p.get('price', 0) or 0.0)
        picks_list[sidx] = new_p
        chosen_total += new_price - old_price
        swap_notes += 1
    chosen_prices = [float(p.get('price', 0) or 0.0) for p in picks_list]
    chosen_total_score = sum((float(p.get('_llm_relevance_score', 0) or 0.0) for p in picks_list))
    within_window = bool(threshold <= chosen_total <= allowed_total)
    used_budget_repair = swap_notes > 0
    final_pids = [str(p.get('product_id', '')) for p in picks_list]
    enriched = _enrich_listings([{'product_id': str(p.get('product_id', '')), 'title': p.get('title', ''), 'price': p.get('price')} for p in picks_list])
    marginal_bands_ctx = [pools[i].get('band') if pools[i] else None for i in range(n_specs)]
    done_ctx: dict = {'selected_products': enriched, 'marginal_bands': marginal_bands_ctx, 'llm_score_floor': VOUCHER_SCORE_FLOOR, 'swap_price_delta': MIN_SWAP_DELTA, 'budget_repair_swaps': swap_notes, 'per_product_prices': [round(p, 2) for p in chosen_prices], 'total_before_discount': round(chosen_total, 2), 'total_llm_score': round(chosen_total_score, 2), 'within_voucher_window': within_window, 'threshold': threshold, 'allowed_total': round(allowed_total, 2), 'budget_constraint': voucher, 'used_budget_repair': used_budget_repair, 'probe_candidate_product_ids': probe_product_ids, 'all_candidate_product_ids': all_voucher_candidates}
    if marginal_done_extra:
        done_ctx.update(marginal_done_extra)
    done_think = f'Multi-spec voucher (marginal bands): `_build_margin_pools` fetched in-band listings per spec (pages 1?2 inside each computed floor/ceiling), scored with `_score_listings` and kept rows = {VOUCHER_SCORE_FLOOR}. Seed = best survivor per spec; then up to {BUDGET_SWAP_LIMIT} `_swap_cheaper` moves that shave = {MIN_SWAP_DELTA} per swap while staying = the score floor. Final pre-discount total={chosen_total:.2f} vs window [threshold={threshold:.2f}, allowed_total={allowed_total:.2f}]. Product_ids={final_pids}. (probe+marginal union?{all_voucher_candidates}.)'
    _pipe_append_step(ctx, done_think, [])
    if not within_window:
        end_soft = f'After marginal retrieval and price-down swaps, cart total={chosen_total:.2f} still violates voucher window [threshold={threshold:.2f}, allowed_total={allowed_total:.2f}]. Returning failure with last cart IDs: {final_pids}.'
        _pipe_finalize(ctx, final_pids, 'failure', think=end_soft)
        return
    _pipe_finalize(ctx, final_pids, 'success', think=done_think)

def _pipe_process_shop_query(ctx, params: dict) -> _ShopResult | None:
    specs = params.get('products', [])
    n_specs = len(specs)
    if not specs:
        _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think='No product specs found in shop query.')
        return None
    voucher_budget: tuple[float, float] | None = None
    if n_specs in (2, 3) and bool(params.get('is_shop_voucher')) and params.get('voucher'):
        vn = _norm_voucher(params.get('voucher'))
        ceiling = _voucher_ceiling(vn)
        if ceiling is not None and ceiling > 0:
            voucher_budget = (float(vn.get('threshold', 0) or 0.0), float(ceiling))
    kw_list = [s.get('keywords') or s.get('q', '') for s in specs]
    max_pages = 3
    collect_cap = TWO_SPEC_COLLECT_CAP
    think_analyze = f"Same-shop task planning: I must find {n_specs} distinct products that all come from the same seller. Retrieval strategy: for each spec I call `find_product` (pages 1?{max_pages}, deduped by product_id, capped at {collect_cap} hits per spec). The cap avoids over-fetching on common queries; three pages is usually sufficient because rare products that only appear later are unlikely to have the right shop anyway. Per-spec keywords: {kw_list}. Parsed price bands: {[s.get('price_range') for s in specs]}. Service filters: {[s.get('service') for s in specs]}. After retrieval the candidate sets are LLM-scored and grouped by shop_id to find shops that cover every spec ? the 'full-coverage' detection step."
    all_results: list[list[ListingRow]] = []
    search_calls: list = []
    for spec in specs:
        base_params = _spec_to_query(spec)
        hits: list[ListingRow] = []
        seen: set[str] = set()
        for pg in range(1, max_pages + 1):
            if collect_cap is not None and len(hits) >= collect_cap:
                break
            response = _call_api('find_product', {**base_params, 'page': pg})
            search_calls.append(response)
            rows = response.get('result') or []
            for row in rows:
                if collect_cap is not None and len(hits) >= collect_cap:
                    break
                product_id = str(row.get('product_id', ''))
                if not product_id or product_id in seen:
                    continue
                hits.append(row)
                seen.add(product_id)
        all_results.append(hits)
    cand_ids_by_spec = [_pid_list(rows) for rows in all_results]
    all_cand_ids = _merged_pids(cand_ids_by_spec)
    cand_details_by_spec = [[{'product_id': str(row.get('product_id', '')), 'shop_id': str(row.get('shop_id', '')), 'price': row.get('price')} for row in rows[:10]] for rows in all_results]
    think_pool = f'Retrieval complete. Per-spec pools (showing pid / shop_id / ?price for the first 5): ' + ' | '.join((f"spec[{i}] '{kw_list[i]}': [{', '.join(('pid=' + str(r.get('product_id', '')) + ' shop=' + str(r.get('shop_id', '')) + ' ?' + str(r.get('price', '')) for r in all_results[i][:5]))}]" for i in range(n_specs))) + f". Union of all candidate ids (used for detail-cache prefetch): {all_cand_ids}. Why show this: the union reveals how many distinct products we are about to judge and lets the scorer load details in one batch. Next step: `_score_listings` assigns each candidate a 0?10 relevance score against its spec's query. Any product scoring below {SHOP_SCORE_MIN} is dropped before the shop-coverage index is built ? this filters noise so shops appear in the coverage map only if they carry *relevant* products, not just keyword-matching ones."
    _pipe_append_step(ctx, think_analyze, search_calls)
    _pipe_append_step(ctx, think_pool, [])
    think_scoring_plan = f"Scoring strategy: for each of the {n_specs} spec pools I call `_score_listings`, which posts all candidates as a batch to the LLM scorer (chain: SCORE_CHAIN models). The scorer assigns an integer relevance score 0?10 per product against each spec's query. I then drop every candidate whose score is below the shop-task threshold of {SHOP_SCORE_MIN}/10. After filtering, I build a `shop_coverage` index keyed by `shop_id`: a shop that appears in every spec's filtered pool is called a 'full-coverage' shop. Outcome of that coverage test drives the resolution path: Case A ? exactly 1 full-coverage shop ? pick first high-scoring product per spec from that shop; Case B ? 2+ full-coverage shops ? rank by heuristic score sum, take top {SHOP_TOP_N}, LLM-elect winner shop + one item per spec; Case C ? no full-coverage shop ? bidirectional two-spec / three-spec pipelines or `_anchor_strategy`."
    _pipe_append_step(ctx, think_scoring_plan, [])
    spec_scored: list[list[tuple[ListingRow, float]]] = []
    spec_scored_full: list[list[tuple[ListingRow, float]]] = []
    score_floor = SHOP_SCORE_MIN
    for sidx, (spec, products) in enumerate(zip(specs, all_results)):
        search_q = spec.get('query') or spec.get('keywords') or ctx.query
        pids = [str(p.get('product_id', '')) for p in products if p.get('product_id')]
        details = _load_details(pids)
        scored_pairs = _score_listings(search_q, products, details, only_product_type=bool(spec.get('only_product_type', False)))
        spec_scored_full.append(list(scored_pairs))
        filtered = [(p, sc) for p, sc in scored_pairs if sc >= score_floor]
        spec_scored.append(filtered)
    filtered_results: list[list[ListingRow]] = [[p for p, _ in scored] for scored in spec_scored]
    shop_coverage = _index_by_shop(filtered_results)
    full_shops = [sid for sid, cov in shop_coverage.items() if len(cov) == n_specs]
    scoring_summary = [{'spec_idx': sidx, 'keywords': specs[sidx].get('keywords', ''), 'total_collected': len(all_results[sidx]), 'passed_threshold': len(spec_scored[sidx]), 'candidate_product_ids': cand_ids_by_spec[sidx], 'top_candidates': [{'product_id': str(p.get('product_id', '')), 'shop_id': str(p.get('shop_id', '')), 'title': p.get('title', ''), 'price': p.get('price'), 'score': sc} for p, sc in spec_scored[sidx][:5]]} for sidx in range(n_specs)]
    fb_scoring = f'Scoring phase: for each spec I called `_score_listings` on that spec?s search results (details from `view_product_information`), then dropped candidates below {score_floor}/10. ' + ' | '.join((f"spec[{sidx}] '{specs[sidx].get('keywords', '')}': {len(spec_scored[sidx])}/{len(all_results[sidx])} passed; top=[{', '.join(('pid=' + str(p.get('product_id', '')) + ' shop=' + str(p.get('shop_id', '')) + ' price=' + str(p.get('price')) + ' score=' + str(round(sc, 1)) for p, sc in spec_scored[sidx][:5]))}]" for sidx in range(n_specs))) + f". I then group surviving rows by `shop_id`: a ?full-coverage? shop appears in every spec?s filtered pool. Count of such shops: {len(full_shops)}. Union of all raw candidate ids: {','.join(all_cand_ids)}."
    think_scoring = fb_scoring
    _pipe_append_step(ctx, think_scoring, [])
    if full_shops:
        case_label = f"Case {('A' if len(full_shops) == 1 else 'B')}"
        case_desc = f'Exactly 1 shop covers every spec ? taking it directly as the winner.' if len(full_shops) == 1 else f'{len(full_shops)} shops each carry products for all {n_specs} specs. I will rank them by summed per-spec heuristic score (`_sort_shops`), keep the top {SHOP_TOP_N}, then let the item-judge LLM (`_choose_shop_llm`) pick the best shop and the best product per spec.'
    else:
        case_label = 'Case C'
        case_desc = f"No shop has products in every spec's filtered pool. " + (f'For {n_specs} specs I use the bidirectional two-spec pipeline (top-{TWO_SPEC_TOP_SHOPS} shops per spec, cross-spec pool ={TWO_SPEC_BIDIR_POOL_CAP}, item-judge both legs, pick direction with highest sum-of-scores).' if n_specs == 2 else f'For {n_specs} specs I classify the sub-case (single-top-scorer ? three-spec pipeline; other modes ? `_anchor_strategy`). `_anchor_strategy` first tries partial-coverage fill (n-1 specs in one shop); if that fails it ranks all shops by best LLM score across specs, picks the top 5 (extending on ties at the boundary), fetches 10 candidates per spec per shop, re-scores them, then applies Case-B logic (full-coverage shops ? `_sort_shops` ? `_choose_shop_llm`) to elect the winning shop and one product per spec.' if n_specs == 3 else f'For {n_specs} specs, `_anchor_strategy` first tries partial-coverage fill (n-1 specs in one shop); if that fails it ranks shops by best LLM score, picks the top 5 (+ ties), fetches 10 candidates per spec per shop, re-scores, then applies Case-B logic to elect the winning shop and one product per spec.')
    think_route = f'Resolution routing: {case_label}. {case_desc}'
    _pipe_append_step(ctx, think_route, [])
    if n_specs not in SKIP_SHOP_FULL_COVERAGE_SPEC_COUNTS and len(full_shops) == 1:
        shop_id = full_shops[0]
        used_ids: set[str] = set()
        chosen_pids: list[str] = []
        chosen_products: list[ListingRow] = []
        for sidx in range(n_specs):
            picked: ListingRow | None = None
            for p in shop_coverage[shop_id].get(sidx, []):
                pid = str(p.get('product_id', ''))
                if pid and pid not in used_ids:
                    chosen_pids.append(pid)
                    used_ids.add(pid)
                    picked = p
                    break
            chosen_products.append(picked or {})
        if len(chosen_pids) == n_specs:
            enriched = _enrich_listings([{'product_id': pid} for pid in chosen_pids])
            leaders_a: list = []
            pools_a: list = []
            for sidx in range(n_specs):
                pool = shop_coverage[shop_id].get(sidx, []) or []
                pools_a.append(pool)
                lead_pid = chosen_pids[sidx]
                lead = next((p for p in pool if str(p.get('product_id', '')) == lead_pid), pool[0] if pool else None)
                leaders_a.append(lead)
            _pipe_weigh_multi(ctx, leaders_a, pools_a, specs)
            cc = None
            if len(specs) == len(enriched):
                cc = [_verify_pick(title=info.get('title', ''), price=info.get('price'), parsed_spec=spec or {}) for spec, info in zip(specs, enriched)]
            ctx_found: dict = {'shop_id': shop_id, 'note': 'Only one shop found covering all product specs.', 'selected_products': enriched, 'candidate_product_ids_by_spec': cand_ids_by_spec, 'all_candidate_product_ids': all_cand_ids}
            if cc is not None:
                ctx_found['constraint_checks'] = cc
            think_found = f'Case A: exactly one shop (shop_id={shop_id}) survived the score threshold of {score_floor}/10 for all {n_specs} specs. No ranking needed. For each spec the first entry in shop_coverage[shop_id][spec_idx] (already ordered by LLM score desc) is committed. Duplicate product_ids are skipped so each spec gets a distinct pick. Chosen product_ids={chosen_pids}. `_pipe_weigh_multi` narrates per-spec comparison; `_verify_pick` checks keywords + price per spec. (Retrieval snapshot: by_spec={cand_ids_by_spec}, union={all_cand_ids}.)'
            return _ShopResult(shop_id=str(shop_id), product_ids=list(chosen_pids), think=think_found, leader_products=[dict(p) if p else {} for p in chosen_products], all_candidate_product_ids=all_cand_ids)
    if n_specs not in SKIP_SHOP_FULL_COVERAGE_SPEC_COUNTS and len(full_shops) > 1:
        preranked = _sort_shops(full_shops, shop_coverage, specs, ctx.query)
        top_pool = preranked[:SHOP_TOP_N]
        shop_id, chosen = _choose_shop_llm(top_pool, shop_coverage, specs, ctx.query)
        chosen_ids = [chosen[sidx]['product_id'] for sidx in range(n_specs) if sidx in chosen]
        if shop_id and len(chosen_ids) == n_specs:
            enriched = _enrich_listings([{'product_id': pid} for pid in chosen_ids])
            llm_reasoning = [{'spec_index': sidx, 'product_id': chosen[sidx]['product_id'], 'reason': chosen[sidx]['reason'], 'relevance_score': chosen[sidx]['score']} for sidx in range(n_specs) if sidx in chosen]
            leaders_b: list = []
            pools_b: list = []
            for sidx in range(n_specs):
                pool = (shop_coverage.get(shop_id) or {}).get(sidx, []) or []
                pools_b.append(pool)
                lead_pid = chosen.get(sidx, {}).get('product_id', '')
                lead = next((p for p in pool if str(p.get('product_id', '')) == lead_pid), pool[0] if pool else None)
                if lead is not None:
                    lead = dict(lead)
                    lead['_llm_reason'] = chosen.get(sidx, {}).get('reason', '')
                    lead['_llm_relevance_score'] = chosen.get(sidx, {}).get('score', 0)
                leaders_b.append(lead)
            _pipe_weigh_multi(ctx, leaders_b, pools_b, specs)
            cc = None
            if len(specs) == len(enriched):
                cc = [_verify_pick(title=info.get('title', ''), price=info.get('price'), parsed_spec=spec or {}) for spec, info in zip(specs, enriched)]
            ctx_found = {'shop_id': shop_id, 'note': f'{len(full_shops)} full-coverage shops prefiltered to top {min(len(full_shops), SHOP_TOP_N)} by heuristic score; LLM relevance ranked the winner.', 'selected_products': enriched, 'llm_reasoning': llm_reasoning, 'candidate_product_ids_by_spec': cand_ids_by_spec, 'all_candidate_product_ids': all_cand_ids}
            if cc is not None:
                ctx_found['constraint_checks'] = cc
            think_found = f'Case B: {len(full_shops)} shops each carry relevant products for all {n_specs} specs. Two-stage selection: (1) `_sort_shops` pre-ranks them by the sum of the best per-spec heuristic title-overlap scores, then the top {len(top_pool)} are kept (cap={SHOP_TOP_N}) to limit LLM calls. (2) `_choose_shop_llm` runs a per-spec LLM judge on each top-N shop simultaneously and elects the shop whose candidates have the highest aggregate relevance scores. This two-stage design avoids calling the expensive LLM judge on every shop while still producing a quality-ranked winner when many shops are equally covered. Winning shop_id={shop_id}, product_ids={chosen_ids}. `_pipe_weigh_multi` narrates the per-spec comparison; `_verify_pick` checks keywords + price per spec. (Retrieval snapshot: by_spec={cand_ids_by_spec}, union={all_cand_ids}.)'
            return _ShopResult(shop_id=str(shop_id), product_ids=list(chosen_ids), think=think_found, leader_products=[dict(lp) if lp else {} for lp in leaders_b], all_candidate_product_ids=all_cand_ids)
    if not full_shops:
        spec_scored_case_c = _filter_spec_floor(spec_scored, TWO_SPEC_SCORE_FLOOR)
        if n_specs == 2:
            pair_res = _pipe_two_spec_resolve(ctx, specs, spec_scored_case_c, cand_ids_by_spec, all_cand_ids, voucher_budget=voucher_budget)
            if pair_res is not None:
                return pair_res
        if n_specs == 3:
            c_mode = _classify_case_c(spec_scored_case_c, specs)
            if c_mode == 1:
                triple_res = _pipe_three_spec_resolve(ctx, specs, spec_scored_case_c, cand_ids_by_spec, all_cand_ids, voucher_budget=voucher_budget)
                if triple_res is not None:
                    return triple_res
    resolved_pids, case_c_ctx = _anchor_strategy(specs, spec_scored, shop_coverage, ctx.query, n_specs, is_shop_voucher=bool(params.get('is_shop_voucher', False)), voucher_budget=voucher_budget)
    if not (resolved_pids and len(resolved_pids) == n_specs):
        is_sv = bool(params.get('is_shop_voucher', False)) or 'same shop' in ctx.query.lower()
        empty_cap = EMPTY_SHOP_ANCHOR_CAP_VOUCHER if is_sv else EMPTY_SHOP_ANCHOR_CAP
        empty_pids, empty_ctx = _shop_empty_fallback_anchor_resolution(specs, spec_scored_full, ctx.query, n_specs, max_anchor_shops=empty_cap)
        if empty_pids and len(empty_pids) == n_specs:
            resolved_pids, case_c_ctx = (empty_pids, empty_ctx)
    if resolved_pids and len(resolved_pids) == n_specs:
        resolution_mode = case_c_ctx.get('resolution_mode', 0)
        if resolution_mode == 4:
            fb_case_c = f"Resolution 4: {case_c_ctx.get('partial_shops_evaluated', 0)} shops covering {n_specs - 1}/{n_specs} specs evaluated. Winner shop {case_c_ctx.get('winner_shop_id')} (score={case_c_ctx.get('winner_shop_score')}). Filled missing spec[{case_c_ctx.get('missing_spec_idx')}] ('{case_c_ctx.get('missing_spec_keywords')}') by searching within that shop. Resolved PIDs: {list(resolved_pids)}."
        elif resolution_mode == 5:
            anchor = case_c_ctx.get('anchor', {})
            fb_case_c = f"Resolution 5 (anchor v2 ? re-score top shops): {case_c_ctx.get('tie_note', '')} {case_c_ctx.get('anchor_full_shops_count', 0)} full-coverage shop(s) after re-scoring {len(case_c_ctx.get('anchor_shops_selected', []))} selected shops. Winning shop_id={anchor.get('shop_id')}. Resolved PIDs: {list(resolved_pids)}."
        else:
            anchor = case_c_ctx.get('anchor', {})
            fb_case_c = f"Resolution {resolution_mode} (attempt {case_c_ctx.get('anchor_attempt', 1)}): {case_c_ctx.get('tie_note', '')} Anchor: spec[{anchor.get('spec_idx')}] '{anchor.get('keywords')}' product_id={anchor.get('product_id')} price={anchor.get('price')} shop_id={anchor.get('shop_id')}. Searched remaining specs within that shop. Resolved PIDs: {list(resolved_pids)}."
        think_case_c = fb_case_c
        _pipe_append_step(ctx, think_case_c, [])
        winner_shop = str(case_c_ctx.get('anchor', {}).get('shop_id') or case_c_ctx.get('winner_shop_id', 'resolved'))
        enriched = _enrich_listings([{'product_id': pid} for pid in resolved_pids])
        leaders_c = [dict(info) for info in enriched]
        pools_c = [[] for _ in range(n_specs)]
        _pipe_weigh_multi(ctx, leaders_c, pools_c, specs)
        cc = None
        if len(specs) == len(enriched):
            cc = [_verify_pick(title=info.get('title', ''), price=info.get('price'), parsed_spec=spec or {}) for spec, info in zip(specs, enriched)]
        pick_lines = '; '.join((f"pid {str(info.get('product_id', ''))} '{str(info.get('title', ''))[:60]}' ₱{info.get('price')}" for info in enriched or []))
        ctx_found: dict = {'shop_id': winner_shop, 'selected_products': enriched, 'llm_reasoning': case_c_ctx.get('filled_specs', []), 'candidate_product_ids_by_spec': cand_ids_by_spec, 'all_candidate_product_ids': all_cand_ids}
        if cc is not None:
            ctx_found['constraint_checks'] = cc
        think_found = f'Case C ? no shop had all specs in the filtered pools, so `_anchor_strategy` ran (resolution_mode={resolution_mode}): ' + (f"partial-coverage fill: shop {case_c_ctx.get('winner_shop_id')} covered {n_specs - 1}/{n_specs} specs; missing spec filled by in-shop search." if resolution_mode == 4 else f'anchor v2: ranked top shops by LLM score, fetched 10 candidates per spec per shop, re-scored, then Case-B logic elected shop_id={winner_shop}.' if resolution_mode == 5 else f'anchor loop: picked anchor listing, fixed its shop, searched remaining specs within that shop.') + f' Final shop_id={winner_shop}; per-spec picks: {pick_lines}. All {len(resolved_pids)} product_ids come from that seller.'
        return _ShopResult(shop_id=winner_shop, product_ids=list(resolved_pids), think=think_found, leader_products=list(leaders_c), all_candidate_product_ids=all_cand_ids)
    is_shop_voucher = bool(params.get('is_shop_voucher', False))
    if is_shop_voucher:
        best_per_spec_pids: list[str] = []
        for scored in spec_scored:
            pid = str(scored[0][0].get('product_id', '')) if scored else ''
            if pid:
                best_per_spec_pids.append(pid)
        if len(best_per_spec_pids) == n_specs:
            fallback_think = f'Shop-voucher mode but anchor resolution failed: I still return the best surviving candidate per spec (first entry in each scored pool after threshold), which may come from different shops: product_ids={best_per_spec_pids}. This is a last-resort fallback ? it does not guarantee one seller for all items.'
            return _ShopResult(shop_id='cross-shop-fallback', product_ids=list(best_per_spec_pids), think=fallback_think, leader_products=[], all_candidate_product_ids=all_cand_ids)
    end_fail = f'Shop task failed: after retrieval, `_score_listings` filtering, Case C bidirectional/three-spec attempts, and `_anchor_strategy`, no seller covers every spec. No cross-shop fallback applies here (non?shop-voucher query). Final candidate ids by spec: {cand_ids_by_spec}; union={all_cand_ids}.'
    _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think=end_fail)
    return None

def _pipe_run_shop(ctx, params: dict) -> None:
    result = _pipe_process_shop_query(ctx, params)
    if result:
        _pipe_finalize(ctx, result.product_ids, 'success', think=result.think)

def _pipe_run_shop_voucher(ctx, params: dict) -> None:
    products = params.get('products') or []
    n_specs = len(products)
    if not products:
        _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think='No product specs found in shop voucher query.')
        return
    if n_specs < 2:
        _pipe_run_voucher(ctx, params)
        return
    voucher = _norm_voucher(params.get('voucher'))
    allowed_total = _voucher_ceiling(voucher)
    if not allowed_total or allowed_total <= 0:
        _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think='Could not calculate allowed total from voucher parameters.')
        return
    threshold = float(voucher.get('threshold', 0) or 0.0)
    result = _pipe_process_shop_query(ctx, params)
    if not result:
        return
    leaders = result.leader_products
    cart_total = sum((float(x.get('price', 0) or 0.0) for x in leaders))
    within_ok = bool(threshold <= cart_total <= allowed_total + 1e-06)
    _window_status = 'within_window' if within_ok else 'below_threshold' if cart_total < threshold else 'above_allowed_total'
    think_window_check = f'Shop-voucher window check: the same-shop resolution produced product_ids={result.product_ids} (shop_id={result.shop_id}) with cart_total={cart_total:.2f}. The voucher math requires threshold={threshold:.2f} = cart_total = allowed_total={allowed_total:.2f}. Status: {_window_status}. ' + ('Cart is within the voucher window ? proceeding to finalize.' if within_ok else f'Cart exceeds allowed_total by {cart_total - allowed_total:.2f} ? triggering marginal repair: I will re-probe min/max prices per spec inside shop_id={result.shop_id} (`_probe_edges_shop`), build tight marginal price bands (`_build_margin_pools`, score_floor={VOUCHER_SCORE_FLOOR}), then `_emit_voucher_result` attempts swap-down until the cart fits [threshold, allowed_total].' if cart_total > allowed_total + 1e-06 else f'Cart is below threshold={threshold:.2f} ? finalizing with failure since the voucher minimum spend is not met.')
    _pipe_append_step(ctx, think_window_check, [])
    if cart_total <= allowed_total + 1e-06:
        if within_ok:
            _pipe_finalize(ctx, result.product_ids, 'success', think=result.think)
        else:
            below_thr = cart_total < threshold
            end_soft = f'Shop voucher check failed after same-shop picks: cart total {cart_total:.2f} must sit in [threshold={threshold:.2f}, allowed_pre_discount_total={allowed_total:.2f}] (allowed_total is the voucher math ceiling from budget/discount/cap). Status failure while keeping cart product_ids={result.product_ids}, shop_id={result.shop_id}.'
            _pipe_finalize(ctx, result.product_ids, 'failure', think=end_soft)
        return
    over_note = f'Shop voucher marginal repair: the same-shop cart total {cart_total:.2f} exceeds allowed_total {allowed_total:.2f}, so I probe min/max per-spec prices inside shop_id={result.shop_id} (`_probe_edges_shop` with priceasc/pricedesc), build tight marginal price bands (`_build_margin_pools`), re-score with `_score_listings` floor {VOUCHER_SCORE_FLOOR}, then `_emit_voucher_result` picks/swap-down until the cart fits or I fail the window check. Starting picks: {result.product_ids}.'
    _pipe_append_step(ctx, over_note, [])
    minima, maxima, probe_calls = _probe_edges_shop(products, allowed_total, result.shop_id)
    probe_pids = _gather_result_ids(probe_calls)
    pools, pool_calls = _build_margin_pools(products, n_specs, threshold, allowed_total, minima, maxima, ctx.query, VOUCHER_SCORE_FLOOR, shop_id=result.shop_id)
    search_calls = probe_calls + pool_calls
    _pipe_emit_voucher_result(ctx, products, n_specs, threshold=threshold, allowed_total=allowed_total, voucher=voucher, pools=pools, search_calls=search_calls, probe_product_ids=probe_pids, extra_candidate_id_lists=[result.all_candidate_product_ids], marginal_done_extra={'shop_id': result.shop_id, 'shop_voucher_marginal_repair': True, 'same_shop_phase_cart_total': round(cart_total, 2)})

def _pipe_run_voucher(ctx, params: dict) -> None:
    products = params.get('products', [])
    n_specs = len(products)
    if not products:
        _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think='No product specs found in voucher query.')
        return
    voucher = _norm_voucher(params.get('voucher'))
    allowed_total = _voucher_ceiling(voucher)
    if not allowed_total or allowed_total <= 0:
        _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think='Could not calculate allowed total from voucher parameters.')
        return
    if n_specs == 2:
        _pipe_run_two_spec_voucher(ctx, params, products, voucher, float(allowed_total))
        return
    kw_list = [spec.get('keywords', '') for spec in products]
    think_analyze = f"Voucher plan ({n_specs} specs, greedy-fill strategy). Allowed pre-discount cart total = {allowed_total:.2f} (derived from: budget={voucher.get('budget')} minus the discount the voucher gives when threshold={float(voucher.get('threshold', 0) or 0):.2f} is met). This step issues one pricedesc scan per spec (price band 1-{allowed_total:.0f}) to observe the highest in-stock price. That max-price probe tells us which spec is most expensive so we can fill the cart in descending-expense order: filling the most expensive slot first minimises the risk of dead-ends where the remaining budget is too small to satisfy a pricey spec after a cheap one has already consumed headroom. Each greedy slot computes a live ceiling = allowed_total - spent_so_far - reserved_lo_from_other_specs, intersects that with the spec's own price_range, fetches pages 1-2, and uses _elect_best to pick the winner."
    scan_calls: list = []
    max_prices: list[float] = []
    for spec in products:
        sp = _spec_to_query(spec, include_price=False)
        sp['price'] = f'1-{allowed_total:.0f}'
        sp['sort'] = 'pricedesc'
        r = _call_api('find_product', sp)
        scan_calls.append(r)
        found = r.get('result') or []
        max_prices.append(float(found[0].get('price', 0)) if found else 0.0)
    _pipe_append_step(ctx, think_analyze, scan_calls)
    remaining: list[int] = sorted(range(n_specs), key=lambda i: max_prices[i], reverse=True)
    priority_order_meta = [{'spec_idx': i, 'keywords': products[i].get('keywords', ''), 'pricedesc_max_observed': max_prices[i], 'fill_priority': rank + 1} for rank, i in enumerate(remaining)]
    think_priority = f'Greedy fill order: I observed the highest-priced listing per spec from the `pricedesc` scan and sort specs by that observed maximum descending ? the spec with the most expensive items is filled first (it consumes the most budget, so constraining it early avoids dead ends). Fill priority: ' + ', '.join((f"spec[{i}] '{products[i].get('keywords', '')}' (max?{max_prices[i]:.0f})" for i in remaining)) + f". For each position the algorithm computes a live price band: ceiling = allowed_total - spent_so_far - sum(parsed_lo for remaining unpicked specs); floor = allowed_total/{n_specs} for the first pick, else 1. The intersection with the spec's own `price_range` gives the final search window. Within that window I call `find_product` (pages 1?2), then `_elect_best` to pick the winner."
    _pipe_append_step(ctx, think_priority, [])
    picked_products: list[ListingRow] = []
    picked_orig_idx: list[int] = []
    picked_pools: list[list] = []
    budget_calls: list = []
    while remaining:
        position = len(picked_products)
        is_anchor = position == 0
        spent = sum((float(p.get('price', 0)) for p in picked_products))
        found_valid = False
        for ci in list(remaining):
            others_ci = [j for j in remaining if j != ci]
            reserved = sum((lo or 0.0 for oi in others_ci for lo, _ in [_parse_price_opt(products[oi].get('price_range'))]))
            ceiling = allowed_total - spent - reserved
            floor = (allowed_total / n_specs if is_anchor else 1.0) if n_specs > 1 else 1.0
            orig_lo, orig_hi = _parse_price_opt(products[ci].get('price_range'))
            if n_specs == 1:
                threshold = float(voucher.get('threshold', 0) or 0)
                final_lo = max(orig_lo if orig_lo is not None else 0.0, threshold)
                final_hi = min(orig_hi if orig_hi is not None else float('inf'), allowed_total)
            else:
                final_lo = max(orig_lo if orig_lo is not None else 0.0, floor)
                final_hi = min(orig_hi if orig_hi is not None else float('inf'), ceiling)
            if final_lo > final_hi:
                continue
            sp = _spec_to_query(products[ci], include_price=False)
            sp['price'] = f'{final_lo:.0f}-{final_hi:.0f}'
            cands_in_range: list[ListingRow] = []
            seen_pids: set[str] = set()
            for pg in range(1, 3):
                r = _call_api('find_product', {**sp, 'page': pg})
                budget_calls.append(r)
                for row in r.get('result') or []:
                    rpid = str(row.get('product_id', ''))
                    if rpid and rpid not in seen_pids:
                        cands_in_range.append(row)
                        seen_pids.add(rpid)
            if not cands_in_range:
                continue
            search_q = products[ci].get('query') or products[ci].get('keywords') or ctx.query
            cpids = [str(row.get('product_id', '')) for row in cands_in_range if row.get('product_id')]
            details = _load_details(cpids)
            chosen = _elect_best(search_q, cands_in_range, details, only_product_type=bool(products[ci].get('only_product_type', False)), model=BACKUP_LLM_)
            if chosen is None:
                chosen = cands_in_range[0]
            picked_products.append(chosen)
            picked_orig_idx.append(ci)
            picked_pools.append(cands_in_range)
            remaining = [j for j in remaining if j != ci]
            found_valid = True
            _pick_price = float(chosen.get('price', 0) or 0.0)
            _new_spent = sum((float(p.get('price', 0) or 0.0) for p in picked_products))
            _remaining_slots = len(remaining)
            _cand_summary = [{'product_id': str(row.get('product_id', '')), 'shop_id': str(row.get('shop_id', '')), 'price': row.get('price')} for row in cands_in_range[:10]]
            think_pick = f"Cart slot {position + 1}/{n_specs} filled: spec[{ci}] '{products[ci].get('keywords', '')}' ? product_id={chosen.get('product_id', '')} shop_id={chosen.get('shop_id', '')} price={_pick_price:.2f} (band {final_lo:.0f}?{final_hi:.0f}, {len(cands_in_range)} candidates: [{', '.join(('pid=' + str(r.get('product_id', '')) + ' shop=' + str(r.get('shop_id', '')) + ' ?' + str(r.get('price', '')) for r in cands_in_range[:5]))}]; `_elect_best` picked the winner). Spent so far: {_new_spent:.2f} / {allowed_total:.2f}. " + (f'Remaining specs to fill: {_remaining_slots}.' if _remaining_slots > 0 else 'All specs filled ? proceeding to final cart check.')
            _pipe_append_step(ctx, think_pick, [])
            break
        if not found_valid:
            already_picked_pids = [str(p.get('product_id', '')) for p in picked_products if p.get('product_id')]
            think_fail = f'Greedy voucher fill stalled at cart slot {position}/{n_specs}: no spec produced listings inside the remaining price band after reserving minima from unpicked specs (spent_so_far={spent:.2f}, hard ceiling allowed_total={allowed_total:.2f}). ' + (f'PIDs already chosen: {already_picked_pids}. ' if already_picked_pids else '') + 'Aborting with failure.'
            _pipe_append_step(ctx, think_fail, budget_calls)
            _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think=f'Could not find a product for cart slot {position}/{n_specs} within the voucher budget constraints (allowed_total={allowed_total:.2f}). ' + (f'Already picked PIDs: {already_picked_pids}.' if already_picked_pids else ''))
            return
    pid_by_spec = {orig_idx: str(picked_products[pos].get('product_id', '')) for pos, orig_idx in enumerate(picked_orig_idx)}
    price_by_spec = {orig_idx: float(picked_products[pos].get('price', 0)) for pos, orig_idx in enumerate(picked_orig_idx)}
    ordered_pids = [pid_by_spec[sidx] for sidx in range(n_specs)]
    total_price = sum(price_by_spec.values())
    enriched = _enrich_listings([{'product_id': pid_by_spec[sidx], 'title': picked_products[picked_orig_idx.index(sidx)].get('title', ''), 'price': picked_products[picked_orig_idx.index(sidx)].get('price')} for sidx in range(n_specs)])
    leaders_v: list = [None] * n_specs
    pools_v: list = [[]] * n_specs
    for pos, orig_idx in enumerate(picked_orig_idx):
        leaders_v[orig_idx] = picked_products[pos]
        pools_v[orig_idx] = picked_pools[pos] if pos < len(picked_pools) else []
    _pipe_weigh_multi(ctx, leaders_v, pools_v, products)
    cc = None
    if len(products) == len(enriched):
        cc = [_verify_pick(title=info.get('title', ''), price=info.get('price'), parsed_spec=spec or {}) for spec, info in zip(products, enriched)]
    fb_done = f"Voucher greedy-fill complete ({n_specs} specs). Pre-discount cart total = {total_price:.2f} vs allowed_total = {allowed_total:.2f} (user budget = {voucher.get('budget')}). The greedy algorithm filled each slot in descending max-price order to minimise dead-ends, constraining each slot to a live price band that reserved minimum spend for remaining specs. `_elect_best` was called per slot to pick the highest-relevance product within the band. `_pipe_weigh_multi` narrated the per-spec candidate comparison; `_verify_pick` checked each winner. Final ordered product_ids={ordered_pids}."
    ctx_done: dict = {'selected_products': enriched, 'total_before_discount': round(total_price, 2), 'budget_constraint': voucher}
    if cc is not None:
        ctx_done['constraint_checks'] = cc
    think_done = fb_done
    _pipe_append_step(ctx, think_done, budget_calls)
    _pipe_finalize(ctx, ordered_pids, 'success')

def _two_spec_clip_marginal_window(lo: float, hi: float, price_range: str | None) -> tuple[float, float] | None:
    band = _clip_band(min(lo, hi), max(lo, hi), price_range)
    if band is None:
        return None
    blo, bhi = band
    if bhi <= 0:
        return None
    return (max(0.0, blo), max(0.0, bhi))

def _two_spec_collect_ranked_pool(ctx, sidx: int, products: list[dict], minima: list, maxima: list, allowed_total: float, threshold: float, search_calls: list) -> tuple[list[dict], dict[str, Any]]:
    other = 1 - sidx
    spec = products[sidx]
    band_a = _two_spec_clip_marginal_window(threshold / 2.0, allowed_total - float(minima[other] or 0.0), spec.get('price_range'))
    band_b = _two_spec_clip_marginal_window(max(threshold - float(maxima[other] or 0.0), 0.0), allowed_total / 2.0, spec.get('price_range'))
    seen: set[str] = set()
    merged: list[dict] = []
    bands_meta: list[dict[str, float]] = []
    for band in (band_a, band_b):
        if band is None:
            continue
        lo, hi = band
        hits, calls = _fetch_band_hits(spec, lo, hi, limit=20)
        search_calls.extend(calls)
        bands_meta.append({'floor': round(lo, 2), 'ceiling': round(hi, 2)})
        for row in hits:
            pid = str(row.get('product_id', '')).strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            merged.append(dict(row))
    q = spec.get('query') or spec.get('keywords') or ctx.query
    pids = [str(x.get('product_id', '')) for x in merged if x.get('product_id')]
    details = _load_details(pids)
    scored = _score_listings(q, merged, details, only_product_type=bool(spec.get('only_product_type', False)))
    ranked_rows: list[dict[str, Any]] = []
    for ridx, (prod, sc) in enumerate(scored, start=1):
        row = dict(prod)
        row['_llm_relevance_score'] = float(sc)
        row['_rank'] = ridx
        ranked_rows.append(row)
    ranked_rows.sort(key=lambda r: (-float(r.get('_llm_relevance_score', 0.0)), int(r.get('_rank', 10000))))
    for ridx, row in enumerate(ranked_rows, start=1):
        row['_rank'] = ridx
    top_pid = str(ranked_rows[0].get('product_id', '')).strip() if ranked_rows else ''
    top_sc = float(ranked_rows[0].get('_llm_relevance_score', 0.0)) if ranked_rows else 0.0
    return (ranked_rows, {'bands': bands_meta, 'count': len(ranked_rows)})

def _two_spec_anchor_partner_within_budget(anchor: dict, partner_rows: list[dict], allowed_total: float) -> dict[str, Any] | None:
    ap = float(anchor.get('price', 0) or 0.0)
    for p in partner_rows:
        if ap + float(p.get('price', 0) or 0.0) <= allowed_total + 1e-06:
            return p
    return None

def _two_spec_seed_pair_candidates(ranked_a: list[dict], ranked_b: list[dict], allowed_total: float) -> list[dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    for a in ranked_a[:5]:
        b = _two_spec_anchor_partner_within_budget(a, ranked_b, allowed_total)
        if b is not None:
            pair_rows.append({'a': a, 'b': b})
    for b in ranked_b[:5]:
        a = _two_spec_anchor_partner_within_budget(b, ranked_a, allowed_total)
        if a is not None:
            pair_rows.append({'a': a, 'b': b})
    return pair_rows

def _two_spec_collate_unique_pairs(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for pr in pair_rows:
        apid = str(pr['a'].get('product_id') or '').strip()
        bpid = str(pr['b'].get('product_id') or '').strip()
        if not apid or not bpid:
            continue
        dedup[apid, bpid] = pr
    return list(dedup.values())

def _two_spec_order_pair_by_richness(pr: dict[str, Any], prio_spec: int, other_spec: int) -> tuple[float, float, int, float, int]:
    a, b = (pr['a'], pr['b'])
    sum_sc = float(a.get('_llm_relevance_score', 0.0)) + float(b.get('_llm_relevance_score', 0.0))
    score_prio = float((a if prio_spec == 0 else b).get('_llm_relevance_score', 0.0))
    rank_prio = int((a if prio_spec == 0 else b).get('_rank', 10000))
    score_other = float((a if other_spec == 0 else b).get('_llm_relevance_score', 0.0))
    rank_other = int((a if other_spec == 0 else b).get('_rank', 10000))
    return (sum_sc, score_prio, -rank_prio, score_other, -rank_other)

def _pipe_run_two_spec_voucher(ctx, params: dict, products: list[dict], voucher: dict, allowed_total: float) -> None:
    if len(products) != 2:
        _pipe_finalize(ctx, [SENTINEL_PID], 'failure')
        return
    threshold = float(voucher.get('threshold', 0) or 0.0)
    think_two_spec_plan = f'Two-spec voucher pipeline start: threshold={threshold:.2f}, allowed_pre_discount_total={allowed_total:.2f}. Strategy ? I first probe min/max prices for each spec via `_probe_edges` (`priceasc` + `pricedesc` queries within the 1?{allowed_total:.0f} window). Those probe prices anchor two marginal price bands per spec: Band A (high-end): [{threshold}/2, allowed_total - other_min] n spec.price_range; Band B (low-end): [max(threshold - other_max, 0), allowed_total/2] n spec.price_range. Both bands are fetched (pages 1?2, =20 items each), merged and deduplicated by product_id, then batch-scored with `_score_listings`. The two ranked lists (spec[0] and spec[1]) are combined into feasible pairs: first I check if top_A + top_B = allowed_total; if not, I enumerate cross-pairs from the top-5 of each list and pick the pair with the highest sum-of-LLM-scores that satisfies threshold = price_sum = allowed_total.'
    _pipe_append_step(ctx, think_two_spec_plan, [])
    minima, maxima, probe_calls = _probe_edges(products, allowed_total)
    search_calls: list = list(probe_calls)
    think_probe = f"Price probes complete. spec[0] '{products[0].get('keywords', '')}': min?{(float(minima[0]) if minima else 0):.2f}, max?{(float(maxima[0]) if maxima else 0):.2f}. spec[1] '{products[1].get('keywords', '')}': min?{(float(minima[1]) if len(minima) > 1 else 0):.2f}, max?{(float(maxima[1]) if len(maxima) > 1 else 0):.2f}. Now computing Band A and Band B for each spec and fetching candidates. spec[0] Band A ceiling = {allowed_total:.2f} - {(float(minima[1]) if len(minima) > 1 else 0):.2f} = {(allowed_total - float(minima[1]) if len(minima) > 1 else allowed_total):.2f}; spec[0] Band B floor = max({threshold:.2f} - {(float(maxima[1]) if len(maxima) > 1 else 0):.2f}, 0) = {(max(threshold - float(maxima[1]), 0) if len(maxima) > 1 else max(threshold, 0)):.2f}."
    _pipe_append_step(ctx, think_probe, probe_calls)
    ranked_a, meta_a = _two_spec_collect_ranked_pool(ctx, 0, products, minima, maxima, allowed_total, threshold, search_calls)
    ranked_b, meta_b = _two_spec_collect_ranked_pool(ctx, 1, products, minima, maxima, allowed_total, threshold, search_calls)
    top_a_preview = [{'product_id': str(r.get('product_id', '')), 'price': r.get('price'), 'score': r.get('_llm_relevance_score')} for r in ranked_a[:3]]
    top_b_preview = [{'product_id': str(r.get('product_id', '')), 'price': r.get('price'), 'score': r.get('_llm_relevance_score')} for r in ranked_b[:3]]
    think_candidates = f"Candidate pools scored. spec[0] '{products[0].get('keywords', '')}': {meta_a.get('count', 0)} candidates from bands {meta_a.get('bands', [])}; top-3 by LLM score: {top_a_preview}. spec[1] '{products[1].get('keywords', '')}': {meta_b.get('count', 0)} candidates from bands {meta_b.get('bands', [])}; top-3 by LLM score: {top_b_preview}. Now selecting the best feasible pair: first try top_A + top_B = {allowed_total:.2f}; if that fails, enumerate pairs from top-5 of each list, score each pair by (sum_llm_scores, prio_spec_score, prio_rank, other_spec_score, other_rank), take the highest-ranked pair satisfying threshold={threshold:.2f} = sum_price = {allowed_total:.2f}."
    _pipe_append_step(ctx, think_candidates, [])
    if not ranked_a or not ranked_b:
        top_a_pid = str(ranked_a[0].get('product_id', '')) if ranked_a else None
        top_b_pid = str(ranked_b[0].get('product_id', '')) if ranked_b else None
        fail = f"Two-spec voucher pipeline: could not populate both marginal candidate lists. `_probe_edges` + dual-band `_fetch_band_hits` per spec returned empty for one side (allowed_total={allowed_total:.2f}, threshold={threshold:.2f}). Diagnostics ? spec[0] top PID={top_a_pid or 'none'}; spec[1] top PID={top_b_pid or 'none'}."
        _pipe_append_step(ctx, fail, search_calls)
        _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think=fail)
        return
    prio_spec = min([0, 1], key=lambda i: (_richness_rank(products[i]), i))
    other_spec = 1 - prio_spec
    top_a, top_b = (ranked_a[0], ranked_b[0])
    pa = float(top_a.get('price', 0) or 0.0)
    pb = float(top_b.get('price', 0) or 0.0)
    if pa + pb <= allowed_total + 1e-06:
        chosen = [top_a, top_b]
    else:
        pair_rows = _two_spec_seed_pair_candidates(ranked_a, ranked_b, allowed_total)
        pairs = _two_spec_collate_unique_pairs(pair_rows)
        if not pairs:
            top_a_pid = str(ranked_a[0].get('product_id', '')) if ranked_a else 'none'
            top_b_pid = str(ranked_b[0].get('product_id', '')) if ranked_b else 'none'
            top_a_price = float(ranked_a[0].get('price', 0) or 0.0) if ranked_a else 0.0
            top_b_price = float(ranked_b[0].get('price', 0) or 0.0) if ranked_b else 0.0
            fail = f'Two-spec voucher: after `_score_listings` ranking, no pair of SKUs had price_sum = allowed_total={allowed_total:.2f} while staying above threshold={threshold:.2f} (pair enumeration from top-ranked rows failed). Top singletons ? spec[0] PID={top_a_pid} @ {top_a_price:.2f}; spec[1] PID={top_b_pid} @ {top_b_price:.2f}.'
            _pipe_append_step(ctx, fail, search_calls)
            _pipe_finalize(ctx, [SENTINEL_PID], 'failure', think=fail)
            return
        pairs.sort(key=lambda pr: _two_spec_order_pair_by_richness(pr, prio_spec, other_spec), reverse=True)
        chosen = [pairs[0]['a'], pairs[0]['b']]
    _pair_method = 'top-pair direct' if float(chosen[0].get('price', 0) or 0.0) + float(chosen[1].get('price', 0) or 0.0) <= float(ranked_a[0].get('price', 0) or 0.0) + float(ranked_b[0].get('price', 0) or 0.0) + 0.001 and str(chosen[0].get('product_id', '')) == str(ranked_a[0].get('product_id', '')) and (str(chosen[1].get('product_id', '')) == str(ranked_b[0].get('product_id', ''))) else 'partner-search fallback'
    _pair_sum = float(chosen[0].get('price', 0) or 0.0) + float(chosen[1].get('price', 0) or 0.0)
    _sc_a = float(chosen[0].get('_llm_relevance_score', 0.0))
    _sc_b = float(chosen[1].get('_llm_relevance_score', 0.0))
    think_pair_sel = f'Pair selection ({_pair_method}). The two ranked candidate lists (spec[0] and spec[1]) were combined into feasible pairs satisfying threshold={threshold:.2f} ? price_sum ? allowed_total={allowed_total:.2f}. ' + (f'The top-ranked items from each list (top_A + top_B) had sum_price ? allowed_total so they were accepted directly without needing partner search. ' if _pair_method == 'top-pair direct' else f'The top pair exceeded allowed_total, so a partner-search enumerated the top-5 of each ranked list and found the cheapest feasible partner for each anchor. Candidate pairs were deduplicated and sorted by: (1) sum of LLM relevance scores, (2) priority-spec score (spec[{prio_spec}] has higher richness rank), (3) priority-spec rank, (4) other-spec score, (5) other-spec rank. The highest-ranked feasible pair was selected. ') + f"Selected: spec[0] pid={str(chosen[0].get('product_id', ''))} " + f"@ ?{chosen[0].get('price')} (LLM score={_sc_a:.1f}), " + f"spec[1] pid={str(chosen[1].get('product_id', ''))} " + f"@ ?{chosen[1].get('price')} (LLM score={_sc_b:.1f}). " + f'Combined pre-discount total = {_pair_sum:.2f}.'
    _pipe_append_step(ctx, think_pair_sel, [])
    ordered_pids = [str(chosen[0].get('product_id', '')).strip(), str(chosen[1].get('product_id', '')).strip()]
    if not ordered_pids[0] or not ordered_pids[1]:
        _pipe_finalize(ctx, [SENTINEL_PID], 'failure')
        return
    total_price = float(chosen[0].get('price', 0) or 0.0) + float(chosen[1].get('price', 0) or 0.0)
    done = f"Two-spec voucher complete. Pipeline: (1) `_probe_edges` sampled priceasc + pricedesc to anchor min/max prices per spec. (2) Two price bands per spec (Band A = high-end headroom, Band B = low-end headroom) were fetched (pages 1-2, ?20 items each), merged and deduplicated. (3) `_score_listings` assigned LLM relevance scores 0-10 to each candidate. (4) Pairs were formed and ranked; the winning pair was selected via '{_pair_method}'. Chosen product_ids={ordered_pids}. Pre-discount total = {total_price:.2f} (threshold={threshold:.2f}, allowed_total={allowed_total:.2f}, user budget={voucher.get('budget')})."
    _pipe_append_step(ctx, done, search_calls)
    _pipe_finalize(ctx, ordered_pids, 'success')
import json
import re
import time
import threading
from itertools import product as itertools_product
from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Sequence
from os import getenv
from typing import Any, NamedTuple
from urllib.parse import quote_plus
from src.agent.proxy_client import ProxyClient
from src.agent.agent_interface import Tool, create_dialogue_step, execute_tool_call
import dataclasses
from dataclasses import dataclass
from typing import Any
import math
import unicodedata
from src.agent import proxy_client as _proxy_client_mod
from src.agent.agent_interface import Tool, create_dialogue_step, execute_tool_call, generate_tool_call_id
from itertools import product as cartesianProduct
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from src.agent import proxy_client as proxyClientModule
P50_CatalogListingDict = dict[str, Any]
P50_ParsedProductSpecDict = dict[str, Any]
P50__CHUTES_INFERENCE_MODELS: dict[str, Any] = {'PRODUCT_PARSE_MODEL': 'deepseek-ai/DeepSeek-V3.1-TEE', 'VOUCHER_PARSE_MODEL': 'deepseek-ai/DeepSeek-V3.1-TEE', 'SHOP_PARSE_MODEL': 'deepseek-ai/DeepSeek-V3.1-TEE', 'FINAL_FALLBACK_MODEL': 'google/gemma-4-31B-turbo-TEE', 'PRODUCT_RANK_MODEL': 'deepseek-ai/DeepSeek-V3-0324-TEE', 'BACKUP_LLM_MODEL': 'deepseek-ai/DeepSeek-V3.1-TEE', 'PICK_CHAIN': ['google/gemma-4-31B-turbo-TEE', 'deepseek-ai/DeepSeek-V3.1-TEE', 'deepseek-ai/DeepSeek-V3-0324-TEE'], 'SCORE_CHAIN': ['deepseek-ai/DeepSeek-V3.1-TEE', 'deepseek-ai/DeepSeek-V3-0324-TEE', 'google/gemma-4-31B-turbo-TEE']}
P50__OPENROUTER_INFERENCE_MODELS: dict[str, Any] = {'PRODUCT_PARSE_MODEL': 'deepseek/deepseek-v3.2', 'VOUCHER_PARSE_MODEL': 'deepseek/deepseek-v3.2', 'SHOP_PARSE_MODEL': 'deepseek/deepseek-v3.2', 'FINAL_FALLBACK_MODEL': 'google/gemma-4-31b-it', 'PRODUCT_RANK_MODEL': 'deepseek/deepseek-v3.2', 'BACKUP_LLM_MODEL': 'deepseek/deepseek-chat-v3.1', 'PICK_CHAIN': ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'deepseek/deepseek-chat-v3.1'], 'SCORE_CHAIN': ['deepseek/deepseek-chat-v3.1', 'deepseek/deepseek-v3.2', 'google/gemma-4-31b-it']}
P50__INFERENCE_MODELS_BY_PROVIDER: dict[str, dict[str, Any]] = {'chutes': P50__CHUTES_INFERENCE_MODELS, 'openrouter': P50__OPENROUTER_INFERENCE_MODELS}

class P50_InferenceProfileMatrix:

    @staticmethod
    def active_vendor_slug() -> str:
        return getenv('INFERENCE_PROVIDER', 'openrouter')

    @staticmethod
    def resolve_model_handle(registry_key: str) -> str:
        slug = P50_InferenceProfileMatrix.active_vendor_slug()
        registry = P50__INFERENCE_MODELS_BY_PROVIDER.get(slug) or P50__OPENROUTER_INFERENCE_MODELS
        return registry[registry_key]

    @staticmethod
    def pick_model_chain() -> list[str]:
        slug = P50_InferenceProfileMatrix.active_vendor_slug()
        registry = P50__INFERENCE_MODELS_BY_PROVIDER.get(slug) or P50__OPENROUTER_INFERENCE_MODELS
        return list(registry['PICK_CHAIN'])

    @staticmethod
    def score_model_chain() -> list[str]:
        slug = P50_InferenceProfileMatrix.active_vendor_slug()
        registry = P50__INFERENCE_MODELS_BY_PROVIDER.get(slug) or P50__OPENROUTER_INFERENCE_MODELS
        return list(registry['SCORE_CHAIN'])

def P50_resolve_inference_model_handle(registry_key: str) -> str:
    return P50_InferenceProfileMatrix.resolve_model_handle(registry_key)
P50_INFERENCE_MODEL_REGISTRY: dict[str, Any] = {'PRODUCT_PARSE_MODEL': 'deepseek/deepseek-v3.2', 'VOUCHER_PARSE_MODEL': 'deepseek/deepseek-v3.2', 'SHOP_PARSE_MODEL': 'deepseek/deepseek-v3.2', 'FINAL_FALLBACK_MODEL': 'google/gemma-4-31b-it', 'PRODUCT_RANK_MODEL': 'z-ai/glm-5.1', 'BACKUP_LLM_MODEL': 'deepseek/deepseek-v3.2', 'PICK_CHAIN': ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'z-ai/glm-5.1'], 'SCORE_CHAIN': ['google/gemma-4-31b-it', 'deepseek/deepseek-v3.2', 'z-ai/glm-5.1']}
P50_MULTI_PRODUCT_CLAUSE_SPLIT_PATTERN = re.compile('(?:,?\\s*and\\s+also\\s+|,?\\s*also,?\\s*|Second(?:ly)?,\\s*|Third(?:ly)?,\\s*|First,\\s*|\\(\\d+\\)\\s*|\\d+\\.\\s*|Additionally,\\s*|Furthermore,\\s*|Moreover,\\s*|In\\s+addition,?\\s*|Plus,\\s*|On\\s+top\\s+of\\s+that,?\\s*|[.]\\s*Next,\\s*|[.]\\s*Lastly,\\s*|[.]\\s*Finally,\\s*|[.]\\s*Last,\\s*|\\bThen\\s*,?\\s*I\\s+(?:need|want|also)\\b|\\bI\\s+also\\s+(?:want|need)\\b)', re.IGNORECASE)
P50_BUDGET_OR_VOUCHER_MENTION_PATTERN = re.compile('(?:My budget|budget is|I have a voucher)', re.IGNORECASE)
P50_RELEVANCE_SCORING_STOPWORDS: frozenset[str] = frozenset({'the', 'a', 'an', 'for', 'with', 'from', 'that', 'this', 'i', 'me', 'my', 'looking', 'show', 'find', 'want', 'need', 'get', 'finish', 'buy', 'also', 'and', 'in', 'is', 'it', 'am', 'im', 'priced', 'pesos', 'php', 'price', 'between', 'than', 'above', 'below', 'more', 'less', 'over', 'under', 'of', 'to', 'or', 'on', 'at', 'by', 'its', 'be', 'can', 'has', 'have', 'will', 'would', 'should', 'item', 'items', 'both', 'these', 'offering', 'sells', 'shop', 'budget', 'voucher', 'discount', 'first', 'second', 'third', 'brand', 'made', 'using', 'available', 'support', 'supports', 'compatible', 'please', 'age', 'use', 'replacement'})
P50_SEARCH_KEYWORD_SYNONYM_MAP = {'ballpoint': 'ball'}
P50_QUERY_TOKENIZATION_STOPWORDS = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was', 'can', 'has', 'have', 'been', 'will', 'find', 'finish', 'looking', 'show', 'want', 'need', 'get', 'buy', 'product', 'products', 'search', 'same', 'shop', 'within', 'budget', 'voucher', 'discount', 'price', 'priced', 'pesos', 'php', 'between', 'than', 'greater', 'less', 'more', 'under', 'over', 'about', 'also', 'both', 'these', 'them', 'each', 'all', 'one', 'two', 'three', 'four', 'use', 'five', 'offering', 'sells', 'using', 'in', 'is', 'it', 'its', 'or', 'at', 'on', 'by', 'be', 'do', 'an', 'my', 'me', 'im', 'items', 'item', 'just', 'first', 'second', 'supports', 'replacement', 'support', 'compatible', 'available', 'made', 'please', 'like', 'of', 'above', 'deals', 'options', 'option', 'delivery', 'shipping', 'offers', 'lazmall', 'lazflash', 'official', 'cash', 'payment', 'pay', 'cost', 'costs', 'via', 'themed', 'such', 'those', 'store', 'stores', 'focus', 'category', 'specifically', 'guaranteed', 'authenticity', 'returns', 'quick', 'perks', 'should', 'help', 'purchase', 'type', 'to', 'named', 'called', 'family', 'belongs', 'comes', 'another', 'lastly', 'benefits', 'you', 'weighing', 'capacity', 'size', 'sized', 'eu', 'fits'}
P50_DIALOGUE_SESSION_TIMEOUT_SECONDS = 250.0
P50_NO_MATCH_PRODUCT_ID_SENTINEL = '0'
P50_FALLBACK_CATALOG_SEARCH_QUERY = 'product'
P50_CATALOG_FIND_PRODUCT_API_PATH = '/search/find_product'
P50_CATALOG_HTTP_MAX_REQUESTS_PER_MINUTE = 90
P50_RATE_LIMIT_WINDOW_SECONDS = 60.0
P50_MIN_SECONDS_BETWEEN_CATALOG_CALLS = 0.7
P50_MIN_SECONDS_BETWEEN_SANDBOX_TOOL_CALLS = 0.5
P50_SANDBOX_TOOL_MAX_RETRY_ATTEMPTS = 3
P50_SANDBOX_TOOL_RETRY_BACKOFF_BASE_SECONDS = 1.0
P50_LLM_COMPLETION_MAX_ATTEMPTS_PER_MODEL = 1
P50_DIALOGUE_TOOL_RESULT_LISTING_CAP = 10
P50_LLM_JUDGE_FAST_ACCEPT_SCORE_THRESHOLD = 8.0
P50_LLM_JUDGE_LOW_CONFIDENCE_SCORE_THRESHOLD = 6.0
P50_SINGLE_PRODUCT_PROBE_MAX_ELAPSED_SECONDS = 220.0
P50_SINGLE_PRODUCT_FINALIZE_MAX_ELAPSED_SECONDS = 250.0
P50_CANDIDATE_POOL_DEFAULT_LIMIT = 10
P50_INTERNAL_PRICE_SCALE_DIVISOR = 100000
P50_SAME_SHOP_LISTING_MIN_LLM_SCORE = 6.0
P50_SAME_SHOP_TOP_SHOP_COUNT = 7
P50_ANCHOR_STRATEGY_MAX_SHOPS_TO_TRY = 12
P50_ANCHOR_STRATEGY_PER_SHOP_TIMEOUT_SECONDS = 10.0
P50_TWO_SPEC_VOUCHER_TOP_SHOP_COUNT = 6
P50_TWO_SPEC_BIDIRECTIONAL_POOL_CAP = 60
P50_TWO_SPEC_COLLECT_PER_SPEC_CAP = 20
P50_TWO_SPEC_MIN_ACCEPTABLE_LLM_SCORE = 5.0
P50_THREE_SPEC_TOP_SHOP_COUNT = 3
P50_THREE_SPEC_CANDIDATE_POOL_CAP = 60
P50_THREE_SPEC_PER_SHOP_LISTING_LIMIT = 10
P50_THREE_SPEC_COLLECT_CAP = 20
P50_SKIP_FULL_COVERAGE_FOR_SPEC_COUNTS: frozenset[int] = frozenset()
P50_SHOP_RANK_SKIP_REASON_SHOP_ID = 1
P50_SHOP_RANK_SKIP_REASON_NO_CROSS_SPEC_HIT = 2
P50_SHOP_RANK_SKIP_REASON_ANCHOR_PRICE = 3
P50_SHOP_RANK_SKIP_REASON_VOUCHER_BAND = 4
P50_SHOP_RANK_SKIP_REASON_VOUCHER_PRICE = 5
P50_VOUCHER_LISTING_MIN_LLM_SCORE = 5.0
P50_VOUCHER_SWAP_MIN_PRICE_IMPROVEMENT = 1.0
P50_VOUCHER_BUDGET_SWAP_MAX_ITERATIONS = 64
P50_SINGLE_PRODUCT_SHORTLIST_SIZE = 10
P50_SINGLE_PRODUCT_BATCH_LLM_SCORE_CAP = 15
P50_SINGLE_PRODUCT_ENABLE_DUAL_JUDGE_CONSISTENCY = True
P50_SINGLE_PRODUCT_DUAL_JUDGE_SCORE_GAP = 1.5
P50_ONLY_PRODUCT_TYPE_SEARCH_NOTE: str = "The query refers to the product type alone with no additional qualifiers (no brand, color, material, or numeric spec). Appending 'only' to the search query narrows results to this exact product type and avoids unrelated products that merely contain this term."
P50_LLM_JSON_INPUT_PREAMBLE = 'Input format: a JSON object with:\n  * "query" ? the raw user request (always present).\n'
P50_LLM_PARSE_RULES_COMMON = 'Rules for keywords:\n  * Concatenate in the same left-to-right order as the raw query.\n  * Include: product type, brand, material, color (with modifiers), quantity + unit, volume/weight, dimensions, capacity, fit, style, length, use-case, packaging hints.\n  * **Use-case / audience / setting (required):** When the query states who or what a product is for—e.g. "for students", "office use", "suitable for", "ideal for kids", "school", "travel", "outdoor"—you **must** put every distinct use-case noun or setting token into `keywords` in left-to-right order. Strip only glue words (`suitable`, `ideal`, `perfect`, `for`, `use`); **never** omit the audience/setting words (`students`, `office`, `kids`, `hiking`, etc.). Example: `suitable for students and office use` → keywords must include `students` and `office`. Do not drop stated use-case tokens to save words; if near the 8-word cap, drop generic filler before dropping any use-case token.\n  * Exclude any service/shipping wording.\n  * Whenever the user gives a number with a physical or commerce unit(e.g. measured quantities and units), **extract it into `keyword`** and normalize to **digits first, unit letters immediately after with no space** (ASCII digits + Latin unit suffix in one token). Cover length, width, height, depth, diameter, screen/TV diagonal, area/volume, weight, capacity, electrical draw (W, V, A, mAh), data size (`128GB`), thread/pitch where numeric, **pack or piece counts** (`6pcs`, `12pk`, `3pack`), multi-axis sizes on one shared unit when natural (`200x300mm`, `10x20cm`). Examples: `2m`, `1.5cm`, `55inch`, `500ml`, `65W`, `19V`, `6pcs`. Never split into a number token plus a spelled-out unit word (`5 meter` → `5m`; `3 pieces` → `3pcs`). Ranges: prefer one compact token when one unit applies (`10-20cm`). Preserve meaningful token order for the rest of the line.\n  * When "any" precedes a descriptor (e.g. "any flavor"), retain the pair verbatim.\n  * When the user quotes a word or phrase (single-quoted or double-quoted), keep that quoted combination verbatim in keywords—including the quote marks and every word inside—in the same left-to-right position as the raw query. Do not strip quotes, split the phrase, or drop inner words. Example: `shoes \'as show\' nike` -> `shoes \'as show\' nike`.\n\nRules for price_range (digit side of the hyphen is mandatory — never invert):\n  * Bounded ("from 1889 to 3315 PHP", "between 500 and 1200"): "lo-hi" e.g. "1889-3315".\n  * Minimum only ("above 1513", "over 1383", "greater than 500", "at least 500"): "lo-" with the number BEFORE the hyphen e.g. "1513-" — NEVER "-1513".\n  * Maximum only ("below 1200", "under 500", "at most 800"): "-hi" with the number AFTER the hyphen e.g. "-1200" — NEVER "1200-".\n  * null when no numeric price bound appears in that product\'s slice.\n\nRules for only_product_type:\n  * true when keywords name a product type alone (including multi-word compound nouns). Append `only` to the keyword if it is true (e.g. `yoga mat` -> `yoga mat only`, `USB hub` -> `USB hub only`).\n  * false when any attribute (brand, color, material, numeric spec, adjective) is present beyond the bare noun.\n\nRules for service (map user wording -> enum):\n  * official store / guaranteed authenticity / quick returns -> "official"\n  * free shipping / free delivery                            -> "freeShipping"\n  * COD / cash on delivery / payment on delivery             -> "COD"\n  * flash deal / limited-time deal / flash sale              -> "flashsale"\n  * Combine multiple with commas; null when none apply.\n\n'
P50_LLM_PARSE_RULES_PRODUCT_ORDER = 'Rules for order:\n  * List products[] in the same left-to-right order as each distinct product intent appears in the raw query. Do not sort or reorder the array by richness or by order.\n  * Single-product requests: use "order": "1st" only.\n  * Multiple products: assign "1st", "2nd", … by decreasing information richness (most specific / constrained = "1st"). Use this only as a richness rank for tie-breaking ? do not move array entries to match it.\n  * Values must be a permutation covering every product exactly once (each rank used once).\n\n'
P50_LLM_PROMPT_PARSE_SINGLE_PRODUCT = P50_LLM_JSON_INPUT_PREAMBLE + 'Task: parse a shopping request into structured search parameters.\n\nOutput schema (strict JSON, no code fence, no prose):\n{\n  "reasoning": "one-sentence summary of the extraction decisions you made",\n  "products": [{\n    "keywords":        "2-8 word search string",\n    "price_range":     "lo-hi" | "lo-" | "-hi" | null,\n    "service":         null | "official" | "freeShipping" | "COD" | "flashsale" | "<csv combination>",\n    "only_product_type": true | false,\n    "constraints":     {"attribute_key": "value", ...},\n    "hypothetical_title": "plausible seller-style product title (8-15 words)"\n  }],\n}\n\n' + P50_LLM_PARSE_RULES_COMMON + 'Rules for constraints (required attribute map):\n  * Extract key-value pairs of product attributes explicitly named in the query: color, size, brand, material, pattern, style, type, model, year, closure, occasion, feature, compatibility, quantity, finish, capacity, dimension, etc.\n  * Use lowercase values. Only include attributes actually stated by the user (never infer).\n  * Empty object {} when no structured attributes are mentioned.\n\nRules for hypothetical_title:\n  * Write a plausible product title a seller would put on a listing that satisfies the query.\n  * Use seller-style vocabulary: include technical descriptors, compatibility cues, and functional terms (e.g. "Replacement Parts", "For X", "Original", "Ribbon", "Cable", "Cover", "Adjustable", "Professional") that sellers commonly add but users rarely say.\n  * 8-15 words, ASCII only, no markdown, no quotes inside.\n  * Use DIFFERENT wording than the raw query so a BM25 probe over this title surfaces seller vocabulary the user\'s phrasing missed.\n\nEmit JSON only.'
P50_LLM_PROMPT_PARSE_SAME_SHOP_MULTI = P50_LLM_JSON_INPUT_PREAMBLE + 'Task: a shopping request names several distinct products the SAME shop must carry. Split it into one entry per product.\n\nOutput schema (strict JSON, no code fence, no prose):\n{\n  "reasoning": "one-sentence summary of how you segmented the query",\n  "products": [{\n    "query":           "the exact slice of the raw query describing this product",\n    "keywords":        "2-8 word search string",\n    "price_range":     "lo-hi" | "lo-" | "-hi" | null,\n    "service":         null | "official" | "freeShipping" | "COD" | "flashsale" | "<csv combination>",\n    "only_product_type": true | false,\n    "order":           "1st" | "2nd" | "3rd" | ...\n  }]\n}\n\n' + P50_LLM_PARSE_RULES_COMMON + P50_LLM_PARSE_RULES_PRODUCT_ORDER + 'Emit JSON only.'
P50_LLM_PROMPT_PARSE_VOUCHER_BUNDLE = P50_LLM_JSON_INPUT_PREAMBLE + 'Task: a shopping request lists one or more products PLUS a voucher/budget constraint. Extract both.\n\nOutput schema (strict JSON, no code fence, no prose):\n{\n  "reasoning": "one-sentence summary of the voucher structure and the products you identified",\n  "products": [{\n    "query":           "the exact slice of the raw query describing this product",\n    "keywords":        "2-8 word search string",\n    "price_range":     "lo-hi" | "lo-" | "-hi" | null,\n    "service":         null | "official" | "freeShipping" | "COD" | "flashsale" | "<csv combination>",\n    "only_product_type": true | false,\n    "order":           "1st" | "2nd" | "3rd" | ...\n  }],\n  "voucher": {\n    "voucher_type":   "platform" | "shop",\n    "discount_type":  "fixed" | "percentage",\n    "discount_value": <number>,\n    "threshold":      <number, minimum spend required>,\n    "cap":            <number, max discount for percentage; 0 when not stated or fixed type>,\n    "budget":         <number, user\'s maximum out-of-pocket>\n  },\n  "is_shop_voucher": true | false\n}\n\n' + P50_LLM_PARSE_RULES_COMMON + P50_LLM_PARSE_RULES_PRODUCT_ORDER + 'Rules for the voucher block:\n  * "42% off" -> discount_type=percentage, discount_value=42.\n  * "PHP 50 off" -> discount_type=fixed, discount_value=50.\n  * threshold defaults to 0 when no minimum is stated.\n  * cap = 0 whenever the voucher is fixed-value or no cap is mentioned.\n  * budget is the user\'s total spending limit BEFORE the voucher applies.\n\nRules for is_shop_voucher:\n  * true when the voucher says the items must come from the same shop; false otherwise.\n\nEmit JSON only.'
P50_LLM_PROMPT_SCORE_CANDIDATE_BATCH = 'Role: candidate-relevance scorer for a multi-product shop-matching task.\n\nInput:  JSON with "request" (the user\'s description), a list of "candidates" (product summaries), and a boolean "only_product_type".\nOutput: JSON ARRAY, one object per candidate in the order received, each with an integer "score" from 0 (no match) to 10 (perfect match).\n\nScoring guidance:\n  * Attributes and sku_options are more trustworthy than the product title. The title can be padded with generic terms.\n  * When the request says "any X", treat it the same as "all X" ? any candidate value satisfies it.\n  * Weigh these factors when present: model/compatibility, material, theme/function, brand, quantity, weight/volume, dimensions, style/fit/length, use-case, service tags, price.\n  * Treat formatting differences (spacing, punctuation, synonyms) as equivalent matches.\n  * When "only_product_type" is true, inspect sku_options and attributes for a "product_type + only" variant ? do not look for it in the title.\n  * Do not reward a candidate just because its title is longer or has more generic matching words.\n  * When multiple candidates equally satisfy one dimension, prefer the one with broader consistency across all other dimensions.\n\nOutput shape (no markdown):\n[{"product_id":<id>,"score":<0-10>}, ...]'
P50_LLM_PROMPT_JUDGE_BEST_LISTING = 'Task: identify the single best candidate product for a shopping request, graded by how exactly the candidate matches what the user asked for.\n\nInputs come as a JSON object with `request` (raw user text), a list of `candidates` (each carrying title, price, service flags, attributes, and a trimmed sku_options_preview), and a boolean `only_product_type`.\n\nJudging principles, applied in order:\n\n(a) Structured signals carry more weight than title prose. The catalogue\'s attributes and sku_options are the seller\'s own labelling and are the source of truth when deciding whether a candidate genuinely carries a requested property.\n\n(b) Each stated user requirement must be accounted for ? compatibility/model, brand, material, colour, quantity/units, weight/volume, dimensions, packaging, fit, style, length, use-case, service tags, and price range all count.\n\n(c) Do not upgrade a candidate just because its title is denser in query words or uses broader generic terms. Title word-count is not evidence.\n\n(d) Treat slight formatting, spacing, punctuation, or tokenisation differences between the user\'s phrasing and the catalogue value as equivalent matches.\n\n(e) When two candidates both clearly satisfy the main requirement, prefer the one whose title + attributes + sku_options agree MORE consistently end-to-end, not the one that happens to pile extra words onto a single attractive field.\n\n(f) When `only_product_type` is true, the bare product type must appear as an `only` variant inside sku_options or attributes. Title-only evidence is insufficient.\n\n(g) Price is a last-resort tiebreaker. Never downgrade a stronger-matching candidate because a weaker one happens to be cheaper.\n\nScoring rubric for `relevance_score` (integer 0 through 10):\n  10 ? every hard requirement satisfied exactly (product type, attributes, sku_options, service, price).\n  8-9 ? every hard requirement satisfied; only cosmetic wording differences remain.\n  6-7 ? most requirements satisfied; exactly one non-critical attribute is unverified.\n  4-5 ? core product type is right but at least one stated attribute or sku value is unsatisfied or unverifiable.\n  2-3 ? partial product-type match with multiple misses.\n  0-1 ? wrong product type or off-target.\n\nBefore settling on the final score, subtract each applicable penalty:\n  -4 when the candidate\'s price falls outside the requested range.\n  -3 for each required service tag the candidate does not offer.\n  -5 when `only_product_type` is true but the product type is qualified (extra attributes attached).\n  -2 for each key attribute that contradicts the request (brand, model, size, material, etc.).\n\nOutput strict JSON, no markdown fences, no prose:\n{\n  "best_product_id": <id>,\n  "reason":          "1-2 sentences citing the specific attribute or sku_option values that decided it",\n  "relevance_score": <integer 0-10>\n}'
P50_HTTP_JOURNAL_ROW_FIELD_ORDER = ('method', 'path', 'status_code', 'duration_ms', 'timestamp', 'params', 'json_data', 'response', 'completion_tokens', 'result_product_ids')
P50_thread_local_http_journal_buffer = threading.local()
P50_SPEC_RICHNESS_RANK_WORST_SENTINEL = 10000
P50_EMPTY_SHOP_ANCHOR_CANDIDATE_CAP = 8
P50_EMPTY_SHOP_ANCHOR_CAP_UNDER_VOUCHER = 4
P50_LLM_PARSE_PROMPT_BY_TASK_KIND: dict[str, str] = {'product': P50_LLM_PROMPT_PARSE_SINGLE_PRODUCT, 'shop': P50_LLM_PROMPT_PARSE_SAME_SHOP_MULTI, 'voucher': P50_LLM_PROMPT_PARSE_VOUCHER_BUNDLE}
P50_LLM_PARSE_MODEL_BY_TASK_KIND: dict[str, str] = {'product': P50_resolve_inference_model_handle('PRODUCT_PARSE_MODEL'), 'shop': P50_resolve_inference_model_handle('SHOP_PARSE_MODEL'), 'voucher': P50_resolve_inference_model_handle('VOUCHER_PARSE_MODEL')}
P50_ENABLE_LLM_SELF_CONSISTENCY_PICK = True
P50_SELF_CONSISTENCY_HEURISTIC_GAP = 1.5
P50_VOUCHER_COMBO_K_PER_SPEC = 12
P50_VOUCHER_COMBO_MAX_COMBOS = 5000
P50_VOUCHER_PRICE_BAND_WIDEN_RATIO = 0.25
P50_ANCHOR_ELECTION_USE_FULL_SCORE_POOL = True

@dataclass
class P50_DialogueRunState:
    pipeline_start_time: float = 0.0
    product_detail_cache: dict[str, dict] = field(default_factory=dict)
    last_tool_call_timestamp: float = 0.0

    def reset_for_run(self) -> None:
        self.pipeline_start_time = time.monotonic()
        self.last_tool_call_timestamp = 0.0
        self.product_detail_cache.clear()
P50_dialogue_run_state = P50_DialogueRunState()

class P50_RequestsPerMinuteGate:

    def __init__(self, max_rpm: int, window: float, min_gap: float) -> None:
        self.max_rpm = max_rpm
        self.window = window
        self.min_gap = min_gap
        self.history: list[float] = []
        self.lock = threading.Lock()

    def compute_delay(self, now: float) -> float:
        expiry = now - self.window
        while self.history and self.history[0] <= expiry:
            self.history.pop(0)
        delay = 0.0
        if self.history:
            gap = now - self.history[-1]
            if gap < self.min_gap:
                delay = self.min_gap - gap
        if len(self.history) >= self.max_rpm:
            delay = max(delay, self.window - (now - self.history[0]))
        return delay

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                wait = self.compute_delay(now)
                if wait <= 0:
                    self.history.append(now)
                    return
            time.sleep(wait)

class P50_JournalingProxyHttpClient:

    def __init__(self, upstream: ProxyClient, label: str) -> None:
        self.upstream = upstream
        self.label = label

    def __getattr__(self, name: str):
        return getattr(self.upstream, name)

    def roundtrip(self, method: str, path: str, params: Any=None, json_data: Any=None, **kw):
        t0 = time.time()
        resp = None
        try:
            if method == 'POST':
                resp = self.upstream.post(path, json_data=json_data, **kw)
            else:
                resp = self.upstream.get(path, params=params, **kw)
            return resp
        finally:
            P50_append_http_roundtrip_journal_event(self.label, method, path, (time.time() - t0) * 1000, resp, params=params, json_data=json_data)

    def post(self, path: str, json_data=None, **kw):
        return self.roundtrip('POST', path, json_data=json_data, **kw)

    def get(self, path: str, params=None, **kw):
        return self.roundtrip('GET', path, params=params, **kw)

class P50_DialogueRunContext:

    def __init__(self) -> None:
        self.steps: list[dict] = []
        self.query: str = ''

class P50_SingleProductRecommendationFlow:

    class EarlyRecommendationSuccessAbort(Exception):
        pass
    __slots__ = ('ctx', 'params', 'spec', 'catalog_search_params', 'constraints', 'unique', 'seen', 'scored_candidates', 'best', 'judge_relevance_score', 'meets_fast_accept_threshold')

    def __init__(self, ctx: 'DialogueRunContext', params: dict) -> None:
        self.ctx = ctx
        self.params = params
        specs = params.get('products', [{}])
        self.spec = specs[0] if specs else {}
        self.catalog_search_params = P50_parsed_spec_to_find_product_params(self.spec)
        self.constraints = self.spec.get('constraints') or {}
        self.unique: list[dict] = []
        self.seen: set[str] = set()
        self.scored_candidates: list[tuple[dict, float]] | None = None
        self.best: dict | None = None
        self.judge_relevance_score = 0.0
        self.meets_fast_accept_threshold = False

    def log_single_product_flow_start(self) -> None:
        pass

    def stage_initial_catalog_search(self) -> None:
        phase1_calls: list = []
        r1 = P50_invoke_sandbox_tool_with_gap_and_retry('find_product', {**self.catalog_search_params, 'page': 1})
        phase1_calls.append(r1)
        P50_merge_find_product_into_candidate_pool(r1, self.unique, self.seen)
        top_preview = [{'title': r.get('title', ''), 'price': r.get('price'), 'product_id': str(r.get('product_id', '') or '')} for r in self.unique[:5]]
        think_search = f"Phase 1 ? initial catalog search. I issued `find_product` for '{self.catalog_search_params.get('q', '')}' (price={self.catalog_search_params.get('price', 'any')}, service={self.catalog_search_params.get('service', 'any')}) with page=1 only. Starting with a single page is intentional: if the LLM judge scores the top result at ?{P50_LLM_JUDGE_FAST_ACCEPT_SCORE_THRESHOLD}/10 we fast-accept and skip the more expensive broadening calls, saving latency and budget. Page 1 returned {len(self.unique)} unique candidates. Top candidates by position: {top_preview}. " + (f"The parser's hypothetical seller title ('{self.spec.get('hypothetical_title', '')}') is available as a secondary probe query if the pool stays small. " if self.spec.get('hypothetical_title') else '') + (f'Structured constraints to satisfy: {self.constraints}. ' if self.constraints else '')
        P50_append_dialogue_step_tool_results(self.ctx, think_search, phase1_calls)

    def stage_initial_llm_judge(self) -> None:
        self.best = P50_llm_judge_best_from_candidate_pool(self.ctx.query, self.unique, self.spec)
        self.judge_relevance_score = float(self.best.get('_llm_relevance_score', 0.0)) if self.best else 0.0
        self.meets_fast_accept_threshold = bool(self.best) and self.judge_relevance_score >= P50_LLM_JUDGE_FAST_ACCEPT_SCORE_THRESHOLD

    def stage_narrate_judge_branch_decision(self) -> None:
        _judge_pid = str(self.best.get('product_id', '') or '') if self.best else 'none'
        _judge_title = str(self.best.get('title', '') or '')[:80] if self.best else ''
        _judge_price = self.best.get('price') if self.best else None
        if self.meets_fast_accept_threshold:
            _decision_branch = 'fast_accept'
            _decision_reason = f"The LLM judge scored the leading candidate pid={_judge_pid} ('{_judge_title}' @ ?{_judge_price}) at {self.judge_relevance_score:.1f}/10, which meets the fast-accept threshold of {P50_LLM_JUDGE_FAST_ACCEPT_SCORE_THRESHOLD}. Decision: fast-accept this pick. A single verification probe with an adapted query will be run next (HyDE seller-vocab / drop service / shorten keywords / page 2) to cross-check the pick against listings the original query may have missed, but the winner is already provisionally chosen."
        elif not self.best:
            _decision_branch = 'broaden_no_pick'
            _decision_reason = f'The LLM judge found no scoreable candidates on page 1. Decision: enter full broadening phase ? page 2, service relaxation, short-keyword trim, and HyDE probe.'
        else:
            _decision_branch = 'broaden_low_score'
            _decision_reason = f"The LLM judge scored the leading candidate pid={_judge_pid} ('{_judge_title}' @ ?{_judge_price}) at {self.judge_relevance_score:.1f}/10, which is at or below the low-confidence threshold of {P50_LLM_JUDGE_LOW_CONFIDENCE_SCORE_THRESHOLD}. Decision: do not fast-accept. Enter broadening phase to widen the candidate pool before re-judging: page 2 adds fresher listings; dropping the service filter tests whether the constraint is too narrow; the short-keyword probe catches sellers using abbreviated titles; the HyDE probe uses seller-style vocabulary from the parser's hypothetical title to surface results user phrasing misses."
        think_judge_decision = _decision_reason
        P50_append_dialogue_step_tool_results(self.ctx, think_judge_decision, [])

    def stage_optional_verification_probes(self) -> None:
        if self.meets_fast_accept_threshold and P50_single_product_may_run_probe_by_time():
            P50_run_fast_accept_verification_probes(self.ctx, self.spec, self.catalog_search_params, self.best, self.judge_relevance_score, self.unique, self.seen)

    def stage_broaden_pool_and_rejudge(self) -> None:
        if not (not self.meets_fast_accept_threshold and (not self.best or self.judge_relevance_score <= P50_LLM_JUDGE_LOW_CONFIDENCE_SCORE_THRESHOLD)):
            return
        phase2_calls: list = []
        probes_allowed = P50_single_product_may_run_probe_by_time()
        if probes_allowed:
            r2 = P50_invoke_sandbox_tool_with_gap_and_retry('find_product', {**self.catalog_search_params, 'page': 2})
            phase2_calls.append(r2)
            P50_merge_find_product_into_candidate_pool(r2, self.unique, self.seen)
            if self.catalog_search_params.get('service'):
                relaxed = {k: v for k, v in self.catalog_search_params.items() if k != 'service'}
                rr = P50_invoke_sandbox_tool_with_gap_and_retry('find_product', {**relaxed, 'page': 1})
                phase2_calls.append(rr)
                P50_merge_find_product_into_candidate_pool(rr, self.unique, self.seen)
            q_raw = (self.catalog_search_params.get('q') or '').replace(' only', '').strip()
            words = q_raw.split()
            if len(words) > 2:
                rs = P50_invoke_sandbox_tool_with_gap_and_retry('find_product', {'q': ' '.join(words[:2]), 'page': 1})
                phase2_calls.append(rs)
                P50_merge_find_product_into_candidate_pool(rs, self.unique, self.seen)
            if len(self.unique) < 10:
                hyde_q = P50_build_seller_vocabulary_hyde_probe_query(self.spec)
                api_q_norm = (self.catalog_search_params.get('q') or '').lower()
                if hyde_q and hyde_q != api_q_norm:
                    hyde_params: dict = {'q': hyde_q, 'page': 1}
                    if self.catalog_search_params.get('price'):
                        hyde_params['price'] = self.catalog_search_params['price']
                    rh = P50_invoke_sandbox_tool_with_gap_and_retry('find_product', hyde_params)
                    phase2_calls.append(rh)
                    P50_merge_find_product_into_candidate_pool(rh, self.unique, self.seen)
        _probes_run: list[str] = []
        if probes_allowed:
            _probes_run.append('page 2 of the same query (catches new listings or pagination gaps)')
            if self.catalog_search_params.get('service'):
                _probes_run.append(f"service-filter dropped (original filter '{self.catalog_search_params.get('service')}' may be too narrow; testing broader inventory)")
            if len((self.catalog_search_params.get('q') or '').replace(' only', '').split()) > 2:
                _probes_run.append("short 2-word keyword query (sellers often use abbreviated titles that don't match the full keyword string)")
            if len(self.unique) < 10:
                _probes_run.append(f"HyDE seller-vocabulary probe ('{self.spec.get('hypothetical_title', 'n/a')}' ? extracted tokens): uses the parser-generated seller-style title to surface listings written in trade vocabulary the user didn't use")
        _broaden_intro = f'Low-confidence judge score ({self.judge_relevance_score:.1f} ? {P50_LLM_JUDGE_LOW_CONFIDENCE_SCORE_THRESHOLD}) on page 1. ' if self.best else 'No usable candidates on page 1. '
        if not probes_allowed:
            _broaden_body = f'Broadening was skipped because elapsed time passed {P50_SINGLE_PRODUCT_PROBE_MAX_ELAPSED_SECONDS:.0f}s ? running under the session deadline. Using whatever was found so far ({len(self.unique)} candidates).'
        else:
            _broaden_body = f'Running {len(_probes_run)} broadening probe(s) in sequence: ' + '; '.join((f'({i + 1}) {p}' for i, p in enumerate(_probes_run))) + f". Each probe's hits are deduplicated by product_id and merged into the pool. After broadening the pool contains {len(self.unique)} distinct products. The LLM judge will re-rank the entire merged pool to select the final winner."
        fallback_broaden = _broaden_intro + _broaden_body
        think_broaden = fallback_broaden
        P50_append_dialogue_step_tool_results(self.ctx, think_broaden, phase2_calls)
        constraints_meaningful = isinstance(self.constraints, dict) and len(self.constraints) >= 2
        cap_slice = self.unique[:P50_SINGLE_PRODUCT_BATCH_LLM_SCORE_CAP]
        if self.unique and constraints_meaningful and P50_single_product_may_finalize_by_time():
            cand_pids = [str(p.get('product_id', '') or '') for p in cap_slice if p.get('product_id')]
            P50_fetch_and_cache_catalog_product_details(cand_pids)
            self.scored_candidates = P50_llm_score_listing_batch(str(self.spec.get('query') or self.spec.get('keywords') or self.ctx.query), cap_slice, P50_dialogue_run_state.product_detail_cache, only_product_type=bool(self.spec.get('only_product_type', False)))
        self.best = P50_llm_judge_best_from_candidate_pool(self.ctx.query, self.unique, self.spec)

    def stage_narrate_attribute_coverage_gate(self) -> None:
        _pre_gate_pid = str(self.best.get('product_id', '') or '') if self.best else None
        self.best = P50_apply_structured_attribute_coverage_gate(self.spec, self.best, self.scored_candidates)
        _post_gate_pid = str(self.best.get('product_id', '') or '') if self.best else None
        if self.best is not None and _pre_gate_pid is not None and (_post_gate_pid != _pre_gate_pid) and self.scored_candidates:
            _gate_pre_cov = P50_weighted_constraint_coverage_score(next((p for p, _ in self.scored_candidates if str(p.get('product_id', '')) == _pre_gate_pid), {}), P50_dialogue_run_state.product_detail_cache.get(_pre_gate_pid, {}), self.spec.get('constraints') or {})
            _gate_post_cov = P50_weighted_constraint_coverage_score(self.best, P50_dialogue_run_state.product_detail_cache.get(_post_gate_pid, {}), self.spec.get('constraints') or {})
            think_gate = f"Attribute-coverage gate: the initial judge pick (pid={_pre_gate_pid}, coverage={_gate_pre_cov * 100:.0f}%) was replaced by pid={_post_gate_pid} (coverage={_gate_post_cov * 100:.0f}%) because the challenger satisfies significantly more of the structured constraints {self.spec.get('constraints') or {}} while still scoring ? 6.0 on the batch scorer. Coverage is measured as the fraction of constraint values that appear in the candidate's title, attributes, or SKU options."
            P50_append_dialogue_step_tool_results(self.ctx, think_gate, [])

    def stage_abort_when_no_acceptable_listing(self) -> None:
        if self.best:
            return
        P50_finalize_dialogue_product_recommendation(self.ctx, [P50_NO_MATCH_PRODUCT_ID_SENTINEL], 'failure', think='No suitable product matched the constraints.')
        raise P50_SingleProductRecommendationFlow.EarlyRecommendationSuccessAbort

    def stage_finalize_successful_recommendation(self) -> None:
        pid = str(self.best.get('product_id', '') or '')
        P50_append_single_product_alternatives_step(self.ctx, self.best, self.unique, self.spec)
        constraint_check = P50_audit_selected_listing_against_spec(title=self.best.get('title', ''), price=self.best.get('price'), parsed_spec=self.spec)
        final_alts = P50_top_alternate_listings_for_narration(self.best, self.unique, self.spec, self.ctx.query, n=2, with_title=True)
        compare_clause = P50_format_single_product_comparison_clause(self.best, final_alts, self.ctx.query, self.spec)
        llm_reason = str(self.best.get('_llm_reason', '') or '').strip()
        _cc_note = ''
        if constraint_check:
            _matched = constraint_check.get('keywords_matched') or []
            _missing = constraint_check.get('keywords_missing') or []
            _price_note = constraint_check.get('price_note') or ''
            _overall = constraint_check.get('overall_note') or ''
            _cc_note = f' Keyword check: matched={_matched}' + (f', missing={_missing}' if _missing else ', no missing keywords') + f'. Price check: {_price_note}. Overall: {_overall}.'
        fb_text = f"Final selection: product_id={pid} title='{str(self.best.get('title', ''))[:100]}' price={self.best.get('price')} service={self.best.get('service')}. " + (f"LLM judge reason: '{llm_reason}'. " if llm_reason else 'No LLM reason recorded ? winner chosen by heuristic score ranking. ') + _cc_note + compare_clause
        detail = P50_dialogue_run_state.product_detail_cache.get(pid, {})
        think_sel = fb_text
        P50_finalize_dialogue_product_recommendation(self.ctx, [pid], 'success', think=think_sel, llm_reason=llm_reason)

    def execute_recommendation_flow(self) -> None:
        try:
            self.log_single_product_flow_start()
            self.stage_initial_catalog_search()
            self.stage_initial_llm_judge()
            self.stage_narrate_judge_branch_decision()
            self.stage_optional_verification_probes()
            self.stage_broaden_pool_and_rejudge()
            self.stage_narrate_attribute_coverage_gate()
            self.stage_abort_when_no_acceptable_listing()
            self.stage_finalize_successful_recommendation()
        except P50_SingleProductRecommendationFlow.EarlyRecommendationSuccessAbort:
            return
P50_acquire_catalog_http_rate_limit_slot = P50_RequestsPerMinuteGate(P50_CATALOG_HTTP_MAX_REQUESTS_PER_MINUTE, P50_RATE_LIMIT_WINDOW_SECONDS, P50_MIN_SECONDS_BETWEEN_CATALOG_CALLS).acquire
P50_journaling_llm_inference_proxy_client = P50_JournalingProxyHttpClient(ProxyClient(timeout=120, max_retries=3), 'inference')
P50_journaling_catalog_search_proxy_client = P50_JournalingProxyHttpClient(ProxyClient(timeout=16, max_retries=2), 'search')

def P50_clear_thread_local_http_journal() -> None:
    setattr(P50_thread_local_http_journal_buffer, 'events', [])

def P50_read_thread_local_http_journal_events() -> list[dict]:
    event_buf = getattr(P50_thread_local_http_journal_buffer, 'events', None)
    if isinstance(event_buf, list):
        return event_buf
    fresh: list[dict] = []
    setattr(P50_thread_local_http_journal_buffer, 'events', fresh)
    return fresh

def P50_parse_completion_token_usage_from_body(response: Any) -> tuple[int | None, dict | None]:
    if not isinstance(response, dict):
        return (None, None)
    usage_block = response.get('usage')
    if not isinstance(usage_block, dict):
        return (None, None)
    return (usage_block.get('completion_tokens'), usage_block)

def P50_parse_product_ids_from_catalog_response(path: str, response: Any) -> list[str]:
    if P50_CATALOG_FIND_PRODUCT_API_PATH not in path or not isinstance(response, list):
        return []
    return [str(rec['product_id']) for rec in response if isinstance(rec, dict) and rec.get('product_id')]

def P50_append_http_roundtrip_journal_event(kind: str, method: str, path: str, elapsed_ms: float, response: Any, params: Any=None, json_data: Any=None) -> None:
    completion_tokens, usage_block = P50_parse_completion_token_usage_from_body(response)
    ts = time.time()
    event: dict = {'kind': kind, 'method': method, 'path': path, 'duration_ms': round(elapsed_ms, 1), 'completion_tokens': completion_tokens, 'status_code': 200 if isinstance(response, (dict, list)) else None, 'timestamp': int(ts * 1000), 't': ts}
    if isinstance(params, dict) and params:
        event['params'] = {k: v for k, v in params.items() if v is not None}
    if isinstance(json_data, dict) and json_data.get('model'):
        event['json_data'] = {'model': json_data['model']}
    if usage_block is not None:
        event['response'] = {'usage': usage_block}
    pids = P50_parse_product_ids_from_catalog_response(path, response)
    if pids:
        event['result_product_ids'] = pids
    P50_read_thread_local_http_journal_events().append(event)

def P50_merge_http_journal_into_first_dialogue_step(steps: list[dict]) -> None:
    if not steps:
        return
    trace = [row for row in ({k: ev[k] for k in P50_HTTP_JOURNAL_ROW_FIELD_ORDER if k in ev} for ev in P50_read_thread_local_http_journal_events()) if row]
    if not trace:
        return
    info = steps[0].get('extra_info')
    if not isinstance(info, dict):
        info = {}
        steps[0]['extra_info'] = info
    info['proxy_calls'] = trace

def P50_catalog_http_get_rate_limit(path: str, params: dict | None=None):
    P50_acquire_catalog_http_rate_limit_slot()
    return P50_journaling_catalog_search_proxy_client.get(path, params)

def P50_dialogue_budget_seconds_remaining() -> float:
    if P50_dialogue_run_state.pipeline_start_time <= 0:
        return P50_DIALOGUE_SESSION_TIMEOUT_SECONDS
    return P50_DIALOGUE_SESSION_TIMEOUT_SECONDS - (time.monotonic() - P50_dialogue_run_state.pipeline_start_time)

def P50_invoke_sandbox_tool_with_gap_and_retry(tool_name: str, params: dict) -> dict:
    registered_name = tool_name if tool_name.startswith('P50_') else f'P50_{tool_name}'
    elapsed_since_last = time.monotonic() - P50_dialogue_run_state.last_tool_call_timestamp
    if elapsed_since_last < P50_MIN_SECONDS_BETWEEN_SANDBOX_TOOL_CALLS:
        time.sleep(P50_MIN_SECONDS_BETWEEN_SANDBOX_TOOL_CALLS - elapsed_since_last)
    attempt_idx = 0
    while True:
        try:
            call_result = {**execute_tool_call(registered_name, params), 'name': tool_name}
            P50_dialogue_run_state.last_tool_call_timestamp = time.monotonic()
            return call_result
        except Exception:
            attempt_idx += 1
            if attempt_idx >= P50_SANDBOX_TOOL_MAX_RETRY_ATTEMPTS:
                raise
            wait_secs = P50_SANDBOX_TOOL_RETRY_BACKOFF_BASE_SECONDS * 2 ** (attempt_idx - 1)
            time.sleep(wait_secs)

def P50_parse_optional_float_from_text(text: str) -> float | None:
    try:
        return float(text)
    except (ValueError, TypeError):
        return None

def P50_normalize_catalog_service_csv_filter(service: str | None) -> str | None:
    if not service:
        return service
    if service == 'default':
        return None
    parts = [p.strip() for p in service.split(',') if p.strip() and p.strip() != 'default']
    return ','.join(parts) or None

def P50_parse_hyphenated_price_range_bounds(price_range: str) -> tuple:
    if not price_range or not isinstance(price_range, str):
        return (None, None)
    left_raw, sep, right_raw = price_range.partition('-')
    if not sep:
        return (None, None)
    left, right = (left_raw.strip(), right_raw.strip())
    return (P50_parse_optional_float_from_text(left) if left else None, P50_parse_optional_float_from_text(right) if right else None)

def P50_parse_optional_price_range_to_float_bounds(price_range: str | None) -> tuple[float | None, float | None]:
    if not price_range:
        return (None, None)
    s = str(price_range).strip()
    if '-' not in s:
        v = P50_parse_optional_float_from_text(s)
        return (None, v) if v is not None else (None, None)
    sep_idx = s.index('-')
    lo_part, hi_part = (s[:sep_idx].strip(), s[sep_idx + 1:].strip())
    return (P50_parse_optional_float_from_text(lo_part) if lo_part else None, P50_parse_optional_float_from_text(hi_part) if hi_part else None)

def P50_strip_stopwords_from_search_keywords(text: str | None) -> str:
    if not text:
        return P50_FALLBACK_CATALOG_SEARCH_QUERY
    unique_tokens = list(dict.fromkeys((P50_SEARCH_KEYWORD_SYNONYM_MAP.get(tok, tok) for tok in text.lower().split() if tok not in P50_RELEVANCE_SCORING_STOPWORDS)))
    return ' '.join(unique_tokens) if unique_tokens else P50_FALLBACK_CATALOG_SEARCH_QUERY

def P50_normalize_keywords_in_parsed_product(prod: dict) -> dict:
    cleaned = dict(prod)
    for field in ('keywords', 'q'):
        if field in cleaned:
            cleaned[field] = P50_strip_stopwords_from_search_keywords(cleaned.get(field))
    return cleaned

def P50_normalize_all_products_in_search_params(params: dict) -> dict:
    out = dict(params)
    raw_products = out.get('products') or []
    cleaned_products = [P50_normalize_keywords_in_parsed_product(p) for p in raw_products if isinstance(p, dict)]
    if cleaned_products:
        out['products'] = cleaned_products
    return out

def P50_unique_non_empty_product_id_strings(ids: list) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in ids:
        val = str(raw).strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out

def P50_join_product_ids_as_csv_ordered(ids: list, expected_order: list=None) -> str:
    deduped = P50_unique_non_empty_product_id_strings(ids)
    if expected_order:
        order_index = {eid: i for i, eid in enumerate(expected_order)}
        fallback = len(expected_order)
        deduped.sort(key=lambda eid: order_index.get(eid, fallback))
    return ','.join(deduped) if deduped else P50_NO_MATCH_PRODUCT_ID_SENTINEL

def P50_fetch_and_cache_catalog_product_details(product_ids: list[str]) -> dict[str, dict]:
    if not product_ids:
        return {}
    missing = [pid for pid in product_ids if pid not in P50_dialogue_run_state.product_detail_cache]
    chunk_size = 10
    idx = 0
    while idx < len(missing):
        chunk = missing[idx:idx + chunk_size]
        idx += chunk_size
        api_result = P50_catalog_http_get_rate_limit('/search/view_product_information', {'product_ids': ','.join(chunk)})
        if isinstance(api_result, list):
            for item in api_result:
                P50_dialogue_run_state.product_detail_cache[str(item.get('product_id', ''))] = item
    return {pid: P50_dialogue_run_state.product_detail_cache[pid] for pid in product_ids if pid in P50_dialogue_run_state.product_detail_cache}

def P50_build_catalog_find_product_api_params(query: str, *, page: int=1, shop_id: str | None=None, price: str | None=None, sort: str | None=None, service: str | None=None) -> dict[str, Any]:
    p: dict[str, Any] = {'q': quote_plus(query), 'page': page}
    if shop_id:
        p['shop_id'] = shop_id
    if price:
        p['price'] = price
    if sort and sort != 'default':
        p['sort'] = sort
    svc = P50_normalize_catalog_service_csv_filter(service)
    if svc:
        p['service'] = svc
    return p

def P50_execute_catalog_product_search(params: dict[str, Any]) -> list[P50_CatalogListingDict]:
    rows = P50_journaling_catalog_search_proxy_client.get('/search/find_product', params) or []
    _oro_record_search(rows)
    return rows

def P50_parsed_spec_to_find_product_params(product: dict, *, include_price: bool=True) -> dict[str, Any]:
    kw = product.get('keywords', 'product')
    svc = product.get('service')
    q = kw + (' only' if 'only' not in kw and (not svc) and bool(product.get('only_product_type')) else '')
    p: dict[str, Any] = {'q': q}
    if include_price and product.get('price_range'):
        p['price'] = product['price_range']
    if svc:
        p['service'] = svc
    return p

def P50_merge_find_product_into_candidate_pool(result: dict, unique: list[dict], seen: set[str]) -> None:
    for prod in (result or {}).get('result') or []:
        if (pid := str(prod.get('product_id', ''))) and pid not in seen:
            seen.add(pid)
            unique.append(prod)

def P50_flatten_listing_and_detail_to_search_text(product: dict, detail: dict | None) -> str:
    fragments = [(product.get('title') or '').lower()]
    if isinstance(detail, dict):
        attrs = detail.get('attributes') or {}
        if isinstance(attrs, dict):
            for k, vs in attrs.items():
                fragments.append(str(k).lower().replace('_', ' '))
                if isinstance(vs, list):
                    fragments.extend((str(v).lower() for v in vs))
                else:
                    fragments.append(str(vs).lower())
        skus = detail.get('sku_options') or {}
        if isinstance(skus, dict):
            for opts in skus.values():
                if isinstance(opts, dict):
                    for k, v in opts.items():
                        fragments.append(str(k).lower().replace('_', ' '))
                        fragments.append(str(v).lower())
    return ' '.join(fragments)

def P50_compact_listings_for_dialogue_trace(items: list) -> list:
    return [{'pid': str(item.get('product_id', '')), 'p': item.get('price'), 's': str(item.get('shop_id', ''))} for item in items[:P50_DIALOGUE_TOOL_RESULT_LISTING_CAP] if isinstance(item, dict)]

def P50_compact_find_product_tool_result_for_trace(tool_call: dict) -> dict:
    if not isinstance(tool_call, dict) or tool_call.get('name') != 'find_product':
        return tool_call
    inner = tool_call.get('result')
    if isinstance(inner, dict) and isinstance(inner.get('result'), list):
        return {**tool_call, 'result': {**inner, 'result': P50_compact_listings_for_dialogue_trace(inner['result'])}}
    if isinstance(inner, list):
        return {**tool_call, 'result': P50_compact_listings_for_dialogue_trace(inner)}
    return tool_call

def P50_llm_model_ids_with_role_fallback(model: str) -> list[str]:
    sandbox = getenv('SANDBOX_MODEL')
    if sandbox:
        return [sandbox]
    return [model, P50_resolve_inference_model_handle('PRODUCT_RANK_MODEL'), P50_resolve_inference_model_handle('FINAL_FALLBACK_MODEL')]

def P50_active_llm_model_chain_for_pick() -> list[str]:
    sandbox = getenv('SANDBOX_MODEL')
    if sandbox:
        return [sandbox]
    return P50_InferenceProfileMatrix.pick_model_chain()

def P50_active_llm_model_chain_for_batch_score() -> list[str]:
    sandbox = getenv('SANDBOX_MODEL')
    if sandbox:
        return [sandbox]
    return P50_InferenceProfileMatrix.score_model_chain()

def P50_tokenize_query_for_relevance_scoring(query_text: str) -> list[str]:
    return list(dict.fromkeys((tok for tok in re.findall('\\b\\w+\\b', query_text.lower()) if len(tok) > 1 and tok not in P50_RELEVANCE_SCORING_STOPWORDS)))

def P50_query_token_matches_title_word_directly(word: str, title_words: set[str]) -> bool:
    if word in title_words:
        return True
    stem = word[:-1] if word.endswith('s') else f'{word}s'
    if stem in title_words:
        return True
    if len(word) < 3:
        return False
    return any((cand.startswith(word) for cand in title_words if len(cand) > len(word)))

def P50_query_token_partially_matches_title_word(word: str, title_words: set[str]) -> bool:
    return any((word.startswith(tw) or tw.startswith(word) for tw in title_words if len(tw) > 2))

def P50_title_match_score(query_words: list[str], title_words: set[str], title: str) -> float:
    score = 0.0
    title_score = 0.0
    for w in query_words:
        if P50_query_token_matches_title_word_directly(w, title_words):
            score += 2.0
        elif P50_query_token_partially_matches_title_word(w, title_words):
            score += 1.0
        elif any((ch.isdigit() for ch in w)) and w in title:
            score += 2.0
        if query_words:
            title_score = score / (2.0 * len(query_words))
    return title_score

def P50_score_title_token_overlap(query_words: list[str], title_words: set[str], title: str) -> float:
    score = 0.0
    for w in query_words:
        if P50_query_token_matches_title_word_directly(w, title_words):
            score += 2
        elif P50_query_token_partially_matches_title_word(w, title_words):
            score += 1
        if any((ch.isdigit() for ch in w)) and w in title:
            score += 2
    return score

def P50_iterate_flat_attribute_pairs_from_detail(detail: P50_CatalogListingDict):
    for key, vals in (detail.get('attributes') or {}).items():
        yield (key, vals)
    for opts in (detail.get('sku_options') or {}).values():
        if isinstance(opts, dict):
            yield from opts.items()

def P50_flatten_detail_to_lowercase_text_and_tokens(detail: P50_CatalogListingDict) -> tuple[str, set[str]]:
    tokens: list[str] = []
    exact_vals: set[str] = set()
    for key, values in (detail.get('attributes') or {}).items():
        tokens.append(key.replace('_', ' '))
        for value in values if isinstance(values, list) else [values]:
            text = str(value).strip().lower()
            tokens.append(text)
            exact_vals.add(text)
    sku_probe = {'attributes': {}, 'sku_options': detail.get('sku_options') or {}}
    for key, value in P50_iterate_flat_attribute_pairs_from_detail(sku_probe):
        text = str(value).strip().lower()
        tokens.extend((key.replace('_', ' '), text))
        exact_vals.add(text)
    return (' '.join(tokens).lower(), exact_vals)

def P50_score_attribute_text_overlap(query_words: list[str], detail: dict) -> float:
    detail_text, exact_vals = P50_flatten_detail_to_lowercase_text_and_tokens(detail)
    detail_words = set(re.findall('\\b\\w+\\b', detail_text))
    total = 0.0
    for w in query_words:
        if f'{w}#' in exact_vals:
            total += 5
        elif w in exact_vals:
            total += 3
        elif w in detail_words:
            total += 2
    return total

def P50_case_sensitive_attribute_score(query_words: list[str], detail: dict, a_coe: float, sku_coe: float) -> float:
    sku_words: dict[str, str] = {}
    attr_words: set[str] = set()
    sku_matched_num = 0.0
    attr_matched_num = 0.0
    for _key, value in (detail.get('attributes') or {}).items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            text = re.findall('\\b\\w+\\b', str(item).strip().lower().replace('_', ' '))
            attr_words.update(text)
    for value in attr_words:
        if value in query_words:
            attr_matched_num += 1.0
    total_attr_num = float(len(attr_words))
    attr_score = attr_matched_num / total_attr_num if total_attr_num > 0 and attr_matched_num else 0.0
    for _key, value in (detail.get('sku_options') or {}).items():
        if isinstance(value, dict):
            sku_words.update(value)
    for _key, value in sku_words.items():
        if value in query_words:
            sku_matched_num += 1.0
    total_sku_num = float(len(sku_words))
    sku_score = sku_matched_num / total_sku_num if total_sku_num > 0 and sku_matched_num else 0.0
    return sku_coe * sku_score + a_coe * attr_score

def P50_heuristic_listing_relevance_score(product: P50_CatalogListingDict, query_text: str, detail: P50_CatalogListingDict | None=None) -> float:
    title = product.get('title', '').lower()
    title_words = set(re.findall('\\b\\w+\\b', title))
    qw = P50_tokenize_query_for_relevance_scoring(query_text)
    score = P50_score_title_token_overlap(qw, title_words, title)
    if detail:
        score += P50_score_attribute_text_overlap(qw, detail)
    return score

def P50_composite_score(product: dict, query_text: str, detail: dict=None, parsed_spec: dict=None) -> float:
    task_kind = P50_classify_shopping_task_kind_from_query(query_text)
    if task_kind == 'voucher':
        t_coe = 2.7
        a_coe = 2.0
        p_coe = 3.4
        s_coe = 1.1
        sku_coe = 0.8
    if task_kind == 'shop':
        t_coe = 3.1
        a_coe = 2.3
        p_coe = 0.6
        s_coe = 2.7
        sku_coe = 1.3
    if task_kind == 'product':
        t_coe = 3.8
        a_coe = 2.7
        p_coe = 1.2
        s_coe = 0.5
        sku_coe = 1.8
    title = product.get('title', '').lower()
    title_words = set(re.findall('\\b\\w+\\b', title))
    qw = P50_tokenize_query_for_relevance_scoring(query_text)
    spec = parsed_spec or {}
    title_score = P50_title_match_score(qw, title_words, title)
    price_val = product.get('price')
    price_range_str = spec.get('price_range')
    price_score = 0.0
    if isinstance(price_val, (int, float)) and price_range_str:
        lo, hi = P50_parse_hyphenated_price_range_bounds(price_range_str)
        outside = lo is not None and price_val < lo or (hi is not None and price_val > hi)
        if outside:
            price_score = 0
        if lo is not None and hi is not None:
            price_score += 1.0 - (float(price_val) - lo) / (hi - lo)
        elif lo is not None and price_val > lo:
            price_score += 1
        elif hi is not None and price_val < hi:
            price_score += 1 - price_val / hi
    service_score = 0.0
    required_svc = spec.get('service')
    if required_svc:
        for svc in (s.strip() for s in required_svc.split(',') if s.strip()):
            if svc == 'official':
                service_score += 0.3
            if svc == 'freeShipping':
                service_score += 0.5
            if svc == 'COD':
                service_score += 0.2
    score = t_coe * title_score + p_coe * price_score + s_coe * service_score
    if detail:
        score += P50_case_sensitive_attribute_score(qw, detail, a_coe, sku_coe)
    return score

def P50_safe_rounded_heuristic_score_or_none(prod: dict, q: str, spec: dict | None) -> float | None:
    try:
        return round(P50_composite_score(prod, q, parsed_spec=spec), 1)
    except Exception:
        return None

def P50_coerce_value_to_optional_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def P50_parse_json_object_from_llm_content(content: str) -> dict | None:
    cleaned = re.sub('<think(?:ing)?>.*?</think(?:ing)?>', '', content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub('<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub('```json?\\s*|```\\s*', '', cleaned).strip()
    try:
        out = json.loads(cleaned)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    start = cleaned.find('{')
    if start != -1:
        depth = 0
        in_str = False
        escape_next = False
        for i, ch in enumerate(cleaned[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_str:
                escape_next = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i + 1]
                    try:
                        out = json.loads(candidate)
                        if isinstance(out, dict):
                            return out
                    except json.JSONDecodeError:
                        break
    brace_match = re.search('\\{.*\\}', content, re.DOTALL)
    if brace_match:
        try:
            out = json.loads(brace_match.group())
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            pass
    return None

def P50_truncate_strings_in_nested_json(value: Any, max_len: int) -> Any:
    if isinstance(value, str):
        return value[:max_len] if len(value) > max_len else value
    if isinstance(value, list):
        return [P50_truncate_strings_in_nested_json(v, max_len) for v in value]
    if isinstance(value, dict):
        return {k: P50_truncate_strings_in_nested_json(v, max_len) for k, v in value.items()}
    return value

def P50_build_llm_batch_score_candidate_dict(product: dict, detail: dict | None, query_text: str) -> dict:
    det = detail or {}
    sku_options = det.get('sku_options') or {}
    query_words = {w for w in re.findall('\\b\\w+\\b', query_text.lower()) if len(w) > 1 and w not in P50_RELEVANCE_SCORING_STOPWORDS}
    ranked_opts: list[tuple[int, dict]] = []
    for opt in sku_options.values():
        if isinstance(opt, dict):
            opt_words = {w for w in re.findall('\\b\\w+\\b', ' '.join((str(v).lower() for v in opt.values()))) if len(w) > 1}
            ranked_opts.append((len(query_words & opt_words), opt))
    seen_keys: set[str] = set()
    sku_preview: list[dict] = []
    for _, opt in sorted(ranked_opts, key=lambda t: t[0], reverse=True):
        key = json.dumps(opt, sort_keys=True, ensure_ascii=False)
        if key not in seen_keys:
            seen_keys.add(key)
            sku_preview.append(opt)
    raw_attrs = det.get('attributes') or {}
    bounded_attrs: dict = {}
    if isinstance(raw_attrs, dict):
        for k, v in list(raw_attrs.items())[:8]:
            bounded_attrs[str(k)[:40]] = P50_truncate_strings_in_nested_json(v, 80)
    raw_title = str(product.get('title', ''))
    title = raw_title[:200] if len(raw_title) > 200 else raw_title
    return {'product_id': str(product.get('product_id', '')).strip(), 'title': title, 'price': product.get('price'), 'service': product.get('service', []), 'attributes': bounded_attrs, 'sku_options_preview': [P50_truncate_strings_in_nested_json(o, 80) for o in sku_preview[:8]]}

def P50_llm_score_listing_batch(query_text: str, candidates: list[P50_CatalogListingDict], details: dict[str, dict], only_product_type: bool=False, model: str=P50_INFERENCE_MODEL_REGISTRY['BACKUP_LLM_MODEL']) -> list[tuple[P50_CatalogListingDict, float]]:
    if not candidates:
        return []
    if P50_dialogue_budget_seconds_remaining() < 35.0:
        return [(p, 7.0) for p in candidates if P50_heuristic_listing_relevance_score(p, query_text) > 0]
    payload = {'request': query_text, 'candidates': [P50_build_llm_batch_score_candidate_dict(p, details.get(str(p.get('product_id', ''))), query_text) for p in candidates], 'only_product_type': only_product_type}
    user_content = json.dumps(payload, ensure_ascii=False)
    for m in P50_active_llm_model_chain_for_batch_score():
        attempt = 0
        while attempt < P50_LLM_COMPLETION_MAX_ATTEMPTS_PER_MODEL:
            attempt += 1
            llm_resp = P50_journaling_llm_inference_proxy_client.post('/inference/chat/completions', json_data={'model': m, 'temperature': 0.5, 'stream': False, 'messages': [{'role': 'system', 'content': P50_LLM_PROMPT_SCORE_CANDIDATE_BATCH}, {'role': 'user', 'content': user_content}]})
            if not (llm_resp and llm_resp.get('choices')):
                continue
            raw_content = llm_resp['choices'][0].get('message', {}).get('content', '')
            stripped = re.sub('```json?\\s*', '', raw_content)
            stripped = re.sub('```\\s*$', '', stripped).strip()
            score_list = None
            try:
                score_list = json.loads(stripped)
            except json.JSONDecodeError:
                array_match = re.search('\\[.*\\]', raw_content, re.DOTALL)
                if array_match:
                    try:
                        score_list = json.loads(array_match.group())
                    except json.JSONDecodeError:
                        pass
            if not isinstance(score_list, list):
                continue
            pid_to_score: dict[str, float] = {}
            for entry in score_list:
                if not isinstance(entry, dict):
                    continue
                pid = str(entry.get('product_id', '')).strip()
                if not pid:
                    continue
                try:
                    pid_to_score[pid] = float(entry.get('score', 0))
                except (TypeError, ValueError):
                    pid_to_score[pid] = 0.0
            scored = [(p, pid_to_score.get(str(p.get('product_id', '')).strip(), 0.0)) for p in candidates]
            scored.sort(key=lambda x: (x[1], str(x[0].get('product_id', ''))), reverse=True)
            return scored
    scored = [(p, 7.0 if P50_heuristic_listing_relevance_score(p, query_text) > 0 else 0.0) for p in candidates]
    scored.sort(key=lambda x: (x[1], str(x[0].get('product_id', ''))), reverse=True)
    return scored

def P50_llm_elect_best_listing_from_pool(query_text: str, candidates: list, details: dict[str, dict], only_product_type: bool=False, model: str=P50_INFERENCE_MODEL_REGISTRY['FINAL_FALLBACK_MODEL'], *, max_candidates: int=10) -> dict | None:
    if P50_dialogue_budget_seconds_remaining() < 35.0:
        return None
    cap = max(1, min(int(max_candidates), 60))
    slice_c = candidates[:cap]
    payload = {'request': query_text, 'candidates': [P50_build_llm_batch_score_candidate_dict(p, details.get(str(p.get('product_id', ''))), query_text) for p in slice_c], 'only_product_type': only_product_type}
    user_content = json.dumps(payload, ensure_ascii=False)
    for m in P50_active_llm_model_chain_for_pick():
        for attempt in range(1, P50_LLM_COMPLETION_MAX_ATTEMPTS_PER_MODEL + 1):
            result = P50_journaling_llm_inference_proxy_client.post('/inference/chat/completions', json_data={'model': m, 'temperature': 0.5, 'stream': False, 'messages': [{'role': 'system', 'content': P50_LLM_PROMPT_JUDGE_BEST_LISTING}, {'role': 'user', 'content': user_content}]})
            if not (result and result.get('choices')):
                continue
            content = result['choices'][0].get('message', {}).get('content', '')
            parsed = P50_parse_json_object_from_llm_content(content)
            if not isinstance(parsed, dict):
                continue
            best_pid = str(parsed.get('best_product_id', '') or '').strip()
            reason = str(parsed.get('reason', '')).strip()
            try:
                rel_score = float(parsed.get('relevance_score', 0))
            except (TypeError, ValueError):
                rel_score = 0.0
            _null_pids = {'', 'none', 'null', '0', 'undefined', 'n/a'}
            if best_pid.lower() in _null_pids:
                continue
            for p in slice_c:
                if str(p.get('product_id', '')).strip() == best_pid:
                    chosen = dict(p)
                    det = details.get(str(p.get('product_id', '')))
                    P50_attach_grounded_llm_reason_to_listing(chosen, reason, rel_score, p, det, query_text)
                    return chosen
    if slice_c:
        fallback = max(slice_c, key=lambda p: P50_heuristic_listing_relevance_score(p, query_text))
        fallback = dict(fallback)
        fallback.setdefault('_llm_relevance_score', 0.0)
        fallback.setdefault('_llm_reason', 'heuristic fallback ? LLM did not return a valid product_id')
        return fallback
    return None

def P50_llm_elect_best_listing_with_self_consistency(query_text: str, candidates: list, details: dict[str, dict], only_product_type: bool=False, *, max_candidates: int=10) -> dict | None:
    if not candidates:
        return None
    first = P50_llm_elect_best_listing_from_pool(query_text, candidates, details, only_product_type=only_product_type, max_candidates=max_candidates)
    if not first or not P50_ENABLE_LLM_SELF_CONSISTENCY_PICK:
        return first
    pid_pick = str(first.get('product_id', '') or '').strip()
    cap = max(1, min(int(max_candidates), 60))
    shortlist = candidates[:cap]
    heur_scores: list[tuple[str, float]] = []
    for row in shortlist:
        pid = str(row.get('product_id', '') or '').strip()
        heur_scores.append((pid, P50_heuristic_listing_relevance_score(row, query_text)))
    heur_scores.sort(key=lambda x: x[1], reverse=True)
    top_gap = heur_scores[0][1] - heur_scores[1][1] if len(heur_scores) > 1 else 10.0
    if top_gap >= P50_SELF_CONSISTENCY_HEURISTIC_GAP:
        return first
    reversed_cands = list(reversed(shortlist))
    second = P50_llm_elect_best_listing_from_pool(query_text, reversed_cands, details, only_product_type=only_product_type, max_candidates=max_candidates)
    if not second:
        return first
    if str(second.get('product_id', '') or '').strip() == pid_pick:
        s1 = float(first.get('_llm_relevance_score', 0))
        s2 = float(second.get('_llm_relevance_score', 0))
        first['_llm_relevance_score'] = min(10.0, s1 + 0.5 * s2 / 10.0)
        return first
    s1 = float(first.get('_llm_relevance_score', 0))
    s2 = float(second.get('_llm_relevance_score', 0))
    return first if s1 >= s2 else second

def P50_find_ungrounded_terms_in_llm_reason(reason: str, product: dict, detail: dict | None, query_text: str) -> tuple[bool, list[str]]:
    haystack = P50_flatten_listing_and_detail_to_search_text(product, detail)
    query_terms = {w for w in re.findall('\\b\\w{4,}\\b', (query_text or '').lower()) if w not in P50_RELEVANCE_SCORING_STOPWORDS}
    if not query_terms:
        return (True, [])
    reason_lower = (reason or '').lower()
    claimed = {t for t in query_terms if t in reason_lower}
    missing = [t for t in claimed if t not in haystack]
    return (len(missing) == 0, missing)

def P50_rewrite_reason_for_ungrounded_terms(original_reason: str, missing: list[str]) -> str:
    ms = ', '.join(sorted(missing))
    return f"Selected as the best available match among returned candidates; the user's requested term(s) ({ms}) could not be confirmed literally in this product's title, attributes, or sku_options, so the match is partial."

def P50_attach_grounded_llm_reason_to_listing(result_product: dict, reason: str, relevance_score: float, product: dict, detail: dict | None, query_text: str) -> None:
    grounded, missing = P50_find_ungrounded_terms_in_llm_reason(reason, product, detail, query_text)
    result_product['_llm_relevance_score'] = relevance_score
    if grounded:
        result_product['_llm_reason'] = reason
        return
    result_product['_llm_reason'] = P50_rewrite_reason_for_ungrounded_terms(reason, missing)
    result_product['_llm_reason_ungrounded_terms'] = missing

def P50_llm_parse_full_shopping_parameters(query: str, task_type: str) -> dict:
    sys_prompt = P50_LLM_PARSE_PROMPT_BY_TASK_KIND.get(task_type, P50_LLM_PROMPT_PARSE_SINGLE_PRODUCT)
    base_model = P50_LLM_PARSE_MODEL_BY_TASK_KIND.get(task_type, P50_INFERENCE_MODEL_REGISTRY['VOUCHER_PARSE_MODEL'])
    for model in P50_llm_model_ids_with_role_fallback(base_model):
        result = P50_journaling_llm_inference_proxy_client.post('/inference/chat/completions', json_data={'model': model, 'temperature': 0, 'stream': False, 'messages': [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': query}]})
        parsed = P50_parse_llm_parameter_json_or_none(result, task_type)
        if parsed is not None:
            return parsed
        msg = 'returned unparseable response' if result and result.get('choices') else 'returned no response'
    return P50_build_regex_fallback_parameter_snapshot(query)

def P50_parse_llm_parameter_json_or_none(result: dict, task_type: str) -> dict | None:
    if not result or not result.get('choices'):
        return None
    content = result['choices'][0].get('message', {}).get('content', '')
    parsed = P50_parse_json_object_from_llm_content(content)
    if parsed is None:
        return None
    if task_type == 'product':
        return P50_normalize_all_products_in_search_params(parsed)
    if task_type == 'shop':
        return P50_normalize_keywords_for_shop_mode_parse(parsed)
    return parsed

def P50_audit_selected_listing_against_spec(*, title: str, price: Any, parsed_spec: dict) -> dict:
    title_lower = (title or '').lower()
    spec = parsed_spec or {}
    kw = [w for w in str(spec.get('keywords', '') or '').lower().split() if w]
    matched = [w for w in kw if w in title_lower]
    missing = [w for w in kw if w not in title_lower]
    price_ok: bool | None = None
    price_note = 'no price range was parsed from the query'
    price_range = spec.get('price_range')
    if price_range:
        try:
            lo, hi = P50_parse_hyphenated_price_range_bounds(str(price_range))
            if price is None:
                price_note = f'no price available to compare against range {price_range}'
            else:
                pv = float(price)
                if lo is not None and pv < lo:
                    price_ok, price_note = (False, f'price {pv} is BELOW lower bound {lo} of range {price_range}')
                elif hi is not None and pv > hi:
                    price_ok, price_note = (False, f'price {pv} is ABOVE upper bound {hi} of range {price_range}')
                else:
                    price_ok, price_note = (True, f'price {pv} fits inside range {price_range}')
        except (TypeError, ValueError):
            price_note = f'price {price!r} is not numeric; could not check range {price_range}'
    has_missing = bool(missing)
    price_bad = price_ok is False
    if not has_missing and (not price_bad):
        note = 'The selected product looks like a genuine match for the parsed query.'
    elif has_missing and price_bad:
        note = f'HONEST MISMATCH: title is missing query terms {missing} and price is outside the requested range. This is the best available candidate, not a clean fit.'
    elif has_missing:
        note = f'HONEST MISMATCH: the selected title is missing query terms {missing}; attributes may still confirm the fit, but the title alone is imperfect.'
    else:
        note = 'HONEST MISMATCH: title matches the keywords but the price does not fit the requested range. Taking it as the closest available option.'
    return {'query_keywords': kw, 'keywords_matched': matched, 'keywords_missing': missing, 'title_contains_all_keywords': not has_missing, 'price_ok': price_ok, 'price_note': price_note, 'overall_note': note}

def P50_build_leader_vs_alternate_reason(leader: dict, alt: dict, query: str='', spec: dict | None=None) -> str:
    spec = spec or {}
    parts: list[str] = []
    lead_llm = P50_coerce_value_to_optional_float(leader.get('_llm_relevance_score'))
    alt_llm = P50_coerce_value_to_optional_float(alt.get('_llm_relevance_score'))
    if lead_llm is not None and alt_llm is not None and (abs(lead_llm - alt_llm) > 0.01):
        parts.append(f"its judge relevance score {lead_llm:.1f} beats the alternative's {alt_llm:.1f}")
    else:
        lead_h = P50_coerce_value_to_optional_float(leader.get('heuristic_score'))
        alt_h = P50_coerce_value_to_optional_float(alt.get('heuristic_score'))
        if lead_h is None or alt_h is None:
            try:
                lead_h = round(P50_heuristic_listing_relevance_score(leader, query), 1)
                alt_h = round(P50_heuristic_listing_relevance_score(alt, query), 1)
            except Exception:
                lead_h = alt_h = None
        if lead_h is not None and alt_h is not None and (abs(lead_h - alt_h) > 0.01):
            parts.append(f"its title-keyword overlap score {lead_h:.1f} is higher than the alternative's {alt_h:.1f}")
    keywords_str = str(spec.get('keywords') or query)
    qtoks = {w for w in re.findall('\\b\\w+\\b', keywords_str.lower()) if len(w) > 1 and w not in P50_RELEVANCE_SCORING_STOPWORDS}
    if qtoks:
        lead_words = set(re.findall('\\b\\w+\\b', str(leader.get('title', '')).lower()))
        alt_words = set(re.findall('\\b\\w+\\b', str(alt.get('title', '')).lower()))
        lead_only = sorted(qtoks & lead_words - alt_words)
        if lead_only:
            parts.append(f"its title carries query term(s) {lead_only} that the alternative's title omits")
    pr_raw = spec.get('price_range')
    if pr_raw:
        lo, hi = P50_parse_optional_price_range_to_float_bounds(str(pr_raw))
        lp = P50_coerce_value_to_optional_float(leader.get('price'))
        ap = P50_coerce_value_to_optional_float(alt.get('price'))
        if lp is not None and ap is not None:
            if hi is not None and ap > hi and (lp <= hi):
                parts.append(f"the alternative's price {ap:.0f} exceeds the requested ceiling {hi:.0f} while the leader's {lp:.0f} fits inside the range")
            elif lo is not None and lp >= lo and (ap < lo):
                parts.append(f"the alternative's price {ap:.0f} is below the requested floor {lo:.0f} while the leader's {lp:.0f} meets the minimum")
    if not parts:
        lp = P50_coerce_value_to_optional_float(leader.get('price'))
        ap = P50_coerce_value_to_optional_float(alt.get('price'))
        if lp is not None and ap is not None and (abs(lp - ap) > 0.01):
            parts.append(f"its price {lp:.2f} differs from the alternative's {ap:.2f} and the heuristic ranking placed it above the alternative on this candidate pool")
        else:
            parts.append("the heuristic ranking placed it above the alternative on this candidate pool's title-token coverage of the spec keywords")
    return '; '.join(parts)

def P50_format_single_product_comparison_clause(leader: dict, alternatives: list, query: str='', spec: dict | None=None) -> str:
    if not leader or not alternatives:
        return ''
    alt = None
    try:
        alt = _oro_p50_outside_alt(spec, query, leader.get('product_id', ''))
    except Exception:
        alt = None
    reason = P50_build_leader_vs_alternate_reason(leader, alt or {}, query, spec)
    alt_price = alt.get('price') if alt else None
    return f" I prefer {_oro_candidate_ref(leader.get('product_id', ''), True)} (price={leader.get('price')}) OVER {_oro_candidate_ref(alt.get('product_id', '') if alt else None, False)} (price={alt_price}) because {reason}."

def P50_format_alternate_listing_for_narration(a: dict, query: str, spec: dict | None, *, with_title: bool=True) -> dict:
    entry: dict = {'product_id': str(a.get('product_id', '') or ''), 'price': a.get('price'), 'heuristic_score': P50_safe_rounded_heuristic_score_or_none(a, query, spec)}
    if with_title:
        entry['title'] = (a.get('title') or '')[:80]
    llm_sc = a.get('_llm_relevance_score')
    if llm_sc is not None:
        entry['_llm_relevance_score'] = llm_sc
    return entry

def P50_top_alternate_listings_for_narration(leader: dict | None, pool: list, spec: dict | None, query: str, n: int=2, *, with_title: bool=True) -> list[dict]:
    if not leader or not pool:
        return []
    lead_pid = str(leader.get('product_id', '') or '')
    others = [p for p in pool if str(p.get('product_id', '') or '') != lead_pid]
    try:
        others.sort(key=lambda p: P50_heuristic_listing_relevance_score(p, query), reverse=True)
    except Exception:
        pass
    return [P50_format_alternate_listing_for_narration(a, query, spec, with_title=with_title) for a in others[:n]]
P50__RICHNESS_RANK_SENTINEL = 10000

def P50_parsed_spec_richness_rank_for_tiebreak(spec: P50_ParsedProductSpecDict) -> int:
    rank = P50_parse_product_order_rank_integer(spec.get('order'))
    return P50__RICHNESS_RANK_SENTINEL if rank is None else rank

def P50_parse_product_order_rank_integer(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return n if n >= 1 else None
    text = str(value).strip().lower()
    if not text:
        return None
    m = re.match('^(\\d+)', text)
    return int(m.group(1)) if m else None

def P50_index_of_most_information_rich_spec(spec_indices: list[int], specs: list[P50_ParsedProductSpecDict]) -> int:

    def _raw(spec: P50_ParsedProductSpecDict) -> tuple[float, int, int]:
        kw_count = len((spec.get('keywords') or '').split())
        price_score = 0.0
        pr = spec.get('price_range') or ''
        if pr and '-' in pr:
            parts = pr.split('-', 1)
            lo, hi = (parts[0].strip(), parts[1].strip())
            price_score = 1.5 if lo and hi else 1.0
        svc_count = len([s.strip() for s in (spec.get('service') or '').split(',') if s.strip()])
        return (price_score, kw_count, svc_count)
    raw = {idx: _raw(specs[idx]) for idx in spec_indices}
    max_kw = max((v[1] for v in raw.values()))
    max_svc = max((v[2] for v in raw.values()))
    final: dict[int, float] = {}
    for idx, (ps, kc, sc) in raw.items():
        score = ps
        if kc == max_kw:
            score += 1.0
        if sc == max_svc:
            score += 1.0
        final[idx] = score
    max_score = max(final.values())
    winners = [idx for idx, sv in final.items() if sv == max_score]
    return min(winners, key=lambda i: (P50_parsed_spec_richness_rank_for_tiebreak(specs[i]), i))
P50_choose_deepest_spec_index = P50_index_of_most_information_rich_spec
P50_EMPTY_SHOP_ANCHOR_CANDIDATE_CAP = 8
P50_EMPTY_SHOP_ANCHOR_CANDIDATE_CAP_UNDER_VOUCHER = 4

def P50_classify_shopping_task_kind_from_query(query: str) -> str:
    query_lower = query.lower()
    voucher_signals = {'voucher', 'budget', 'discount'}
    if any((sig in query_lower for sig in voucher_signals)):
        return 'voucher'
    shop_keywords = re.search('\\b(both|these|offering|offers|sells|same|together|along\\s+with)\\b', query_lower)
    if 'shop' in query_lower and (shop_keywords is not None or P50_MULTI_PRODUCT_CLAUSE_SPLIT_PATTERN.search(query) is not None):
        return 'shop'
    return 'product'

def P50_extract_keyword_tokens_from_query(text: str) -> list[str]:
    text_lower = text.lower()
    alpha_words = [w for w in re.findall('\\b[a-zA-Z]{2,}\\b', text_lower) if w not in P50_QUERY_TOKENIZATION_STOPWORDS]
    mixed_tokens = re.findall('\\b\\d+[a-zA-Z]+\\b|\\b[a-zA-Z]+\\d+[a-zA-Z]*\\b', text_lower)
    kw_tokens = alpha_words[:6]
    for tok in mixed_tokens[:2]:
        if tok not in kw_tokens:
            kw_tokens.append(tok)
    for num_token in re.findall('(\\d+)#', text)[:2]:
        if num_token not in kw_tokens:
            kw_tokens.append(num_token)
    return kw_tokens

def P50_extract_price_range_phrase_from_query(text: str) -> str | None:
    if not text or not isinstance(text, str):
        return None
    from_to = re.search('(?:priced\\s+)?from\\s+(\\d{1,6})\\s+to\\s+(\\d{1,6})', text, re.I)
    if from_to:
        return f'{from_to.group(1)}-{from_to.group(2)}'
    between_match = re.search('between\\s+(\\d{1,6})\\s+and\\s+(\\d{1,6})', text, re.I)
    if between_match:
        return f'{between_match.group(1)}-{between_match.group(2)}'
    range_match = re.search('(\\d{1,6})\\s*(?:to|and|-)\\s*(\\d{1,6})\\s*(?:pesos|php)', text, re.I)
    if range_match:
        return f'{range_match.group(1)}-{range_match.group(2)}'
    min_match = re.search('(?:greater|more|over|above|at\\s+least|minimum|min\\.?|>)\\s*(?:than\\s*)?(\\d{1,6})', text, re.I)
    if min_match:
        return f'{min_match.group(1)}-'
    max_match = re.search('(?:less|under|below|at\\s+most|maximum|max\\.?|<)\\s*(?:than\\s*)?(\\d{1,6})', text, re.I)
    if max_match:
        return f'-{max_match.group(1)}'
    if re.search('(?:price|pesos|php|cost)', text, re.I):
        range_match2 = re.search('(\\d{1,6})\\s+(?:to|and)\\s+(\\d{1,6})', text)
        if range_match2:
            return f'{range_match2.group(1)}-{range_match2.group(2)}'
    return None

def P50_extract_service_tags_csv_from_query(text_lower: str) -> str | None:
    svc_parts: list[str] = []
    service_signals = [('official', ('lazmall', 'official')), ('freeShipping', ('free shipping', 'free delivery')), ('flashsale', ('lazflash', 'flash sale', 'flashsale')), ('COD', ('cash on delivery', 'cod'))]
    for svc_name, markers in service_signals:
        if any((marker in text_lower for marker in markers)):
            svc_parts.append(svc_name)
    return ','.join(svc_parts) if svc_parts else None

def P50_regex_extract_lightweight_product_spec(text: str) -> dict:
    text_lower = text.lower()
    kw_tokens = P50_extract_keyword_tokens_from_query(text)
    keywords = ' '.join(kw_tokens) or 'product'
    return {'keywords': keywords, 'price_range': P50_extract_price_range_phrase_from_query(text), 'service': P50_extract_service_tags_csv_from_query(text_lower)}

def P50_build_regex_fallback_parameter_snapshot(query: str) -> dict:
    task_type = P50_classify_shopping_task_kind_from_query(query)
    product_text = P50_BUDGET_OR_VOUCHER_MENTION_PATTERN.split(query)[0].strip()
    if not product_text or len(product_text) < 15:
        product_text = query
    parts = [p.strip() for p in P50_MULTI_PRODUCT_CLAUSE_SPLIT_PATTERN.split(product_text) if p and len(p.strip()) > 10]
    if not parts:
        parts = [query]
    products = [P50_regex_extract_lightweight_product_spec(p) for p in parts]
    products = [s for s in products if len(s['keywords'].split()) >= 2] or products
    is_shop = task_type == 'shop' or (task_type == 'voucher' and 'same shop' in query.lower())
    return {'task_type': task_type, 'products': products, 'is_shop_voucher': is_shop}

def P50_normalize_keywords_for_shop_mode_parse(parsed: dict) -> dict:
    for prod in parsed.get('products', []):
        kw = prod.get('keywords')
        if not kw:
            continue
        if isinstance(kw, list):
            kw = ' '.join((str(t) for t in kw))
        prod['keywords'] = ' '.join((w for w in str(kw).split() if w.lower() not in P50_RELEVANCE_SCORING_STOPWORDS))
    return parsed

def P50_build_seller_vocabulary_hyde_probe_query(spec: dict) -> str | None:
    title = str(spec.get('hypothetical_title') or '').strip()
    if not title:
        return None
    uniq = list(dict.fromkeys((w for w in re.findall('\\b\\w+\\b', title.lower()) if w not in P50_QUERY_TOKENIZATION_STOPWORDS and w not in P50_RELEVANCE_SCORING_STOPWORDS and (len(w) > 1) and (not w.isdigit()))))
    return ' '.join(uniq[:10]) if len(uniq) >= 3 else None

def P50_weighted_constraint_coverage_score(product: dict, detail: dict | None, constraints: dict) -> float:
    if not constraints:
        return 1.0
    haystack: set[str] = set()
    title = str(product.get('title', '')).lower()
    haystack.update(re.findall('\\b\\w+\\b', title))
    if isinstance(detail, dict):
        for _k, vs in (detail.get('attributes') or {}).items():
            for v in vs if isinstance(vs, list) else [vs]:
                haystack.update(re.findall('\\b\\w+\\b', str(v).lower()))
        for _sid, opts in (detail.get('sku_options') or {}).items():
            if isinstance(opts, dict):
                for _k, v in opts.items():
                    haystack.update(re.findall('\\b\\w+\\b', str(v).lower()))
    matched = 0
    for _k, v in constraints.items():
        value_tokens = re.findall('\\b\\w+\\b', str(v).lower())
        if not value_tokens:
            continue
        if all((t in haystack for t in value_tokens)):
            matched += 1
    return matched / max(len(constraints), 1)

def P50_single_product_flow_elapsed_seconds() -> float:
    if P50_dialogue_run_state.pipeline_start_time <= 0:
        return 0.0
    return time.monotonic() - P50_dialogue_run_state.pipeline_start_time

def P50_single_product_may_run_probe_by_time() -> bool:
    return P50_single_product_flow_elapsed_seconds() < P50_SINGLE_PRODUCT_PROBE_MAX_ELAPSED_SECONDS

def P50_single_product_may_finalize_by_time() -> bool:
    return P50_single_product_flow_elapsed_seconds() < P50_SINGLE_PRODUCT_FINALIZE_MAX_ELAPSED_SECONDS

def P50_llm_final_judge_over_shortlisted_pool(products: list[dict], query_text: str, *, top_count: int=10, parsed_spec: dict | None=None) -> dict | None:
    if not products:
        return None
    spec = parsed_spec or {}
    pids_for_sort = [str(p.get('product_id', '') or '') for p in products if p.get('product_id')]
    details_for_sort = P50_fetch_and_cache_catalog_product_details(pids_for_sort)
    products.sort(key=lambda p: P50_composite_score(p, query_text, details_for_sort.get(str(p.get('product_id', '') or '')), spec), reverse=True)
    top = products[:top_count]
    if not top:
        return None
    pids = [str(p.get('product_id', '') or '') for p in top if p.get('product_id')]
    details = P50_fetch_and_cache_catalog_product_details(pids)
    llm = P50_llm_elect_best_listing_with_self_consistency(query_text, top, details, only_product_type=bool(spec.get('only_product_type', False)))
    if llm is not None:
        return llm
    return max(top, key=lambda p: P50_composite_score(p, query_text, details.get(str(p.get('product_id', '') or '')), spec))

def P50_llm_judge_best_from_candidate_pool(query_text: str, pool: list[dict], spec: dict) -> dict | None:
    if not pool:
        return None
    if P50_single_product_may_finalize_by_time():
        return P50_llm_final_judge_over_shortlisted_pool(pool, query_text, top_count=10, parsed_spec=spec)
    pids = [str(p.get('product_id', '') or '') for p in pool if p.get('product_id')]
    details = P50_fetch_and_cache_catalog_product_details(pids)
    q_for_heur = str(spec.get('keywords') or query_text)
    return max(pool, key=lambda p: P50_composite_score(p, q_for_heur, details.get(str(p.get('product_id', '') or '')), spec))

def P50_build_task_intro_narration_fallback(task_type: str, ctx: 'DialogueRunContext', keyword_list: list, price_list: list, service_list: list) -> str:
    base = f"Task type: {task_type}. Query (prefix): '{ctx.query[:300]}'. Parsed search keywords per product line: {keyword_list}. Parsed price_range strings: {price_list}. Parsed service filters: {service_list}. "
    if task_type == 'shop':
        return base + ' Next: same-shop flow runs per-spec catalog retrieval, `llm_score_listing_batch` thresholding, full-coverage shop detection, then Case C / anchor logic if needed.'
    if task_type == 'voucher':
        return base + ' Next: voucher flow computes `allowed_total` from discount/threshold/cap/budget, then searches price bands, scores candidates, and enforces cart window [threshold, allowed_total].'
    if task_type == 'product':
        return base + ' Next: single-product flow searches, judges, and may broaden before recommending.'
    return base

def P50_run_fast_accept_verification_probes(ctx, spec: dict, catalog_search_params: dict, best: dict, judge_relevance_score: float, unique: list[dict], seen: set[str]) -> None:
    hyde_q = P50_build_seller_vocabulary_hyde_probe_query(spec)
    if hyde_q and hyde_q != (catalog_search_params.get('q') or '').lower():
        verify_params: dict = {'q': hyde_q, 'page': 1}
        if catalog_search_params.get('price'):
            verify_params['price'] = catalog_search_params['price']
        adapt_note = f"reframed using seller-vocabulary phrasing ('{hyde_q}') to test whether alternative listing styles surface a stronger candidate the user-vocab query missed"
    elif catalog_search_params.get('service'):
        verify_params = {k: v for k, v in catalog_search_params.items() if k != 'service'}
        verify_params['page'] = 1
        adapt_note = f"dropped the service filter ('{catalog_search_params.get('service')}') to test breadth"
    else:
        q_words = (catalog_search_params.get('q') or '').replace(' only', '').split()
        if len(q_words) > 2:
            verify_params = {'q': ' '.join(q_words[:2]), 'page': 1}
            if catalog_search_params.get('price'):
                verify_params['price'] = catalog_search_params['price']
            adapt_note = f"trimmed keywords from '{catalog_search_params.get('q', '')}' to '{verify_params['q']}' for a broader semantic match"
        else:
            verify_params = {**catalog_search_params, 'page': 2}
            adapt_note = 'advanced to page 2 of the same query (single-token query ? no broader trim available)'
    rv = P50_invoke_sandbox_tool_with_gap_and_retry('find_product', verify_params)
    P50_merge_find_product_into_candidate_pool(rv, unique, seen)
    adapted_top = [{'title': r.get('title', ''), 'price': r.get('price'), 'product_id': str(r.get('product_id', '') or '')} for r in (rv or {}).get('result', [])[:3]]
    new_count = len((rv or {}).get('result', []))
    think_v = f"Fast-accept pick (pid {best.get('product_id', '')}, score {judge_relevance_score:.1f}). Verification probe: {adapt_note}. Returned {new_count} candidates; top: {adapted_top}. Pool now {len(unique)}."
    P50_append_dialogue_step_tool_results(ctx, think_v, [rv])

def P50_apply_structured_attribute_coverage_gate(spec: dict, best: dict | None, scored_candidates: list[tuple[dict, float]] | None) -> dict | None:
    constraints = spec.get('constraints') or {}
    if not (best and isinstance(constraints, dict) and (len(constraints) >= 2) and scored_candidates):
        return best
    best_pid = str(best.get('product_id', '') or '')
    best_cov = P50_weighted_constraint_coverage_score(best, P50_dialogue_run_state.product_detail_cache.get(best_pid), constraints)
    judge_now = float(best.get('_llm_relevance_score') or 0.0)
    if judge_now < 8.0 and best_cov < 0.3:
        challenger: tuple[dict, float, float] | None = None
        for cand, sc in scored_candidates[:10]:
            cand_pid = str(cand.get('product_id', '') or '')
            if cand_pid == best_pid or sc < 6.0:
                continue
            cov = P50_weighted_constraint_coverage_score(cand, P50_dialogue_run_state.product_detail_cache.get(cand_pid), constraints)
            if cov - best_cov < 0.3:
                continue
            if challenger is None or cov > challenger[1] or (cov == challenger[1] and sc > challenger[2]):
                challenger = (cand, cov, sc)
        if challenger is not None:
            return challenger[0]
    return best

def P50_run_single_product_task_branch(ctx, params: dict) -> None:
    P50_SingleProductRecommendationFlow(ctx, params).execute_recommendation_flow()

def P50_dispatch_task_to_branch_handler(ctx: 'DialogueRunContext', task_type: str, params: dict) -> None:
    try:
        P50_run_single_product_task_branch(ctx, params)
    except Exception:
        try:
            P50_finalize_dialogue_product_recommendation(ctx, [P50_NO_MATCH_PRODUCT_ID_SENTINEL], 'failure')
        except Exception:
            pass

def P50_execute_dialogue_from_parsed_parameters(ctx: 'DialogueRunContext') -> None:
    try:
        task_type = P50_classify_shopping_task_kind_from_query(ctx.query)
        params = P50_llm_parse_full_shopping_parameters(ctx.query, task_type)
        products_info = params.get('products', [])
        keyword_list = [e.get('keywords') or e.get('q', '') for e in products_info]
        price_list = [e.get('price_range') for e in products_info]
        service_list = [e.get('service') for e in products_info]
        init_fallback = P50_build_task_intro_narration_fallback(task_type, ctx, keyword_list, price_list, service_list)
        init_ctx: dict = {'keywords': keyword_list, 'price_constraints': price_list, 'service_filters': service_list}
        if products_info and bool(products_info[0].get('only_product_type')):
            init_ctx['only_product_type'] = True
            init_ctx['only_product_type_reason'] = P50_ONLY_PRODUCT_TYPE_SEARCH_NOTE
        if params.get('voucher'):
            voucher_info = params['voucher']
            init_ctx['budget_constraint'] = {'discount_type': voucher_info.get('discount_type'), 'discount_value': voucher_info.get('discount_value'), 'threshold': voucher_info.get('threshold'), 'cap': voucher_info.get('cap'), 'budget': voucher_info.get('budget')}
        P50_append_dialogue_step_tool_results(ctx, init_fallback, [])
        P50_dispatch_task_to_branch_handler(ctx, task_type, params)
    except Exception:
        try:
            P50_finalize_dialogue_product_recommendation(ctx, [P50_NO_MATCH_PRODUCT_ID_SENTINEL], 'failure')
        except Exception:
            pass

def P50_append_dialogue_step_tool_results(ctx, think: str, tool_results: list, response: str='') -> None:
    compact = [P50_compact_find_product_tool_result_for_trace(tc) for tc in tool_results or []]
    ctx.steps.append(create_dialogue_step(think, compact, response, ctx.query, len(ctx.steps) + 1))

def P50_finalize_dialogue_product_recommendation(ctx, product_ids: list, status: str, think: str='', llm_reason: str='') -> None:
    fmt_ids = P50_join_product_ids_as_csv_ordered(product_ids)
    qprev = str(getattr(ctx, 'query', '') or '')[:240]
    rec = P50_invoke_sandbox_tool_with_gap_and_retry('recommend_product', {'product_ids': fmt_ids})
    term = P50_invoke_sandbox_tool_with_gap_and_retry('terminate', {'status': status})
    if not think:
        reason_part = f'{llm_reason} ' if llm_reason else ''
        fb = f'I am recommending product(s) {fmt_ids} for the query. {reason_part}Status: {status}.'
        narrate_ctx: dict = {'recommended_product_ids': fmt_ids, 'status': status, 'note': 'Finalising recommendation and terminating the session.'}
        if llm_reason:
            narrate_ctx['llm_reason'] = llm_reason
        think = fb
    P50_append_dialogue_step_tool_results(ctx, think, [rec, term], 'Done.')

def P50_append_single_product_alternatives_step(ctx, leader: dict | None, pool: list, spec: dict | None, n_alts: int=3) -> None:
    if not leader or not pool:
        return
    lead_pid = str(leader.get('product_id', ''))
    lead_heur = P50_safe_rounded_heuristic_score_or_none(leader, ctx.query, spec)
    others = [p for p in pool if str(p.get('product_id', '')) != lead_pid]
    try:
        others = sorted(others, key=lambda p: P50_composite_score(p, ctx.query, parsed_spec=spec), reverse=True)
    except Exception:
        pass
    alts = [{'product_id': str(a.get('product_id', '')), 'title': (a.get('title') or '')[:80], 'price': a.get('price'), 'heuristic_score': P50_safe_rounded_heuristic_score_or_none(a, ctx.query, spec)} for a in others[:n_alts]]
    step_data = {'weighing': {'leader': {'product_id': lead_pid, 'title': (leader.get('title') or '')[:80], 'price': leader.get('price'), 'heuristic_score': lead_heur, 'llm_reason': leader.get('_llm_reason', ''), 'relevance_score': leader.get('_llm_relevance_score', 0)}, 'alternatives': alts}, 'query_constraints': {'keywords': (spec or {}).get('keywords'), 'price_range': (spec or {}).get('price_range'), 'service': (spec or {}).get('service')}}
    alts_fmt = ', '.join((f"pid={a['product_id']} price={a['price']} score={a['heuristic_score']}" for a in alts)) or 'none'
    outside_alt = None
    try:
        outside_alt = _oro_p50_outside_alt(spec, ctx.query, lead_pid)
    except Exception:
        outside_alt = None
    prefer = ''
    if alts or outside_alt:
        alt_pid = outside_alt.get('product_id', '') if outside_alt else None
        alt_price = outside_alt.get('price') if outside_alt else None
        alt_score = P50_safe_rounded_heuristic_score_or_none(outside_alt, ctx.query, spec) if outside_alt else None
        prefer = f" I prefer {_oro_candidate_ref(lead_pid, True)} (price={leader.get('price')}, score={lead_heur}) OVER {_oro_candidate_ref(alt_pid, False)} (price={alt_price}, score={alt_score}) because the leader has higher heuristic score and tighter alignment with the parsed query."
    fb = f"I am weighing the top candidates. The current leader is product_id={lead_pid}, price={leader.get('price')}, heuristic_score={lead_heur}. LLM reason: {leader.get('_llm_reason', '')}. Alternatives considered: {alts_fmt}.{prefer}"
    P50_append_dialogue_step_tool_results(ctx, fb, [])

@Tool
def P50_find_product(q: str, page: int=1, shop_id: str | None=None, price: str | None=None, sort: str | None=None, service: str | None=None) -> list[dict]:
    p = {'q': quote_plus(q), 'page': page, 'shop_id': shop_id, 'price': price, 'sort': sort, 'service': service}
    if p.get('sort') == 'default':
        p.pop('sort')
    norm_svc = P50_normalize_catalog_service_csv_filter(p.get('service'))
    if norm_svc is not None:
        p['service'] = norm_svc
    elif 'service' in p:
        p.pop('service')
    result = P50_catalog_http_get_rate_limit('/search/find_product', p) or []
    if not result and p.get('service'):
        retry = dict(p)
        retry.pop('service', None)
        result = P50_catalog_http_get_rate_limit('/search/find_product', retry) or []
    return result

@Tool
def P50_calculate_voucher(product_prices: str, voucher_type: str, discount_value: float, threshold: float, budget: float, cap: float=0) -> dict:
    try:
        prices = [float(x.strip()) for x in str(product_prices).split(',')]
    except ValueError:
        return {'error': 'Invalid product_prices format. Use comma-separated numbers.'}
    total = sum(prices)
    discount = 0.0
    applied = False
    if total >= threshold:
        applied = True
        if voucher_type == 'fixed':
            discount = discount_value
        elif voucher_type == 'percentage':
            discount = total * (discount_value / 100.0)
            if cap > 0:
                discount = min(discount, cap)
    final = total - discount
    out = {'prices': prices, 'total_before': round(total, 2), 'discount_amount': round(discount, 2), 'total_after': round(final, 2), 'within_budget': final <= budget, 'voucher_applied': applied, 'budget': budget}
    return out

@Tool
def P50_recommend_product(product_ids: str) -> str:
    return f'Having recommended the products to the user: {product_ids}.'

@Tool
def P50_terminate(status: str='success') -> str:
    return f'The interaction has been completed with status: {status}'

def P50_agent_main(problem_data: dict) -> list[dict]:
    ctx = P50_DialogueRunContext()

    def execute_shopping_dialogue_pipeline(ctx: 'DialogueRunContext', problem_data: dict) -> list[dict]:
        P50_dialogue_run_state.reset_for_run()
        P50_clear_thread_local_http_journal()
        ctx.steps = []
        ctx.query = problem_data.get('query', '')
        try:
            P50_execute_dialogue_from_parsed_parameters(ctx)
        except Exception:
            try:
                P50_finalize_dialogue_product_recommendation(ctx, [P50_NO_MATCH_PRODUCT_ID_SENTINEL], 'failure')
            except Exception:
                ctx.steps.append(create_dialogue_step('Done.', [], 'Done.', ctx.query, len(ctx.steps) + 1))
        if not ctx.steps:
            ctx.steps.append(create_dialogue_step('Done.', [], 'Done.', ctx.query, 1))
        P50_merge_http_journal_into_first_dialogue_step(ctx.steps)
        return ctx.steps
    return execute_shopping_dialogue_pipeline(ctx, problem_data)

def agent_main(problem_data: dict) -> list[dict]:
    problem_data = _EmptyProblemDataProcessor.ensure(problem_data)
    _oro_reset_problem()
    query = problem_data.get('query', '') if isinstance(problem_data, dict) else ''
    if _route_task_kind(query) == 'product':
        return P50_agent_main(problem_data)
    ctx = _PipeCtx()
    return run(ctx, problem_data)
