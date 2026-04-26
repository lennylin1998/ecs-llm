"""
Responsibility: load the raw twcs.csv, reconstruct customer→brand reply threads,
and emit a clean DataFrame of *single-turn* complaint-response pairs.

"Single-turn" means the customer tweet is the FIRST message in the exchange —
not a follow-up to an earlier brand reply.  This matters because follow-up
customer tweets carry different frustration signals (already partially resolved,
context-dependent) that would contaminate the FIS scoring in Step 3.

Public API (everything else is implementation detail):
    build_pairs(csv_path) -> pd.DataFrame

Output schema
─────────────
pair_id             str   tweet_id of the agent reply (unique per pair)
domain              str   "airline" | "technology"
customer_text       str   raw tweet text from the customer
agent_text          str   raw tweet text from the brand agent
customer_tweet_id   str   tweet_id of the customer complaint
agent_tweet_id      str   tweet_id of the brand reply (== pair_id)
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Domain configuration
#
# Handles were determined by inspecting all distinct outbound author_ids in the
# dataset (2.8 M tweets) and keeping only those that unambiguously belong to
# one of the two target domains.  Ambiguous handles were excluded.
#
# "technology" is a broad domain covering both telecom carriers and consumer
# tech/platform companies (Apple, Amazon).  They share a common pattern:
# high-volume transactional support where tone calibration is commercially
# significant, making them a coherent contrast class against airlines.
# ---------------------------------------------------------------------------
AIRLINE_HANDLES: frozenset[str] = frozenset({
    "AmericanAir",     # American Airlines
    "Delta",           # Delta Air Lines
    "British_Airways", # British Airways
    "SouthwestAir",    # Southwest Airlines
    "JetBlue",         # JetBlue Airways
    "AlaskaAir",       # Alaska Airlines
    "VirginAtlantic",  # Virgin Atlantic
    "VirginAmerica",   # Virgin America (acquired by Alaska; historical tweets remain)
})

# Telecom carriers + major consumer tech platforms grouped as "technology".
# Rationale: both sub-groups are high-volume, transactional, digitally native
# support contexts — the tonal norms are more similar to each other than to airlines.
TECHNOLOGY_HANDLES: frozenset[str] = frozenset({
    # Telecom / cable / internet
    "comcastcares",    # Comcast
    "TMobileHelp",     # T-Mobile
    "Ask_Spectrum",    # Charter Spectrum
    "sprintcare",      # Sprint
    "VerizonSupport",  # Verizon
    "ATT",             # AT&T
    "CoxHelp",         # Cox Communications
    "CenturyLinkHelp", # CenturyLink
    # Consumer tech & platforms
    "AmazonHelp",      # Amazon customer support
    "AppleSupport",    # Apple Support
})

ALL_HANDLES: frozenset[str] = AIRLINE_HANDLES | TECHNOLOGY_HANDLES


# ---------------------------------------------------------------------------
# Dataset acquisition
# ---------------------------------------------------------------------------

KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"
_CSV_SUBPATH   = "twcs/twcs.csv"   # relative path inside the kagglehub cache dir


def get_csv_path() -> Path:
    """
    Return a local Path to twcs.csv, downloading via kagglehub if not already cached.

    kagglehub stores the download under ~/.cache/kagglehub, so repeated calls
    are instant — it only fetches if the version is missing or outdated.
    No manual Kaggle API key setup is needed for public datasets.
    """
    import kagglehub
    base = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    return base / _CSV_SUBPATH


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Read twcs.csv with all columns as strings to prevent tweet-ID precision loss.

    The 'inbound' column appears as the Python bool True/False after CSV round-trips,
    but also as the string "True"/"False" depending on how the file was exported.
    We normalise both forms here so the rest of the pipeline can use plain bool logic.
    """
    df = pd.read_csv(csv_path, dtype=str)

    bool_map = {"True": True, "False": False, "true": True, "false": False}
    df["inbound"] = df["inbound"].map(bool_map)

    # Replace pandas NaN-strings so downstream .get() checks work cleanly
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].where(
        df["in_response_to_tweet_id"].notna() & (df["in_response_to_tweet_id"] != "nan"),
        other=None,
    )

    return df


def _build_tweet_index(df: pd.DataFrame) -> dict[str, dict]:
    """
    Build an O(1) lookup table: tweet_id → row dict.

    We need this to trace parent chains without repeated DataFrame scans.
    Using a plain dict (not a second DataFrame join) keeps the hot loop fast
    on a 3M-row dataset.
    """
    return df.set_index("tweet_id").to_dict("index")


# ---------------------------------------------------------------------------
# Single-turn classification
# ---------------------------------------------------------------------------

def _is_first_contact(customer_tweet: dict, tweet_index: dict[str, dict]) -> bool:
    """
    Return True if the customer tweet is the opening message of an exchange.

    Strategy: walk one step up the customer tweet's parent chain.
    - If the parent does not exist or is itself inbound (customer), the
      customer tweet started the thread → single-turn ✓
    - If the parent is an outbound (brand) tweet, the customer tweet is a
      follow-up reply to a prior brand message → multi-turn, discard ✗

    We only need to look one level up because the brand tweet we are
    evaluating is already the *direct* reply to this customer tweet.
    The question is only whether *this customer tweet* was first contact.
    """
    parent_id = customer_tweet.get("in_response_to_tweet_id")

    if not parent_id or parent_id not in tweet_index:
        # No traceable parent → opening tweet
        return True

    parent = tweet_index[parent_id]

    # If the parent is outbound (brand), this customer tweet is a follow-up
    return bool(parent.get("inbound", True))


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def build_pairs(csv_path: str | Path) -> pd.DataFrame:
    """
    Load twcs.csv and return a DataFrame of single-turn complaint-response pairs.

    Extraction logic
    ────────────────
    1. Index all tweets for O(1) parent lookup.
    2. Iterate over outbound (brand) tweets only — these are the agent side.
    3. For each brand tweet, verify:
       a. The brand is one of the 7 target handles.
       b. It has a traceable parent customer tweet (inbound=True).
       c. The customer tweet is first-contact (not a follow-up reply).
       d. Neither text is empty/null.
    4. Assign domain label from handle membership.
    5. Deduplicate on customer_tweet_id: if two brand agents replied to the
       same complaint, keep only the first reply (lowest tweet_id sort order)
       to avoid double-counting the same complaint.

    Parameters
    ----------
    csv_path : path to twcs.csv (the raw Kaggle download)

    Returns
    -------
    pd.DataFrame with columns: pair_id, domain, customer_text, agent_text,
                                customer_tweet_id, agent_tweet_id
    """
    df = load_csv(csv_path)
    tweet_index = _build_tweet_index(df)

    # Only outbound rows can be agent replies
    outbound = df[df["inbound"] == False].copy()  # noqa: E712

    records: list[dict] = []

    for _, agent_row in outbound.iterrows():
        handle = str(agent_row.get("author_id", "")).strip()

        # (a) Target brand only
        if handle not in ALL_HANDLES:
            continue

        parent_id = agent_row.get("in_response_to_tweet_id")
        if not parent_id or parent_id not in tweet_index:
            continue

        customer_tweet = tweet_index[parent_id]

        # (b) Parent must be inbound (customer)
        if not customer_tweet.get("inbound", False):
            continue

        # (c) Customer tweet must be first contact
        if not _is_first_contact(customer_tweet, tweet_index):
            continue

        # (d) Both texts must be non-empty
        customer_text = str(customer_tweet.get("text", "")).strip()
        agent_text = str(agent_row.get("text", "")).strip()
        if not customer_text or not agent_text:
            continue

        domain = "airline" if handle in AIRLINE_HANDLES else "technology"

        records.append(
            {
                "pair_id": str(agent_row["tweet_id"]),
                "domain": domain,
                "customer_text": customer_text,
                "agent_text": agent_text,
                "customer_tweet_id": str(parent_id),
                "agent_tweet_id": str(agent_row["tweet_id"]),
            }
        )

    pairs = pd.DataFrame(records)

    # (5) Deduplicate: keep first brand reply per unique customer tweet
    # Sort by agent_tweet_id (ascending) so the earliest reply wins
    pairs = (
        pairs.sort_values("agent_tweet_id")
        .drop_duplicates(subset="customer_tweet_id", keep="first")
        .reset_index(drop=True)
    )

    return pairs


# ---------------------------------------------------------------------------
# CLI convenience
#   With argument:    python -m src.pair_builder path/to/twcs.csv
#   Without argument: python -m src.pair_builder   (auto-downloads via kagglehub)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        print("No CSV path provided — downloading dataset via kagglehub …")
        csv_path = get_csv_path()

    out_path = Path("data/pairs_raw.csv")
    out_path.parent.mkdir(exist_ok=True)

    print(f"Loading {csv_path} …")
    pairs = build_pairs(csv_path)

    print(f"\nExtracted {len(pairs):,} single-turn pairs")
    print(pairs.groupby("domain").size().to_string())

    pairs.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
