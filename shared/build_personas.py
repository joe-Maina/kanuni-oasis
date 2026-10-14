"""Build the shared persona file used by both simulators.

Inputs:
  - Qhala persona CSV (30 hand-written personas), default
    ../camel-oasis/oasis/personas/personas_v1.csv
  - personas_v1_added_fields.csv: fields the brief needs that the CSV lacks
    (connectivity, urban/rural, livelihood, education, trust, media habits).
    These are ASSUMPTIONS, recorded per persona in `assumed_fields`. Edit
    that CSV, not the output.
  - Synthetic fill: extra ordinary-public personas up to --n total
    (placeholders until Afrobarometer / KNBS sampling exists).

Columns about the AI bill (rl, rlLabel, sections, satisfaction, inclusion,
service, ecology) and 2039 projections are not carried over: they belong to a
different bill and would leak post-2024 information into the backtest.

Output: personas.json (list of objects; `agent_id` is the shared ID).
Usage: python build_personas.py [--n 100] [--seed 0]
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_SRC = HERE.parent / "camel-oasis" / "oasis" / "personas" / \
    "personas_v1.csv"
ADDED = HERE / "personas_v1_added_fields.csv"

# Synthetic fill: ordinary members of the public, mostly online, because the
# hand-written set is weighted towards officials.
FILL_CONNECTIVITY = {"heavy_online": 0.86, "light_online": 0.08,
                     "offline": 0.06}
FILL_LIVELIHOODS = [
    # (livelihood, income text, education choices, urban share)
    ("boda boda rider", "KES 10K–25K/mo", ["primary", "secondary"], 0.6),
    ("mama mboga (vegetable vendor)", "KES 6K–15K/mo",
     ["primary", "secondary"], 0.7),
    ("smallholder farmer", "KES 5K–12K/mo (seasonal)",
     ["primary", "secondary"], 0.1),
    ("informal trader", "KES 10K–30K/mo", ["primary", "secondary"], 0.7),
    ("university student", "KES 3K–10K/mo (HELB + family)", ["tertiary"],
     0.8),
    ("salaried worker", "KES 40K–120K/mo", ["secondary", "tertiary"], 0.8),
    ("casual labourer", "KES 5K–12K/mo", ["primary", "secondary"], 0.6),
    ("unemployed graduate", "KES 0–5K/mo (family support)", ["tertiary"],
     0.8),
    ("small business owner", "KES 30K–80K/mo", ["secondary", "tertiary"],
     0.7),
    ("matatu driver", "KES 15K–30K/mo", ["primary", "secondary"], 0.8),
]
COUNTIES = {
    "Nairobi": ["Kibra", "Embakasi", "Kasarani", "Roysambu", "Langata",
                "Dagoretti"],
    "Mombasa": ["Nyali", "Likoni", "Changamwe"],
    "Kisumu": ["Kisumu East", "Nyando"], "Nakuru": ["Naivasha", "Njoro"],
    "Kiambu": ["Thika", "Ruiru", "Juja"], "Machakos": ["Athi River", "Mavoko"],
    "Uasin Gishu": ["Kapseret", "Turbo"], "Kakamega": ["Lurambi", "Mumias"],
    "Kisii": ["Bobasi", "Nyaribari"], "Meru": ["Imenti North", "Buuri"],
    "Nyeri": ["Othaya", "Mathira"], "Kilifi": ["Malindi", "Kilifi North"],
    "Kajiado": ["Kitengela", "Ngong"], "Bungoma": ["Kanduyi", "Webuye"],
    "Garissa": ["Garissa Township"], "Kirinyaga": ["Kerugoya"],
}
FIRST = {
    "female": ["Wanjiru", "Akinyi", "Njeri", "Chebet", "Achieng", "Wairimu",
               "Nyambura", "Atieno", "Fatuma", "Halima", "Nafula", "Moraa",
               "Nyaboke", "Kendi", "Mumbua", "Wanjiku", "Awino", "Jebet"],
    "male": ["Otieno", "Kamau", "Mwangi", "Kiprono", "Mutua", "Omondi",
             "Kibet", "Juma", "Wekesa", "Baraka", "Ekai", "Kipchoge",
             "Odhiambo", "Njuguna", "Onyango", "Kiplagat", "Abdi", "Barasa"],
}
LAST = ["Kariuki", "Ochieng", "Wambui", "Korir", "Mutiso", "Odera", "Kimani",
        "Njoroge", "Langat", "Barasa", "Abdi", "Hassan", "Nyamweya", "Gitau",
        "Owino", "Makokha", "Lokwang", "Mwende", "Ndegwa", "Kosgei"]
ETHNICITY_BY_COUNTY = {
    "Nairobi": ["Kikuyu", "Luo", "Luhya", "Kamba", "Kalenjin", "Kisii"],
    "Mombasa": ["Mijikenda", "Swahili", "Luo", "Kamba"],
    "Kisumu": ["Luo"], "Nakuru": ["Kikuyu", "Kalenjin"],
    "Kiambu": ["Kikuyu"], "Machakos": ["Kamba"], "Uasin Gishu": ["Kalenjin"],
    "Kakamega": ["Luhya"], "Kisii": ["Kisii/Gusii"], "Meru": ["Meru"],
    "Nyeri": ["Kikuyu"], "Kilifi": ["Mijikenda"], "Kajiado": ["Maasai",
                                                             "Kikuyu"],
    "Bungoma": ["Luhya/Bukusu"], "Garissa": ["Somali-Kenyan"],
    "Kirinyaga": ["Kikuyu"],
}
MEDIA = {
    "heavy_online": [["x_twitter", "tiktok", "whatsapp"],
                     ["x_twitter", "whatsapp", "youtube"],
                     ["tiktok", "instagram", "whatsapp"],
                     ["x_twitter", "facebook", "whatsapp"]],
    "light_online": [["whatsapp", "radio"], ["whatsapp", "tv"]],
    "offline": [["radio"], ["radio", "tv"]],
}


def age_group(age: int) -> str:
    if age < 18:
        return "under_18"
    for lo, hi in [(18, 24), (25, 34), (35, 44), (45, 54)]:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "55+"


def income_band(text: str) -> str:
    """Band by the lower bound of the monthly KES range."""
    m = re.search(r"KES\s*([\d.]+)\s*K", text or "")
    if not m:
        return "dependent" if "N/A" in (text or "") else "unknown"
    k = float(m.group(1))
    if k < 10:
        return "very_low"
    if k < 30:
        return "low"
    if k < 100:
        return "middle"
    if k < 300:
        return "upper_middle"
    return "high"


def num(x):
    try:
        return float(x) if "." in x else int(x)
    except (TypeError, ValueError):
        return None


def from_source(row: dict, added: dict) -> dict:
    name = row["name"]
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return {
        "agent_id": int(row["id"]),
        "name": name,
        "user_name": f"{slug}_{row['id']}",
        "age": int(row["age"]),
        "age_group": age_group(int(row["age"])),
        "gender": row["gender"].lower(),
        "county": row["county"],
        "sub_county": row["subCounty"],
        "urban_rural": added["urban_rural"],
        "livelihood": added["livelihood"],
        "role": row["role"],
        "ethnicity": row["eth"],
        "income": row["income"],
        "income_band": income_band(row["income"]),
        "education": added["education"],
        "trust_in_government": added["trust_in_government"],
        "media_habits": added["media_habits"].split(";"),
        "connectivity": added["connectivity"],
        "archetype": row["archetype"],
        "pop2024": num(row["pop2024"]),
        "pct2024": num(row["pct2024"]),
        "source": "qhala_personas_v1",
        "assumed_fields": ["urban_rural", "livelihood", "education",
                           "trust_in_government", "media_habits",
                           "connectivity"],
    }


def synthetic(rng: random.Random, agent_id: int, connectivity: str) -> dict:
    gender = rng.choice(["female", "male"])
    livelihood, income, edu, urban_share = rng.choice(FILL_LIVELIHOODS)
    age = rng.randint(19, 26) if "student" in livelihood else rng.randint(
        18, 58)
    county = rng.choice(list(COUNTIES))
    first, last = rng.choice(FIRST[gender]), rng.choice(LAST)
    urban = county == "Nairobi" or rng.random() < urban_share
    return {
        "agent_id": agent_id,
        "name": f"{first} {last}",
        "user_name": f"{first.lower()}_{last.lower()}_{agent_id}",
        "age": age,
        "age_group": age_group(age),
        "gender": gender,
        "county": county,
        "sub_county": rng.choice(COUNTIES[county]),
        "urban_rural": "urban" if urban else "rural",
        "livelihood": livelihood,
        "role": livelihood,
        "ethnicity": rng.choice(ETHNICITY_BY_COUNTY[county]),
        "income": income,
        "income_band": income_band(income),
        "education": rng.choice(edu),
        "trust_in_government": rng.choices(["low", "medium", "high"],
                                           [0.5, 0.35, 0.15])[0],
        "media_habits": rng.choice(MEDIA[connectivity]),
        "connectivity": connectivity,
        "archetype": None,
        "pop2024": None,
        "pct2024": None,
        "source": "synthetic_fill",
        "assumed_fields": ["all"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "personas.json"))
    a = p.parse_args()

    with open(ADDED, newline="", encoding="utf-8") as f:
        added = {int(r["id"]): r for r in csv.DictReader(f)}
    with open(a.src, newline="", encoding="utf-8") as f:
        src = list(csv.DictReader(f))
    missing = [r["id"] for r in src if int(r["id"]) not in added]
    if missing:
        raise SystemExit(f"No added fields for persona ids {missing}; "
                         f"add rows to {ADDED.name}.")

    personas = [from_source(r, added[int(r["id"])]) for r in src]
    rng = random.Random(a.seed)
    next_id = max(p["agent_id"] for p in personas) + 1
    tags = list(FILL_CONNECTIVITY)
    weights = list(FILL_CONNECTIVITY.values())
    while len(personas) < a.n:
        personas.append(synthetic(rng, next_id,
                                  rng.choices(tags, weights)[0]))
        next_id += 1

    Path(a.out).write_text(json.dumps(personas, indent=2,
                                      ensure_ascii=False) + "\n")
    counts = {}
    for p_ in personas:
        counts[p_["connectivity"]] = counts.get(p_["connectivity"], 0) + 1
    print(f"wrote {len(personas)} personas to {a.out}; connectivity {counts}")


if __name__ == "__main__":
    main()
