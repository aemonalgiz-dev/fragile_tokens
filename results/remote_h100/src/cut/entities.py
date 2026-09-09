"""Build a bank of REAL multi-token entities the model must reproduce exactly.

The hallucination bridge asks whether token-level fragility predicts the
plausible-near-miss hallucination on rare entities -- the fabricated API name
that almost exists, the author with one syllable swapped. That needs entities
that (a) are real and were in the training distribution, (b) span a wide rarity
range, and (c) are multi-token, so constituent fragility is something to
aggregate.

Code identifiers come from the live interpreter -- Python's standard library,
torch, transformers -- so every one is a real API. Proper nouns are a fixed list
of real places, people and substances across scripts. Each entity is tokenised
exactly as it will appear in the bridge prompt (code inside backticks, no
leading space; nouns preceded by a space) so the constituent ids match.
"""
from __future__ import annotations
import argparse, json, random, sys, importlib, re
from pathlib import Path
from transformers import AutoTokenizer

STDLIB_MODULES = ["os", "os.path", "sys", "re", "json", "itertools", "functools",
                  "collections", "subprocess", "pathlib", "shutil", "datetime", "math",
                  "random", "string", "typing", "asyncio", "threading", "logging",
                  "argparse", "unittest", "socket", "struct", "hashlib", "base64",
                  "urllib.parse", "http.client", "email", "csv", "sqlite3", "statistics",
                  "decimal", "fractions", "heapq", "bisect", "textwrap", "difflib",
                  "inspect", "dataclasses", "contextlib", "operator", "codecs", "gzip",
                  "zipfile", "tarfile", "tempfile", "glob", "fnmatch", "pickle", "copy",
                  "enum", "abc", "queue", "signal", "select", "mmap", "ctypes", "array",
                  "weakref", "traceback", "warnings", "locale", "gettext", "calendar",
                  "time", "zoneinfo", "uuid", "secrets", "hmac", "ssl", "ipaddress",
                  "xml.etree.ElementTree", "html", "html.parser", "webbrowser", "cmd",
                  "shlex", "pprint", "reprlib", "numbers", "cmath", "io", "tokenize",
                  "ast", "dis", "importlib", "pkgutil", "platform", "errno", "getpass",
                  "curses", "concurrent.futures", "multiprocessing", "sched", "graphlib"]

PROPER_NOUNS = [
    "Ouagadougou", "Thiruvananthapuram", "Antananarivo", "Ulaanbaatar", "Reykjavík",
    "Tbilisi", "Ljubljana", "Bratislava", "Vientiane", "Nouakchott", "Bujumbura",
    "Lilongwe", "Paramaribo", "Tegucigalpa", "Ashgabat", "Bishkek", "Dushanbe",
    "Podgorica", "Mbabane", "Yamoussoukro", "Kinshasa", "Brazzaville", "Windhoek",
    "Gaborone", "Maseru", "Moroni", "Funafuti", "Nukuʻalofa", "Port Moresby",
    "Honiara", "Tarawa", "Majuro", "Ngerulmud", "Palikir", "Wellington", "Canberra",
    "Kraków", "Wrocław", "Gdańsk", "Poznań", "Szczecin", "Białystok", "Częstochowa",
    "Düsseldorf", "Nürnberg", "Würzburg", "Göttingen", "Tübingen", "Saarbrücken",
    "Malmö", "Göteborg", "Linköping", "Jönköping", "Umeå", "Tromsø", "Ålesund",
    "Odense", "Aalborg", "Esbjerg", "Tampere", "Jyväskylä", "Kuopio", "Rovaniemi",
    "Zaragoza", "Málaga", "Córdoba", "Alicante", "Valladolid", "Logroño", "Badajoz",
    "Marseille", "Toulouse", "Strasbourg", "Montpellier", "Grenoble", "Besançon",
    "Perugia", "Cagliari", "Trieste", "Brescia", "Bergamo", "Reggio Emilia",
    "Coimbra", "Braga", "Aveiro", "Évora", "Funchal", "Guimarães",
    "Thessaloniki", "Heraklion", "Ioannina", "Volos", "Patras", "Kalamata",
    "Novosibirsk", "Yekaterinburg", "Vladivostok", "Krasnoyarsk", "Arkhangelsk",
    "Nietzsche", "Tchaikovsky", "Dostoevsky", "Kierkegaard", "Wittgenstein",
    "Schopenhauer", "Rachmaninoff", "Shostakovich", "Mussorgsky", "Prokofiev",
    "Sibelius", "Dvořák", "Janáček", "Smetana", "Szymanowski", "Lutosławski",
    "Chandrasekhar", "Ramanujan", "Raman", "Bose", "Saha", "Bhabha",
    "Michelangelo", "Caravaggio", "Botticelli", "Brunelleschi", "Donatello",
    "acetaminophen", "ibuprofen", "amoxicillin", "azithromycin", "ciprofloxacin",
    "metformin", "atorvastatin", "lisinopril", "omeprazole", "levothyroxine",
    "amlodipine", "hydrochlorothiazide", "prednisone", "gabapentin", "sertraline",
    "Tyrannosaurus", "Velociraptor", "Archaeopteryx", "Pachycephalosaurus",
    "Ankylosaurus", "Parasaurolophus", "Brachiosaurus", "Diplodocus",
    "Kilimanjaro", "Aconcagua", "Denali", "Elbrus", "Vinson", "Puncak Jaya",
    "Kangchenjunga", "Dhaulagiri", "Annapurna", "Nanga Parbat", "Manaslu",
]


def public_names(modname):
    try:
        m = importlib.import_module(modname)
    except Exception:
        return []
    out = []
    for n in dir(m):
        if n.startswith("_") or len(n) < 4:
            continue
        out.append(f"{modname}.{n}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--per-bucket", type=int, default=220,
                    help="code entities per token-count bucket (2..6)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/entities.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]

    code = set()
    for m in STDLIB_MODULES:
        code.update(public_names(m))
    code.update(f"{m}" for m in getattr(sys, "stdlib_module_names", []))
    for pkg in ("torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.utils.data",
                "torch.distributions", "transformers", "numpy", "numpy.linalg", "numpy.random"):
        code.update(public_names(pkg))
    code = [c for c in code if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", c) and 4 <= len(c) <= 48]
    print(f"code identifiers collected: {len(code)}")

    items = []
    buckets = {k: [] for k in range(2, 7)}
    for c in code:
        ids = enc(c)
        if 2 <= len(ids) <= 6:
            buckets[len(ids)].append((c, ids))
    for k, lst in buckets.items():
        rng.shuffle(lst)
        for c, ids in lst[:a.per_bucket]:
            src = c.split(".")[0]
            items.append({"entity": c, "type": "code", "source": src, "ids": ids,
                          "prompt_form": c})
    print("code entities by token count: " + ", ".join(
        f"{k}: {min(len(v), a.per_bucket)}" for k, v in buckets.items()))

    n_noun = 0
    for nme in PROPER_NOUNS:
        ids = enc(" " + nme)
        if 2 <= len(ids) <= 8:
            items.append({"entity": nme, "type": "noun", "source": "noun", "ids": ids,
                          "prompt_form": " " + nme})
            n_noun += 1
    print(f"proper nouns kept (2-8 tokens): {n_noun} of {len(PROPER_NOUNS)}")

    all_ids = sorted({i for it in items for i in it["ids"]})
    print(f"entities: {len(items)}; distinct constituent tokens: {len(all_ids)}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "items": items, "token_ids": all_ids},
              open(a.out, "w"), indent=1)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
