"""
build_jmdict_db.py — Phase 3.1 data pipeline
Parses JMdict XML + kanjium pitch accent data + Waller JLPT word lists
and writes a merged CSV ready for load_jmdict_db.py.

Run from repo root:
    python scripts/build_jmdict_db.py \
        --jmdict path/to/JMdict_e.xml \
        --kanjium path/to/kanjium_accents.txt \
        --waller-dir path/to/waller_jlpt/ \
        --output data/jmdict_merged.csv

Input file expectations:
    JMdict XML:
        Standard JMdict distribution from https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project
        Filename: JMdict_e (English glosses only, smaller) or JMdict (all languages).
        Use JMdict_e — ~170k entries, English only.

    Kanjium accents:
        From https://github.com/mifunetoshiro/kanjium  (data/accents.txt)
        Format: word TAB reading TAB downstep_number
        Example:
            食べる\tたべる\t2
        Downstep 0 = heiban (flat). Downstep N = drop after mora N.

    Waller JLPT directory:
        From http://www.tanos.co.uk/jlpt/ — one file per level.
        Expected filenames in --waller-dir:
            N5.txt, N4.txt, N3.txt, N2.txt, N1.txt
        Each file: one word per line (kanji form or kana if no kanji).

Output CSV columns (matches Supabase jmdict_entries schema):
    id, text, reading, jlpt_level, pitch_accent, meanings, parts_of_speech

    meanings and parts_of_speech are JSON arrays (strings).
    pitch_accent is a simplified LH contour string, e.g. 'LHL', 'LHHL'.
    jlpt_level is 'N5'|'N4'|'N3'|'N2'|'N1'|'' (empty = not in any list).
"""

import argparse
import csv
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JLPT_LEVELS = ("N1", "N2", "N3", "N4", "N5")
# Waller-style .txt filenames (fallback — often Word docs, unreliable)
WALLER_FILENAMES = {level: f"{level}.txt" for level in JLPT_LEVELS}
# davidluzgouveia JSON filenames
JLPT_JSON_FILENAMES = {
    "N1": "jlpt_n1.json",
    "N2": "jlpt_n2.json",
    "N3": "jlpt_n3.json",
    "N4": "jlpt_n4.json",
    "N5": "jlpt_n5.json",
}
# Bluskyo CSV filenames (https://github.com/Bluskyo/JLPT_Vocabulary)
# Format: kanji,reading  (one word per row, header row "Kanji,Reading")
JLPT_CSV_FILENAMES = {
    "N1": "n1_vocab_cleaned.csv",
    "N2": "n2_vocab_cleaned.csv",
    "N3": "n3_vocab_cleaned.csv",
    "N4": "n4_vocab_cleaned.csv",
    "N5": "n5_vocab_cleaned.csv",
}
# stephenmk combined vocab file (https://github.com/stephenmk/yomitan-jlpt-vocab)
# Format: Kanji,Reading,Level  where Level is integer 1-5
STEPHENMK_COMBINED_CSV = "JLPT_vocab_ALL.csv"
STEPHENMK_LEVEL_MAP = {"1": "N1", "2": "N2", "3": "N3", "4": "N4", "5": "N5"}

# Parts-of-speech entity abbreviations used in JMdict XML.
# The XML uses SGML entities like &n; &v1; etc. — ElementTree expands these
# if the DTD is loaded, but when parsing without DTD they survive as text.
# We normalise here to human-readable strings.
POS_ENTITY_MAP: dict[str, str] = {
    "n": "noun",
    "v1": "ichidan verb",
    "v5u": "godan verb (u)",
    "v5k": "godan verb (ku)",
    "v5g": "godan verb (gu)",
    "v5s": "godan verb (su)",
    "v5t": "godan verb (tsu)",
    "v5n": "godan verb (nu)",
    "v5b": "godan verb (bu)",
    "v5m": "godan verb (mu)",
    "v5r": "godan verb (ru)",
    "v5aru": "godan verb (aru, special)",
    "vk": "kuru verb (irregular)",
    "vs-i": "suru verb (irregular)",
    "vs-s": "suru verb (special class)",
    "vs": "noun or participle: takes suru",
    "adj-i": "i-adjective",
    "adj-na": "na-adjective",
    "adj-no": "no-adjective",
    "adv": "adverb",
    "adv-to": "adverb (to)",
    "exp": "expression",
    "int": "interjection",
    "conj": "conjunction",
    "prt": "particle",
    "pref": "prefix",
    "suf": "suffix",
    "aux": "auxiliary",
    "aux-v": "auxiliary verb",
    "aux-adj": "auxiliary adjective",
    "num": "numeric",
    "ctr": "counter",
    "pn": "pronoun",
}

# ---------------------------------------------------------------------------
# Data classes (no Pydantic dependency — pipeline is standalone)
# ---------------------------------------------------------------------------


class JmdictRawEntry:
    """Minimal data extracted from one JMdict <entry>."""

    __slots__ = ("seq_id", "kanji_forms", "reading_forms", "meanings", "pos_tags")

    def __init__(
        self,
        seq_id: str,
        kanji_forms: list[str],
        reading_forms: list[str],
        meanings: list[str],
        pos_tags: list[str],
    ) -> None:
        self.seq_id = seq_id
        self.kanji_forms = kanji_forms
        self.reading_forms = reading_forms
        self.meanings = meanings
        self.pos_tags = pos_tags

    @property
    def primary_text(self) -> str:
        """Kanji form if present, otherwise first reading."""
        return self.kanji_forms[0] if self.kanji_forms else self.reading_forms[0]

    @property
    def primary_reading(self) -> str:
        return self.reading_forms[0] if self.reading_forms else ""


class MergedEntry:
    """Final merged row ready for CSV output."""

    __slots__ = (
        "id",
        "text",
        "reading",
        "jlpt_level",
        "pitch_accent",
        "meanings",
        "parts_of_speech",
    )

    def __init__(
        self,
        id: str,
        text: str,
        reading: str,
        jlpt_level: str | None,
        pitch_accent: str | None,
        meanings: list[str],
        parts_of_speech: list[str],
    ) -> None:
        self.id = id
        self.text = text
        self.reading = reading
        self.jlpt_level = jlpt_level
        self.pitch_accent = pitch_accent
        self.meanings = meanings
        self.parts_of_speech = parts_of_speech

    def to_csv_row(self) -> dict[str, str]:
        return {
            "id": self.id,
            "text": self.text,
            "reading": self.reading,
            "jlpt_level": self.jlpt_level or "",
            "pitch_accent": self.pitch_accent or "",
            "meanings": json.dumps(self.meanings, ensure_ascii=False),
            "parts_of_speech": json.dumps(self.parts_of_speech, ensure_ascii=False),
        }


# ---------------------------------------------------------------------------
# JMdict XML parser
# ---------------------------------------------------------------------------


def _normalise_pos(raw: str) -> str:
    """
    JMdict XML encodes POS as SGML entities. When parsed without the DTD,
    ElementTree returns the raw entity name (e.g. 'n', 'v1').
    Map to a human-readable string; fall back to raw value if unknown.
    """
    cleaned = raw.strip().lstrip("&").rstrip(";")
    return POS_ENTITY_MAP.get(cleaned, cleaned)


def parse_jmdict_xml(xml_path: Path) -> Iterator[JmdictRawEntry]:
    """
    Stream-parse JMdict XML using iterparse to keep memory low.
    Yields one JmdictRawEntry per <entry>.

    Why iterparse: JMdict_e is ~70 MB uncompressed. Loading the full tree
    into memory at once works but is wasteful. iterparse lets us process
    and discard each <entry> element immediately.
    """
    log.info("Parsing JMdict XML: %s", xml_path)
    entry_count = 0
    context = ET.iterparse(str(xml_path), events=("end",))

    current: dict = {}

    for event, elem in context:
        tag = elem.tag

        if tag == "ent_seq":
            current["seq_id"] = (elem.text or "").strip()
        elif tag == "keb":
            current.setdefault("kanji_forms", []).append((elem.text or "").strip())
        elif tag == "reb":
            current.setdefault("reading_forms", []).append((elem.text or "").strip())
        elif tag == "gloss":
            # Only include English glosses (xml:lang="eng" or absent).
            lang = elem.get("{http://www.w3.org/XML/1998/namespace}lang", "eng")
            if lang == "eng":
                text = (elem.text or "").strip()
                if text:
                    current.setdefault("meanings", []).append(text)
        elif tag == "pos":
            raw = (elem.text or "").strip()
            normalised = _normalise_pos(raw)
            current.setdefault("pos_tags", []).append(normalised)
        elif tag == "entry":
            seq_id = current.get("seq_id", "")
            kanji_forms = current.get("kanji_forms", [])
            reading_forms = current.get("reading_forms", [])
            meanings = current.get("meanings", [])
            pos_tags = list(dict.fromkeys(current.get("pos_tags", [])))  # dedupe, keep order

            if seq_id and reading_forms and meanings:
                yield JmdictRawEntry(
                    seq_id=seq_id,
                    kanji_forms=kanji_forms,
                    reading_forms=reading_forms,
                    meanings=meanings,
                    pos_tags=pos_tags,
                )
                entry_count += 1

            current = {}
            elem.clear()  # free memory immediately

    log.info("JMdict parse complete: %d entries yielded", entry_count)


# ---------------------------------------------------------------------------
# Kanjium pitch accent parser
# ---------------------------------------------------------------------------


def _downstep_to_lh_string(reading: str, downstep: int) -> str:
    """
    Convert a kanjium downstep number to a simplified LH contour string.

    Pitch accent rules (Tokyo dialect):
        Mora 1 is always the opposite pitch of mora 2.
        Downstep N means mora N is high and mora N+1 drops to low.
        Downstep 0 (heiban) = rises after mora 1, stays high, no drop.

    We approximate the mora count using the reading length.
    Compound morae (っ, long vowels ー) count as 1 mora each.

    Examples:
        食べる (taberu, 4 morae) downstep 2 → LHLL
        雨 (ame, 2 morae) downstep 1 → HL
        橋 (hashi, 3 morae) downstep 0 → LHH (heiban)
        箸 (hashi, 3 morae) downstep 2 → LHL
    """
    # Approximate mora count from the reading string.
    # Each character is one mora (imprecise for okurigana but fine for display).
    mora_count = len(reading) if reading else 1
    mora_count = max(mora_count, 1)

    if downstep == 0:
        # Heiban: L followed by all H
        pattern = "L" + "H" * (mora_count - 1)
    elif downstep == 1:
        # Atamadaka: H followed by all L
        pattern = "H" + "L" * (mora_count - 1)
    else:
        # Nakadaka / odaka: L, then H up to downstep, then L after
        highs = min(downstep, mora_count) - 1
        lows = mora_count - 1 - highs
        pattern = "L" + "H" * highs + "L" * lows

    return pattern


def load_kanjium_accents(kanjium_path: Path) -> dict[str, str]:
    """
    Load kanjium accents.txt into a lookup dict: {word: lh_pattern}.
    Key is the word (kanji form if present).

    File format (tab-separated, no header):
        word TAB reading TAB downstep
    Lines beginning with # are comments.

    Returns a dict. When multiple entries exist for the same word,
    the first is kept (most common form first in kanjium).
    """
    log.info("Loading kanjium pitch accent data: %s", kanjium_path)
    accents: dict[str, str] = {}
    skipped = 0

    with kanjium_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                log.debug("kanjium line %d: unexpected format, skipping: %r", lineno, line)
                skipped += 1
                continue

            word, reading, downstep_raw = parts[0], parts[1], parts[2]
            try:
                downstep = int(downstep_raw.strip())
            except ValueError:
                log.debug("kanjium line %d: non-integer downstep %r, skipping", lineno, downstep_raw)
                skipped += 1
                continue

            if word not in accents:
                accents[word] = _downstep_to_lh_string(reading, downstep)

    log.info(
        "kanjium load complete: %d entries loaded, %d lines skipped",
        len(accents),
        skipped,
    )
    return accents


# ---------------------------------------------------------------------------
# Waller JLPT list parser
# ---------------------------------------------------------------------------


_OLE2_MAGIC = b"\xd0\xcf\x11\xe0"  # OLE2 Compound Document header (old .xls / .doc)


def _is_ole2(filepath: Path) -> bool:
    """Return True if the file starts with the OLE2 magic bytes."""
    with filepath.open("rb") as fh:
        return fh.read(4) == _OLE2_MAGIC


def _read_xls_first_column(filepath: Path) -> list[str]:
    """
    Read the first column of the first sheet of an old-format .xls file.
    Waller JLPT files from tanos.co.uk are Excel 97-2003 files renamed to .txt.
    Requires: pip install xlrd
    """
    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError(
            f"{filepath.name} is an Excel file but xlrd is not installed.\n"
            "Run: pip install xlrd"
        ) from exc

    book = xlrd.open_workbook(str(filepath))
    sheet = book.sheet_by_index(0)
    lines = []
    for row_idx in range(sheet.nrows):
        cell = sheet.cell(row_idx, 0)
        value = str(cell.value).strip()
        if value:
            # xlrd returns floats for numeric cells — strip trailing .0
            if value.endswith(".0") and value[:-2].isdigit():
                value = value[:-2]
            lines.append(value)
    return lines


def _detect_and_read_lines(filepath: Path) -> list[str]:
    """
    Read a word-list file regardless of format:
      - OLE2 binary (.xls renamed to .txt) parsed via xlrd
      - UTF-8, UTF-8-BOM, UTF-16 text decoded normally
      - Latin-1 fallback (always succeeds)

    Waller JLPT files from tanos.co.uk are Excel 97-2003 binaries
    saved with a .txt extension, so OLE2 detection must come first.
    """
    if _is_ole2(filepath):
        return _read_xls_first_column(filepath)

    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return filepath.read_text(encoding=encoding).splitlines()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Could not decode {filepath} with any known encoding")


def _load_stephenmk_combined_csv(filepath: Path) -> dict[str, str]:
    """
    Load the stephenmk JLPT_vocab_ALL.csv combined vocabulary file.
    Format: Kanji,Reading,Level  (header row, Level is integer 1-5)
    Returns {word: jlpt_level} directly — handles all levels in one pass.
    https://github.com/stephenmk/yomitan-jlpt-vocab
    """
    import csv as _csv
    word_to_level: dict[str, str] = {}
    with filepath.open(encoding="utf-8-sig", newline="") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            kanji = (row.get("Kanji") or "").strip()
            reading = (row.get("Reading") or "").strip()
            level_raw = (row.get("Level") or "").strip()
            level = STEPHENMK_LEVEL_MAP.get(level_raw)
            if not level:
                continue
            word = kanji if kanji else reading
            if word:
                word_to_level[word] = level
    return word_to_level


def _load_jlpt_csv(filepath: Path) -> list[str]:
    """
    Load JLPT word list from a Bluskyo-format CSV file.
    Format: two columns (Kanji, Reading), optional header row.
    https://github.com/Bluskyo/JLPT_Vocabulary
    Returns a list of word strings (kanji column; falls back to reading if blank).
    """
    import csv as _csv
    words = []
    with filepath.open(encoding="utf-8-sig", newline="") as fh:
        reader = _csv.reader(fh)
        for row in reader:
            if not row:
                continue
            kanji = row[0].strip()
            # Skip header row
            if kanji.lower() in ("kanji", "word", "vocab", "vocabulary", ""):
                reading = row[1].strip() if len(row) > 1 else ""
                if not reading or reading.lower() in ("reading", "kana", "furigana"):
                    continue
            word = kanji if kanji else (row[1].strip() if len(row) > 1 else "")
            if word:
                words.append(word)
    return words


def _load_jlpt_json(filepath: Path, level: str) -> list[str]:
    """
    Load JLPT word list from a davidluzgouveia-format JSON file.
    Format: array of objects, each with a "word" key.
    Example: [{"word": "食べる", "meaning": "to eat", ...}, ...]
    Returns a list of word strings.
    """
    import json as _json
    data = _json.loads(filepath.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{filepath.name}: expected a JSON array, got {type(data)}")
    words = []
    for item in data:
        if isinstance(item, str):
            words.append(item.strip())
        elif isinstance(item, dict):
            word = (item.get("word") or item.get("kanji") or "").strip()
            if word:
                words.append(word)
    return words


def load_waller_jlpt_lists(waller_dir: Path) -> dict[str, str]:
    """
    Load JLPT vocabulary lists from a directory.
    Returns {word: jlpt_level} for all words found.

    Supports multiple source formats — tries in order:
      1. stephenmk combined CSV: JLPT_vocab_ALL.csv (Kanji,Reading,Level)
         https://github.com/stephenmk/yomitan-jlpt-vocab
      2. Bluskyo per-level CSVs: n5_vocab_cleaned.csv ... n1_vocab_cleaned.csv
         https://github.com/Bluskyo/JLPT_Vocabulary
      3. davidluzgouveia JSON: jlpt_n5.json ... jlpt_n1.json
      4. Plain text fallback: N5.txt ... N1.txt

    When a word appears in multiple lists, the lowest level wins.
    """
    log.info("Loading JLPT word lists from: %s", waller_dir)
    word_to_level: dict[str, str] = {}
    level_priority = {"N5": 5, "N4": 4, "N3": 3, "N2": 2, "N1": 1}

    # Try the stephenmk combined CSV first — one file covers all levels
    combined_path = waller_dir / STEPHENMK_COMBINED_CSV
    if combined_path.exists():
        try:
            combined = _load_stephenmk_combined_csv(combined_path)
            for word, level in combined.items():
                existing = word_to_level.get(word)
                if existing is None or level_priority[level] > level_priority[existing]:
                    word_to_level[word] = level
            counts = {lvl: sum(1 for v in word_to_level.values() if v == lvl) for lvl in JLPT_LEVELS}
            for lvl in JLPT_LEVELS:
                log.info("  %s: %d words loaded (combined CSV)", lvl, counts[lvl])
            log.info("JLPT list load complete: %d unique words with level", len(word_to_level))
            return word_to_level
        except Exception as exc:
            log.warning("Combined CSV load failed (%s), falling back to per-level files", exc)

    for level in JLPT_LEVELS:
        # Try JSON format first (davidluzgouveia)
        json_path = waller_dir / JLPT_JSON_FILENAMES[level]
        txt_path = waller_dir / WALLER_FILENAMES[level]

        csv_path = waller_dir / JLPT_CSV_FILENAMES[level]

        if csv_path.exists():
            try:
                words = _load_jlpt_csv(csv_path)
                for word in words:
                    existing = word_to_level.get(word)
                    if existing is None or level_priority[level] > level_priority[existing]:
                        word_to_level[word] = level
                log.info("  %s: %d words loaded (CSV)", level, len(words))
                continue
            except Exception as exc:
                log.warning("  %s: CSV load failed (%s), trying JSON fallback", level, exc)

        if json_path.exists():
            try:
                words = _load_jlpt_json(json_path, level)
                for word in words:
                    existing = word_to_level.get(word)
                    if existing is None or level_priority[level] > level_priority[existing]:
                        word_to_level[word] = level
                log.info("  %s: %d words loaded (JSON)", level, len(words))
                continue
            except Exception as exc:
                log.warning("  %s: JSON load failed (%s), trying text fallback", level, exc)

        if txt_path.exists():
            try:
                count = 0
                for line in _detect_and_read_lines(txt_path):
                    word = line.split("\t")[0].strip()
                    if not word or word.startswith("#"):
                        continue
                    existing = word_to_level.get(word)
                    if existing is None or level_priority[level] > level_priority[existing]:
                        word_to_level[word] = level
                        count += 1
                log.info("  %s: %d words loaded (text)", level, count)
                continue
            except Exception as exc:
                log.warning("  %s: text load failed (%s)", level, exc)

        log.warning("  %s: no usable file found (tried %s, %s, %s)", level, csv_path.name, json_path.name, txt_path.name)

    log.info("JLPT list load complete: %d unique words with level", len(word_to_level))
    return word_to_level


# ---------------------------------------------------------------------------
# Merge pipeline
# ---------------------------------------------------------------------------


def merge_entries(
    jmdict_entries: Iterator[JmdictRawEntry],
    kanjium_accents: dict[str, str],
    waller_jlpt: dict[str, str],
) -> Iterator[MergedEntry]:
    """
    Merge JMdict entries with kanjium and Waller data.

    Merge strategy:
        - pitch_accent: look up entry.primary_text in kanjium_accents;
          fall back to entry.primary_reading lookup if no kanji form match.
        - jlpt_level: look up entry.primary_text in waller_jlpt;
          fall back to entry.primary_reading lookup if no kanji form match.
        - meanings: deduplicated, capped at 5 to keep rows concise.
        - parts_of_speech: deduplicated.

    Yields one MergedEntry per valid JMdict entry.
    """
    for raw in jmdict_entries:
        text = raw.primary_text
        reading = raw.primary_reading

        # Pitch accent — try kanji form first, then reading
        pitch = kanjium_accents.get(text) or kanjium_accents.get(reading)

        # JLPT level — try kanji form first, then reading
        jlpt = waller_jlpt.get(text) or waller_jlpt.get(reading)

        # Cap meanings at 5 (prevents bloated rows for entries with 20+ glosses)
        meanings = raw.meanings[:5]

        yield MergedEntry(
            id=raw.seq_id,
            text=text,
            reading=reading,
            jlpt_level=jlpt,
            pitch_accent=pitch,
            meanings=meanings,
            parts_of_speech=raw.pos_tags,
        )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "id",
    "text",
    "reading",
    "jlpt_level",
    "pitch_accent",
    "meanings",
    "parts_of_speech",
]


def write_csv(entries: Iterator[MergedEntry], output_path: Path) -> int:
    """
    Write merged entries to CSV. Returns total row count written.
    meanings and parts_of_speech are serialised as JSON arrays.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing output CSV: %s", output_path)

    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.to_csv_row())
            count += 1
            if count % 10_000 == 0:
                log.info("  %d rows written...", count)

    return count


# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------


def log_stats(output_path: Path) -> None:
    """
    Read the output CSV and log a quick sanity-check summary.
    Separate pass so it doesn't interrupt the write stream.
    """
    total = 0
    with_jlpt = 0
    with_pitch = 0
    jlpt_counts: dict[str, int] = {level: 0 for level in JLPT_LEVELS}

    with output_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            if row["jlpt_level"]:
                with_jlpt += 1
                jlpt_counts[row["jlpt_level"]] = jlpt_counts.get(row["jlpt_level"], 0) + 1
            if row["pitch_accent"]:
                with_pitch += 1

    log.info("--- Output summary ---")
    log.info("  Total rows:     %d", total)
    log.info("  With JLPT level: %d (%.1f%%)", with_jlpt, 100 * with_jlpt / max(total, 1))
    log.info("  With pitch accent: %d (%.1f%%)", with_pitch, 100 * with_pitch / max(total, 1))
    for level in JLPT_LEVELS:
        log.info("    %s: %d entries", level, jlpt_counts.get(level, 0))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build JMdict merged CSV from JMdict XML + kanjium + Waller JLPT lists.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--jmdict",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to JMdict_e.xml (download from https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project)",
    )
    parser.add_argument(
        "--kanjium",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to kanjium accents.txt (https://github.com/mifunetoshiro/kanjium — data/accents.txt)",
    )
    parser.add_argument(
        "--waller-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing Waller JLPT lists: N5.txt N4.txt N3.txt N2.txt N1.txt",
    )
    parser.add_argument(
        "--output",
        default=Path("data/jmdict_merged.csv"),
        type=Path,
        metavar="PATH",
        help="Output CSV path (default: data/jmdict_merged.csv)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate inputs exist before doing any work
    for flag, path in [("--jmdict", args.jmdict), ("--kanjium", args.kanjium), ("--waller-dir", args.waller_dir)]:
        if not path.exists():
            log.error("%s path does not exist: %s", flag, path)
            return 1

    log.info("=== build_jmdict_db.py ===")

    kanjium_accents = load_kanjium_accents(args.kanjium)
    waller_jlpt = load_waller_jlpt_lists(args.waller_dir)

    jmdict_stream = parse_jmdict_xml(args.jmdict)
    merged_stream = merge_entries(jmdict_stream, kanjium_accents, waller_jlpt)

    total = write_csv(merged_stream, args.output)
    log.info("Write complete: %d rows → %s", total, args.output)

    log_stats(args.output)
    log.info("Done. Run load_jmdict_db.py next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
