import json
import os
import random
import time
import threading
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
INPUT_PATH  = Path(r"C:\Users\SL\OneDrive\Desktop\OneDrive - National University of Sciences & Technology\qwen-alif-xcot\data\processed\separated_categories\generation.json")

# ✅ HARDCODED absolute output path — no more relative path bugs
OUTPUT_PATH = Path(r"C:\Users\SL\OneDrive\Desktop\OneDrive - National University of Sciences & Technology\qwen-alif-xcot\src\data\data\processed\final_training_dataset2.jsonl")

API_KEYS = [
    os.getenv("NVIDIA_KEY_1"),
    os.getenv("NVIDIA_KEY_2"),
    os.getenv("NVIDIA_KEY_3"),
    os.getenv("NVIDIA_KEY_4"),
]

# Set to an integer e.g. 2 for a quick test, None for the full run
TEST_SLICE  = None

# ✅ Timeout in seconds per API call — prevents infinite hangs
API_TIMEOUT = 120

# ─────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────

QWEN_STUDENT_SYSTEM_PROMPT = (
    "You are a multilingual reasoning AI. "
    "You are a helpful reasoning assistant. Think step-by-step in English before answering."
)

NVIDIA_TEACHER_PROMPT_TEMPLATE = """System: You are an elite Bilingual AI Architect and a master of Urdu linguistics.

Task: Analyze the provided prompt and generate a response using the specified XML architecture. 

Directives:
1. XML-Only: Output ONLY the requested XML structure. Do NOT include markdown code blocks (e.g., ```xml), introductory remarks, or concluding fluff.
2. Linguistic Firewall: The execution must be in native, idiomatic Urdu. Strictly NO English words, NO Roman Urdu, and NO translations that retain English grammatical structures in the execution block.
3. Logical Scaffolding: The <planning_core> must be in English to ensure high-fidelity reasoning.

<planning_core>
[Theme]: [Core conceptual vector, 1-5 words]
[Tone]: [Narrative voice, 1-3 adjectives]
[Constraints]: [Identify logical, formatting, or stylistic boundaries]
[Roadmap]: [Concise linear progression of the output]
</planning_core>

<execution_core>
[Fluent Urdu response here. Follow the Roadmap exactly. Use formatting appropriate for the context. Maintain absolute linguistic purity.]
</execution_core>

Question: {question}"""

# ─────────────────────────────────────────────
#  THREAD-SAFE GLOBALS
# ─────────────────────────────────────────────
write_lock  = threading.Lock()
skip_lock   = threading.Lock()
skipped_log = []


# ─────────────────────────────────────────────
#  INPUT LOADING — auto-detect JSON array vs JSONL
# ─────────────────────────────────────────────
def load_input(path: Path) -> list:
    """Load input file — handles both JSON array and JSONL formats."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)

    with open(path, "r", encoding="utf-8") as f:
        if first_char == "[":
            rows = json.load(f)
            print(f"📂  Detected format : JSON array")
            return rows
        else:
            rows = []
            bad  = 0
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Input line {line_num}: JSON error — {e}")
                    bad += 1
            print(f"📂  Detected format : JSONL  ({bad} bad lines skipped)")
            return rows


# ─────────────────────────────────────────────
#  SINGLE SOURCE OF TRUTH — question extractor
# ─────────────────────────────────────────────
def get_question(row: dict) -> str:
    """Extract the question string from a row, regardless of field name."""
    return (
        row.get("instruction")
        or row.get("user")
        or row.get("prompt")
        or row.get("question")
        or ""
    ).strip()


# ─────────────────────────────────────────────
#  CHECKPOINT — robust with full diagnostics
# ─────────────────────────────────────────────
def get_done_set() -> set:
    done = set()

    if not OUTPUT_PATH.exists():
        print("📄  No existing output file found — starting fresh.")
        return done

    total_lines = 0
    parsed_ok   = 0
    bad_lines   = 0

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj      = json.loads(line)
                messages = obj.get("messages", [])

                # Expected structure: [system(0), user(1), assistant(2)]
                if (
                    isinstance(messages, list)
                    and len(messages) == 3
                    and all(isinstance(m, dict) for m in messages)
                ):
                    user_content = messages[1].get("content", "").strip()
                    if user_content:
                        done.add(user_content)
                        parsed_ok += 1
                    else:
                        print(f"⚠️  Line {line_num}: user content is empty")
                        bad_lines += 1
                else:
                    print(f"⚠️  Line {line_num}: bad structure — len={len(messages)}")
                    bad_lines += 1

            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: JSON parse error — {e} | raw: {line[:80]}")
                bad_lines += 1

    print(f"\n📊  Checkpoint stats:")
    print(f"    Output path  : {OUTPUT_PATH}")
    print(f"    Total lines  : {total_lines}")
    print(f"    Parsed OK    : {parsed_ok}")
    print(f"    Bad/skipped  : {bad_lines}")
    print(f"    Unique done  : {len(done)}\n")

    return done


# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
def is_valid_structure(text: str) -> bool:
    required_tags = [
        "<planning_core>", "</planning_core>",
        "<execution_core>", "</execution_core>",
    ]
    return all(tag in text for tag in required_tags)


# ─────────────────────────────────────────────
#  API CALL — with timeout to prevent infinite hangs
# ─────────────────────────────────────────────
def call_api(api_key: str, prompt: str, key_label: str, retries: int = 3):
    # ✅ timeout= kills the request after API_TIMEOUT seconds
    client = OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        timeout=API_TIMEOUT,
    )

    payload = {
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "messages": [
            {
                "role": "system",
                "content": "You are a strict structured text generation machine. Output only the requested format with no extra commentary.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.35,
        "top_p": 0.95,
        "max_tokens": 2048,
        # ✅ thinking=True is REQUIRED for deepseek-v4-flash on NVIDIA NIM
        # Without it the API hangs indefinitely
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "low",
            }
        },
    }

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(**payload)
            message  = response.choices[0].message

            # ✅ message is a Pydantic object — must use getattr, NOT .get()
            content = (
                getattr(message, "content", None)
                or getattr(message, "reasoning_content", None)
                or ""
            ).strip()

            if not content:
                raise Exception("Empty response — no content or reasoning_content")

            # Jitter: stay safely under 40 RPM per key
            time.sleep(1.5 + random.uniform(0, 1.0))
            return content

        except Exception as e:
            if attempt == retries:
                print(f"\n{key_label} ❌  Max retries hit, skipping. Error: {e}")
                return None
            wait = 6 + random.uniform(0, 3)
            print(f"\n{key_label} ⚠️  Attempt {attempt + 1} failed ({e}). Retrying in {wait:.1f}s…")
            time.sleep(wait)

    return None


# ─────────────────────────────────────────────
#  WORKER — one thread per API key
# ─────────────────────────────────────────────
def worker(key_index: int, rows_slice: list, done_set: set, progress_bar: tqdm):
    key   = API_KEYS[key_index]
    label = f"[Key {key_index + 5}]"   # matches env var name NVIDIA_KEY_5..8

    if not key:
        print(f"{label} ❌  not found in .env — skipping.")
        progress_bar.update(len(rows_slice))
        return

    # Stagger startup so threads don't all hit the API at the same second
    time.sleep(key_index * 3)

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_file:
        for item in rows_slice:
            q = get_question(item)

            if not q:
                progress_bar.update(1)
                continue

            # ✅ Check done_set BEFORE making the API call
            if q in done_set:
                progress_bar.update(1)
                continue

            prompt = NVIDIA_TEACHER_PROMPT_TEMPLATE.format(question=q)
            result = call_api(key, prompt, label)

            if result and is_valid_structure(result):
                record = {
                    "messages": [
                        {"role": "system",    "content": QWEN_STUDENT_SYSTEM_PROMPT},
                        {"role": "user",      "content": q},
                        {"role": "assistant", "content": f"<::>\n{result}\n</::>"},
                    ]
                }
                with write_lock:
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_file.flush()
                    done_set.add(q)   # ✅ prevent duplicate writes across threads
            else:
                reason = "Invalid Format" if result else "API Drop"
                with skip_lock:
                    skipped_log.append({
                        "key":              key_index + 5,
                        "question_snippet": q[:80],
                        "reason":           reason,
                    })

            progress_bar.update(1)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    # Show key load status
    for i, k in enumerate(API_KEYS):
        status = f"{k[:12]}... ✅" if k else "MISSING ❌"
        print(f"NVIDIA_KEY_{i + 5}: {status}")

    print(f"\nOUTPUT PATH: {OUTPUT_PATH}")

    if not INPUT_PATH.exists():
        print(f"\n❌  Input file not found: {INPUT_PATH}")
        return

    # ── Load input ──
    rows = load_input(INPUT_PATH)
    print(f"    Loaded {len(rows)} rows total")

    # ── Diagnose field names ──
    if rows:
        print(f"\n🔍  Sample row keys : {list(rows[0].keys())}")
        print(f"🔍  Sample question : {get_question(rows[0])[:120]}")

    # ── Warn about unextractable rows ──
    none_count = sum(1 for r in rows if not get_question(r))
    if none_count:
        print(f"⚠️  WARNING: {none_count} rows have no question field — will be skipped!")

    if TEST_SLICE:
        rows = rows[:TEST_SLICE]
        print(f"\n🧪  TEST MODE — processing first {TEST_SLICE} rows only.")

    # ── Checkpoint ──
    done_set  = get_done_set()
    remaining = [r for r in rows if get_question(r) not in done_set]

    print(f"📦  Total rows      : {len(rows)}")
    print(f"✅  Already done    : {len(done_set)}")
    print(f"⏳  Left to process : {len(remaining)}")

    if not remaining:
        print("🎉  Nothing left to process!")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Launch workers ──
    active_keys = [i for i, k in enumerate(API_KEYS) if k]
    num_workers = len(active_keys)

    if num_workers == 0:
        print("❌  No API keys found in .env.")
        return

    print(f"\n🚀  Launching {num_workers} worker(s) | timeout={API_TIMEOUT}s per call\n")

    slices = [remaining[i::num_workers] for i in range(num_workers)]

    # ✅ tqdm bars use total= so they show real progress
    progress_bars = [
        tqdm(
            total=len(slices[i]),
            desc=f"KEY_{active_keys[i] + 5}",
            position=i,
            leave=True,
            dynamic_ncols=True,
        )
        for i in range(num_workers)
    ]

    threads = []
    for i in range(num_workers):
        t = threading.Thread(
            target=worker,
            args=(active_keys[i], slices[i], done_set, progress_bars[i]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for bar in progress_bars:
        bar.close()

    print(f"\n✅  Done! Output → {OUTPUT_PATH}")
    print(f"⚠️  Skipped rows : {len(skipped_log)}")

    if skipped_log:
        skip_path = OUTPUT_PATH.parent / "skipped_rows.json"
        with open(skip_path, "w", encoding="utf-8") as f:
            json.dump(skipped_log, f, ensure_ascii=False, indent=2)
        print(f"📄  Skipped log  → {skip_path}")


if __name__ == "__main__":
    run()