import json
import os
import random
import time
import threading
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  CONFIG — edit these paths and keys only
# ─────────────────────────────────────────────
BASE = Path(__file__).parent

INPUT_PATH  = BASE / "C:\\Users\\SL\\OneDrive\\Desktop\\OneDrive - National University of Sciences & Technology\\qwen-alif-xcot\\data\\processed\\separated_categories\\reasoning.json"
OUTPUT_PATH = BASE / "data/processed/final_training_dataset.jsonl"

API_KEYS = [
    os.getenv("NVIDIA_KEY_1"),
    os.getenv("NVIDIA_KEY_2"),
    os.getenv("NVIDIA_KEY_3"),
    os.getenv("NVIDIA_KEY_4")
]

# Set to an integer (e.g. 10) to run a quick test slice, None for full run
TEST_SLICE = None

# ─────────────────────────────────────────────
#  PROMPTS
# ─────────────────────────────────────────────
QWEN_STUDENT_SYSTEM_PROMPT = (
    "You are a multilingual reasoning AI. "
    "Execute literal token mapping and structural logic inside <::> tags before answering."
)

NVIDIA_TEACHER_PROMPT_TEMPLATE = """Task: Generate the internal xCoT reasoning chain for an Urdu machine learning dataset.
The input question is written in native Nastaliq (Arabic script). You must read it, but your output must follow the 4-header architecture below.
Output ONLY the four headers and their analysis. Do NOT write introductory remarks or conversational fluff.

Follow this architecture word-for-word:

[Translation & Query Decomposition]
Step 1: Literal Token Glossing (Read the incoming Nastaliq script, isolate the words, and map them to English using equal signs. Use the phonetic concept of: 'aik dukan = one shop', 'do anda = two eggs').
Step 2: Equation Assembly (String the literal English tokens together sequentially).
Step 3: Syntactic Correction (Fix the raw string into perfect English grammar).

[Premise & Variable Verification]
(Identify constants, numeric parameters, and isolate what the query specifically asks to solve).

[Logic Execution]
(Perform step-by-step mathematical or logical calculation entirely in English prose and equations).

[Linguistic Target Mapping]
(Map the numerical or logical result back into fluent, natural Urdu script grammar structures).

Input Parameters to Process:
Urdu Question: {question}
Urdu Answer: {answer}"""

# ─────────────────────────────────────────────
#  THREAD-SAFE WRITE LOCK + SKIP SET
# ─────────────────────────────────────────────
write_lock  = threading.Lock()
skip_lock   = threading.Lock()
skipped_log = []


# ─────────────────────────────────────────────
#  CHECKPOINT: load already-finished questions
# ─────────────────────────────────────────────
def get_done_set() -> set:
    done = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q = obj["messages"][1]["content"]
                    done.add(q)
                except Exception:
                    pass
    return done


# ─────────────────────────────────────────────
#  API CALL with capped retries + jitter
# ─────────────────────────────────────────────
def call_api(client: OpenAI, prompt: str, key_label: str, retries: int = 3):
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b",
                messages=[
                    {"role": "system", "content": "You are a strict text generation machine."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2500,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 1024,
                },
            )
            result = response.choices[0].message.content.strip()

            # Random jitter between calls: 1.5 – 2.5 s  (keeps each key well under 40 RPM)
            time.sleep(1.5 + random.uniform(0, 1.0))
            return result

        except Exception as e:
            if attempt == retries:
                print(f"\n{key_label} ❌ Max retries hit, skipping. Last error: {e}")
                return None
            wait = 6 + random.uniform(0, 3)
            print(f"\n{key_label} ⚠️  Attempt {attempt + 1} failed ({e}). Retrying in {wait:.1f}s…")
            time.sleep(wait)

    return None


# ─────────────────────────────────────────────
#  WORKER — one thread per API key
# ─────────────────────────────────────────────
def worker(key_index: int, rows_slice: list, done_set: set, progress_bars: list):
    key   = API_KEYS[key_index]
    label = f"[Key {key_index + 1}]"

    if not key:
        print(f"{label} ❌  No API key found in .env — skipping this worker.")
        return

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)

    # Stagger startup: key N waits N*3 seconds so bursts don't overlap at t=0
    time.sleep(key_index * 3)

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_file:
        for item in progress_bars[key_index]:
            q = item.get("instruction") or item.get("user")
            a = item.get("output")      or item.get("assistant")

            if not q or not a:
                continue

            # Skip if already in checkpoint
            if q in done_set:
                continue

            prompt = NVIDIA_TEACHER_PROMPT_TEMPLATE.format(question=q, answer=a)
            result = call_api(client, prompt, label)

            if result:
                payload = {
                    "messages": [
                        {"role": "system",    "content": QWEN_STUDENT_SYSTEM_PROMPT},
                        {"role": "user",      "content": q},
                        {"role": "assistant", "content": f"<::>\n{result}\n</::>\n\n{a}"},
                    ]
                }
                with write_lock:
                    out_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    out_file.flush()          # flush after every write — safe on crash
            else:
                with skip_lock:
                    skipped_log.append({"key": key_index + 1, "question_snippet": q[:80]})


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    if not INPUT_PATH.exists():
        print(f"❌  Input file not found: {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if TEST_SLICE:
        rows = rows[:TEST_SLICE]
        print(f"🧪  TEST MODE — processing first {TEST_SLICE} rows only.")

    done_set = get_done_set()
    remaining = [r for r in rows if (r.get("instruction") or r.get("user")) not in done_set]

    print(f"\n📦  Total rows      : {len(rows)}")
    print(f"✅  Already done    : {len(done_set)}")
    print(f"⏳  Left to process : {len(remaining)}")

    if not remaining:
        print("🎉  Nothing left to process!")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Split remaining rows round-robin across available keys
    num_keys   = sum(1 for k in API_KEYS if k)
    slices     = [remaining[i::num_keys] for i in range(num_keys)]

    # One tqdm bar per worker — position=i keeps them stacked neatly
    progress_bars = [
        tqdm(slices[i], desc=f"Key {i + 1}", position=i, leave=True)
        for i in range(num_keys)
    ]

    threads = []
    for i in range(num_keys):
        t = threading.Thread(target=worker, args=(i, slices[i], done_set, progress_bars), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Close all progress bars cleanly
    for bar in progress_bars:
        bar.close()

    print(f"\n✅  Generation complete! Output → {OUTPUT_PATH}")
    print(f"⚠️   Skipped rows    : {len(skipped_log)}")

    if skipped_log:
        skip_path = OUTPUT_PATH.parent / "skipped_rows.json"
        with open(skip_path, "w", encoding="utf-8") as f:
            json.dump(skipped_log, f, ensure_ascii=False, indent=2)
        print(f"📄  Skipped log saved → {skip_path}")


if __name__ == "__main__":
    run()