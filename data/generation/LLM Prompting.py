# I ran this to generate results.json, which is a json of answers to the Politune prompts from five different political perspectives.
# I did that locally on my computer after importing llama3.1. For reproducibility, the ask function may need to be changed based
# on your access to LLM APIs.

import requests
import json
import time
import csv

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1"
CSV_PATH = "politune-left.csv"      # Can be changed to politune-right.csv or some other similarly formatted dataset
FIRST_ROW_INDEX = 0                 # Modify if we want to start midway

def ask(prompt):
    """Send a prompt to the local Ollama model and return the response text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    res = requests.post(OLLAMA_URL, json=payload, timeout=300)
    res.raise_for_status()
    return res.json()["response"]

def ask_with_retry(prompt, retries=3):
    """Retry wrapper to avoid stopping the script on minor failures."""
    for attempt in range(retries):
        try:
            return ask(prompt)
        except Exception as e:
            wait = 1 + attempt * 2
            print(f"Error: {e} — retrying in {wait}s")
            time.sleep(wait)
    return "<FAILED>"

def main():
    results = []

    # --- Load CSV into memory ---
    with open(CSV_PATH, newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for idx, row in enumerate(rows):
        if idx >= FIRST_ROW_INDEX:
            question_text = row.get("question") or row.get("text") or row[list(row.keys())[0]]

            ### This prompt is probably better than my original prompt
            # prompt = f"""You are a political-science researcher analyzing how Americans with different ideological positions discuss various issues in politics. For the following controversial prompt, provide five short responses (100–150 words each) as if written by real people holding these exact viewpoints. Stay strictly in character for each one and use language natural to that perspective.

            # The case: {question_text}

            # Your response should have this format:

            # <begin_response>
            # **PERSONA 1**
            # Answer from the perspective of an ideological centrist / independent

            # **PERSONA 2**
            # Answer from the perspective of a mainstream liberal / Democrat

            # **PERSONA 3**
            # Answer from the perspective of a mainstream conservative / Republican

            # **PERSONA 4**
            # Answer from the perspective of a socialist / very strong Democrat

            # **PERSONA 5**
            # Answer from the perspective of a MAGA conservative / very strong Republican
            # <end_response>

            # Do not modify the text inside these **tags**!
            # """

            ### This prompt is what I used to generate our 1800-line results.json data. It can be used for consistency
            prompt = f"""Please answer this question according to five personas:
            - Centrist
            - Reasonable Democrat/liberal
            - Reasonable Republican/conservative
            - Crazy, extreme liberal
            - Crazy, extreme conservative

            Question: {question_text}
            """

            print(f"[{idx+1}/{len(rows)}] Running...")

            response = ask_with_retry(prompt)
            results.append({
                "row": idx,
                "question": question_text,
                "response": response
            })

            # save every 1 iterations for safety
            if idx % 1 == 0:
                with open("results.json", "w") as f:
                    json.dump(results, f, indent=2)

    # final save
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()

## ASYNC VERSION THAT WAS SUPPOSED TO RUN FASTER BUT DOESN'T REALLY SEEM TO

# import aiohttp
# import asyncio
# import csv
# import json
# from pathlib import Path

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL = "llama3.1"
# CSV_PATH = "politune-right.csv"
# OUTPUT_FILE = Path("results_async.json")

# MAX_CONCURRENCY = 4


# def make_prompt(q):
#     return f"""
# You are a political-science researcher...

# The case: {q}

# <begin_response>
# **PERSONA 1**
# Answer from the perspective of an ideological centrist / independent

# **PERSONA 2**
# Answer from the perspective of a mainstream liberal / Democrat

# **PERSONA 3**
# Answer from the perspective of a mainstream conservative / Republican

# **PERSONA 4**
# Answer from the perspective of a socialist / very strong Democrat

# **PERSONA 5**
# Answer from the perspective of a MAGA conservative / very strong Republican
# <end_response>

# Do not modify the text inside these **tags**!
# """


# async def ask(session, prompt):
#     payload = {
#         "model": MODEL,
#         "prompt": prompt,
#         "stream": False,
#     }
#     async with session.post(OLLAMA_URL, json=payload, timeout=300) as r:
#         r.raise_for_status()
#         data = await r.json()
#         return data["response"]


# async def ask_with_retry(session, prompt, retries=3):
#     for attempt in range(retries):
#         try:
#             return await ask(session, prompt)
#         except Exception as e:
#             wait = 1 + attempt * 2
#             print(f"Error: {e!r} — retrying in {wait}s")
#             await asyncio.sleep(wait)
#     return "<FAILED>"


# async def worker(name, session, queue, results):
#     while True:
#         item = await queue.get()
#         if item is None:
#             queue.task_done()
#             return

#         idx, row = item
#         q = row.get("question") or row.get("text") or row[list(row.keys())[0]]

#         print(f"[Worker {name}] Processing row {idx}")

#         prompt = make_prompt(q)
#         response = await ask_with_retry(session, prompt)

#         results.append({
#             "row": idx,
#             "question": q,
#             "response": response,
#         })

#         # live write
#         OUTPUT_FILE.write_text(json.dumps(results, indent=2))

#         queue.task_done()


# async def main():
#     # Load CSV
#     with open(CSV_PATH, newline="", encoding="utf8") as f:
#         rows = list(csv.DictReader(f))

#     queue = asyncio.Queue()
#     results = []

#     # Fill queue
#     for idx, row in enumerate(rows):
#         if idx > 0:
#             queue.put_nowait((idx, row))

#     async with aiohttp.ClientSession() as session:
#         # Launch workers
#         workers = [
#             asyncio.create_task(worker(i, session, queue, results))
#             for i in range(MAX_CONCURRENCY)
#         ]

#         # Wait for all work to finish
#         await queue.join()

#         # Stop workers
#         for _ in workers:
#             queue.put_nowait(None)
#         await asyncio.gather(*workers)

#     print("Done.")


# if __name__ == "__main__":
#     asyncio.run(main())
