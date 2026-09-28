import json
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

DOC_ID = "9a5a4b4a-488e-4a38-a171-07160cbc735b"

questions = [
    "What is a README file, and what is its purpose in a software project?",
    "What are branches in Git and why are they useful?",
    "What is the purpose of the .gitignore file?",
    "What are GitHub Pull Requests?",
    "What is git stash and when should you use it?",
]

results_summary = []

for i, q in enumerate(questions):
    print(f"\n===== Question {i+1}/{len(questions)} =====")
    print(f"Q: {q}")
    try:
        data = json.dumps({"document_id": DOC_ID, "question": q}).encode()
        req = Request(
            "http://127.0.0.1:5000/ask",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = urlopen(req, timeout=180)
        result = json.loads(resp.read().decode())

        status = result.get("status", "?")
        gate = result.get("gate", {})
        coverage = gate.get("coverage", 0)
        supported = gate.get("supported_requirements", 0)
        total = gate.get("total_requirements", 0)

        print(f"Status: {status} | Coverage: {coverage*100:.0f}% ({supported}/{total})")

        answer = result.get("answer", "No answer")
        print(f"A: {answer[:400]}")

        evidence = result.get("evidence", [])
        if evidence:
            print(f"Evidence items: {len(evidence)}")
            for j, ev in enumerate(evidence):
                page = ev.get("page", "?")
                req_text = ev.get("requirement", "")[:80]
                print(f"  [{j+1}] Page {page} - {req_text}")

        verdict = "PASS" if status == "SUFFICIENT" else "PARTIAL/FAIL"
        print(verdict)
        results_summary.append((q, status, f"{supported}/{total}", verdict))

    except HTTPError as e:
        body = e.read().decode()[:200]
        print(f"HTTP {e.code}: {body}")
        results_summary.append((q, f"HTTP {e.code}", "N/A", "ERROR"))
    except Exception as e:
        print(f"Error: {e}")
        results_summary.append((q, str(e)[:60], "N/A", "ERROR"))

    # Wait between questions to avoid rate limits
    if i < len(questions) - 1:
        print("(waiting 8s before next question...)")
        time.sleep(8)

print("\n\n========== TEST SUMMARY ==========")
passes = sum(1 for r in results_summary if r[3] == "PASS")
total_q = len(results_summary)
print(f"Results: {passes}/{total_q} PASS\n")
for q, status, coverage, verdict in results_summary:
    print(f"  [{verdict:12s}] {coverage:5s}  {q}")
print("\n===== ALL TESTS COMPLETE =====")
