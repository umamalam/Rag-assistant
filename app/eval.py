"""
eval.py
A simple evaluation script: define test questions with known expected
answers (or expected sources), run them through the RAG pipeline, and
measure how well it does. This produces the kind of concrete numbers
worth mentioning in an interview -- "90% retrieval accuracy on N test
questions" -- instead of just "it seems to work."

Edit TEST_CASES below to match your own documents before running.

Run with:  python3 eval.py
"""

from rag_engine import ask

# Each test case: a question, and either the exact source file you expect
# to be cited (retrieval check), or a keyword you expect in the answer
# (answer quality check). Fill these in based on YOUR actual documents.
TEST_CASES = [
    {
        "question": "What is horizontal scaling?",
        "expected_source_contains": "system_design",  # filename should contain this
        "expected_answer_contains": "servers",          # answer should mention this
    },
    {
        "question": "What color is the sky on Mars?",  # a question NOT covered by any document
        "expect_no_answer": True,  # the system should say "I don't have enough information"
    },
    # Add more test cases here based on your actual 10 books:
    # {
    #     "question": "...",
    #     "expected_source_contains": "...",
    #     "expected_answer_contains": "...",
    # },
]


def run_eval():
    passed = 0
    total = len(TEST_CASES)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{total}] Q: {case['question']}")
        result = ask(case["question"])
        answer = result["answer"]
        sources = [s["source"] for s in result["sources"]]

        print(f"  Answer: {answer[:150]}...")
        print(f"  Sources: {sources}")

        ok = True

        if case.get("expect_no_answer"):
            if "don't have enough information" not in answer.lower() and \
               "do not have enough information" not in answer.lower():
                print("  FAIL: expected a 'not enough information' response, "
                      "but got a confident answer -- possible hallucination")
                ok = False

        if "expected_source_contains" in case:
            if not any(case["expected_source_contains"] in s for s in sources):
                print(f"  FAIL: expected a source containing "
                      f"'{case['expected_source_contains']}', got {sources}")
                ok = False

        if "expected_answer_contains" in case:
            if case["expected_answer_contains"].lower() not in answer.lower():
                print(f"  FAIL: expected answer to mention "
                      f"'{case['expected_answer_contains']}'")
                ok = False

        if ok:
            print("  PASS")
            passed += 1

    print(f"\n{'=' * 40}")
    print(f"Result: {passed}/{total} passed ({passed / total * 100:.0f}%)")


if __name__ == "__main__":
    run_eval()
