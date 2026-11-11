"""
JobShield Latency Benchmark
Measures warm inference latency over 50 iterations on representative text inputs.
Verifies p95 < 2000ms local responsiveness target.
"""
import time
import platform
import numpy as np
import pytest

from src.predictor import JobShieldPredictor, JobAdvertisementInput


def test_inference_latency_benchmark():
    """Measures warm-inference latency and asserts p95 latency is well below 2.0s."""
    predictor = JobShieldPredictor()

    representative_ad = JobAdvertisementInput(
        title="Full Stack Software Engineer",
        company_name="CloudMatrix Technologies",
        description=(
            "CloudMatrix Technologies is looking for a Full Stack Software Engineer to join our growing engineering hub. "
            "In this role, you will architect microservices using Python and FastAPI, build interactive frontend interfaces "
            "with TypeScript, optimize PostgreSQL database queries, and automate CI/CD delivery pipelines in Docker. "
            "We value collaborative problem solving, thorough code reviews, and pragmatic architectural designs."
        ),
        requirements=(
            "B.S. or M.S. in Computer Science or equivalent practical experience. "
            "3+ years of production experience in Python, relational databases, and REST APIs. "
            "Familiarity with containerization and cloud infrastructure."
        ),
        benefits=(
            "Competitive salary, annual performance bonus, comprehensive health insurance, "
            "hybrid work policy, continuous learning budget, and generous paid time off."
        ),
    )

    # Warmup runs (3 iterations)
    for _ in range(3):
        _ = predictor.predict(representative_ad)

    # Benchmark runs (50 iterations)
    latencies_ms = []
    iterations = 50
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = predictor.predict(representative_ad)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))

    print(f"\n--- Warm Inference Latency Benchmark ({iterations} runs) ---")
    print(f"Hardware: {platform.machine()} ({platform.system()} {platform.release()})")
    print(f"Input size: {len(representative_ad.build_full_text())} characters")
    print(f"Min: {min_ms:.2f} ms")
    print(f"Median: {median_ms:.2f} ms")
    print(f"p95: {p95_ms:.2f} ms")
    print(f"p99: {p99_ms:.2f} ms")
    print(f"Max: {max_ms:.2f} ms")

    # Assert local responsiveness target (<2000 ms)
    assert p95_ms < 2000.0, f"p95 latency {p95_ms:.2f} ms exceeds target of 2000 ms"
    assert median_ms < 500.0, f"Median latency {median_ms:.2f} ms exceeds 500 ms"
