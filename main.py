import asyncio
import time
from src.inference import InferenceEngine


async def main():
    engine = InferenceEngine()

    test_input = "I'm feeling really anxious about this deadline."

    start = time.time()

    try:
        result = await engine.evaluate_turn(test_input)
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return

    elapsed = time.time() - start

    print("\n=== Evaluation Summary ===")
    print(f"Input: {test_input}")
    print(f"Facets evaluated: {result.total_facets_evaluated}")
    print(f"Model: {result.model_used}")
    print(f"Time taken: {elapsed:.2f}s")

    print("\n=== Sample Results ===")

    for ev in result.evaluations[:5]:
        name = getattr(ev, "facet_name", ev.facet_id)  # fallback
        print(f"{name}: {ev.get_final_score()} (conf: {ev.confidence:.2f})")

    # 🔥 Bonus: low-confidence facets
    low_conf = result.get_low_confidence_facets(0.6)
    if low_conf:
        print(f"\n⚠️ Low confidence facets ({len(low_conf)}): {low_conf[:5]}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        # fallback for notebook environments
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())