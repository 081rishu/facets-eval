import streamlit as st
import asyncio
import time
from src.inference import InferenceEngine

# ---------------------------
# Async runner helper (IMPORTANT)
# ---------------------------
def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(
    page_title="Conversation Evaluator",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Conversation Evaluation Engine")
st.markdown("Evaluate a conversation turn across hundreds of behavioral facets.")

# ---------------------------
# Input
# ---------------------------

user_input = st.text_area(
    "Enter a conversation turn:",
    value="I'm feeling really anxious about this deadline.",
    height=120
)

run_button = st.button("Evaluate")

# ---------------------------
# Engine (cached)
# ---------------------------

@st.cache_resource
def load_engine():
    return InferenceEngine(config_path="configs/model.yaml")

engine = load_engine()

# ---------------------------
# Run inference
# ---------------------------

if run_button and user_input.strip():

    with st.spinner("Evaluating..."):
        start = time.time()

        try:
            result = run_async(engine.evaluate_turn(user_input))
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

        elapsed = time.time() - start

    # ---------------------------
    # Summary
    # ---------------------------

    st.success("✅ Evaluation complete")

    col1, col2, col3 = st.columns(3)

    col1.metric("Facets Evaluated", result.total_facets_evaluated)
    col2.metric("Model", result.model_used or "N/A")
    col3.metric("Time (s)", f"{elapsed:.2f}")

    # ---------------------------
    # Confidence stats
    # ---------------------------

    if result.evaluations:
        avg_conf = sum(ev.confidence for ev in result.evaluations) / len(result.evaluations)
        st.metric("Avg Confidence", f"{avg_conf:.2f}")

    # ---------------------------
    # Results table
    # ---------------------------

    st.subheader("📊 Facet Scores")

    table_data = []

    for ev in result.evaluations:
        name = getattr(ev, "facet_name", ev.facet_id)

        table_data.append({
            "Facet": name,
            "Score": ev.get_final_score(),
            "Confidence": round(ev.confidence, 2)
        })

    st.dataframe(table_data, use_container_width=True)

    # ---------------------------
    # Low confidence section
    # ---------------------------

    low_conf = result.get_low_confidence_facets(0.6)

    if low_conf:
        st.subheader("⚠️ Low Confidence Facets")
        st.write(low_conf[:10])

    # ---------------------------
    # Expandable detailed view
    # ---------------------------

    with st.expander("🔍 Detailed Results (Raw)"):
        st.json({
            "total": result.total_facets_evaluated,
            "evaluations": [ev.dict() for ev in result.evaluations]
        })