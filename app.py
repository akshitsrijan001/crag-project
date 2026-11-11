"""
CRAG — Self-Corrective Retrieval-Augmented Generation
Streamlit UI
"""
import os
import sys
import streamlit as st
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.config import (DOCS_DIR, INDEX_PATH, CHUNKS_PATH,
                         UPPER_THRESHOLD, LOWER_THRESHOLD)

# --- Page Config ---
st.set_page_config(
    page_title="CRAG — Self-Corrective RAG",
    page_icon="🔍",
    layout="wide",
)

# --- Cache the embedding model ---
@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    from src.config import EMBEDDING_MODEL
    return SentenceTransformer(EMBEDDING_MODEL)

# Pre-load model in background
load_embedding_model()

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")

    # Build / Rebuild Index
    if st.button("🔨 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Building index..."):
            from src.ingest import build_index
            try:
                n_chunks = build_index()
                st.success(f"✅ Index built: {n_chunks} chunks")
                st.session_state["index_built"] = True
            except Exception as e:
                st.error(f"Error: {e}")

    # File uploader
    st.markdown("---")
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Add .txt or .pdf files to the knowledge base",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uf in uploaded_files:
            dest = os.path.join(DOCS_DIR, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            st.caption(f"Saved: {uf.name}")
        st.info("Click 'Build / Rebuild Index' to include new files.")

    # Threshold sliders
    st.markdown("---")
    st.subheader("🎚️ Thresholds")
    upper_t = st.slider("CORRECT threshold (upper)", 0.0, 1.0, UPPER_THRESHOLD, 0.05)
    lower_t = st.slider("INCORRECT threshold (lower)", 0.0, 1.0, LOWER_THRESHOLD, 0.05)

    # How CRAG works
    st.markdown("---")
    st.subheader("ℹ️ How CRAG Works")
    st.markdown("""
    1. **Retrieve** relevant chunks from the knowledge base
    2. **Evaluate** each chunk's relevance with an LLM
    3. **Route** based on the evaluation:
       - 🟢 **CORRECT** → Refine context → Generate
       - 🟠 **AMBIGUOUS** → Refine + Web search → Generate
       - 🔴 **INCORRECT** → Rewrite query → Web search → Generate
    4. **Generate** the final answer from the processed context
    """)

# --- Main Area ---
st.title("🔍 CRAG — Self-Corrective RAG")
st.caption("Compare baseline RAG with self-corrective retrieval-augmented generation")

# Auto-build index on first run
if not os.path.exists(INDEX_PATH):
    with st.spinner("Building index for the first time..."):
        try:
            from src.ingest import build_index
            n_chunks = build_index()
            st.session_state["index_built"] = True
            st.toast(f"Index built: {n_chunks} chunks", icon="✅")
        except Exception as e:
            st.error(f"Could not build index: {e}")

# Check API key
from src.config import GEMINI_API_KEY
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_key_here":
    st.error("⚠️ GEMINI_API_KEY not set. Please add it to your .env file and restart the app.")
    st.stop()

# Check index
if not os.path.exists(INDEX_PATH):
    st.warning("No index found. Click 'Build / Rebuild Index' in the sidebar.")
    st.stop()

# --- Tabs ---
tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Evaluation Results"])

with tab1:
    # Example questions
    st.subheader("Example Questions")
    example_cols = st.columns(3)
    with example_cols[0]:
        st.markdown("**🟢 Should be CORRECT:**")
        if st.button("What is RAG and who introduced it?", key="ex1"):
            st.session_state["query"] = "What is RAG and who introduced it?"
        if st.button("How does CRAG evaluate retrieved docs?", key="ex2"):
            st.session_state["query"] = "How does CRAG evaluate retrieved documents before generation?"
    with example_cols[1]:
        st.markdown("**🟠 Should be AMBIGUOUS:**")
        if st.button("How does Self-RAG differ from CRAG?", key="ex3"):
            st.session_state["query"] = "How does Self-RAG differ from CRAG in handling retrieval evaluation?"
        if st.button("Advanced prompt engineering for RAG?", key="ex4"):
            st.session_state["query"] = "What are some advanced prompt engineering techniques for improving RAG systems?"
    with example_cols[2]:
        st.markdown("**🔴 Should be INCORRECT:**")
        if st.button("Who won the FIFA World Cup?", key="ex5"):
            st.session_state["query"] = "Who won the most recent FIFA World Cup?"
        if st.button("How to make carbonara pasta?", key="ex6"):
            st.session_state["query"] = "How do you make traditional Italian carbonara pasta?"

    st.markdown("---")

    # Query input
    query = st.text_input("🔎 Enter your question:", value=st.session_state.get("query", ""))

    if st.button("Ask", type="primary", use_container_width=True) and query:
        st.session_state["query"] = query

        col1, col2 = st.columns(2)

        # Run baseline
        with col1:
            st.subheader("📋 Baseline RAG")
            with st.spinner("Running baseline RAG..."):
                try:
                    from src.baseline_rag import run_baseline
                    baseline = run_baseline(query)
                    st.markdown(baseline["answer"])
                    st.caption(f"⏱️ Latency: {baseline['latency_seconds']}s")
                    with st.expander("📄 Retrieved Chunks"):
                        for i, c in enumerate(baseline["chunks"]):
                            st.markdown(f"**Chunk {i+1}** (score: {c['score']:.3f}, source: {c['source']})")
                            st.text(c["text"][:300])
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Baseline error: {e}")

        # Run CRAG
        with col2:
            st.subheader("🔍 CRAG")
            with st.spinner("Running CRAG pipeline..."):
                try:
                    from src.crag import run_crag
                    crag = run_crag(query, upper_threshold=upper_t, lower_threshold=lower_t)

                    # Verdict badge
                    verdict = crag["verdict"]
                    if verdict == "CORRECT":
                        st.success(f"🟢 Verdict: **{verdict}**")
                    elif verdict == "AMBIGUOUS":
                        st.warning(f"🟠 Verdict: **{verdict}**")
                    else:
                        st.error(f"🔴 Verdict: **{verdict}**")

                    # Pipeline trace
                    st.caption(f"📍 Pipeline: {crag['branch_used']}")

                    # Answer
                    st.markdown(crag["answer"])
                    st.caption(f"⏱️ Latency: {crag['latency_seconds']}s")

                    # Evaluator scores
                    with st.expander("📊 Evaluator Scores"):
                        for i, (score, reason) in enumerate(zip(crag["scores"], crag["reasons"])):
                            color = "🟢" if score >= upper_t else ("🟠" if score >= lower_t else "🔴")
                            st.markdown(f"{color} **Chunk {i+1}**: {score:.2f} — {reason}")

                    # Rewritten query
                    if crag["rewritten_query"]:
                        st.info(f"🔄 Rewritten query: *{crag['rewritten_query']}*")

                    # Web results
                    if crag["web_results"]:
                        with st.expander("🌐 Web Search Results"):
                            for r in crag["web_results"]:
                                st.markdown(f"**{r['title']}**")
                                st.caption(r["snippet"][:200])
                                st.markdown(f"[{r['url']}]({r['url']})")
                                st.markdown("---")

                    # Retrieved chunks
                    with st.expander("📄 Retrieved Chunks (Raw)"):
                        for i, c in enumerate(crag["local_chunks"]):
                            st.markdown(f"**Chunk {i+1}** (cosine: {c['score']:.3f}, source: {c['source']})")
                            st.text(c["text"][:300])
                            st.markdown("---")

                except Exception as e:
                    st.error(f"CRAG error: {e}")

with tab2:
    st.subheader("📊 Evaluation Results")
    results_path = os.path.join(os.path.dirname(__file__), "eval", "results.csv")

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)

        # Summary metrics
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Baseline Correctness", f"{df['baseline_correctness'].mean():.2f}")
        with mcol2:
            st.metric("CRAG Correctness", f"{df['crag_correctness'].mean():.2f}")
        with mcol3:
            st.metric("Baseline Avg Latency", f"{df['baseline_latency'].mean():.1f}s")
        with mcol4:
            st.metric("CRAG Avg Latency", f"{df['crag_latency'].mean():.1f}s")

        st.markdown("---")

        # Branch distribution chart
        st.subheader("Branch Activation Rates")
        branch_counts = df['actual_verdict'].value_counts()
        st.bar_chart(branch_counts)

        # Correctness comparison chart
        st.subheader("Correctness Comparison")
        comparison_df = pd.DataFrame({
            "Baseline": [df['baseline_correctness'].mean(), df['baseline_faithfulness'].mean()],
            "CRAG": [df['crag_correctness'].mean(), df['crag_faithfulness'].mean()],
        }, index=["Correctness", "Faithfulness"])
        st.bar_chart(comparison_df)

        # Detailed results table
        st.subheader("Detailed Results")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No evaluation results yet. Run: `python -m src.evaluate` to generate them.")
