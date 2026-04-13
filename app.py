import streamlit as st
import Graph

st.set_page_config(
    page_title = "TruthGate",
    page_icon  = "🔍",
    layout     = "centered"
)

st.title("🔍 TruthGate: Three Stage Moderation Framework")
st.write("Enter a comment or text-based post to analyze its toxicity, domain, and misinformation verdict.")

input_text = st.text_area("Enter the Comment / Text based post:", height=150)

if st.button("Analyze"):
    if input_text.strip():
        with st.spinner("Analyzing..."):
            result = Graph.graph.invoke({"original": input_text.strip()})

        st.subheader("Analysis Results")

        # ── Row 1: Verdict / Flag / Severity / Discrepancy ────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Verdict",           result.get("verdict",           "N/A"))
        col2.metric("Toxicity Flag",      result.get("toxicity_flag",     "N/A"))
        col3.metric("Severity",           result.get("severity",          "N/A"))
        col4.metric("Discrepancy Score",  result.get("discrepancy_score", "N/A"))

        # ── Misleading Angle ───────────────────────────────────────────────
        if result.get("misleading_angle"):
            st.markdown("### ⚠️ What is Misleading People")
            st.error(result["misleading_angle"])

        # ── Misinformation Type + Target ──────────────────────────────────
        col_a, col_b = st.columns(2)
        col_a.write(f"**Misinformation Type:** {result.get('misinformation_type', 'N/A')}")
        col_b.write(f"**Target Audience:** {result.get('target_audience', 'N/A')}")

        # ── Corrected Claim ────────────────────────────────────────────────
        if result.get("corrected_claim"):
            st.markdown("### ✅ What the Truth Actually Is")
            st.success(result["corrected_claim"])

        # ── Summary ───────────────────────────────────────────────────────
        if result.get("summary"):
            st.markdown("### 📋 Summary")
            st.info(result["summary"])

        # ── Key Differences ───────────────────────────────────────────────
        if result.get("differences"):
            st.markdown("### 🔎 Key Differences Found")
            for diff in result["differences"]:
                st.warning(f"• {diff}")

        # ── Extracted Details ──────────────────────────────────────────────
        st.markdown("### 📌 Extracted Details")
        st.write(f"**Primary Domain:** {result.get('primary_domain', 'N/A')}")
        st.write(f"**Claim:** {result.get('claim', 'N/A')}")

        # ── Scores ────────────────────────────────────────────────────────
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if result.get("toxic_scores"):
                st.markdown("**Toxic Scores**")
                st.json(result["toxic_scores"])
        with col_s2:
            if result.get("domain_scores"):
                st.markdown("**Domain Scores**")
                st.json(result["domain_scores"])

        # ── ReAct Trace ───────────────────────────────────────────────────
        with st.expander("🧠 ReAct Trace (Thoughts / Actions / Observations)"):
            st.markdown("**Thoughts**")
            for t in result.get("thoughts", []):
                st.write(f"• {t}")

            st.markdown("**Actions**")
            for a in result.get("actions", []):
                st.write(f"• [{a.get('timestamp', '')}] {a.get('action', '')}")

            st.markdown("**Observations**")
            for o in result.get("observations", []):
                st.write(f"• {o}")

        # ── Raw State ─────────────────────────────────────────────────────
        with st.expander("📦 Full Raw State"):
            st.json(result)

    else:
        st.warning("Please enter some text to analyze.")