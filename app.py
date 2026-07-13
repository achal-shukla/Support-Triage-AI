import streamlit as st

from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain


# --- Page Setup ---
st.set_page_config(
    page_title="Support Triage AI"
)


# --- Modern dashboard styling ---
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(79,70,229,.18), transparent 28%),
                radial-gradient(circle at 85% 15%, rgba(6,182,212,.12), transparent 25%),
                #0b1020;
}
.block-container {max-width: 1150px; padding-top: 5rem;}
[data-testid="stSidebar"] {background:#111827; border-right:1px solid rgba(255,255,255,.08);}
.hero {
    font-size: clamp(2.7rem, 6vw, 4.8rem); line-height:1.02; font-weight:850;
    letter-spacing:-.05em; margin:.3rem 0 1rem;
    background:linear-gradient(90deg,#fff,#c7d2fe,#67e8f9);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.subtitle {color:#94a3b8; font-size:1.08rem; max-width:760px; margin-bottom:2rem;}
.kicker {color:#818cf8; font-size:.76rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase;}
.pill {display:inline-block; padding:.4rem .75rem; border-radius:999px;
       color:#6ee7b7; background:rgba(16,185,129,.1); border:1px solid rgba(52,211,153,.25);
       font-size:.8rem; font-weight:800; margin:.8rem 0;}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:rgba(15,23,42,.78); border:1px solid rgba(148,163,184,.14);
    border-radius:18px; box-shadow:0 18px 45px rgba(0,0,0,.18);
}
.stTextArea textarea {background:rgba(2,6,23,.7); border-radius:14px;}
.stButton>button {
    width:100%; min-height:3rem; border:0; border-radius:12px; font-weight:800;
    background:linear-gradient(90deg,#4f46e5,#0891b2); color:white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="kicker">AI CUSTOMER OPERATIONS</div>', unsafe_allow_html=True)
st.markdown('<div class="pill">● AI SUPPORT WORKFLOW</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Turn support emails<br>into ready-to-send replies.</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze the issue, detect sentiment, choose the right tone, and generate a concise customer response in one AI workflow.</div>', unsafe_allow_html=True)



# --- API Key Input ---
with st.sidebar:
    st.markdown("##  Support Triage AI")
    st.caption("AI-powered customer support workflow")
    cohere_api_key = st.text_input(
        "Cohere API key",
        type="password",
        placeholder="Enter your API key"
    )
    st.divider()
    st.markdown("**Workflow**")
    st.caption("① Extract issue & sentiment")
    st.caption("② Select response tone")
    st.caption("③ Draft the final reply")


if cohere_api_key:

    # --- Initialize LLM ---
    llm_cohere = ChatCohere(
        cohere_api_key=cohere_api_key,
        temperature=0.7,
        max_tokens=256
    )


    # --- CHAIN 1: Extract Issue & Sentiment ---
    template_1 = """
    Analyze the following customer email.
    Identify the core issue in one sentence, and determine the
    customer's sentiment.

    Email: {email}

    Format your response as:
    Issue: [Core issue]
    Sentiment: [Sentiment]
    """

    prompt_1 = PromptTemplate(
        input_variables=["email"],
        template=template_1
    )

    chain_1 = LLMChain(
        llm=llm_cohere,
        prompt=prompt_1,
        output_key="analysis"
    )


    # --- CHAIN 2: Determine Tone ---
    template_2 = """
    Based on the customer's issue and sentiment,
    determine the appropriate tone for our response.

    Choose exactly one from:
    Apologetic, Technical, Sales-oriented, or Friendly.

    Customer Analysis: {analysis}

    Response Tone:
    Analysis:
    """

    prompt_2 = PromptTemplate(
        input_variables=["analysis"],
        template=template_2
    )

    chain_2 = LLMChain(
        llm=llm_cohere,
        prompt=prompt_2,
        output_key="tone"
    )


    # --- CHAIN 3: Generate Response ---
    template_3 = """
   You are a professional customer support agent.

    Write a reply to the customer addressing their issue.
    Adopt the requested tone.

    Guidelines:
    - For technical issues regarding the game download button
      or payment integration, provide troubleshooting steps.
    - If the customer is angry, use an empathetic and apologetic tone.
    - Keep the response under 150 words.

    Customer Email: {email}
    Analysis: {analysis}
    Required Tone: {tone}

    Draft Response:
    """

    prompt_3 = PromptTemplate(
        input_variables=["email", "analysis", "tone"],
        template=template_3
    )

    chain_3 = LLMChain(
        llm=llm_cohere,
        prompt=prompt_3,
        output_key="final_response"
    )


    # --- Sequential Chain ---
    triage_chain = SequentialChain(
        chains=[chain_1, chain_2, chain_3],
        input_variables=["email"],
        output_variables=[
            "analysis",
            "tone",
            "final_response"
        ],
        verbose=True
    )


    # --- Modern dashboard UI ---
    with st.container(border=True):
        st.markdown('<div class="kicker">INCOMING TICKET</div>', unsafe_allow_html=True)
        st.subheader(" Customer Email")
        customer_email = st.text_area(
            "Customer email",
            height=210,
            placeholder="Paste the customer support email here...",
            label_visibility="collapsed"
        )
        process_email = st.button("⚡ Analyze & Draft Response", use_container_width=True)

    if process_email:
        if customer_email.strip():
            with st.spinner("AI is triaging the ticket and drafting a reply..."):
                try:
                    st.session_state["triage_result"] = triage_chain.invoke(
                        {"email": customer_email}
                    )
                except Exception as error:
                    st.error(f"Error: {error}")
        else:
            st.warning("Please enter an email to process.")

    result = st.session_state.get("triage_result")

    if result:
        st.markdown("---")
        st.markdown('<div class="kicker">TRIAGE RESULTS</div>', unsafe_allow_html=True)
        st.header("AI Decision Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.subheader(" Issue & Sentiment")
                st.caption("Stage 01 · Extraction")
                st.write(result["analysis"])

        with col2:
            with st.container(border=True):
                st.subheader(" Recommended Tone")
                st.caption("Stage 02 · Routing")
                st.write(result["tone"])

        with st.container(border=True):
            st.subheader(" AI-Generated Response")
            st.caption("Stage 03 · Final draft")
            st.write(result["final_response"])
            st.download_button(
                "⬇ Download Response",
                data=result["final_response"],
                file_name="support_response.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.caption("Review AI-generated responses before sending them to customers.")

else:
    st.info(
        "Please enter your Cohere API key "
        "in the sidebar to begin."
    )

