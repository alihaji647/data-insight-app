import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-lite"
INSTRUCTION = (
    "You summarize data for a manager. Write three short bullet "
    "points using only the figures in the data supplied, and name "
    "the figure each point is based on."
)

st.set_page_config(page_title="Data Insight App", layout="wide")
st.title("Data Insight App")


def read_key():
    """Retrieves the API key from Streamlit secrets, or "" if none."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


# Read the API key from the secrets, or show a sidebar field for it
api_key = read_key()
if not api_key:
    api_key = st.sidebar.text_input("Gemini API key", type="password")

# Front end: show an uploader for a CSV file
uploaded = st.file_uploader("Upload a CSV file", type="csv")
if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

# Data: read the file into a pandas DataFrame
df = pd.read_csv(uploaded)
st.write(f"The file has {len(df):,} rows and {df.shape[1]} columns.")
st.dataframe(df, height=250)

# List the number columns and the text columns of up to 30 values
num_cols = df.select_dtypes("number").columns.tolist()
text_cols = [c for c in df.columns
             if c not in num_cols and df[c].nunique() <= 30]
if not text_cols or not num_cols:
    st.warning("The file needs a text column and a number column.")
    st.stop()

# Filter: keep the rows that have one of the selected values
st.sidebar.header("Filter")
filter_col = st.sidebar.selectbox("Filter column", text_cols)
values = sorted(df[filter_col].dropna().unique())
chosen = st.sidebar.multiselect("Values to keep", values,
                                default=values)
filtered = df[df[filter_col].isin(chosen)]
st.write(f"The filtered table has {len(filtered):,} rows.")

# Chart: add up the Value column for each group
st.subheader("Chart")
group_cols = [c for c in df.columns if c not in num_cols]
group_col = st.selectbox("Group by", group_cols,
                         index=min(1, len(group_cols) - 1))
value_col = st.selectbox("Value", num_cols,
                         index=len(num_cols) - 1)
totals = (filtered.groupby(group_col)[value_col].sum()
          .sort_values(ascending=False))
st.bar_chart(totals, horizontal=True, sort=False)


@st.cache_data(ttl=3600)
def summarize(prompt, _key):
    """Submits the prompt, then returns the text and token counts."""
    client = genai.Client(api_key=_key)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTION),
    )
    usage = response.usage_metadata
    # Add any thinking tokens, which are billed as output tokens
    tokens_out = (usage.candidates_token_count
                  + (usage.thoughts_token_count or 0))
    return response.text, usage.prompt_token_count, tokens_out


# Model API: request a summary only when the button is pressed
st.subheader("Summary")
send = st.radio("Data sent to the model",
                ["Totals table", "Filtered rows"], horizontal=True)
if send == "Totals table":
    data_text = totals.to_csv()
else:
    data_text = filtered.to_csv(index=False)
prompt = (f"Filter: {filter_col} is one of {', '.join(chosen)}\n"
          f"Rows after filtering: {len(filtered)}\n\n{data_text}")

if st.button("Summarize with Gemini"):
    if not api_key:
        st.error("Add a Gemini API key first.")
    else:
        try:
            text, tokens_in, tokens_out = summarize(prompt, api_key)
            st.markdown(text)
            st.caption(f"Input tokens: {tokens_in:,} · "
                       f"Output tokens: {tokens_out:,}")
        except Exception as error:
            st.error(f"The request failed: {error}")
