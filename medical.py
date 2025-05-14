import streamlit as st 
from transformers import pipeline
import fitz  # PyMuPDF
import pandas as pd
import io

qa = pipeline('question-answering')
summary_pipeline = pipeline('summarization')

st.title("🩺 Medical Report Summarizer")

uploaded_file = st.file_uploader("Upload a medical file [PDF, TXT, DATA, CSV]", type=['pdf', 'txt', 'data', 'csv'])

# Extract text based on file type
def extract_text(file_bytes, file_name):
    if file_name.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif file_name.endswith((".txt", ".data")):
        return file_bytes.decode("utf-8")
    elif file_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.to_string(index=False)
    else:
        return ""

# Extract and store the text only once
if uploaded_file and "extracted_text" not in st.session_state:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    text = extract_text(file_bytes, file_name)

    if text.strip():
        st.session_state['extracted_text'] = text
        st.success("✅ Text extracted successfully.")
    else:
        st.warning("⚠️ No readable text found.")

# Show extracted text
if "extracted_text" in st.session_state:
    st.subheader("📄 Raw Extracted Text")
    st.text_area("Extracted Text", value=st.session_state['extracted_text'], height=300)

    st.subheader("❓ Ask a Question About the Report")
    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer") and user_question:
        with st.spinner("Searching for answer..."):
            answer = qa(question=user_question, context=st.session_state['extracted_text'])
            st.subheader("🔎 Answer:")
            st.write(answer['answer'] if 'answer' in answer else "❌ No answer found.")
            
    if st.button("Get Report Summary"):
        if "extracted_text" in st.session_state and st.session_state["extracted_text"].strip():
            with st.spinner("Summarizing report..."):
                # Break the long text into chunks for summarization
                text = st.session_state["extracted_text"]
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                summary_text = ""
                for chunk in chunks:
                    summary = summary_pipeline(chunk, max_length=130, min_length=30, do_sample=False)
                    summary_text += summary[0]['summary_text'] + " "

                st.subheader("📝 Report Summary:")
                st.write(summary_text.strip())

                # Store summary in session_state for Q&A
                st.session_state["summary"] = summary_text.strip()
        else:
            st.warning("Please upload a file and extract text first.")
