import pickle
import streamlit as st
from utils import utils
from gemini import gemini

svc_model = pickle.load(open('./dumps/clf.pkl', 'rb'))
tfidf = pickle.load(open('./dumps/tfidf.pkl', 'rb'))
le = pickle.load(open('./dumps/encoder.pkl', 'rb'))

# Function to predict the category of a resume
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = utils.cleanResume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name


def main():
    st.set_page_config(page_title="Resume Category Prediction",
                       page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown(
        "Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")

    uploaded_file = st.file_uploader(
        "Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = utils.handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Make prediction
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(
                f"The predicted category of the uploaded resume is: **{category}**")

            # Get rating and improvements from Gemini
            gemini_response_text = gemini.getgemini(resume_text)
            rating, improvements = gemini.parse_gemini_response(
                gemini_response_text)

            st.subheader("Resume Rating")
            st.write(f"**Rating:** {rating}/10")

            st.subheader("Suggested Improvements")
            for i, improvement in enumerate(improvements, start=1):
                st.markdown(f"**{i}.** {improvement}")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
