import pickle
import streamlit as st


model = pickle.load(open("spam.pkl","rb"))
cv = pickle.load(open("vectorizer.pkl","rb"))


def main():
    st.title("Email Spam Classification Apps")
    st.subheader("build With Streamlit and Python")
    msg = st.text_input("Enter a Text:")
    if st.button("Predict"):
        data = [msg]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]
        if result == 1:
            st.error("This is a Spam Mail")
        else:
            st.success("This is a Ham Mail")
main()