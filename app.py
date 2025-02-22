import streamlit as st
import pandas as pd
import os
import kagglehub
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

st.title("Disease Prediction & Chatbot Assistant 🤖🏥")
st.write("Chat with the bot or enter symptoms to predict a disease!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

path = kagglehub.dataset_download("itachi9604/disease-symptom-description-dataset")
df_disease = pd.read_csv(os.path.join(path, "dataset.csv"))

df_symptoms = df_disease.iloc[:, :4]
df_symptoms.columns = ["Disease", "Symptom_1", "Symptom_2", "Symptom_3"]

df_symptoms_clean = df_symptoms.applymap(lambda x: x.strip().lower().replace(" ", "_") if isinstance(x, str) else x)
df_symptoms_clean["Symptoms"] = df_symptoms_clean[["Symptom_1", "Symptom_2", "Symptom_3"]].values.tolist()
df_symptoms_clean = df_symptoms_clean[["Disease", "Symptoms"]]

all_symptoms = set(df_symptoms_clean["Symptoms"].explode().dropna().unique())

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_symptoms_clean["Symptoms"])
y = df_symptoms_clean["Disease"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

def handle_query(user_input):
    user_input = user_input.lower()

    if "," in user_input:
        user_symptoms = [s.strip().lower().replace(" ", "_") for s in user_input.split(",") if s.strip()]
        valid_symptoms = [s for s in user_symptoms if s in all_symptoms]

        if not valid_symptoms:
            return "❌ Invalid symptoms. Please enter valid symptoms from the dataset."

        symptoms_vector = mlb.transform([valid_symptoms])

        if symptoms_vector.sum() == 0:
            return "❌ No matching symptoms found. Try different symptoms."

        predicted_disease = model.predict(symptoms_vector)[0]
        return f"✅ Based on your symptoms, the possible disease is **{predicted_disease}**. However, consult a doctor for proper diagnosis."

    elif "hello" in user_input or "hi" in user_input:
        return "Hello! How can I assist you with disease prediction?"
    elif "thank you" in user_input or "thanks" in user_input:
        return "You're welcome! Stay healthy. 😊"
    elif "bye" in user_input:
        return "Goodbye! Take care. 👋"
    elif "symptoms" in user_input:
        return "Enter your symptoms separated by commas to get a disease prediction."
    else:
        return "I'm not sure I understand. Try entering symptoms like 'fever, cough, headache'."

user_message = st.text_input("You:", "")

if st.button("Send"):
    if user_message:
        response = handle_query(user_message)
        st.session_state.chat_history.append(("You", user_message))
        st.session_state.chat_history.append(("🤖 Bot", response))

for role, message in st.session_state.chat_history:
    st.write(f"**{role}:** {message}")
