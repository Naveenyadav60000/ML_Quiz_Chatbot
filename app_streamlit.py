import streamlit as st
import random
import pandas as pd
import joblib

# Load question dataset
df = pd.read_csv("ml_quiz_dataset.csv")

# Load ML model for difficulty prediction
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# List of student names
students = ["Naveen Yadav", "Rajini Kanth", "Nithin", "Mahitha", "Abhishik"]

st.title("🎓 Professor's Spin Wheel Quiz Game")

# Step 1: Spin the wheel
if st.button("🎡 Spin to select a student"):
    chosen_student = random.choice(students)
    st.session_state["selected_student"] = chosen_student
    st.session_state["question_index"] = random.randint(0, len(df) - 1)

# Display student
if "selected_student" in st.session_state:
    st.subheader(f"🎯 Selected Student: {st.session_state['selected_student']}")

    # Step 2: Show random question from CSV
    q = df.iloc[st.session_state["question_index"]]
    st.write("**📖 Question:**", q["question"])
    options = {
        "A": q["option_a"],
        "B": q["option_b"],
        "C": q["option_c"],
        "D": q["option_d"]
    }
    for key, val in options.items():
        st.write(f"**{key}.** {val}")

    # Step 3: Student chooses an answer
    student_answer = st.radio("Select the student's answer:", options=["A", "B", "C", "D"])

    if st.button("✅ Submit Answer"):
        correct_option = q["answer"].strip().lower()
        submitted_option = student_answer.strip().lower()

        if correct_option == submitted_option:
            st.success("✅ Correct Answer!")
        else:
            st.error(f"❌ Wrong Answer! Correct answer is: **{q['answer'].upper()}**")

    # Optional: Predict difficulty using ML
    if st.button("🤖 Predict Question Difficulty"):
        text = f"{q['question']} {q['option_a']} {q['option_b']} {q['option_c']} {q['option_d']}"
        vec = vectorizer.transform([text])
        pred = model.predict(vec)
        difficulty = le.inverse_transform(pred)[0]
        st.info(f"📊 ML Predicted Difficulty: **{difficulty}**")
