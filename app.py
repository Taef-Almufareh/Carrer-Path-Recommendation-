import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier

st.title("🎓 Career Recommendation System")

# Upload dataset
file = st.file_uploader("Upload dataset (Excel or CSV)", type=["xlsx","csv"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Prepare data
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    X = df.drop(columns=["career"])
    y = df["career"]

    X_encoded = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = RandomForestClassifier()
    model.fit(X_scaled, y_encoded)

    st.header("Enter Student Data")

    programming = st.slider("Programming", 40,100,80)
    algorithms = st.slider("Algorithms", 40,100,75)
    databases = st.slider("Databases", 40,100,70)
    networks = st.slider("Networks", 40,100,65)
    software_engineering = st.slider("Software Engineering", 40,100,80)
    machine_learning = st.slider("Machine Learning", 40,100,75)
    security = st.slider("Security", 40,100,60)



if st.button("Recommend"):
    new_data = pd.DataFrame([{
        "programming":programming,
        "algorithms":algorithms,
        "databases":databases,
        "networks":networks,
        "software_engineering":software_engineering,
        "machine_learning":machine_learning,
        "security":security
    }])

    new_data = pd.get_dummies(new_data)
    new_data = new_data.reindex(columns=X_encoded.columns, fill_value=0)

    new_scaled = scaler.transform(new_data)

    # 🔥 Get probability for all careers
    probs = model.predict_proba(new_scaled)[0]

    # 🔥 Get top 3 indices
    top_3_idx = np.argsort(probs)[-3:][::-1]

    # 🔥 Convert to career names
    top_3_careers = label_encoder.inverse_transform(top_3_idx)
    top_3_scores = probs[top_3_idx]

    st.success("Top 3 Recommended Careers:")

    for i in range(3):
        st.write(f"{i+1}. {top_3_careers[i]} — Confidence: {top_3_scores[i]:.2f}")
