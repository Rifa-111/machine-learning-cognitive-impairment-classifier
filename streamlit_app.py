import streamlit as st
import pandas as pd
import joblib

# Load model and expected feature columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Extract all known scan types from column names
scan_options = sorted([
    col.replace("Description_", "")
    for col in model_columns if col.startswith("Description_")
])

# Page setup
st.set_page_config(page_title="Cognitive Classifier", layout="centered")

# App title and instructions
#st.image("https://upload.wikimedia.org/wikipedia/commons/6/6a/Brain_icon.png", width=60)
st.title("🧠 Cognitive Impairment Classifier")
st.markdown("Enter patient demographics and scan type to predict cognitive status.")
st.divider()

# User inputs
age = st.number_input("👵 Age", min_value=0.0, max_value=120.0, value=70.0)
sex = st.selectbox("⚧ Gender", ["Female", "Male"])
scan_type = st.selectbox("🧪 MRI Scan Description", scan_options)

# Prediction button
if st.button("Predict"):
    # Build input dictionary
    input_data = {
        'Age': [age],
        'Sex': [1 if sex == "Male" else 0]
    }

    # Add all known scan columns with 0
    for col in model_columns:
        if col.startswith("Description_"):
            input_data[col] = [0]

    # Activate selected scan column
    scan_col = f"Description_{scan_type}"
    if scan_col in model_columns:
        input_data[scan_col] = [1]

    # Create DataFrame and align columns
    input_df = pd.DataFrame(input_data)
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    # Predict class and probability
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities)

    # Class labels
    label_map = {0: "CN", 1: "EMCI", 2: "LMCI", 3: "MCI"}
    result = label_map[prediction]

    # Output results
    st.success(f"🧠 Predicted Cognitive State: **{result}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    # Optional: Probability chart
    st.subheader("Prediction Probabilities")
    st.bar_chart(pd.DataFrame(probabilities, index=label_map.values(), columns=["Probability"]))

















# import streamlit as st
# import pandas as pd
# import joblib

# # Load trained model and expected columns
# model = joblib.load("model.pkl")
# model_columns = joblib.load("model_columns.pkl")

# st.title("🧠 Cognitive Impairment Classifier")
# st.write("Enter patient demographics and scan type to predict cognitive status.")

# # Basic user inputs
# age = st.number_input("Age", min_value=0.0, max_value=120.0, value=65.0)
# sex = st.selectbox("Sex", ["Female", "Male"])
# scan_type = st.text_input("MRI Scan Description", "3-plane localizer")

# # Predict on click
# if st.button("Predict"):
#     # Start with Age and Sex
#     input_data = {
#         'Age': [age],
#         'Sex': [1 if sex == "Male" else 0]
#     }

#     # Add all known scan description columns from training as 0
#     for col in model_columns:
#         if col.startswith("Description_"):
#             input_data[col] = [0]

#     # Activate the one matching the user's input
#     scan_col = f"Description_{scan_type}"
#     if scan_col in model_columns:
#         input_data[scan_col] = [1]
#     else:
#         st.warning(f"Scan type '{scan_type}' was not seen during training and may reduce accuracy.")

#     # Build DataFrame with all expected columns
#     input_df = pd.DataFrame(input_data)

#     # Add missing columns (if any) and reorder
#     for col in model_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0
#     input_df = input_df[model_columns]

#     # Predict
#     prediction = model.predict(input_df)[0]
#     label_map = {0: "CN", 1: "EMCI", 2: "LMCI", 3: "MCI"}

#     st.success(f"🧠 Predicted Cognitive State: **{label_map[prediction]}**")














# import streamlit as st
# import pandas as pd
# import joblib

# import streamlit as st

# st.title("✅ Streamlit Test")
# st.write("If you see this, Streamlit is working!")


# # Load the trained model
# model = joblib.load('model.pkl')

# # List of all known scan types from training (simplified list — update with full set if needed)
# scan_options = [
#     "3-plane localizer", "Axial T2-FLAIR", "MPRAGE", "Field Mapping",
#     "SURVEY", "B1-Calibration Body", "B1-Calibration PA", 
#     "ADNI Brain PET: Raw AV45", "ADNI Brain PET: Raw FDG", "Axial T2-Star"
# ]

# # Streamlit UI
# st.title("Cognitive Impairment Classifier")
# st.write("Enter patient info and scan description to predict cognitive status.")

# # Input fields
# age = st.number_input("Age", min_value=0.0, max_value=120.0, value=65.0)
# sex = st.selectbox("Sex", options=["Female", "Male"])
# scan_type = st.selectbox("MRI Scan Description", options=scan_options)

# # Prepare input
# if st.button("Predict"):
#     # Convert inputs to DataFrame
#     input_data = {
#         'Age': [age],
#         'Sex': [1 if sex == 'Male' else 0],
#     }

#     # Add scan description one-hot encoding
#     for scan in scan_options:
#         input_data[f'Description_{scan}'] = [1 if scan == scan_type else 0]

#     input_df = pd.DataFrame(input_data)

#     # Predict
#     prediction = model.predict(input_df)[0]
#     label_map = {0: "CN", 1: "EMCI", 2: "LMCI", 3: "MCI"}

#     st.success(f"Predicted Cognitive State: **{label_map[prediction]}**")
