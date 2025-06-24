import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and preprocessors
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

# Load dataset
df = pd.read_csv("beer-servings.csv")
df['continent'].fillna(df['continent'].mode()[0], inplace=True)

# Title
st.title("🍺 Alcohol Consumption Predictor")

# Infographic
st.subheader("Average Alcohol Consumption by Continent")
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=df, x="continent", y="total_litres_of_pure_alcohol", estimator='mean', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("🔢 Enter Your Details")

# Inputs
beer = st.slider("Beer Servings", 0, 500, 100)
spirit = st.slider("Spirit Servings", 0, 500, 50)
wine = st.slider("Wine Servings", 0, 500, 30)
country = st.selectbox("Country", sorted(df['country'].unique()))
continent = st.selectbox("Continent", sorted(df['continent'].unique()))

# Input encoding
def encode_inputs(df, country, continent):
    df_temp = df.copy()
    df_temp['country'] = country
    df_temp['continent'] = continent
    df_temp = pd.get_dummies(df_temp, columns=['country', 'continent'], drop_first=True)
    return df_temp.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("Predict"):
    input_df = df.iloc[[0]].copy()
    input_df['beer_servings'] = beer
    input_df['spirit_servings'] = spirit
    input_df['wine_servings'] = wine
    encoded = encode_inputs(input_df, country, continent)
    X = imputer.transform(encoded)
    X = scaler.transform(X)
    result = model.predict(X)[0]
    st.success(f"Predicted Pure Alcohol Consumption: **{result:.2f} litres**")
