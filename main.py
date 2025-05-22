from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import os

# Load dataset
data = pd.read_csv("Updated_Bipolar_Dataset.csv")
data = data.fillna("")  # Replace NaN values with empty strings

# Encode gender (Male -> 1, Female -> 0)
gender_map = {"Male": 1, "Female": 0}
data["Gender_numeric"] = data["Gender"].map(gender_map)

# Get unique age ranges from dataset
age_ranges = sorted(data["Age Range"].unique().tolist())

# Function to determine age range
def get_age_range(age):
    for range_str in age_ranges:
        min_age, max_age = map(int, range_str.split('-'))
        if min_age <= age <= max_age:
            return range_str
    return None  # Return None if age doesn't fit into any range

# One-hot encode 'Bipolar Stage', 'Mood', and 'Age Range'
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[["Bipolar Stage", "Mood", "Age Range"]])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(["Bipolar Stage", "Mood", "Age Range"]))
data = data.join(encoded_df)

# TF-IDF vectorization for text-based recommendations
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(data["Recommended Activities"] + " " + data["Activity Description"])

# Combine all features for similarity comparison
combined_features = np.hstack((text_features.toarray(), encoded_features, data[["Gender_numeric"]].to_numpy()))

# Define FastAPI app
app = FastAPI()

# Define input model

class UserInput(BaseModel):
    mood: str
    bipolar_stage: str
    age: int
    gender: str

    @field_validator("gender")
    def validate_gender(cls, value):
        if value not in gender_map:
            raise ValueError("Invalid gender. Please choose 'Male' or 'Female'.")
        return value

    @field_validator("bipolar_stage")
    def validate_bipolar_stage(cls, value):
        valid_bipolar_stages = data["Bipolar Stage"].unique().tolist()
        if value not in valid_bipolar_stages:
            raise ValueError(f"Invalid bipolar stage. Valid options are: {', '.join(valid_bipolar_stages)}.")
        return value

    @field_validator("mood")
    def validate_mood(cls, value):
        valid_moods = data["Mood"].unique().tolist()
        if value not in valid_moods:
            raise ValueError(f"Invalid mood. Valid options are: {', '.join(valid_moods)}.")
        return value

    @field_validator("age")
    def validate_age(cls, value):
        age_range = get_age_range(value)
        if age_range is None:
            raise ValueError(f"No matching age range found for age {value}. Available ranges: {', '.join(age_ranges)}")
        return value

@app.post("/recommendations")
@app.post("/recommendations/")
def get_recommendations(user_input: UserInput):
    try:
        # Convert age to age range
        user_age_range = get_age_range(user_input.age)
        if not user_age_range:
            raise HTTPException(status_code=400, detail="Invalid age. No matching range found.")

        # Encode user input
        user_encoded = encoder.transform([[user_input.bipolar_stage, user_input.mood, user_age_range]])
        user_input_vector = np.hstack((np.zeros(text_features.shape[1]), user_encoded.flatten(), [gender_map[user_input.gender]]))

        # Ensure the input vector is the correct shape
        user_input_vector = user_input_vector.reshape(1, -1)

        # Calculate similarity using precomputed similarity matrix
        similarity = cosine_similarity(combined_features, user_input_vector)
        sorted_indices = similarity.flatten().argsort()[::-1]  # Sort by descending similarity

        # Collect unique recommendations
        recommendations = []
        seen_activities = set()  # Track unique activity names

        for idx in sorted_indices:
            if len(recommendations) >= 5:
                break  # Stop when we have 5 unique recommendations

            activity_name = data["Recommended Activities"].iloc[idx]
            activity_gender = data["Gender"].iloc[idx]

            # Skip the activity if it does not match the user's gender
            if user_input.gender == "Male" and activity_gender != "Male":
                continue
            elif user_input.gender == "Female" and activity_gender != "Female":
                continue

            if activity_name in seen_activities:
                continue  # Skip duplicate activities

            # Directly use the image URL from the dataset
            image_url = data["Image URL"].iloc[idx]

            recommendation = {
                "activity": activity_name,
                "description": data["Activity Description"].iloc[idx],
                "duration": int(data["Duration (minutes)"].iloc[idx]),  # Convert to integer
                "image_url": image_url  # Use the Firebase Storage URL directly
            }

            seen_activities.add(activity_name)  # Mark activity as added
            recommendations.append(recommendation)

        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found based on the provided input.")

        return {
            "message": f"Top 5 Recommendations for {user_input.bipolar_stage}",
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
