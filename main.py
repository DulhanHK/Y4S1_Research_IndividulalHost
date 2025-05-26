from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv("Updated_Bipolar_Dataset.csv")
data = data.fillna("")  # Replace NaN with empty string

# Ensure Activity Type column exists and fill missing with 'other'
data["Activity Type"] = data.get("Activity Type", pd.Series(["other"] * len(data))).fillna("other")

# Encode gender
gender_map = {"Male": 1, "Female": 0}
data["Gender_numeric"] = data["Gender"].map(gender_map)

# Age range helper
age_ranges = sorted(data["Age Range"].unique().tolist())

def get_age_range(age: int) -> str | None:
    for range_str in age_ranges:
        min_age, max_age = map(int, range_str.split('-'))
        if min_age <= age <= max_age:
            return range_str
    return None

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[["Bipolar Stage", "Mood", "Age Range"]])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(["Bipolar Stage", "Mood", "Age Range"]))
data = data.join(encoded_df)

# TF-IDF vectorization for text features
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(data["Recommended Activities"] + " " + data["Activity Description"])

# Combine all features
combined_features = np.hstack((text_features.toarray(), encoded_features, data[["Gender_numeric"]].to_numpy()))

app = FastAPI()

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
        valid_stages = data["Bipolar Stage"].unique().tolist()
        if value not in valid_stages:
            raise ValueError(f"Invalid bipolar stage. Valid options: {', '.join(valid_stages)}")
        return value

    @field_validator("mood")
    def validate_mood(cls, value):
        valid_moods = data["Mood"].unique().tolist()
        if value not in valid_moods:
            raise ValueError(f"Invalid mood. Valid options: {', '.join(valid_moods)}")
        return value

    @field_validator("age")
    def validate_age(cls, value):
        if get_age_range(value) is None:
            raise ValueError(f"No matching age range for age {value}. Available ranges: {', '.join(age_ranges)}")
        return value

@app.post("/recommendations/")
def get_recommendations(user_input: UserInput):
    try:
        # Encode user input as DataFrame to avoid sklearn warnings
        user_df = pd.DataFrame(
            [[user_input.bipolar_stage, user_input.mood, get_age_range(user_input.age)]],
            columns=["Bipolar Stage", "Mood", "Age Range"]
        )
        user_encoded = encoder.transform(user_df)

        user_input_vector = np.hstack((
            np.zeros(text_features.shape[1]),  # No text features for user input
            user_encoded.flatten(),
            [gender_map[user_input.gender]]
        )).reshape(1, -1)

        similarity = cosine_similarity(combined_features, user_input_vector)
        sorted_indices = similarity.flatten().argsort()[::-1]  # descending similarity

        recommendations = []
        seen_activities = set()

        # Iterate over sorted indices to collect unique, gender-matching recommendations
        for idx in sorted_indices:
            if len(recommendations) >= 5:
                break

            activity_name = data.at[idx, "Recommended Activities"]
            activity_gender = data.at[idx, "Gender"]

            if user_input.gender != activity_gender:
                continue  # skip if gender doesn't match

            if activity_name in seen_activities:
                continue  # skip duplicates

            image_url = data.at[idx, "Image URL"]

            recommendations.append({
                "activity": activity_name,
                "description": data.at[idx, "Activity Description"],
                "duration": int(data.at[idx, "Duration (minutes)"]),
                "image_url": image_url
            })
            seen_activities.add(activity_name)

        # If less than 5 found, fill from remaining regardless of gender but avoid duplicates
        if len(recommendations) < 5:
            for idx in sorted_indices:
                if len(recommendations) >= 5:
                    break

                activity_name = data.at[idx, "Recommended Activities"]
                if activity_name in seen_activities:
                    continue

                image_url = data.at[idx, "Image URL"]

                recommendations.append({
                    "activity": activity_name,
                    "description": data.at[idx, "Activity Description"],
                    "duration": int(data.at[idx, "Duration (minutes)"]),
                    "image_url": image_url
                })
                seen_activities.add(activity_name)

        return {"message": "Top 5 Recommendations", "recommendations": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
