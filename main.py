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
data["Activity Type"] = data.get("Activity Type", pd.Series(["other"]*len(data))).fillna("other")

# Encode gender
gender_map = {"Male": 1, "Female": 0}
data["Gender_numeric"] = data["Gender"].map(gender_map)

# Age range helper
age_ranges = sorted(data["Age Range"].unique().tolist())

def get_age_range(age):
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

# Clinical activity type filtering function
def get_allowed_activity_types(bipolar_stage):
    if bipolar_stage in ["Manic", "Hypomanic"]:
        return ["routine", "other"]  # Avoid happiness and many physical activities
    elif bipolar_stage == "Depression":
        return ["happiness", "physical", "routine", "other"]  # Encourage happiness and physical activities
    elif bipolar_stage == "Euthymia":
        return ["routine", "other"]  # Continue routine
    else:
        return ["routine", "other"]  # Default safe choice

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

@app.post("/recommendations")
def get_recommendations(user_input: UserInput):
    try:
        user_age_range = get_age_range(user_input.age)
        if not user_age_range:
            raise HTTPException(status_code=400, detail="Invalid age range")

        # Encode user input
        user_encoded = encoder.transform([[user_input.bipolar_stage, user_input.mood, user_age_range]])
        user_vector = np.hstack((
            np.zeros(text_features.shape[1]),  # no user text features
            user_encoded.flatten(),
            [gender_map[user_input.gender]]
        )).reshape(1, -1)

        # Similarity calculation
        similarity = cosine_similarity(combined_features, user_vector)
        sorted_indices = similarity.flatten().argsort()[::-1]

        allowed_types = get_allowed_activity_types(user_input.bipolar_stage)

        recommendations = []
        seen_activities = set()

        for idx in sorted_indices:
            if len(recommendations) >= 5:
                break

            activity_name = data["Recommended Activities"].iloc[idx]
            activity_gender = data["Gender"].iloc[idx]
            activity_type = data["Activity Type"].iloc[idx]

            # Clinical filtering by activity type
            if activity_type not in allowed_types:
                continue

            # Gender filtering
            if user_input.gender == "Male" and activity_gender != "Male":
                continue
            elif user_input.gender == "Female" and activity_gender != "Female":
                continue

            if activity_name in seen_activities:
                continue

            recommendations.append({
                "activity": activity_name,
                "description": data["Activity Description"].iloc[idx],
                "duration": int(data["Duration (minutes)"].iloc[idx]),
                "image_url": data["Image URL"].iloc[idx]
            })

            seen_activities.add(activity_name)

        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        return {
            "message": f"Top 5 Recommendations for {user_input.bipolar_stage}",
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
