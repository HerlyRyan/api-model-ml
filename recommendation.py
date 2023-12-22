import os
import uvicorn
import traceback
import tensorflow as tf
import pandas as pd

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sklearn.neighbors import NearestNeighbors

model_path = "./recommendation_model.h5"
model = tf.keras.models.load_model(model_path)

# Load the dataset
data = pd.read_csv("data_rating3.csv") 

# Remove duplicates from the dataset
data_no_duplicates = data.drop_duplicates(subset=['user_id', 'product'])

# Create a FastAPI app
app = FastAPI()

# Define the request model for prediction with product_name
class RatingPredictionRequest(BaseModel):
    user_id: int
    product_name: str

# Define the response model for prediction
class RatingPredictionResponse(BaseModel):
    prediction: float

# Define the request model for recommendation
class RatingRecommendationRequest(BaseModel):
    user_id: int

# Define the response model for recommendation with product names and ratings
class RatingRecommendationResponseWithNamesAndRatings(BaseModel):
    recommendations: dict
    average_rating: float

# Define the endpoint for prediction
@app.post("/predict_rating", response_model=RatingPredictionResponse)
def predict_rating(request: RatingPredictionRequest):
    try:
        # Find the product_id based on product_name
        product_rating = data_no_duplicates[(data_no_duplicates['product'] == request.product_name) & (data_no_duplicates['user_id'] == request.user_id)]['rating'].values
        if len(product_rating) == 0:
            raise HTTPException(status_code=404, detail="Product not found for the given user")
        result = product_rating.mean()
        # Return the prediction
        return {"prediction": float(result)}

    except HTTPException as e:
        raise  # Reraise HTTPException to maintain status code and detail
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.get("/get_recommendation/{user_id}", response_model=RatingRecommendationResponseWithNamesAndRatings)
def get_recommendation(user_id: int):
    try:
        # Aggregate ratings for duplicate entries (taking the average)
        user_ratings = data.groupby(['user_id', 'product'], as_index=False)['rating'].mean()

        # Filter data based on the user's rating
        user_ratings_filtered = user_ratings[user_ratings['user_id'] == user_id][['user_id', 'product', 'rating']]

        # Check if there are enough samples for the specified user
        if len(user_ratings_filtered) >= 3:
            # Create a pivot table for user-item matrix
            user_item_matrix = user_ratings_filtered.pivot(index='user_id', columns='product', values='rating').fillna(0)

            # Fit Nearest Neighbors model
            knn_model = NearestNeighbors(metric='cosine', algorithm='brute')

            # Check if there are enough samples for fitting the model
            if len(user_item_matrix) >= 2:
                knn_model.fit(user_item_matrix)

                # Find k-neighbors for the given user
                _, indices = knn_model.kneighbors(user_item_matrix.loc[[user_id]], n_neighbors=1)

                # Get recommended products with their ratings
                recommended_products_data = user_ratings.loc[user_ratings['user_id'].isin(indices[0])][['product', 'rating']].drop_duplicates()

                # Create a dictionary of recommended products with their ratings
                recommended_products_dict = {}
                for _, row in recommended_products_data.iterrows():
                    recommended_products_dict[row["product"]] = float(row["rating"])

                # Calculate the average rating of recommended products
                if recommended_products_data.empty:
                    average_rating = 0.0
                else:
                    average_rating = recommended_products_data['rating'].mean()

                return {"recommendations": recommended_products_dict, "average_rating": float(average_rating)}
            else:
                return {"recommendations": {}, "average_rating": 0.0}
        else:
            return {"recommendations": {}, "average_rating": 0.0}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run the FastAPI app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
