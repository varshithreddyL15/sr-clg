import streamlit as st
import numpy as np
import os.path
from PIL import Image

# Set page configuration
st.set_page_config(page_title="House Price Prediction", layout="centered")

def load_model():
    """Safely load the prediction model with error handling"""
    try:
        import pickle
        # Try different potential paths for the model
        potential_paths = ["models/model.pkl", "model.pkl", "./model.pkl"]
        
        for path in potential_paths:
            if os.path.exists(path):
                with open(path, "rb") as pickle_in:
                    return pickle.load(pickle_in)
        
        # If model not found, raise exception
        raise FileNotFoundError("Model file not found in any of the expected locations")
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure the model file exists in the correct location.")
        return None

# Prediction function with error handling
def predict(model, house_age, distance_to_nearest_metro, no_of_nearby_stores):
    """Make prediction with the loaded model"""
    try:
        inputs = np.array([house_age, distance_to_nearest_metro, no_of_nearby_stores], dtype=np.float64)
        prediction = model.predict([inputs])
        return prediction[0]
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

# Main function to run the Streamlit app
def main():
    st.title("🏡 House Price Prediction")

    # Custom header with HTML styling
    st.markdown("""
        <div style='background-color: #1E3A8A; padding: 10px; border-radius: 10px'>
            <h2 style='color: white; text-align: center;'>HOUSE PRICE PREDICTION</h2>
        </div>
    """, unsafe_allow_html=True)

    # Show house image
    try:
        img_paths = ["house.jpg", "images/house.jpg", "./house.jpg"]
        img_loaded = False
        
        for img_path in img_paths:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, width=300, caption="House-worth")
                img_loaded = True
                break
                
        if not img_loaded:
            st.info("Image 'house.jpg' not found. The app will function without the image.")
    except Exception as e:
        st.warning(f"Cannot display image: {str(e)}")

    # Load the model
    model = load_model()
    
    if model is None:
        st.error("The prediction model could not be loaded. Please check the model file.")
        return

    # Create columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        house_age = st.number_input("House Age (in years)", min_value=0.0, step=1.0)
    
    with col2:
        distance_to_nearest_metro = st.number_input("Distance to Nearest Metro (in km)", min_value=0.0, step=0.1)
    
    no_of_nearby_stores = st.number_input("Number of Nearby Stores", min_value=0, step=1)

    # Add a prediction button
    if st.button("Predict House Price", key="predict_button", type="primary"):
        # Show a spinner while making prediction
        with st.spinner("Calculating prediction..."):
            result = predict(model, house_age, distance_to_nearest_metro, no_of_nearby_stores)
            
            if result is not None:
                st.success(f"🏠 Predicted House Price: ₹{result:,.2f}")
                
                # Display additional information in an expander
                with st.expander("Prediction Details"):
                    st.write("Input parameters:")
                    st.write(f"- House Age: {house_age} years")
                    st.write(f"- Distance to Metro: {distance_to_nearest_metro} km")
                    st.write(f"- Nearby Stores: {no_of_nearby_stores}")

# Run the app
if __name__ == '__main__':
    main()