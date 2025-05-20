import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model for car price prediction
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define categories and models mapping
category_model_mapping = {
    'Jeep': ['Santa FE', 'Tucson', 'Elantra', 'Sonata', 'Kona', 'H1', 'IX35', 'Tucson SE', 'Veracruz', 'Galloper'],
    'Sedan': ['Accent', 'Azera', 'Elantra', 'Elantra GLS', 'Elantra GT', 'Elantra Limited', 'Elantra SE', 'Elantra Sport', 'Genesis',
              'Getz', 'Grandeur', 'H1', 'I30', 'Lantra', 'Santa FE', 'Sonata', 'Sonata SE', 'Sonata Sport', 'Tucson', 'Veloster'],
    'Universal': ['H1', 'I30'],
    'Minivan': ['H1', 'H1 Grand Starex', 'Santa FE'],
    'Coupe': ['Elantra', 'Elantra GS', 'Genesis', 'Veloster'],
    'Hatchback': ['Accent', 'Accent GS', 'Accent SE', 'Elantra', 'Elantra GT', 'I30', 'Getz', 'Ioniq', 'Sonata', 'Veloster', 'Veloster Turbo'],
    'Microbus': ['H1'],
    'Goods wagon': ['I30'],
    'Pickup': ['H1', 'I30']
}

# Sample lists for other fields
leather_interior_options = ['Yes', 'No']
fuel_type_options = ['CNG', 'Diesel', 'Hybrid', 'Hydrogen', 'LPG', 'Petrol']
colors = ['Beige', 'Black', 'Blue', 'Brown', 'Carnelian Red', 'Golden', 'Green', 'Grey', 'Orange', 'Purple', 'Red', 'Silver', 'Sky Blue',
           'White', 'Yellow']


# Function to filter options based on search term
def filter_options(search_term, options):
    if search_term:
        return [option for option in options if search_term.lower() in option.lower()]
    return options

# Function to make predictions (updated based on actual feature names)
def predict_car_price(data):
    # Example mapping, ensure this matches your actual preprocessing
    feature_order = ['Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type',
                     'Engine volume', 'Mileage','Cylinders', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 
                     'Color', 'Airbags', ]

    data['Manufacturer'] = data['Manufacturer'].map({'HYUNDAI': 0})
    # Encode 'Model' using label encoding
    model_map = {model: idx for idx, model in enumerate(pd.Series(data['Model']).unique())}
    data['Model'] = data['Model'].map(model_map)
    data['Category'] = data['Category'].map({'Coupe': 0, 'Goods wagon': 1, 'Hatchback': 2, 'Jeep': 3, 'Microbus': 4, 'Minivan': 5, 'Pickup': 6, 'Sedan': 7, 'Universal': 8})
    data['Wheel']=data['Wheel'].map({'Left Wheel': 0, 'Right Wheel': 1})
    data['Leather interior'] = data['Leather interior'].map({'No': 0, 'Yes': 1})
    data['Fuel type'] = data['Fuel type'].map({'CNG': 0, 'Diesel': 1, 'Hybrid': 2, 'Hydrogen': 3, 'LPG': 4, 'Petrol': 5})
    data['Gear box type'] = data['Gear box type'].map({'Manual': 0, 'Automatic': 1,'Triptronic':2,'Variator':3})
    data['Drive wheels'] = data['Drive wheels'].map({'Front': 0, 'Rear': 1})
    data['Color']=data['Color'].map({'Beige': 0,'Black': 1,'Blue': 2,'Brown': 3,'Carnelian Red': 4,'Golden': 5,'Green': 6,'Grey': 7,
                                     'Orange': 8,'Purple': 9,'Red': 10,'Silver': 11,'Sky Blue': 12,'White': 13,'Yellow': 14})

   #doors

    # Ensure correct column order
    data = data[feature_order]

    # Use the loaded scaler to transform new data
    new_data_scaled = scaler.transform(data)

    # Use the loaded model to predict the price
    prediction = model.predict(new_data_scaled)
    return prediction[0]

# Streamlit App
def main():
    st.header("Vehicle Price Prediction App",divider='rainbow')
    st.subheader("This app predicts the vehicle price based on its specifications.")
    st.divider()

    st.write("Below are features used to predict the vehicle price:")
    st.write("""<table>
            <tr><th>Features</th><th>Description</th></tr>
            <tr><th>Manufacturer</th><td>The brand or manufacturer of the vehicle</td></tr>
            <tr><th>Model</th><td>The specific model of the vehicle</td></tr>
            <tr><th>Prod. year</th><td>The year the vehicle was manufactured</td></tr>
            <tr><th>Category</th><td>The type or class of the vehicle</td></tr>
            <tr><th>Leather interior</th><td>Indicates whether the vehicle has a leather interior</td></tr>
            <tr><th>Fuel type</th><td>The type of fuel the vehicle uses</td></tr>
            <tr><th>Engine volume</th><td>The volume of the vehicle’s engine</td></tr>
            <tr><th>Mileage</th><td>The distance the vehicle has traveled</td></tr>
            <tr><th>Gear box type</th><td>The type of transmission the vehicle has</td></tr>
            <tr><th>Drive wheels</th><td>Which wheels are driven by the engine</td></tr>
            <tr><th>Doors</th><td>The no of doors the vehicle has</td></tr>
            <tr><th>Wheel</th><td>The type or configuration of the wheels</td></tr>
            <tr><th>Color</th><td>The color of the vehicle</td></tr>
            <tr><th>Airbags</th><td>The number of airbags in the vehicle</td></tr>
            <tr><th>Cylinders</th><td>The number of cylinders in the vehicle’s engine</td></tr>
            </table><br>""", unsafe_allow_html=True)

    st.divider()
    st.write('Please enter the vehicle details in the sidebar and click the button below to predict the price.')

    # Sidebar for user inputs
    st.sidebar.header("Insert Vehicle Information")

    def user_input_features():
        # Manufacturer Field - Read-only Text Field with "Hyundai"
        st.sidebar.text_input('Manufacturer', value='HYUNDAI', disabled=True)

        # Category and Model Selection
        category = st.sidebar.selectbox('Select Category', list(category_model_mapping.keys()))
        
        model_options = category_model_mapping.get(category, [])
        model = st.sidebar.selectbox('Select Model', model_options)

        production_year = st.sidebar.number_input('Production Year', 1990, 2024, 2015)
        leather_interior = st.sidebar.selectbox('Leather Interior', leather_interior_options)
        fuel_type = st.sidebar.selectbox('Fuel Type', fuel_type_options)
        doors = st.sidebar.number_input('Number of Doors', 2, 5, 4)
        mileage = st.sidebar.number_input('Mileage (Km)', 0, None, 50000, step=1000)
        airbags = st.sidebar.number_input('Number of Airbags', 0, 16, 2)
        engine_volume = st.sidebar.number_input('Engine Volume (L)', 0.0, 10.0, 1.8, step=0.1)
        cylinders = st.sidebar.number_input('Number of Cylinders', 1, 16, 4)
        gearbox_type = st.sidebar.selectbox('Gearbox Type', ('Manual', 'Automatic','Triptronic','Variator'))
        drive_wheels = st.sidebar.selectbox('Drive Wheels', ('Front', 'Rear'))
        wheel_type = st.sidebar.selectbox('Wheel Type', ('Left Wheel', 'Right Wheel'))
        color = st.sidebar.selectbox('Color',('Beige', 'Black', 'Blue', 'Brown', 'Carnelian Red', 'Golden', 'Green',
                                               'Grey','Orange', 'Purple', 'Red', 'Silver', 'Sky Blue', 'White', 'Yellow'))
        
        
        
       

        # Create a dictionary for user input with exact feature names
        data = {
            'Manufacturer': 'HYUNDAI',  # Fixed value
            'Model': model,
            'Prod. year': production_year,
            'Category': category,
           'Leather interior': leather_interior,
            'Fuel type': fuel_type,
            'Engine volume': engine_volume,
            'Mileage': mileage,
            'Cylinders': cylinders,
            'Gear box type': gearbox_type,
            'Drive wheels': drive_wheels,
            'Doors': doors,
            'Wheel': wheel_type,
            'Color': color,
            'Airbags': airbags,
             }

        # Convert to DataFrame
        return pd.DataFrame(data, index=[0])

    # Get user input
    user_input = user_input_features()

    # Button to trigger predictions
    if st.button('Predict Car Price'):
        if user_input is not None:
            # Make predictions
            predicted_price = predict_car_price(user_input)

            # Display the prediction
            st.subheader('Predicted Car Price:')
            st.success(f"${predicted_price:,.2f}")
        else:
            st.warning('Please insert values first')

if __name__ == '__main__':
    main()
