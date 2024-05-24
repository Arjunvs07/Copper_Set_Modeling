import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from io import StringIO
import uuid
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("C:\\Users\\ADMIN\\New folder\\Copper_Set.csv")

def data_preprocessing(df):
    #Preprocessing
    uuids = [uuid.uuid4() for _ in range(df['id'].isnull().sum())]
    # Assign UUIDs to null IDs
    df.loc[df['id'].isnull(), 'id'] = uuids

    # Define a function to convert the float date to a proper date format
    def convert_to_date(date_float):
        # Check if the value is NaN
        if pd.isna(date_float):
            return None
        else:
            date_str = str(int(float(date_float)))
            if len(date_str) == 8:
                try:
                    # Parse string into datetime object
                    return pd.to_datetime(date_str, format='%Y%m%d')
                except ValueError:
                    # Return None for invalid date values
                    return None
            else:
                # Return None for invalid date values
                return None

    # Apply the conversion function to the 'delivery date' column
    df['item_date'] = df['item_date'].apply(convert_to_date)

    default_date = df['item_date'].min()  # or max() for the latest date
    df['item_date'].fillna(default_date, inplace=True)

    # Convert datetime objects to numerical representation (day count from a reference date)
    df['numerical_item_date'] = (df['item_date'] - df['item_date'].min()).dt.days

    # Reshape numerical representation to be a 2D array (required by MinMaxScaler)
    numerical_dates = df['numerical_item_date'].values.reshape(-1, 1)

    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    df['item_date'] = scaler.fit_transform(numerical_dates)

    customer_med = df['customer'].median()
    df['customer'].fillna(customer_med, inplace = True)

    # Instantiate the SimpleImputer with a strategy (e.g., most frequent)
    imputer = SimpleImputer(strategy='most_frequent')

    # Fit the imputer to the column 'country' (assuming it's numeric)
    imputer.fit(df[['country']])

    # Transform the 'country' column, replacing NaN values with the most frequent value
    df['country'] = imputer.transform(df[['country']])

    mode_status = df['status'].mode()[0]
    df['status'].fillna(mode_status, inplace=True)

    mode_application = df['application'].mode()[0]
    df['application'].fillna(mode_application, inplace=True)

    mode_thickness = df['thickness'].mode()[0]
    df['thickness'].fillna(mode_thickness, inplace=True)

    mask = df['material_ref'].str.startswith('00000').fillna(False)
    # Replace the values that match the condition with None
    df.loc[mask, 'material_ref'] = None
    default_code = 'UNKNOWN'
    df['material_ref'].fillna(default_code, inplace=True)

    # Define a function to convert the float date to a proper date format
    def convert_to_date(date_float):
        # Check if the value is NaN
        if pd.isna(date_float):
            return None
        else:
            date_str = str(int(float(date_float)))
            if len(date_str) == 8:
                try:
                    # Parse string into datetime object
                    return pd.to_datetime(date_str, format='%Y%m%d')
                except ValueError:
                    # Return None for invalid date values
                    return None
            else:
                # Return None for invalid date values
                return None

    # Apply the conversion function to the 'delivery date' column
    df['delivery date'] = df['delivery date'].apply(convert_to_date)

    default_date = df['delivery date'].min()  # or max() for the latest date
    df['delivery date'].fillna(default_date, inplace=True)

    # Convert datetime objects to numerical representation (day count from a reference date)
    df['numerical_item_date'] = (df['delivery date'] - df['delivery date'].min()).dt.days

    # Reshape numerical representation to be a 2D array (required by MinMaxScaler)
    numerical_dates = df['numerical_item_date'].values.reshape(-1, 1)

    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    df['delivery date'] = scaler.fit_transform(numerical_dates)

    mean_selling_price = df['selling_price'].mean()
    df['selling_price'].fillna(mean_selling_price, inplace=True)

    df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
    quantity = df['quantity tons'].mean()
    df['quantity tons'].fillna(quantity, inplace= True)

    return(df)

def quantity_tons_plot(df):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("### with outlier")
    # Plot boxplot with outliers
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
        plt.boxplot(df['quantity tons'], patch_artist=True, showfliers=True)
        plt.title('Boxplot of Quantity Tons with Outliers')
        plt.ylabel('Quantity (Tons)')
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'quantity tons'
        plt.hist(df['quantity tons'], bins=20)
        plt.xlabel('quantity tons')
        plt.ylabel('Frequency')
        st.pyplot()

    # Interquartile Range (IQR) method
    Q1 = df['quantity tons'].quantile(0.25)
    Q3 = df['quantity tons'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    median = df['quantity tons'].median()

    # Replace outliers with the median value
    df['quantity tons'] = df['quantity tons'].apply(lambda x: median if x < lower_bound or x > upper_bound else x)

    df['quantity tons'] = np.cbrt(df['quantity tons'])

    st.write("### without outliers")

    col1,col2 = st.columns(2)
    with col1:
        # Plot boxplot without outliers
        plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
        plt.boxplot(df['quantity tons'], patch_artist=True, showfliers=True)
        plt.title('Boxplot of Quantity Tons without Outliers')
        plt.ylabel('Quantity (Tons)')
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'quantity tons'
        plt.hist(df['quantity tons'], bins=20)
        plt.xlabel('quantity tons')
        plt.ylabel('Frequency')
        st.pyplot()
    return(df)

def plot_country(df):
    st.write("## with outliers")
    col1,col2 = st.columns(2)
    with col1:
        # Create a boxplot with outliers
        plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
        plt.boxplot(df['country'], patch_artist=True, showfliers=True)
        plt.title('Boxplot of Country with Outliers')
        plt.xlabel('Country')
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'quantity tons'
        plt.hist(df['country'], bins=20)
        plt.xlabel('country')
        plt.ylabel('Frequency')
        st.pyplot()

    # Perform Box-Cox transformation
    transformed_data, lambda_value = boxcox(df['country'])

    # Assign the transformed values to a new column in the DataFrame
    df['country'] = transformed_data
    st.write("## without outliers")
    col1,col2 = st.columns(2)
    with col1:
        # Create a boxplot with outliers
        plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
        plt.boxplot(df['country'], patch_artist=True, showfliers=True)
        plt.title('Boxplot of Country without Outliers')
        plt.xlabel('Country')
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'quantity tons'
        plt.hist(df['country'], bins=20)
        plt.xlabel('country')
        plt.ylabel('Frequency')
        st.pyplot()
    return(df)

def plot_customer(df):
    st.write("## with outliers")
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8,6))
        plt.boxplot(df['customer'], patch_artist=True,showfliers=True)
        plt.title("Boxplot of customer with outliers")
        plt.xlabel("Customer")
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'customer'
        plt.hist(df['customer'], bins=20)
        plt.xlabel('customer')
        plt.ylabel('Frequency')
        st.pyplot()

    def treat_outliers_isolation_forest(data, column):
        # Reshape the data for Isolation Forest
        X = data[column].values.reshape(-1, 1)
        
        # Instantiate IsolationForest
        clf = IsolationForest(contamination=0.1, random_state=42)  # Contamination is the expected proportion of outliers
        
        # Fit the model
        clf.fit(X)
        
        # Predict outliers
        outliers = clf.predict(X)
        
        # Marking outliers and replacing them with NaNs
        data[column] = np.where(outliers == -1, np.nan, data[column])
        
        return data

    df = treat_outliers_isolation_forest(df, 'customer')

    mean_value = df['customer'].mean()
    df['customer'].fillna(mean_value, inplace=True)

    df['customer'] = np.log(df['customer'])
    st.write("## without outliers")
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8,6))
        plt.boxplot(df['customer'], patch_artist=True,showfliers=True)
        plt.title("Box plot of customer without outliers")
        plt.xlabel("Customer")
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'customer'
        plt.hist(df['customer'], bins=20)
        plt.xlabel('customer')
        plt.ylabel('Frequency')
        st.pyplot()
    return(df)

def plot_application(df):
    st.write("## with outliers")
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8,6))
        plt.boxplot(df['application'], patch_artist=True,showfliers=True)
        plt.title("Boxplot of application with outliers")
        plt.xlabel("Application")
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'customer'
        plt.hist(df['application'], bins=20)
        plt.xlabel('Application')
        plt.ylabel('Frequency')
        st.pyplot()

    df['application'] = np.log(df['application'])

    st.write("## without outliers")
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8,6))
        plt.boxplot(df['application'], patch_artist=True,showfliers=True)
        plt.title("Boxplot of application without outliers")
        plt.xlabel("Application")
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'customer'
        plt.hist(df['application'], bins=20)
        plt.xlabel('Application')
        plt.ylabel('Frequency')
        st.pyplot()
    return(df)

def plot_thickness(df):
    st.write("## with outliers")
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8,6))
        plt.boxplot(df['thickness'],patch_artist=True,showfliers=True)
        plt.title("Boxplot of thickness with outliers")
        plt.xlabel("Thickness")
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'Thickness'
        plt.hist(df['thickness'], bins=20)
        plt.xlabel('Thickness')
        plt.ylabel('Frequency')
        st.pyplot()

    def treat_outliers_isolation_forest(data, column):
        # Reshape the data for Isolation Forest
        X = data[column].values.reshape(-1, 1)
        
        # Instantiate IsolationForest
        clf = IsolationForest(contamination=0.1, random_state=42)  # Contamination is the expected proportion of outliers
        
        # Fit the model
        clf.fit(X)
        
        # Predict outliers
        outliers = clf.predict(X)
        
        # Marking outliers and replacing them with NaNs
        data[column] = np.where(outliers == -1, np.nan, data[column])
        
        return data

    df = treat_outliers_isolation_forest(df, 'thickness')

    mean_value = df['thickness'].mean()
    df['thickness'].fillna(mean_value, inplace=True)

    df['thickness'] = np.log(df['thickness'])
    st.write("## without outliers")
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8,6))
        plt.boxplot(df['thickness'], patch_artist=True,showfliers=True)
        plt.title("Boxplot of thickness without outliers")
        plt.xlabel("Thickness")
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'Thickness'
        plt.hist(df['thickness'], bins=20)
        plt.xlabel('Thickness')
        plt.ylabel('Frequency')
        st.pyplot()
    return(df)


def plot_width(df):
    st.write("## with outliers")
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8,6))
        plt.boxplot(df['width'], patch_artist=True,showfliers=True)
        plt.title("Boxplot of width with outliers")
        plt.xlabel("width")
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'width'
        plt.hist(df['width'], bins=20)
        plt.xlabel('width')
        plt.ylabel('Frequency')
        st.pyplot()

    def treat_outliers_isolation_forest(data, column):
        # Reshape the data for Isolation Forest
        X = data[column].values.reshape(-1, 1)
        
        # Instantiate IsolationForest
        clf = IsolationForest(contamination=0.1, random_state=42)  # Contamination is the expected proportion of outliers
        
        # Fit the model
        clf.fit(X)
        
        # Predict outliers
        outliers = clf.predict(X)
        
        # Marking outliers and replacing them with NaNs
        data[column] = np.where(outliers == -1, np.nan, data[column])
        
        return data

    df = treat_outliers_isolation_forest(df, 'width')

    mean_value = df['width'].mean()
    df['width'].fillna(mean_value, inplace=True)

    st.write("## without outliers")
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8,6))
        plt.boxplot(df['width'], patch_artist=True,showfliers=True)
        plt.title("Boxplot of width without outliers")
        plt.xlabel("width")
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'width'
        plt.hist(df['width'], bins=20)
        plt.xlabel('width')
        plt.ylabel('Frequency')
        st.pyplot()
    return(df)

def plot_selling_price(df):
    st.write("## with outliers")
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8,6))
        plt.boxplot(df['selling_price'], patch_artist=True,showfliers=True)
        plt.title("Boxplot of selling_price with outliers")
        plt.xlabel("selling_price")
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'selling_price'
        plt.hist(df['selling_price'], bins=20)
        plt.xlabel('selling_price')
        plt.ylabel('Frequency')
        st.pyplot()

    def treat_outliers_isolation_forest(data, column):
        # Reshape the data for Isolation Forest
        X = data[column].values.reshape(-1, 1)
        
        # Instantiate IsolationForest
        clf = IsolationForest(contamination=0.1, random_state=42)  # Contamination is the expected proportion of outliers
        
        # Fit the model
        clf.fit(X)
        
        # Predict outliers
        outliers = clf.predict(X)
        
        # Marking outliers and replacing them with NaNs
        data[column] = np.where(outliers == -1, np.nan, data[column])
        
        return data

    df = treat_outliers_isolation_forest(df, 'selling_price')

    mean_value = df['selling_price'].mean()
    df['selling_price'].fillna(mean_value, inplace=True)

    st.write("## without outliers")
    col1,col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(8,6))
        plt.boxplot(df['selling_price'], patch_artist=True,showfliers=True)
        plt.title("Boxplot of selling_price without outliers")
        plt.xlabel("selling_price")
        st.pyplot()
    with col2:
        # Assuming df is your DataFrame with the column 'selling_price'
        plt.hist(df['selling_price'], bins=20)
        plt.xlabel('selling_price')
        plt.ylabel('Frequency')
        st.pyplot()
    return(df)

def train_test_split1(df,sp,ts):
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Apply Label Encoding to 'status', 'material_ref', and 'item_type' columns
    df['status_encoded'] = label_encoder.fit_transform(df['status'])
    df['material_ref_encoded'] = label_encoder.fit_transform(df['material_ref'])
    df['item_type_encoded'] = label_encoder.fit_transform(df['item type'])

    df['product_ref'] = df['product_ref'].astype('category')
    df_encoded = pd.get_dummies(df,columns = ['product_ref'],prefix = 'product')

    df_encoded = df_encoded.drop(columns = ['id','status','item type','numerical_item_date','material_ref'])

    # Split the data into features (X) and target variable (y)
    X = df_encoded.drop(sp, axis=1)
    y = df_encoded[sp]

    st.write("Features (X)",X)
    st.write("Target varaiable (y)",y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Display the splits
    st.write("### Train-Test Split")
    st.write("Train Set:")
    st.write(X_train)
    st.write("Test Set:")
    st.write(X_test)
    col1,col2 = st.columns(2)
    with col1:
        st.write("Train Labels:")
        st.write(y_train)
    with col2:
        st.write("Test Labels:")
        st.write(y_test)



def randomforrestregression(df,sp,tsize):
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Apply Label Encoding to 'status', 'material_ref', and 'item_type' columns
    df['status_encoded'] = label_encoder.fit_transform(df['status'])
    df['material_ref_encoded'] = label_encoder.fit_transform(df['material_ref'])
    df['item_type_encoded'] = label_encoder.fit_transform(df['item type'])

    df['product_ref'] = df['product_ref'].astype('category')
    df_encoded = pd.get_dummies(df,columns = ['product_ref'],prefix = 'product')

    df_encoded = df_encoded.drop(columns = ['id','status','item type','numerical_item_date','material_ref'])

    # Split the data into features (X) and target variable (y)
    X = df_encoded.drop(sp, axis=1)
    y = df_encoded[sp]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize, random_state=42)

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model on the training data
    rf_regressor.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = rf_regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("## Random Forest Regression Results:")
    st.write("Mean Squared Error:", mse)
    st.write("R^2 Score:", r2)

    st.subheader("Regression Analysis Plot")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs Predicted Selling Price')
    plt.xlim(0,3000)
    plt.ylim(0,3000)
    st.pyplot()

    st.write("## Dataframe of Actual and Predicted Selling Price")
    results_df = pd.DataFrame({'Actual Selling Price': y_test, 'Predicted Selling Price': y_pred})
    results_df = results_df.reset_index(drop=True)

    st.write(results_df)

def train_test_split2(df,sts,ts):
    label_encoder = LabelEncoder()

    # Apply Label Encoding to 'status', 'material_ref', and 'item_type' columns
    df['status_encoded'] = label_encoder.fit_transform(df[sts])
    df['material_ref_encoded'] = label_encoder.fit_transform(df['material_ref'])
    df['item_type_encoded'] = label_encoder.fit_transform(df['item type'])

    new_df = df.drop(columns = ['id','item type','material_ref','numerical_item_date','status_encoded'])
    df['product_ref'] = df['product_ref'].astype('category')
    new_df = pd.get_dummies(new_df,columns = ['product_ref'],prefix = 'product')
    # Mapping status values to Success or Failure
    new_df[sts] = new_df[sts].map({'Won': 'Success', 'Lost': 'Failure'})

    # Dropping rows with status values other than 'Success' or 'Failure'
    new_df = new_df[new_df[sts].isin(['Success', 'Failure'])]
    # 2. Split data into features (X) and target variable (y)
    X = new_df.drop(sts, axis=1)
    y = new_df[sts]

    st.write("Features (X)",X)
    st.write("Target varaiable (y)",y)

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Display the splits
    st.write("### Train-Test Split")
    st.write("Train Set:")
    st.write(X_train)
    st.write("Test Set:")
    st.write(X_test)
    col1,col2 = st.columns(2)
    with col1:
        st.write("Train Labels:")
        st.write(y_train)
    with col2:
        st.write("Test Labels:")
        st.write(y_test)





def randomforrestclassifier(df,sts,ts):
    label_encoder = LabelEncoder()

    # Apply Label Encoding to 'status', 'material_ref', and 'item_type' columns
    df['status_encoded'] = label_encoder.fit_transform(df[sts])
    df['material_ref_encoded'] = label_encoder.fit_transform(df['material_ref'])
    df['item_type_encoded'] = label_encoder.fit_transform(df['item type'])

    new_df = df.drop(columns = ['id','item type','material_ref','numerical_item_date','status_encoded'])
    df['product_ref'] = df['product_ref'].astype('category')
    new_df = pd.get_dummies(new_df,columns = ['product_ref'],prefix = 'product')
    # Mapping status values to Success or Failure
    new_df[sts] = new_df[sts].map({'Won': 'Success', 'Lost': 'Failure'})

    # Dropping rows with status values other than 'Success' or 'Failure'
    new_df = new_df[new_df[sts].isin(['Success', 'Failure'])]

    # 2. Split data into features (X) and target variable (y)
    X = new_df.drop(sts, axis=1)
    y = new_df[sts]

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # 4. Initialize the RandomForestClassifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # 5. Train the model
    rf_classifier.fit(X_train, y_train)

    # 6. Predict on the testing data
    y_pred = rf_classifier.predict(X_test)

    # 7. Evaluate the model
    st.write("## Results of Random Forest Classifier")
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    # Plot confusion matrix
    st.write("### Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot()

    # Plot classification report
    st.write("### Classification Report:")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    plt.figure(figsize=(8, 6))
    sns.heatmap(class_report_df.iloc[:-1, :-1], annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.title('Classification Report')
    st.pyplot()

    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    result_df = result_df.reset_index(drop=True)
    st.write("## Actual and Predicted Values")
    st.write(result_df)




# Streamlit part

st.title(" INDUSTRIAL COPPER MODELING ")
with st.sidebar:
    select = option_menu("Main Menu",["HOME","DATA PREPROCESSING","MACHINE LEARNING"])
if 'data_7' not in st.session_state:
    st.session_state.data_7 = None
if select == "HOME":

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Display the uploaded data
        st.write("Uploaded CSV file:")
        st.write(df)
        
        # Show success message
        st.success("Upload completed successfully!")

elif select == "DATA PREPROCESSING":

    data = data_preprocessing(df)
    st.write("## Preprocessed Data")
    buffer = StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    st.write("## Quantity Tons")
    data_1 = quantity_tons_plot(data)
    st.write("## Country")
    data_2 = plot_country(data_1)
    st.write("## Customer")
    data_3 = plot_customer(data_2)
    st.write("## Application")
    data_4 = plot_application(data_3)
    st.write("## Thickness")
    data_5 = plot_thickness(data_4)
    st.write("## Width")
    data_6 = plot_width(data_5)
    st.write("## Selling Price")
    data_7  = plot_selling_price(data_6)
    if data_7 is not None:
        st.session_state.data_7 = pickle.dumps(data_7)

    st.write("## Data preprocessing completed !!")

elif select == "MACHINE LEARNING":
    data_7 = pickle.loads(st.session_state.data_7)
    selected_option = st.selectbox('Select the target variable :',data_7.columns.unique())
    test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, 0.05)
    if data_7 is not None:  # Ensure data_7 has been assigned a value
        if selected_option in ['selling_price', 'status']:           
            if selected_option == 'selling_price':
                ndf = train_test_split1(data_7,selected_option,test_size)
                option = st.radio("## Select the type of Machine learning", ("None", "REGRESSION", "CLASSIFICATION"))
                if option == "None":
                    pass
                elif option == "REGRESSION":
                    randomforrestregression(data_7,selected_option,test_size)
                elif option == "CLASSIFICATION":
                    with st.spinner("Warning"):
                        st.warning("the selected target variable is non categorical !!")
                
            else:
                n1df = train_test_split2(data_7,selected_option,test_size) 
                option = st.radio("## Select the type of Machine learning", ("None", "REGRESSION", "CLASSIFICATION"))
                if option == "None":
                    pass
                elif option == "REGRESSION":
                    with st.spinner("Warning"):
                        st.warning("the selected target variable is numerical !!")
                elif option == "CLASSIFICATION":
                    randomforrestclassifier(data_7,selected_option,test_size)
        else:
            st.write("Please select a valid target variable from the available options !!!")
    else:
        st.write("Please preprocess data before performing machine learning.")

