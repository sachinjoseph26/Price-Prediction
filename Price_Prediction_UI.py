import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from xgboost import XGBRegressor
def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def get_features(data):
    
    luxury_brands = ["MCM", "MCM Worldwide", "Louis Vuitton", "Burberry", "Burberry London", "Burberry Brit", "HERMES", "Tieks",
                     "Rolex", "Apple", "Gucci", "Valentino", "Valentino Garavani", "RED Valentino", "Cartier", "Christian Louboutin",
                     "Yves Saint Laurent", "Saint Laurent", "YSL Yves Saint Laurent", "Georgio Armani", "Armani Collezioni", "Emporio Armani"]
    data['is_luxurious'] = (data['brand_name'].isin(luxury_brands)).astype(np.int8)

    expensive_brands = ["Michael Kors", "Louis Vuitton", "Lululemon", "LuLaRoe", "Kendra Scott", "Tory Burch", "Apple", "Kate Spade",
                  "UGG Australia", "Coach", "Gucci", "Rae Dunn", "Tiffany & Co.", "Rock Revival", "Adidas", "Beats", "Burberry",
                  "Christian Louboutin", "David Yurman", "Ray-Ban", "Chanel"]
    data['is_expensive'] = (data['brand_name'].isin(expensive_brands)).astype(np.int8)
    return data

def get_ohe_single(sample, col_name):
    vect = CountVectorizer()
    if isinstance(sample[col_name], pd.Series):
        ohe = vect.fit_transform(sample[col_name].values.astype('U'))  # Convert to Unicode
    else:
        ohe = vect.fit_transform([sample[col_name]])
    return ohe

def get_text_encodings_single(sample, col_name, min_val, max_val):
    vect = TfidfVectorizer(min_df=min_val, ngram_range=(min_val, max_val), max_features=1000000)
    if isinstance(sample[col_name], pd.Series):
        encoding = vect.fit_transform(sample[col_name].values.astype('U'))
    else:
        encoding = vect.fit_transform([sample[col_name]])
    return encoding

def generate_encodings_single(sample):
    ohe_category_0 = get_ohe_single(sample, 'category_0')
    ohe_category_1 = get_ohe_single(sample, 'category_1')
    ohe_category_2 = get_ohe_single(sample, 'category_2')

    trans = csr_matrix(pd.get_dummies(sample[['shipping', 'item_condition_id','is_luxurious','is_expensive']], sparse=True).values)

    name_encoding = get_text_encodings_single(sample, 'name', 1, 1)
    text_encoding = get_text_encodings_single(sample, 'item_description', 1, 2)

    data = hstack((ohe_category_0, ohe_category_1, ohe_category_2, trans, name_encoding, text_encoding)).tocsr().astype('float32')

    return data

def process_category(data):
    for i in range(3):
        
        def get_part(x):
            
            if type(x) != str:
                return np.nan
        
            parts = x.split('/')
            
            if i >= len(parts):
                return np.nan
            else:
                return parts[i]

        field_name = 'category_' + str(i)
        
        data[field_name] = data['category_name'].apply(get_part)
    
    return data

def clean_text(text):
    if not isinstance(text, str):
        return ''  # Return empty string for non-string inputs
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove single quotes, double quotes
    text = re.sub(r"[\'\"]", "", text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    text = decontracted(text)
    # Remove emojis
    text = emoji.demojize(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in list(string.punctuation)]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text

def preprocess(data):
    data = process_category(data)
    data['name'] = data['name'].apply(lambda x: clean_text(x))
    data['item_description'] = data['item_description'].apply(lambda x: clean_text(x))
    
    data = get_features(data)
    
    data.fillna({'brand_name': ' ', 'category_0': 'other', 'category_1': 'other', 'category_2': 'other'}, inplace=True)
    
    #Concat columns
    data['name'] = data['name'] + ' ' + data['brand_name']  # Assuming 'brand_name' is important for encoding
    
    return data

def load_predict_model(input_data):
    #print('Processing Data...')
    processed_data = preprocess(input_data)
    #print('Generating Encodings...')
    encoded_data = generate_encodings_single(processed_data)
    #print(encoded_data.shape)
    target_num_features = 1000
    reshaped_arr = np.zeros((1, target_num_features))
    # Copy the existing data into the new array
    reshaped_arr[:, :encoded_data.shape[1]] = encoded_data.toarray()
    loaded_model = XGBRegressor()
    loaded_model.load_model('XGBoost_PricePrediction_Model.hdf5')
    pred = loaded_model.predict(reshaped_arr)
    # Display predicted price
    st.subheader("Predicted Price:")
    st.write("$", round(pred[0], 2))

def main():
    st.title("Product Price Predictor")
    
    # Input features
    name = st.text_input("Product Name",'Levis Black Leggings, Womens. Size L.')
    item_condition_id = st.slider("Item Condition ID", 1, 5, 1)
    category_name = st.text_input("Category Name",'Women/Athletic Apparel/Pants, Tights, Leggings')
    brand_name = st.text_input("Brand Name",'Levis')
    shipping = st.radio("Shipping",("Yes", "No"), index=0)
    item_description = st.text_area("Item Description",'Adorable gym wear from a well known brand. In great condition. Size L. Black and stretchable.')
    
    # Button to trigger prediction
    if st.button("Get Price"):
        # Preprocess input features
        data = [{'name': name,
                'item_condition_id': item_condition_id,
                'category_name': category_name,
                'brand_name': brand_name,
                'shipping': 0 if shipping == "Yes" else 1,
                'item_description': item_description}]
        input_data = pd.DataFrame.from_dict(data)
        print(input_data)
        load_predict_model(input_data)

if __name__ == "__main__":
    main()