import streamlit as st
import pandas as pd
from tensorflow import keras
import numpy as np
# new= CSA_ANN.py
# old = csat_ann_model_new.keras
# ------------------------------
# ✅ Load Model (final clean version)
# ------------------------------
MODEL_PATH = "csat_ann_model_new.keras"
model = keras.models.load_model(MODEL_PATH, compile=False)
import base64

# ------------------------------
# ✅ Set Background Image
# ------------------------------

def add_bg_image(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
st.set_page_config(page_title="Customer Satisfaction Prediction", page_icon="🤖", layout="centered")
# Call this function with your downloaded image path
add_bg_image("bgimg.png")  # Replace with your image file name
st.markdown("""
<h1 style='text-align:center; color:#000000;'>
🤖 Customer Satisfaction Prediction
</h1>
""", unsafe_allow_html=True)
st.write("Fill in the details below to predict the Customer Satisfaction (CSAT) score.")
st.markdown(
    """
    <style>
    /* Make all selectboxes black very transparent */
    div[data-baseweb="select"] > div {
        background-color: rgba(0, 0, 0, 0.3) !important;  /* black with 30% opacity */
        color: white;
        border-radius: 8px;
    }

    /* Change hover effect */
    div[data-baseweb="select"] > div:hover {
        background-color: rgba(0, 0, 0, 0.5) !important;  /* slightly darker on hover */
    }

    /* Dropdown options styling */
    div[data-baseweb="menu"] {
        background-color: rgba(0, 0, 0, 0.5) !important;  /* black with 50% opacity */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# 🔽 Input Sections
# ------------------------------

channel = st.selectbox("Channel Name", ['Outcall', 'Inbound', 'Email'])
category = st.selectbox("Category", ['Product Queries', 'Order Related', 'Returns', 'Cancellation',
       'Shopzilla Related', 'Payments related', 'Refund Related',
       'Feedback', 'Offers & Cashback', 'Onboarding related', 'Others',
       'App/website'])
subcategory = st.selectbox("Sub-Category", ['Life Insurance', 'Product Specific Information',
       'Installation/demo', 'Reverse Pickup Enquiry', 'Not Needed',
       'Fraudulent User', 'Exchange / Replacement', 'Missing',
       'General Enquiry', 'Return request', 'Delayed',
       'Service Centres Related', 'Payment related Queries',
       'Order status enquiry', 'Return cancellation', 'Unable to track',
       'Seller Cancelled Order', 'Wrong', 'Invoice request',
       'Priority delivery', 'Refund Related Issues', 'Signup Issues',
       'Online Payment Issues', 'Technician Visit',
       'UnProfessional Behaviour', 'Damaged', 'Product related Issues',
       'Refund Enquiry', 'Customer Requested Modifications',
       'Instant discount', 'Card/EMI', 'Shopzila Premium Related',
       'Account updation', 'COD Refund Details', 'Seller onboarding',
       'Order Verification', 'Other Cashback', 'Call disconnected',
       'Wallet related', 'PayLater related', 'Call back request',
       'Other Account Related Issues', 'App/website Related',
       'Affiliate Offers', 'Issues with Shopzilla App', 'Billing Related',
       'Warranty related', 'Others', 'e-Gift Voucher',
       'Shopzilla Rewards', 'Unable to Login', 'Non Order related',
       'Service Center - Service Denial', 'Payment pending',
       'Policy Related', 'Self-Help', 'Commission related'])
manager = st.selectbox("Manager", ['Jennifer Nguyen', 'Michael Lee', 'William Kim', 'John Smith',
       'Olivia Tan', 'Emily Chen'])
tenure = st.selectbox("Tenure", ['On Job Training', '>90', '0-30', '31-60', '61-90'])
shift = st.selectbox("Agent Shift", ['Morning', 'Evening', 'Split', 'Afternoon', 'Night'])
customer_remarks = st.selectbox("Customer Remark", ['No remark', 'Positive', 'Negative', 'Neutral'])
response_time = st.number_input("Response Time (hours)", min_value=0.0, step=0.1)
product_category = st.selectbox("Product Category", ['LifeStyle', 'Electronics', 'Mobile', 'Home Appliences',
       'Furniture', 'Home', 'Books & General merchandise', 'GiftCard',
       'Affiliates'])
customer_city = st.selectbox("Customer City", ['NAGPUR', 'RANCHI', 'BETIA', 'NEW DELHI', 'KODERMA',
       'FARIDABAD', 'KANYAKUMARI', 'BURDWAN', 'JHUJHUNU', 'RANGAREDDY',
       'DHENKANAL', 'VERAVAL', 'PATTAMUNDAI', 'PALAMU', 'HARIDWAR',
       'GAYA', 'RASULABAD', 'VIRAMGAM', 'DHULE', 'BHUBANESWAR',
       'CHANGANACHERRY', 'BANGALORE', 'DARBHANGA', 'HALDWANI', 'PUNE',
       'CHANDRAPUR', 'JAMNAGAR', 'CUDDAPAH', 'AURANGABAD', 'MUNGER',
       'BARAUT', 'GOPALGANJ', 'MUZAFFARNAGAR', 'UNNAO', 'SAWER',
       'DHARAMASALA', 'AGRA', 'BERHAMPUR', 'UDAIPUR', 'BELLARY',
       'BANKURA', 'JALALPUR', 'JHUMRI TELAIYA', 'CUTTACK', 'JAJPUR ROAD',
       'HYDERABAD', 'JHANSI', 'BHIWANDI', 'DIMAPUR', 'BETTIAH',
       'SRINAGAR', 'VIDISHA', 'BILASPUR', 'LUCKNOW', 'PURNIA', 'KOLKATA',
       'MIRZAPUR', 'MADHUBANI', 'GHAZIABAD', 'AMBAD', 'KHAMGAON',
       'AMRAOTI', 'SRIPERUMBUDUR', 'BASTI', 'NALGONDA', 'BARMER',
       'CHARKHI DADRI', 'TIRUPATI', 'MUMBAI', 'BHOPAL', 'KATIHAR',
       'KHANDWA', 'MEDINIPUR', 'AHMEDABAD', 'CHENNAI', 'DURGAPUR',
       'BHIWADI', 'AKOLA', 'VISHAKHAPATNAM', 'ALLAHABAD', 'ULHASNAGAR',
       'AJMER', 'NAVI MUMBAI', 'JODHPUR', 'HATTA', 'ALWAR', 'ANAND',
       'AGARTALA', 'MATHURA', 'PALAKKAD', 'NAINITAL', 'SURAT', 'Dhansura',
       'CHATRAPUR', 'MORBI', 'AZAMGARH', 'HAMIRPUR', 'NAWADA',
       'BALESHWAR', 'VADODARA', 'CHIKHLI', 'GUNTUR', 'THANE',
       'VIJAYAWADA', 'BIDHUNA', 'AMBEDKAR NAGAR', 'GIRIDIH', 'GURGAON',
       'DOHAD', 'PURI', 'GANDHINAGAR', 'GAUTAM BUDDHA NAGAR', 'FAIZABAD',
       'GOLAGHAT', 'Garur', 'AMRELI', 'GUWAHATI', 'BALLIA',
       'GREATER NOIDA', 'KURNOOL', 'SAMASTIPUR', 'JAJPUR', 'JAMSHEDPUR',
       'DHOLPUR', 'PANVEL', 'CHIRAWA', 'COIMBATORE', 'BAREILLY',
       'MALIHABAD', 'KATHUA', 'BOKARO', 'NOKHA', 'PUTTUR', 'PURULIA',
       'KISHANGANJ', 'JADCHERLA', 'SHAJAPUR', 'ALIGARH', 'ARARIA',
       'TIJARA', 'RAIPUR', 'SANGLI', 'BIKANER', 'NARNAUND', 'BARDOLI',
       'KALOL', 'GONDAL', 'BHADOHI', 'HAZARIBAGH', 'PALAMANER', 'ROHTAK',
       'KOILWAR', 'AMBERNATH', 'INDORE', 'NAJIBABAD', 'GAIRKATA',
       'OSMANABAD', 'BEGUSARAI', 'KOTA', 'PATNA', 'JAIPUR', 'GORAKHPUR',
       'BIHAR SHARIF', 'VAPI', 'SOLAPUR', 'AMBIKAPUR', 'NANDED', 'ETAH',
       'BEED', 'RAJKOT', 'BHIWANI', 'ROURKELA', 'BHARTHANA', 'VARANASI',
       'NAGAUR', 'ALUVA', 'MADHEPURA', 'THRISSUR', 'CHAS', 'BHARUCH',
       'KANPUR', 'AIZAWL', 'VIRAR', 'DHANBAD', 'BHAGALPUR', 'ZAMANIA',
       'DARJILING', 'RAJAHMUNDRY', 'SILCHAR', 'RAJPUR', 'TUMKUR', 'CHURU',
       'JABALPUR', 'BIJNOR', 'GIDDERBAHA', 'MAHABUB NAGAR', 'SONIPAT',
       'NAWANSHAHAR', 'SIWAN', 'FATEHPUR', 'BULANDSHAHR', 'DEORIA',
       'MITHAPUR', 'SIKAR', 'ALIBAG', 'RAIPUR RANI', 'TINDIVANAM',
       'VEDASANDUR', 'MANDI', 'KOHIMA', 'BALUGAON', 'BANSWARA',
       'TULSIPUR', 'GANGAPUR CITY', 'RUDRA PRAYAG', 'JATNI', 'NELLORE',
       'JORHAT', 'TINSUKIA', 'VILLUPURAM', 'KULTI', 'SHAHAPUR', 'MAU',
       'MULBAGAL', 'MAHOBA', 'HAZIPUR', 'JHARSUGUDA', 'PINDWARA', 'PORSA',
       'JHARIA', 'KODUR', 'GOHANA', 'WANKANER', 'MUZAFFARPUR',
       'YAMUNANAGAR', 'DAUSA', 'MORADABAD', 'KANGRA', 'ERNAKULAM',
       'KISHTWAR', 'JANJGIR - CHAMPA', 'BELGAUM', 'WEST DINAJPUR',
       'SAKTI', 'BISWANATH CHARIALI', 'GANNAVARAM', 'KOTTAYAM',
       'BASAVAKALYAN', 'SAGAR', 'ASANSOL', 'SUNDERBANI', 'KURUKSHETRA',
       'NARWANA', 'LINGOTAM', 'TRICHY', 'PATTI', 'HASSAN', 'DAHOD',
       'PATHANAMTHITTA', 'UDAIPURWATI', 'RAE BARELI', 'ZIRAKPUR', 'DATIA',
       'SASARA', 'BUDNI', 'JUNAGADH', 'JHANJHARPUR', 'SARAN', 'Birbhum',
       'HUZURNAGAR', 'KOKRAJHAR', 'ALIPORE', 'BUXAR', 'SAHARANPUR',
       'BARIPARA', 'PHALODI', 'HINDAUN CITY', 'KANINA', 'DINDIGUL',
       'PILIBHIT', 'KAIRANA', 'KENDRAPARA', 'ANAPARTHI', 'BEHROR',
       'NALLASOPARA', 'SAHARSA', 'SATTENAPALLI', 'BERHAMPORE',
       'ANANDAPUR', 'ANANTAPUR', 'PINAHAT', 'UNA', 'BACHHRAWAN',
       'WARANGAL', 'MALDA', 'PANJIM', 'MAHESANA', 'AMARPUR', 'GWALIOR',
       'TIRUNELVELI', 'SAMBA', 'PANDUA', 'MADURAI', 'RECKONG PEO',
       'NASHIK', 'KHARAR', 'NAGERCOIL', 'NAIHATI', 'SEHORE', 'GUHLA',
       'NAGAON', 'BHUSAWAL', 'HOWRAH', 'PADRAUNA', 'PARATWADA',
       'MANGALDOI', 'MUNDGOD', 'BHARATPUR', 'NIRMAL', 'NANDYAL',
       'BARDHAMAN', 'LEH', 'LUDHIANA', 'SIRSA', 'Aundha Nagnath',
       'MUSAFIRKHANA', 'RISHRA', 'AURAIYA', 'SANQUELIM', 'ELURU',
       'BHINMAL', 'JALGAON', 'ZAHEERABAD', 'SATHYAMANGALAM', 'W26',
       'KOLKATTA', 'AMBAH', 'KILVELUR', 'TAMBARAM', 'Barkagaon', 'SIDHI',
       'SAILU', 'RAGHOGARH', 'KAITHAL', 'KATRA', 'FATEHGARH SAHIB',
       'SAMBALPUR', 'DHAR', 'MAHENDRAGARH', 'BONGAIGAON', 'HAPUR',
       'ARAKKONAM', 'SHAHGANJ', 'ADIPUR', 'PHULPUR', 'PATAN', 'SONBHADRA',
       'JALANDHAR', 'KASGANJ', 'BHAVNAGAR', 'MOTIHARI', 'LAHAR',
       'HAFLONG', 'PASIGHAT', 'GODDA', 'TALCHER', 'NANPARA', 'CHITRAKOOT',
       'BELDA', 'MEERUT', 'RATIA', 'RUDRAPUR', 'SANGAREDDY', 'BURHANPUR',
       'RAXAUL', 'DURG', 'CHHATARPUR', 'SHYAMNAGAR', 'GANGANAGAR',
       'KOLHAPUR', 'BHUJ', 'DALTONGANJ', 'ADILABAD', 'BUDAUN', 'MAKRANA',
       'MUKHED', 'MANGALORE', 'BISWAN', 'SASARAM', 'SILVASSA', 'PHAGWARA',
       'AHMED NAGAR', 'SANDILA', 'SINGTAM', 'RANAGHAT', 'BAHRAICH',
       'BUDHANA', 'BANSGAON', 'DEWA', 'RAIGARH', 'BHADRAK', 'JHARGRAM',
       'LOHAGHAT', 'MOGA', 'BANDA', 'SITAPUR', 'ROORKEE', 'DHARWAD'])

# ------------------------------
# 🔮 Predict Button
# ------------------------------
if st.button("Predict CSAT Score"):
    # Placeholder input data (replace with your actual encoded features)
    input_data = np.random.rand(1, model.input_shape[1])
    pred = float(model.predict(input_data)[0][0])

    # Limit to [0,5] range
    csat = max(0, min(pred, 5))

    # 🌟 Convert to stars (with partial fill)
    full_stars = int(csat)
    partial_star = csat - full_stars
    empty_stars = 5 - full_stars - (1 if partial_star > 0 else 0)

    # Build star display
    stars_display = "⭐" * full_stars
    if partial_star > 0:
        stars_display += "✨"  # ✨ = partial star effect
    stars_display += "☆" * empty_stars

    # ------------------------------
    # 🎨 Display
    # ------------------------------
    st.markdown("---")
    st.subheader("Predicted Customer Satisfaction Score:")
    st.markdown(f"<h3 style='text-align: center;'>CSAT score: {csat:.2f} / 5.00</h3>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center; color:gold;'>{stars_display}</h2>", unsafe_allow_html=True)
    st.success("Success ✅")
    



