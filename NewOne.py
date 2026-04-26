import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import time


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Travel Itinerary Generator",
    page_icon="🧭",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS (IMPROVED FRONTEND)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to   {opacity: 1; transform: translateY(0px);}
}

.main-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 900;
    color: #0E6B72;
    margin-top: 10px;
}

.subtext {
    text-align: center;
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 25px;
}

.fade-card {
    animation: fadeIn 0.8s ease-in-out;
    background: white;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0px 2px 15px rgba(0,0,0,0.10);
    transition: transform 0.2s ease-in-out;
}

.fade-card:hover {
    transform: scale(1.01);
}

.center-box {
    background: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    margin-top: 15px;
    margin-bottom: 25px;
}

.center-heading {
    text-align: center;
    font-size: 2.0rem;
    font-weight: 800;
    color: #0E6B72;
    margin-top: 15px;
    margin-bottom: 20px;
}

.small-divider {
    height: 3px;
    background: #0E6B72;
    border-radius: 10px;
    margin: auto;
    width: 120px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
LOCATION_MAP = {
    "North":  ["Delhi", "Jaipur", "Shimla", "Leh", "Amritsar", "Chandigarh", "Rishikesh"],
    "East":   ["Kolkata", "Gangtok", "Puri", "Agartala", "Ranchi", "Imphal"],
    "West":   ["Mumbai", "Goa", "Surat", "Daman", "Indore", "Pune"],
    "South":  ["Bengaluru", "Chennai", "Kochi", "Hyderabad", "Coimbatore"],
    "Island": ["Port Blair"]
}

SEASON_MAP = {
    "Summer":  ["April", "May", "June"],
    "Winter":  ["November", "December", "January", "February"],
    "Monsoon": ["July", "August", "September"],
    "Spring":  ["February", "March", "April"],
    "Autumn":  ["September", "October", "November"]
}


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    itin_df = pd.read_csv("india_travel_itinerary_dataset.csv")
    hotel_df = pd.read_csv("india_hotels_dataset.csv")
    rest_df = pd.read_csv("india_restaurants_dataset.csv")

    itin_df.columns = [c.strip() for c in itin_df.columns]
    hotel_df.columns = [c.strip() for c in hotel_df.columns]
    rest_df.columns = [c.strip() for c in rest_df.columns]

    if "Unnamed: 0" in itin_df.columns:
        itin_df.drop(columns=["Unnamed: 0"], inplace=True)

    itin_df.rename(columns={
        "Name": "Activity",
        "Google review rating": "Rating",
        "Entrance Fee in INR": "Estimated Budget (INR)",
        "time needed to visit in hrs": "Time Needed (hrs)"
    }, inplace=True)

    hotel_df.rename(columns={
        "Price": "Hotel Budget",
        "Address": "Hotel Address"
    }, inplace=True)

    rest_df.rename(columns={
        "restaurant_name": "Restaurant Name",
        "rating": "Restaurant Rating",
        "average_price": "Restaurant Budget",
        "location": "City"
    }, inplace=True)

    if "Best Time to Visit" not in itin_df.columns:
        itin_df["Best Time to Visit"] = "All Year"

    if "Activity Address" not in itin_df.columns:
        itin_df["Activity Address"] = itin_df["State"]

    itin_df["Estimated Budget (INR)"] = pd.to_numeric(itin_df["Estimated Budget (INR)"], errors="coerce")
    itin_df["Rating"] = pd.to_numeric(itin_df["Rating"], errors="coerce")

    hotel_df["Hotel Budget"] = pd.to_numeric(hotel_df["Hotel Budget"], errors="coerce")

    rest_df["Restaurant Budget"] = pd.to_numeric(rest_df["Restaurant Budget"], errors="coerce")
    rest_df["Restaurant Rating"] = pd.to_numeric(rest_df["Restaurant Rating"], errors="coerce")

    itin_df["Estimated Budget (INR)"] = itin_df["Estimated Budget (INR)"].fillna(itin_df["Estimated Budget (INR)"].median())
    itin_df["Rating"] = itin_df["Rating"].fillna(4.0)

    hotel_df["Hotel Budget"] = hotel_df["Hotel Budget"].fillna(hotel_df["Hotel Budget"].median())

    rest_df["Restaurant Budget"] = rest_df["Restaurant Budget"].fillna(rest_df["Restaurant Budget"].median())
    rest_df["Restaurant Rating"] = rest_df["Restaurant Rating"].fillna(4.0)

    return itin_df, hotel_df, rest_df


itin_df, hotel_df, rest_df = load_data()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_season(d):
    m = d.month
    if m in [4, 5, 6]:
        return "Summer"
    if m in [11, 12, 1, 2]:
        return "Winter"
    if m in [7, 8, 9]:
        return "Monsoon"
    if m in [2, 3, 4]:
        return "Spring"
    return "Autumn"


def choose_cities(location, duration):
    cities = LOCATION_MAP.get(location, [])
    if not cities:
        return []
    return [cities[i % len(cities)] for i in range(duration)]


def score_df(df):
    df = df.copy().dropna(subset=["Rating", "Estimated Budget (INR)"])
    if df.empty:
        return df

    scaler = MinMaxScaler()
    df[["R_sc", "C_sc"]] = scaler.fit_transform(df[["Rating", "Estimated Budget (INR)"]])

    df["Score_mm"] = 0.7 * df["R_sc"] + 0.3 * (1 - df["C_sc"])
    cos = cosine_similarity(df[["R_sc", "C_sc"]].values, np.array([[1.0, 0.0]])).flatten()
    df["Score_cos"] = cos

    df["Final_score"] = 0.6 * df["Score_mm"] + 0.4 * df["Score_cos"]
    return df


# ─────────────────────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS (WITH FALLBACK)
# ─────────────────────────────────────────────────────────────
def pick_activities(itin_df, city, months, budget, n=3):
    pool = itin_df[itin_df["City"].str.lower() == city.lower()].copy()

    if pool.empty:
        pool = itin_df.copy()

    filt = pool[pool["Estimated Budget (INR)"] <= budget]

    if filt.empty:
        filt = pool

    scored = score_df(filt)
    if scored.empty:
        return []

    scored = scored.sort_values("Final_score", ascending=False).head(n)

    return [
        {
            "name": row.get("Activity", "—"),
            "address": row.get("Activity Address", "—"),
            "time": row.get("Best Time to Visit", "All Year"),
            "budget": row.get("Estimated Budget (INR)", 0)
        }
        for _, row in scored.iterrows()
    ]


def pick_hotels(hotel_df, city, budget, n=3):
    pool = hotel_df[hotel_df["City"].str.lower() == city.lower()].copy()

    if pool.empty:
        pool = hotel_df.copy()

    filt = pool[pool["Hotel Budget"] <= budget]

    if filt.empty:
        filt = pool

    filt = filt.sort_values("Hotel Budget").head(n)

    return [
        {
            "name": row.get("Hotel Name", row.get("Name", "Hotel")),
            "address": row.get("Hotel Address", "—"),
            "budget": row.get("Hotel Budget", None)
        }
        for _, row in filt.iterrows()
    ]


def pick_restaurants(rest_df, city, budget, n=4):
    pool = rest_df[rest_df["City"].str.lower() == city.lower()].copy()

    if pool.empty:
        pool = rest_df.copy()

    filt = pool[pool["Restaurant Budget"] <= budget]

    if filt.empty:
        filt = pool

    filt = filt.sort_values("Restaurant Rating", ascending=False).head(n)

    return [
        {
            "name": row.get("Restaurant Name", "—"),
            "rating": row.get("Restaurant Rating", None),
            "budget": row.get("Restaurant Budget", None)
        }
        for _, row in filt.iterrows()
    ]


# ─────────────────────────────────────────────────────────────
# ITINERARY GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_itinerary(total_budget, members, visit_date, duration, location):
    per_head = total_budget / members

    season = get_season(visit_date)
    months = SEASON_MAP.get(season, [])

    city_seq = choose_cities(location, duration)

    days = []
    for i, city in enumerate(city_seq, 1):

        activities = pick_activities(itin_df, city, months, per_head, n=3)

        days.append({
            "day": i,
            "city": city,
            "activities": activities,
            "hotels": pick_hotels(hotel_df, city, per_head, n=3),
            "restaurants": pick_restaurants(rest_df, city, per_head, n=4)
        })

    return days, season, per_head


def fmt_inr(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"₹{x:,.0f}"


def fmt_stars(r):
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "N/A"
    return f"⭐ {r:.1f}"


# ─────────────────────────────────────────────────────────────
# UI MAIN
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🧠 Intelligent Travel Itinerary Generator</div>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Enter your preferences and get a complete travel itinerary with recommendations.</p>", unsafe_allow_html=True)

st.markdown("<div class='center-box'>", unsafe_allow_html=True)
st.subheader("📝 Enter Trip Details")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Your Name", placeholder="Enter your name")
    age_range = st.text_input("Age Range (Custom)", placeholder="Example: 18-25 or 20-30")
    location = st.selectbox("Region of India", list(LOCATION_MAP.keys()))
    duration = st.slider("Trip Duration (Days)", 1, 15, 3)

with col2:
    trip_type = st.selectbox("Trip Type", ["Solo", "Couple", "Friends", "Family"])

    if trip_type == "Solo":
        members = 1
    elif trip_type == "Couple":
        members = 2
    elif trip_type == "Friends":
        members = st.number_input("Number of Friends", min_value=3, max_value=15, value=4)
    else:
        members = st.number_input("Number of Family Members", min_value=3, max_value=15, value=4)

    budget = st.slider("Total Budget (₹)", 5000, 500000, 50000, step=5000)
    visit_date = st.date_input("Date of Travel", value=date.today(), min_value=date.today())

st.markdown("</div>", unsafe_allow_html=True)

generate_btn = st.button("✈️ Generate Itinerary")


# ─────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────
if generate_btn:

    with st.spinner("⏳ Generating your itinerary... Please wait..."):
        time.sleep(6)

        days, season, per_head = generate_itinerary(
            budget, members, visit_date, duration, location
        )

    st.success(f"✅ Itinerary Generated Successfully for {name if name else 'Traveller'}!")

    st.subheader("📌 Trip Summary")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Traveller Name", name if name else "Traveller")
    col2.metric("Age Range", age_range if age_range else "Not Entered")
    col3.metric("Region", location)
    col4.metric("Season", season)
    col5.metric("Members", members)
    col6.metric("Per Person Budget", fmt_inr(per_head))

    st.markdown("---")

    st.markdown("<div class='center-heading'>🗺️ Your {}-Day Itinerary Plan</div>".format(duration), unsafe_allow_html=True)
    st.markdown("<div class='small-divider'></div>", unsafe_allow_html=True)

    for d in days:
        st.markdown("<div class='fade-card'>", unsafe_allow_html=True)

        st.markdown(f"## Day {d['day']} - 📍 {d['city']}")

        st.markdown("### 🎯 Activities (Top 3)")
        for idx, act in enumerate(d["activities"], 1):
            st.write(f"**{idx}.** ✅ {act['name']} ({fmt_inr(act['budget'])})")
            st.write(f"📍 Location: {act['address']}")
            st.write("")

        st.markdown("### 🏨 Hotels (Recommended)")
        for h in d["hotels"]:
            st.write(f"🏨 **{h['name']}**")
            st.write(f"📍 {h['address']}")
            st.write(f"💰 {fmt_inr(h['budget'])}/night")
            st.write("")

        st.markdown("### 🍽️ Restaurants (Recommended)")
        for r in d["restaurants"]:
            st.write(f"🍽️ **{r['name']}** | {fmt_stars(r['rating'])} | {fmt_inr(r['budget'])}/meal")

        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # BUDGET BREAKDOWN + GRAPH
    # ─────────────────────────────────────────────────────────────
    total_act = sum(a["budget"] for d in days for a in d["activities"] if a["budget"])

    hotel_vals = [h["budget"] for d in days for h in d["hotels"]
                  if h["budget"] is not None and not np.isnan(h["budget"])]

    rest_vals = [r["budget"] for d in days for r in d["restaurants"]
                 if r["budget"] is not None and not np.isnan(r["budget"])]

    avg_hotel = np.mean(hotel_vals) * len(days) if hotel_vals else 0
    avg_rest = np.mean(rest_vals) * len(days) if rest_vals else 0

    grand_total = total_act + avg_hotel + avg_rest

    st.subheader("💰 Estimated Budget Breakdown (Per Person)")

    st.write(f"🎯 Activities Total: **{fmt_inr(total_act)}**")
    st.write(f"🏨 Hotels Total: **{fmt_inr(avg_hotel)}**")
    st.write(f"🍽️ Restaurants Total: **{fmt_inr(avg_rest)}**")
    st.write(f"📊 Grand Total: **{fmt_inr(grand_total)}**")

    # Pie Chart Only
    st.subheader("📊 Budget Distribution Chart")

    labels = ["Activities", "Hotels", "Restaurants"]
    values = [total_act, avg_hotel, avg_rest]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Budget Distribution")
    st.pyplot(fig)

    # ─────────────────────────────────────────────────────────────
    # DOWNLOAD OPTION
    # ─────────────────────────────────────────────────────────────
    st.subheader("📥 Download Itinerary")

    out_data = []
    for d in days:
        out_data.append({
            "Day": d["day"],
            "City": d["city"],
            "Activities": ", ".join([a["name"] for a in d["activities"]]),
            "Hotels": ", ".join([h["name"] for h in d["hotels"]]),
            "Restaurants": ", ".join([r["name"] for r in d["restaurants"]]),
        })

    out_df = pd.DataFrame(out_data)

    st.download_button(
        label="📥 Download Itinerary as CSV",
        data=out_df.to_csv(index=False),
        file_name="yatra_itinerary.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Fill trip details above and click **Generate Itinerary**.")