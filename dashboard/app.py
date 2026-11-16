import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
st.set_page_config(
    page_title="GBFS Trip Destination Prediction",
    page_icon="🚲",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    h1, h2, h3 {
        color: #172554;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚲 Smart Mobility Operations")
st.subheader("GBFS-Based Trip Destination Prediction System")

st.write(
    "An intelligent decision-support platform for predicting "
    "the likely destination of shared mobility vehicles."
)

st.divider()

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Operations Overview",
        "Predict Destination",
        "Candidate Explorer",
        "Map Visualization",
        "Model Performance",
        "Explainable AI"
    ]
)

if page == "Operations Overview":

    st.header("Operations Overview")

    st.write(
        "Monitor candidate generation, destination prediction, "
        "and vehicle return analysis."
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Prediction Model", "XGBoost")

    with col2:
        st.metric("Selected Configuration", "E11")

    with col3:
        st.metric("Candidate Distance Limit", "5 km")

    with col4:
        st.metric("Candidate Time Limit", "30 min")

    st.divider()

    st.subheader("Prediction Workflow")

    steps = [
        ("01", "Rental Information"),
        ("02", "Candidate Generation"),
        ("03", "E11 Candidate Filtering"),
        ("04", "XGBoost Prediction"),
        ("05", "Destination Selection"),
        ("06", "Map Visualization")
    ]

    cols = st.columns(3)

    for index, (number, title) in enumerate(steps):
        with cols[index % 3]:
            st.info(f"**{number} — {title}**")

    st.divider()

    st.subheader("System Description")

    st.write("""
    This platform helps shared mobility operators estimate the
    likely return location of rented vehicles.

    The system uses:
    - GBFS-based rental and return information.
    - Coverage-aware candidate filtering.
    - A pretrained XGBoost destination prediction model.
    - Geographical candidate selection.
    - Explainable AI using SHAP.
    """)

elif page == "Predict Destination":

    st.header("Predict Destination")
    st.write(
        "Select a rental event to identify possible vehicle return locations."
    )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    candidate_file = (
        PROJECT_ROOT / "data" / "processed" / "expanded_candidates.csv"
    )

    if not candidate_file.exists():
        st.error("Candidate dataset not found.")
        st.stop()

    candidates = pd.read_csv(candidate_file)

    st.success("Candidate dataset connected successfully.")

    # --------------------------------
    # Rental Event Selection
    # --------------------------------

    st.subheader("1. Select Rental Information")

    rental_ids = sorted(
        candidates["spoofed_id_rental"].dropna().unique().tolist()
    )

    selected_rental = st.selectbox(
        "Select Rental Event ID",
        rental_ids
    )

    # Retrieve candidate records belonging to this rental
    rental_candidates = candidates[
        candidates["spoofed_id_rental"] == selected_rental
    ].copy()

    if rental_candidates.empty:
        st.warning("No candidate records found for this rental.")
        st.stop()

    # Get rental information
    rental = rental_candidates.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rental Vehicle ID", rental["vehicle_id_rental"])

    with col2:
        st.metric(
            "Rental Latitude",
            f"{rental['lat_rental']:.5f}"
        )

    with col3:
        st.metric(
            "Rental Longitude",
            f"{rental['lng_rental']:.5f}"
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Battery Level", rental["battery_rental"])

    with col5:
        st.metric("Remaining Range", rental["range_rental"])

    with col6:
        st.metric(
            "Rental Time",
            str(rental["timestamp_rental"])
        )

    st.divider()

    # --------------------------------
    # E11 Candidate Filtering
    # --------------------------------

    st.subheader("2. E11 Candidate Filtering")

    st.write("""
    E11 applies the following candidate constraints:

    - Maximum distance: 5,000 metres
    - Maximum time difference: 1,800 seconds
    - Maximum battery difference: 10
    - Maximum distance-based speed: 20 km/h
    """)

    if st.button("Generate E11 Candidates", type="primary"):

        filtered = rental_candidates.copy()

        # E11 distance and time constraints
        filtered = filtered[
            (filtered["distance"] <= 5000) &
            (filtered["time_diff"] <= 1800) &
            (filtered["battery_diff"] <= 10)
        ]

        # Speed constraint, when the column is available
        if "speed" in filtered.columns:
            filtered = filtered[
                filtered["speed"] <= 20
            ]

        st.session_state["filtered_candidates"] = filtered
        st.session_state["selected_rental"] = selected_rental

        st.success("E11 filtering completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Candidates Before Filtering",
                len(rental_candidates)
            )

        with col2:
            st.metric(
                "Candidates After E11 Filtering",
                len(filtered)
            )

        st.divider()

        st.subheader("3. Filtered Candidate Locations")

        if not filtered.empty:

            display_columns = [
                col for col in [
                    "vehicle_id_return",
                    "spoofed_id_return",
                    "lat_return",
                    "lng_return",
                    "timestamp_return",
                    "distance",
                    "time_diff",
                    "battery_diff",
                    "range_diff"
                ]
                if col in filtered.columns
            ]

            st.dataframe(
                filtered[display_columns],
                use_container_width=True
            )

        else:
            st.warning("No candidates satisfy the E11 constraints.")

    # Display previously generated results
    elif "filtered_candidates" in st.session_state:

        if st.session_state["selected_rental"] == selected_rental:

            filtered = st.session_state["filtered_candidates"]

            st.subheader("Previously Generated E11 Candidates")

            st.metric("Filtered Candidates", len(filtered))

            if not filtered.empty:
                st.dataframe(filtered, use_container_width=True)
            else:
                st.warning("No candidates were found.")

    # --------------------------------
    # XGBoost Destination Prediction
    # --------------------------------

    st.divider()
    st.subheader("4. XGBoost Destination Prediction")

    if st.button("Predict Destination with XGBoost", type="primary"):

        if "filtered_candidates" not in st.session_state:
            st.warning("Please generate E11 candidates first.")

        else:
            filtered = st.session_state["filtered_candidates"].copy()

            if filtered.empty:
                st.warning("No candidates available for prediction.")

            else:
                try:
                    # Load existing trained model and scaler
                    model_path = PROJECT_ROOT / "ml" / "model.pkl"
                    scaler_path = PROJECT_ROOT / "ml" / "scaler.pkl"

                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)

                    # Features used by the trained model
                    model_features = [
                        "lat_lend",
                        "lng_lend",
                        "time_diff",
                        "battery_diff",
                        "range_diff",
                        "distance",
                        "mean_speed_distance_based",
                        "mean_speed_range_based"
                    ]

                    scaled_features = [
                        "time_diff",
                        "battery_diff",
                        "range_diff",
                        "distance",
                        "mean_speed_distance_based",
                        "mean_speed_range_based"
                    ]

                    # Prepare model input
                    X = filtered.copy()

                    # Map rental coordinates
                    X["lat_lend"] = X["lat_rental"]
                    X["lng_lend"] = X["lng_rental"]

                    # Map speed features only if the expected columns are absent
                    if "mean_speed_distance_based" not in X.columns:
                        X["mean_speed_distance_based"] = X["speed"]

                    if "mean_speed_range_based" not in X.columns:
                        X["mean_speed_range_based"] = X["speed_range_based"]

                    # Check required columns
                    missing = [
                        col for col in model_features
                        if col not in X.columns
                    ]
                    st.write("Available dataset columns:")
                    st.write(X.columns.tolist())

                    st.write("Available speed-related columns:")
                    st.write([
                        col for col in X.columns
                        if "speed" in col.lower()
                   ])

                    if missing:
                        st.error(
                            f"Missing model features: {missing}"
                        )

                    else:
                        # Prepare model input
                        X_model = X[model_features].copy()

                        X_model = X_model.apply(
                            pd.to_numeric,
                            errors="coerce"
                        )

                        if X_model.isnull().any().any():
                            st.error(
                                "Some candidate records contain missing "
                                "model feature values."
                            )

                        else:
                            # Apply the existing scaler
                            X_model[scaled_features] = scaler.transform(
                                X_model[scaled_features]
                            )

                            # Predict destination coordinates
                            predictions = model.predict(X_model)

                            predicted_lat = float(predictions[0][0])
                            predicted_lng = float(predictions[0][1])

                            # Find the candidate closest to predicted location
                            X["prediction_distance_m"] = X.apply(
                                lambda row: geodesic(
                                    (predicted_lat, predicted_lng),
                                    (row["lat_return"], row["lng_return"])
                                ).meters,
                                axis=1
                            )

                            best_candidate = X.loc[
                                X["prediction_distance_m"].idxmin()
                            ]

                            # Store prediction
                            st.session_state["destination_prediction"] = {
                                "latitude": predicted_lat,
                                "longitude": predicted_lng,
                                "candidate": best_candidate
                            }

                            st.success(
                                "Destination prediction completed!"
                            )

                except Exception as e:
                    st.error(f"Prediction error: {e}")

    # Display prediction result
    if "destination_prediction" in st.session_state:

        result = st.session_state["destination_prediction"]
        best = result["candidate"]

        st.subheader("5. Predicted Destination")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Latitude",
                f"{result['latitude']:.6f}"
            )

        with col2:
            st.metric(
                "Predicted Longitude",
                f"{result['longitude']:.6f}"
            )

        st.success("Best matching return candidate selected.")

        st.write("### Selected Candidate Details")

        st.write({
            "Return Vehicle ID": best["vehicle_id_return"],
            "Return Latitude": best["lat_return"],
            "Return Longitude": best["lng_return"],
            "Distance from Model Prediction (metres)": round(
                best["prediction_distance_m"], 2
            )
        })
elif page == "Candidate Explorer":

    st.header("🔍 Candidate Explorer")

    if "filtered_candidates" not in st.session_state:

        st.warning(
            "Please generate E11 candidates from the Predict Destination page first."
        )

    else:

        candidates = st.session_state["filtered_candidates"].copy()

        st.write(
            "Explore the possible return locations remaining after E11 filtering."
        )

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Filtered Candidates", len(candidates))

        if "distance" in candidates.columns:
            col2.metric(
                "Average Distance",
                f"{candidates['distance'].mean():.2f} m"
            )

        if "time_diff" in candidates.columns:
            col3.metric(
                "Average Time Difference",
                f"{candidates['time_diff'].mean():.2f} sec"
            )

        st.divider()

        if not candidates.empty:

            st.subheader("Candidate Records")

            display_columns = [
                col for col in [
                    "vehicle_id_return",
                    "spoofed_id_return",
                    "lat_return",
                    "lng_return",
                    "distance",
                    "time_diff",
                    "battery_diff",
                    "range_diff",
                    "mean_speed_distance_based"
                ]
                if col in candidates.columns
            ]

            # Highlight selected candidate
            if "destination_prediction" in st.session_state:

                selected_id = (
                    st.session_state["destination_prediction"]
                    ["candidate"]["spoofed_id_return"]
                )

                candidates["Selection Status"] = candidates[
                    "spoofed_id_return"
                ].apply(
                    lambda x: "Selected Destination"
                    if x == selected_id
                    else "Alternative Candidate"
                )

            st.dataframe(
                candidates[display_columns + (
                    ["Selection Status"]
                    if "Selection Status" in candidates.columns
                    else []
                )],
                width="stretch"
            )

            st.divider()

            st.subheader("Candidate Distance Comparison")

            if "distance" in candidates.columns:

                chart_data = candidates[
                    ["vehicle_id_return", "distance"]
                ].copy()

                chart_data["vehicle_id_return"] = (
                    chart_data["vehicle_id_return"].astype(str)
                )

                st.bar_chart(
                    chart_data.set_index("vehicle_id_return")
                )

        else:

            st.warning("No candidates available for this rental event.")

elif page == "Map Visualization":

    st.header("🗺️ Destination Map Visualization")

    if (
        "destination_prediction" not in st.session_state
        or "filtered_candidates" not in st.session_state
    ):
        st.warning(
            "Please generate candidates and predict a destination first."
        )

    else:
        result = st.session_state["destination_prediction"]
        candidates = st.session_state["filtered_candidates"]

        best = result["candidate"]

        # Rental coordinates
        rental_lat = float(best["lat_rental"])
        rental_lng = float(best["lng_rental"])

        # Predicted destination
        predicted_lat = result["latitude"]
        predicted_lng = result["longitude"]

        # Selected candidate
        selected_lat = float(best["lat_return"])
        selected_lng = float(best["lng_return"])

        st.subheader("Predicted Destination Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Available Candidates", len(candidates))

        col2.metric(
            "Predicted Latitude",
            f"{predicted_lat:.6f}"
        )

        col3.metric(
            "Predicted Longitude",
            f"{predicted_lng:.6f}"
        )

        # Create map
        map_center = [
            predicted_lat,
            predicted_lng
        ]

        destination_map = folium.Map(
            location=map_center,
            zoom_start=14
        )

        # Rental marker
        folium.Marker(
            [rental_lat, rental_lng],
            popup="Rental Location",
            tooltip="Rental Location",
            icon=folium.Icon(color="blue", icon="bicycle")
        ).add_to(destination_map)

        # Predicted destination marker
        folium.Marker(
            [predicted_lat, predicted_lng],
            popup="XGBoost Predicted Destination",
            tooltip="Predicted Destination",
            icon=folium.Icon(color="red", icon="star")
        ).add_to(destination_map)

        # Selected candidate marker
        folium.Marker(
            [selected_lat, selected_lng],
            popup="Selected Return Candidate",
            tooltip="Selected Candidate",
            icon=folium.Icon(color="green", icon="ok")
        ).add_to(destination_map)

        # Other candidate locations
        for _, row in candidates.iterrows():

            folium.CircleMarker(
                location=[
                    row["lat_return"],
                    row["lng_return"]
                ],
                radius=6,
                color="orange",
                fill=True,
                fill_opacity=0.7,
                tooltip="Available Return Candidate"
            ).add_to(destination_map)

        # Display map
        st_folium(
            destination_map,
            width=1000,
            height=550
        )

        st.success(
            "Map generated using the rental location, "
            "XGBoost prediction and available return candidates."
        )

elif page == "Model Performance":

    st.header("📊 Model Performance Analysis")

    st.write(
        "Comparison of Baseline and E11 candidate filtering "
        "using the held-out test results."
    )

    # Actual test results
    metrics = pd.DataFrame({
        "Metric": [
            "Candidate Availability (%)",
            "Candidate Coverage - 1 km (%)",
            "Average Candidates",
            "End-to-End Accuracy - 1 km (%)"
        ],
        "Baseline": [
            95.71,
            86.63,
            2.59,
            70.07
        ],
        "E11": [
            97.28,
            90.75,
            3.12,
            69.98
        ]
    })

    st.subheader("Baseline vs E11")

    st.dataframe(
        metrics,
        hide_index=True,
        width="stretch"
    )

    st.divider()

    st.subheader("Performance Comparison")

    chart_metrics = metrics.set_index("Metric")

    st.bar_chart(chart_metrics)

    st.divider()

    st.subheader("Test Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Test Rentals", "2,282")

    col2.metric("Baseline Candidate Pairs", "5,917")

    col3.metric("E11 Candidate Pairs", "7,111")

    st.divider()

    st.subheader("Key Findings")

    st.success(
        "E11 improved candidate availability by 1.58 percentage points."
    )

    st.success(
        "E11 improved candidate coverage by 4.12 percentage points."
    )

    st.info(
        "E11 end-to-end accuracy was 69.98%, compared with "
        "70.07% for Baseline. The difference was not statistically "
        "significant in the paired McNemar test (p = 0.8601)."
    )

    st.write(
        "The results indicate that E11 improves the availability "
        "and geographic coverage of candidate return locations, "
        "while maintaining prediction accuracy close to the baseline."
    )

elif page == "Explainable AI":

    st.header("🧠 Explainable AI - SHAP Analysis")

    st.write(
        "Understand which input features influence the "
        "XGBoost destination prediction model."
    )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    metrics_dir = PROJECT_ROOT / "results" / "metrics"

    latitude_file = metrics_dir / "shap_latitude_importance.csv"
    longitude_file = metrics_dir / "shap_longitude_importance.csv"

    if latitude_file.exists() and longitude_file.exists():

        latitude_data = pd.read_csv(latitude_file)
        longitude_data = pd.read_csv(longitude_file)

        st.subheader("1. Latitude Prediction - Feature Importance")

        st.dataframe(
            latitude_data,
            width="stretch",
            hide_index=True
        )

        st.bar_chart(
            latitude_data.set_index(
                latitude_data.columns[0]
            )
        )

        st.divider()

        st.subheader("2. Longitude Prediction - Feature Importance")

        st.dataframe(
            longitude_data,
            width="stretch",
            hide_index=True
        )

        st.bar_chart(
            longitude_data.set_index(
                longitude_data.columns[0]
            )
        )

        st.divider()

        st.subheader("3. Interpretation")

        st.info("""
        **Latitude prediction:**
        Rental latitude is the most influential feature, followed by
        distance and rental longitude.

        **Longitude prediction:**
        Rental longitude is the most influential feature, followed by
        distance and rental latitude.

        Speed-related features have smaller contributions, while
        time difference, battery difference and range difference
        show relatively low importance in this SHAP analysis.
        """)

        st.caption(
            "Feature importance is based on the previously computed "
            "SHAP permutation analysis."
        )

    else:

        st.error(
            "SHAP importance files were not found in results/metrics."
        )