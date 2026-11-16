import sys
from pathlib import Path

import pandas as pd
from geopy.distance import geodesic

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_PATH


# --------------------------------------------------
# STEP 1: LOAD PROCESSED DATA
# --------------------------------------------------

def load_processed_data():

    df = pd.read_csv(PROCESSED_DATA_PATH)

    df["timestamp_lend"] = pd.to_datetime(
        df["timestamp_lend"], utc=True
    )

    df["timestamp_returned"] = pd.to_datetime(
        df["timestamp_returned"], utc=True
    )

    print("Dataset loaded successfully!")
    print("Total records:", len(df))

    return df


# --------------------------------------------------
# STEP 2: CREATE RENTAL AND RETURN EVENTS
# --------------------------------------------------

def create_events(df):

    all_rentals = {}
    all_returns = {}

    for index, row in df.iterrows():

        date = str(row["source_date"])

        rental_time = int(row["timestamp_lend"].timestamp())
        return_time = int(row["timestamp_returned"].timestamp())

        # Original rule: exclude trips longer than 3 hours
        if return_time - rental_time > 3 * 60 * 60:
            continue

        rental_battery = row["pedelec_battery_lend"]
        return_battery = row["pedelec_battery_returned"]

        # Handle missing battery values and 100% return battery
        if (
            pd.isna(rental_battery)
            or pd.isna(return_battery)
            or return_battery == 100
        ):
            rental_battery = 0
            return_battery = 0

        rental_battery = int(rental_battery)
        return_battery = int(return_battery)

        # Original rule: skip battery increases
        if return_battery > rental_battery:
            continue

        # Create date-wise event lists
        if date not in all_rentals:
            all_rentals[date] = []

        if date not in all_returns:
            all_returns[date] = []

        # Unique, reproducible event IDs
        rental_id = f"{date}_rental_{index}"
        return_id = f"{date}_return_{index}"

        all_rentals[date].append({

            "vehicle_id": row["vehicle_id"],
            "spoofed_id": rental_id,

            "timestamp": rental_time,

            "lat": float(row["lat_lend"]),
            "lng": float(row["lng_lend"]),

            "lat_returned_true": float(row["lat_returned"]),
            "lng_returned_true": float(row["lng_returned"]),

            "battery": rental_battery,
            "range": int(row["current_range_meters_lend"])

        })

        all_returns[date].append({

            "vehicle_id": row["vehicle_id"],
            "spoofed_id": return_id,

            "timestamp": return_time,

            "lat": float(row["lat_returned"]),
            "lng": float(row["lng_returned"]),

            "battery": return_battery,
            "range": int(row["current_range_meters_returned"])

        })

    print("\nEvent creation completed!")

    print("Rental events:", sum(len(v) for v in all_rentals.values()))
    print("Return events:", sum(len(v) for v in all_returns.values()))

    return all_rentals, all_returns


# --------------------------------------------------
# STEP 3: FILTER RETURNS WITH 100% BATTERY
# --------------------------------------------------

def filter_returns(all_returns):

    filtered_returns = {}

    removed_count = 0

    for date, returns in all_returns.items():

        filtered_returns[date] = []

        for ret in returns:

            if ret["battery"] < 100:
                filtered_returns[date].append(ret)

            else:
                removed_count += 1

    print("\nReturn filtering completed!")

    print("Returns removed:", removed_count)

    print(
        "Returns remaining:",
        sum(len(v) for v in filtered_returns.values())
    )

    return filtered_returns


# --------------------------------------------------
# STEP 4: GENERATE CANDIDATES
# --------------------------------------------------

def generate_candidates(all_rentals, filtered_returns):

    rental_to_possible_returns = {}

    # Prevent reuse of returns matched to special non-moving trips
    blacklist = set()

    total_candidates = 0

    for date, rentals in all_rentals.items():

        returns = filtered_returns.get(date, [])

        rental_to_possible_returns[date] = {}

        for rental in rentals:

            possible_returns = []

            rental_time = rental["timestamp"]
            rental_battery = rental["battery"]

            for ret in returns:

                # Skip already blacklisted return events
                if ret["spoofed_id"] in blacklist:
                    continue

                return_time = ret["timestamp"]
                return_battery = ret["battery"]

                time_diff = return_time - rental_time

                battery_diff = rental_battery - return_battery

                # Rule 1: Return must not be before rental
                if rental_time > return_time:
                    continue

                # Rule 2: Return battery cannot exceed rental battery
                if return_battery > rental_battery:
                    continue

                # Rule 3: Maximum time difference = 50 minutes
                if time_diff > 3000:
                    continue

                # Rule 4: Maximum battery difference = 10
                if battery_diff > 10:
                    continue

                # Calculate geodesic distance in metres
                distance = int(
                    geodesic(
                        (rental["lat"], rental["lng"]),
                        (ret["lat"], ret["lng"])
                    ).meters
                )

                # Calculate range difference
                range_diff = rental["range"] - ret["range"]

                # Calculate average speeds
                if time_diff > 0:

                    mean_speed = round(
                        (distance / 1000) / (time_diff / 3600), 2
                    )

                    mean_speed_range = round(
                        (range_diff / 1000) / (time_diff / 3600), 2
                    )

                else:
                    mean_speed = 0
                    mean_speed_range = 0

                # Rule 5: Maximum speed = 20 km/h
                if mean_speed > 20:
                    continue

                # Rule 6: Maximum distance = 5000 metres
                if distance > 5000:
                    continue

                # Rule 7: Special non-moving trip
                if (
                    time_diff <= 6
                    and range_diff == 0
                    and battery_diff <= 1
                    and distance < 1000
                ):

                    possible_returns = []

                    blacklist.add(ret["spoofed_id"])

                    possible_returns.append(
                        build_candidate(
                            rental, ret, date,
                            time_diff, battery_diff,
                            range_diff, distance,
                            mean_speed, mean_speed_range
                        )
                    )

                    break

                # Additional zero-time condition
                if time_diff == 0 and range_diff >= 1000:
                    continue

                # Add valid candidate
                possible_returns.append(
                    build_candidate(
                        rental, ret, date,
                        time_diff, battery_diff,
                        range_diff, distance,
                        mean_speed, mean_speed_range
                    )
                )

            rental_to_possible_returns[date][
                rental["spoofed_id"]
            ] = possible_returns

            total_candidates += len(possible_returns)

    print("\nCandidate generation completed!")

    print("Total rental events:", sum(len(v) for v in all_rentals.values()))

    print("Total candidate pairs:", total_candidates)

    return rental_to_possible_returns


# --------------------------------------------------
# STEP 5: BUILD CANDIDATE RECORD
# --------------------------------------------------

def build_candidate(
    rental,
    ret,
    date,
    time_diff,
    battery_diff,
    range_diff,
    distance,
    mean_speed,
    mean_speed_range
):

    return {

        "vehicle_id_rental": rental["vehicle_id"],
        "vehicle_id_return": ret["vehicle_id"],

        "spoofed_id_rental": rental["spoofed_id"],
        "spoofed_id_return": ret["spoofed_id"],

        "lat_rental": rental["lat"],
        "lng_rental": rental["lng"],

        "lat_return": ret["lat"],
        "lng_return": ret["lng"],

        "lat_returned_true": rental["lat_returned_true"],
        "lng_returned_true": rental["lng_returned_true"],

        "timestamp_rental": rental["timestamp"],
        "timestamp_return": ret["timestamp"],

        "battery_rental": rental["battery"],
        "battery_return": ret["battery"],

        "range_rental": rental["range"],
        "range_return": ret["range"],

        "distance": distance,
        "time_diff": time_diff,

        "battery_diff": battery_diff,
        "range_diff": range_diff,

        "speed": mean_speed,
        "speed_range_based": mean_speed_range,

        "date": date

    }


# --------------------------------------------------
# STEP 6: MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    # Load data
    trips = load_processed_data()

    # Create rental and return events
    all_rentals, all_returns = create_events(trips)

    # Remove 100% battery returns
    filtered_returns = filter_returns(all_returns)

    # Generate candidates
    candidates = generate_candidates(
        all_rentals,
        filtered_returns
    )

    # Convert candidate dictionary into a DataFrame
    candidate_records = []

    for date, rentals in candidates.items():

        for rental_id, returns in rentals.items():

            for candidate in returns:

                candidate_records.append(candidate)

    candidates_df = pd.DataFrame(candidate_records)

    # Save results
    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "expanded_candidates.csv"
    )

    candidates_df.to_csv(output_path, index=False)

    print("\n===================================")
    print("BASELINE PIPELINE FINISHED")
    print("===================================")

    print("Total candidate pairs:", len(candidates_df))

    print("Saved to:", output_path)

    print("===================================")