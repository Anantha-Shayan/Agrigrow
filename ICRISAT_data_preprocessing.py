import pandas as pd
import json

def clean_and_transform_icrisat(file_path):
    df = pd.read_csv(file_path)

    base_cols = ["State Name", "Dist Name", "Year"]
    crop_cols = [c for c in df.columns if c not in ["Dist Code", "State Code"] + base_cols]

    # Melt dataset to long format
    long_df = df.melt(
        id_vars=base_cols,
        value_vars=crop_cols,
        var_name="metric",
        value_name="value"
    )

    # Extract crop + metric type
    def split_metric(metric):
        parts = metric.split(" ")
        crop = []
        metric_type = []
        for p in parts:
            if p.upper() in ["AREA", "PRODUCTION", "YIELD"]:
                metric_type.append(p)
            else:
                crop.append(p)
        return " ".join(crop).strip(), " ".join(metric_type).strip()

    long_df[["crop", "metric_type"]] = long_df["metric"].apply(
        lambda x: pd.Series(split_metric(x))
    )
    long_df.drop(columns=["metric"], inplace=True)

    # Pivot so each crop-year-district has AREA, PRODUCTION, YIELD in columns
    cleaned_df = long_df.pivot_table(
        index=["State Name", "Dist Name", "Year", "crop"],
        columns="metric_type",
        values="value",
        aggfunc="first"
    ).reset_index()

    cleaned_df = cleaned_df.rename(columns={
        "State Name": "state",
        "Dist Name": "district",
        "Year": "year",
        "AREA": "area_1000ha",
        "PRODUCTION": "production_1000tons",
        "YIELD": "yield_kg_per_ha"
    })

    return cleaned_df


def generate_crop_mappings(file_path):
    cleaned_df = clean_and_transform_icrisat(file_path)

    # Drop rows with no production
    cleaned_df = cleaned_df.dropna(subset=["production_1000tons"])
    # Keep crops with some nonzero production
    filtered_df = cleaned_df[cleaned_df["production_1000tons"] > 0]

    # District → Crops
    district_crop_map = (
        filtered_df.groupby(["state", "district"])["crop"]
        .unique()
        .apply(list)  # ensure list not numpy array
        .to_dict()
    )

    # Crop → Districts
    crop_district_map = (
        filtered_df.groupby("crop")[["state", "district"]]
        .apply(lambda x: x.drop_duplicates().to_dict(orient="records"))
        .to_dict()
    )

    return cleaned_df, district_crop_map, crop_district_map


if __name__ == "__main__":
    file_path = "ICRISAT-District Level Data.csv"

    cleaned_df, district_crop_map, crop_district_map = generate_crop_mappings(file_path)

    # Save both JSONs
    with open("district_crop_map.json", "w") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in district_crop_map.items()}, f, indent=2)

    with open("crop_district_map.json", "w") as f:
        json.dump(crop_district_map, f, indent=2)

    print("✅ Saved district_crop_map.json and crop_district_map.json")