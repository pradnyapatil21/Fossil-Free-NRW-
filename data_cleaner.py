import pandas as pd

# Load your CSV file
df = pd.read_csv("coal mine data.csv")

# Strip column names to remove any unwanted whitespace
df.columns = df.columns.str.strip()

# Define key column names
plant_col = 'Coal Plant/Lignite power plants'
power_col = 'Installed power  MW'
phase_col = 'phase'

# Define the full set of expected phases
all_phases = {1, 2, 3}

# Collect new rows for missing phases
new_rows = []

# Group by plant name
for plant, group in df.groupby(plant_col):
    existing_phases = set(group[phase_col])
    missing_phases = all_phases - existing_phases
    installed_power = group[power_col].iloc[0]

    for phase in missing_phases:
        new_row = {
            plant_col: plant,
            power_col: installed_power,
            phase_col: phase
        }
        # Set all other columns to zero
        for col in df.columns:
            if col not in new_row:
                new_row[col] = 0
        new_rows.append(new_row)

# Create a DataFrame from new rows and append it
df_extended = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# Sort the result for readability
df_extended = df_extended.sort_values(by=[plant_col, phase_col]).reset_index(drop=True)

solar_cost_map = { 
    '1': 182.5, 
    '2': 160.0, 
    '3': 135.5 
}

# Convert phase to string (if it's not already)
df_extended['phase'] = df_extended['phase'].astype(str)

# Apply the cost map to 'Solar PV (€/MWh)'
df_extended['Solar PV (€/MWh)'] = df_extended['phase'].map(solar_cost_map)

# Save to CSV if needed

wind_cost_map = {
    "1": 155.0,
    "2": 139.0,
    "3": 116.5
}

# Convert phase to string (if it's not already)
df_extended['phase'] = df_extended['phase'].astype(str)

# Apply the cost map to 'Solar PV (€/MWh)'
df_extended['Wind (€/MWh)'] = df_extended['phase'].map(wind_cost_map)

numeric_cols = [
    "Coal MW replaced( demand)",
    "Solar MW(generation capacity)",
    "Wind MW(generation capacity)",
    "Panels needed",
    "Turbines Needed",
    "Solar PV (€/MWh)",
    "Wind (€/MWh)"
]

for col in numeric_cols:
    df_extended[col] = df_extended[col].replace(",", "", regex=True).astype(float)
# Save to CSV if needed
df_extended.to_csv("coal_mine_data_cleaned.csv", index=False, encoding='utf-8-sig')
