import pandas as pd

# Read the CSV file
df = pd.read_csv('Processed_Plant_Data.csv')

# Display the first few rows
print(df.to_string())

# Check for missing values in each column
print(df.isnull().sum())

import pandas as pd
import gamspy as gp

# === Load Data ===
df = pd.read_csv("Processed_Plant_Data.csv")
df.columns = df.columns.str.strip()  # remove trailing spaces

# === Prepare Data ===
df['Plant_ID'] = df['Plant_ID'].astype(str)
df['phase'] = df['phase'].astype(str)
df['Coal Plant/Lignite power plants'] = df['Coal Plant/Lignite power plants'].astype(str)

# === Sets ===
plants = df['Plant_ID'].unique().tolist()  # I
demand_nodes = df['Coal Plant/Lignite power plants'].unique().tolist()  # J
periods = sorted(df['phase'].unique().tolist())  # T

# === Create Model Container ===
m = gp.Container()

I = gp.Set(m, name="I", description="Candidate plant sites", records=plants)
J = gp.Set(m, name="J", description="Demand nodes", records=demand_nodes)
T = gp.Set(m, name="T", description="Planning periods", records=periods)

# === Parameters ===
c = gp.Parameter(m, name="c", domain=[I, J, T])
d = gp.Parameter(m, name="d", domain=[J, T])
s = gp.Parameter(m, name="s", domain=[I, T])

# === Scalars ===
lo = 0.3  # Minimum utilization (can be updated)
p = 3     # Number of renewable plants per period (example; update as needed)

# === Assign Values to Parameters ===
for _, row in df.iterrows():
    i = row['Plant_ID']
    j = row['Coal Plant/Lignite power plants']
    t = row['phase']
    c[i, j, t] = float(row['Cost'])
    d[j, t] = float(row['Installed power  MW (demand)'])
    s[i, t] = float(row['Generation_Capacity'])

# === Binary Variables ===
x = gp.Variable(m, name="x", domain=[I, J, T], type="binary")
y = gp.Variable(m, name="y", domain=[I, T], type="binary")

# === Objective Function ===
obj_expr = gp.Sum([t for t in T], gp.Sum([i for i in I], gp.Sum([j for j in J], c[i, j, t] * x[i, j, t])))
m.Equations['Obj'] = gp.Equation(m, name="Obj", body=obj_expr)
m.Objective = gp.Objective(m, name="TotalCost", equation=m.Equations['Obj'], sense=gp.minimize)

# === Constraints ===

# 1. Demand allocation
for t in T:
    for j in J:
        m.Equations[f"demand_alloc_{t}_{j}"] = gp.Equation(
            m,
            name=f"demand_alloc_{t}_{j}",
            body=gp.Sum([i for i in I], x[i, j, t]) == 1
        )

# 2. Facility capacity constraint
for t in T:
    for i in I:
        m.Equations[f"cap_{t}_{i}"] = gp.Equation(
            m,
            name=f"cap_{t}_{i}",
            body=gp.Sum([j for j in J], d[j, t] * x[i, j, t]) <= s[i, t] * y[i, t]
        )

# 3. Minimum utilization
for t in T:
    for i in I:
        m.Equations[f"util_{t}_{i}"] = gp.Equation(
            m,
            name=f"util_{t}_{i}",
            body=gp.Sum([j for j in J], d[j, t] * x[i, j, t]) >= lo * s[i, t] * y[i, t]
        )

# 4. Policy constraint
for t in T:
    m.Equations[f"policy_{t}"] = gp.Equation(
        m,
        name=f"policy_{t}",
        body=gp.Sum([i for i in I], y[i, t]) == p
    )

# === Solve ===
m.solve(solver="cbc")  # or "gurobi" if available

# === Results ===
print("\nObjective Value:", m.Objective.level)

# Extract operational plants
results = []
for i in I:
    for t in T:
        if y[i, t].level > 0.5:
            results.append({"Plant": i, "Period": t, "Active": 1})
df_results = pd.DataFrame(results)

print("\nActive Plants per Period:")
print(df_results)
