import os
import gurobipy as gp
import pandas as pd
import numpy as np
import math
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense, Options

#os.environ['GRB_LICENSE_FILE'] = 'path to gurobi license'

#print(f"GRB_LICENSE_FILE: {os.environ.get('GRB_LICENSE_FILE')}")

# Read the CSV file
df = pd.read_csv('coal_mine_data_cleaned.csv')

condition = (
    (df['Coal MW replaced( demand)'] != 0) &
    (df['Solar MW(generation capacity)'] != 0)
)

df = df[condition]

# Get unique names
unique_names = df["Coal Plant/Lignite power plants"].unique()

# Show unique names
#print(unique_names)

# Get total number of unique names
total_unique = df["Coal Plant/Lignite power plants"].nunique()
print("Total unique names:", total_unique)

# Configuration - adjust these as needed
num_i = total_unique    
num_j = total_unique    
#num_t = 3  

#t_labels = [f"t{t+1}" for t in range(num_t)]
present_phases = sorted(df["phase"].unique())
t_labels = [f"t{p}" for p in present_phases]
i_labels = unique_names.astype(str).tolist()
#print(i_labels)
j_labels = unique_names.astype(str).tolist()
#print(j_labels)

cost_input = df[["Coal Plant/Lignite power plants", "phase", "Total Solar cost of allocating demand node j to facility i in period t (in €)"]]
cost_input.columns = ["i", "t", "value"]

# Convert t to t1, t2, etc.
cost_input["t"] = cost_input["t"].apply(lambda x: f"t{x}")
cost_input["i"] = cost_input["i"].astype(str)

# Create all combinations of i, j, t
multi_index = pd.MultiIndex.from_product(
    [i_labels, j_labels, t_labels],
    names=['i', 'j', 't']
)

# Make DataFrame from MultiIndex
assignment_cost = pd.DataFrame(index=multi_index).reset_index()

# Merge in the cost info (from cost_input)
assignment_cost = assignment_cost.merge(cost_input, on=['i', 't'], how='left')
assignment_cost = assignment_cost.dropna(subset=["value"])

# Set proper MultiIndex again
assignment_cost = assignment_cost.set_index(['i', 'j', 't'])[['value']]

print(assignment_cost.head())


p = 40 # number of located facilities
#p = 0 # number of factories that can be running
#print(p)

k = num_j # scalar needed for coupling constraint
#print(k)

m = Container()

#sets
j = Set(container=m,
        name="j",
        description="demand locations",
        records= assignment_cost.index.get_level_values('j').unique()
        )
#j.records
#print(j.records)

i = Set(container=m,
        name="i",
        description="candidate locations",
        records= assignment_cost.index.get_level_values('i').unique()
        )
i.records
#print(i.records)

t = Set(container=m,
        name="t",
        description="dperiods",
        records= assignment_cost.index.get_level_values('t').unique()
        )
t.records
#print(t.records)

#parameters
#print(assignment_cost.to_string())

c = Parameter(
    container=m,
    name="c",
    domain=[i, j, t],
    description="cost of allocating demand node j to facility i in period t",
)
# Assign values from DataFrame
c.setRecords(assignment_cost.reset_index())
#print(c.records.to_string())

#Create GAMSpy parameters  sit  and  djt .
s = Parameter(
    container=m,
    name="s",
    domain=[i, t],
    description="energy supply of facility i in period t",
)

solar_df = df[["Coal Plant/Lignite power plants", "phase", "Solar MW(generation capacity)"]]
solar_df.columns = ["i", "t", "value"]
solar_df["t"] = solar_df["t"].apply(lambda x: f"t{x}")

cf_solar = 0.15
# hours_per_year = 8760
# solar_df["value"] = solar_df["value"] * cf_solar * hours_per_year
solar_df["value"] = solar_df["value"] * cf_solar
s.setRecords(solar_df)

# print(s.records)

d = Parameter(
    container=m,
    name="d",
    domain=[j, t],
    description="energy demand of demand point j in period t",
)

demand_solar_df = df[["Coal Plant/Lignite power plants", "phase", "Coal MW replaced( demand)"]]
demand_solar_df.columns = ["j", "t", "value"]
demand_solar_df["t"] = demand_solar_df["t"].apply(lambda x: f"t{x}")
d.setRecords(demand_solar_df)

#print(d.records)

lo = Parameter(m, name="lo", description="minimum utilization", records= 0.8)
print(lo.records)

varP = Parameter(m, name="varP")
varP.setRecords(p)
varP.records

x = Variable(
    container=m,
    name="x",
    domain=[i, j, t],
    type="binary",
    description="= 1, if allocating demand node j to facility i in period t (0, otherwise)",
)

y = Variable(
    container=m,
    name="y",
    domain=[i, t], 
    type="binary",
    description="= 1, facility i is available in period t (0, otherwise)",
)

obj = Sum((i, j, t), c[i, j, t] * x[i, j, t]) #objective function

assign = Equation(
    container=m,
    name="assign",
    domain=[j,t],
    description="assign each demand point j to exactly one facility i in period t"
)

#assign[j,t] = Sum(i, x[i, j, t]) == 1 ## assignment (1st constraint)
# Build a[j,t] = 1 if d[j,t] > 0 else 0
a = Parameter(
    container=m,
    name="a",
    domain=[j, t],
    description="1 if demand at (j,t) > 0, else 0",
)

# Turn d into a DataFrame, compute indicator, then set records
d_df = demand_solar_df.copy()               # has columns: ["j","t","value"]
a_df = d_df.copy()
a_df["value"] = (a_df["value"] > 0).astype(int)
a.setRecords(a_df)

assign[j,t] = Sum(i, x[i, j, t]) == a[j,t]

maxutil = Equation(
    container=m,
    name="maxutil",
    domain=[i,t],
    description="maximum utilization of facility i in period t"
)

maxutil[i,t] = Sum(j, d[j,t] * x[i,j,t]) <= s[i,t] * y[i,t]

minutil = Equation(
   container=m,
   name="minutil",
   domain=[i,t],
   description="minimum utilization of facility i in period t"
)

minutil[i,t] = Sum(j, d[j,t] * x[i,j,t]) >= lo * s[i,t] * y[i,t]


facility_count = Equation(
    container=m,
    name="facility_count",
    domain=[t],
    description="limit the total number of open facilities to p for each period t",
)

facility_count[t] = Sum(i, y[i, t]) <= varP  ## p constraint (3rd constraint) using GAMSpy parameter varP

mppmp = Model(
    m,
    name="mppmp",
    equations=[assign, minutil, maxutil, facility_count],
    problem="MIP",
    sense=Sense.MIN,
    objective=obj,
)

import sys
mppmp.solve(output=sys.stdout,options=Options(relative_optimality_gap=0))

y.records.set_index(["i", "t"])
x.records.set_index(["i","j","t"])
assignment = pd.DataFrame(x.records.set_index(["i","j","t"]))
print(assignment.head())

import networkx as nx
import matplotlib.pyplot as plt

active_assignments = assignment[assignment['level'] > 0]
print(active_assignments.head())

# Iterate over each unique 't' and plot the graph
for t in assignment.index.get_level_values('t').unique():
    # Get assignments for the current 't'
    edges = active_assignments.xs(t, level='t', drop_level=False)

    if not edges.empty:
        # Create a directed graph
        G = nx.DiGraph()

        # Add edges (j -> i) with weight = level
        for (i, j, _), row in edges.iterrows():
            G.add_edge(j, i, weight=row['level'])

        # Draw the graph
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)  # Layout for consistent visualization
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(G, pos, font_size=12)

        # Add edge labels (weights)
        edge_labels = {(j, i): f"{row['level']}" for (i, j, _), row in edges.iterrows()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

        plt.title(f"Assignments for t = {t}")
        plt.axis('off')
        plt.show()
    else:
        print(f"No assignments for t = {t}")

mppmp_analysis = Model(
    m,
    name="mppmp_analysis",
    equations=[assign, minutil, maxutil, facility_count],
    problem="MIP",
    sense=Sense.MIN,
    objective=obj,
)

located = list(range(0, 40, 1)) # second argument = card(j) +1
located

# Initialize an empty list to store results
results = []

mppmp_analysis.freeze(modifiables=[varP]) # keep all constant except varP

for p_value in located:
    varP.setRecords(p_value)
    mppmp_analysis.solve()
     # Store results in a dictionary
    result_entry = {
        'p_value': p_value,
        'objective_value': mppmp_analysis.objective_value
    }
    results.append(result_entry)  # Append to results list

mppmp_analysis.unfreeze()

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame
print(results_df)

import matplotlib.pyplot as plt

# Assuming `results_df` is your DataFrame from the previous step
plt.figure(figsize=(8, 5))  # Set figure size

# Plot dashed line with points
plt.plot(
    results_df['p_value'],
    results_df['objective_value'],
    marker='o',          # Show points as circles
    linestyle='--',      # Dashed line
    color='blue',        # Line color
    markersize=8,        # Point size
    linewidth=2          # Line thickness
)


# Label axes
plt.xlabel("Number of located facilities", fontsize=12)
plt.ylabel("Total assignment cost", fontsize=12)

# Add grid for better readability
plt.grid(True, linestyle=':', alpha=0.7)

# Title (optional)
plt.title("Objective Value vs. Number of Facilities", fontsize=14)

# Show plot
plt.show()