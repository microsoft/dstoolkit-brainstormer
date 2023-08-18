# %%

import json
from pathlib import Path

import pandas as pd
import plotly.express as px

# %%
data_path = Path("data/test_data_for_ui.json")

with data_path.open() as data_file:
    data = json.load(data_file)

data = data["ideation_stage"]["results"]
use_cases = []
business_values = []
feasibilities = []

for use_case_key in data.keys():
    use_cases.append(data[use_case_key]["use_case_name"])
    business_values.append(data[use_case_key]["business_value"])
    feasibilities.append(data[use_case_key]["feasibility"])

df = pd.DataFrame(
    {
        "use_case": use_cases,
        "business_value": business_values,
        "feasibility": feasibilities,
    }
)

value_mapping = {"High": 100, "Moderate": 50, "Low": 10}
df["business_value_score"] = df["business_value"].replace(value_mapping)
df["feasibility_score"] = df["feasibility"].replace(value_mapping)
df["overall_score"] = df["business_value_score"] + df["feasibility_score"]
df = df.drop(columns=["business_value_score", "feasibility_score"])

fig = px.scatter(
    df,
    y="business_value",
    x="feasibility",
    color="use_case",
    symbol="use_case",
    size="overall_score",
    category_orders={
        "feasibility": ["Low", "Moderate", "High"],
        "business_value": ["High", "Moderate", "Low"],
    },
    
)
fig.show()
fig.write_html("data/sample_vis.html")
