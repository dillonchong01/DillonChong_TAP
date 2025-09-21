import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

resale_df = pd.read_csv("Question 1/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")
agent_df = pd.read_csv("Question 1/CEASalespersonsPropertyTransactionRecordsresidential.csv")

## Resale Dataframe
# Convert month col to quarter
resale_df["month"] = pd.to_datetime(resale_df["month"], format="%Y-%m")
resale_df["quarter"] = resale_df["month"].dt.to_period("Q").dt.to_timestamp()
# Aggregate no. of sales per quarter
sales_per_quarter = resale_df.groupby("quarter").size()


## Graph 1 - HDB Resale Quantity over time
# Best Fit Line
x = np.arange(len(sales_per_quarter))
y = sales_per_quarter.values
coeffs = np.polyfit(x, y, 1)
trendline = np.polyval(coeffs, x)

# Plot Graph
plt.plot(sales_per_quarter.index, sales_per_quarter.values,
         marker="o")
plt.plot(sales_per_quarter.index, trendline, color="red")
plt.title("No. of HDB Resale Transactions over Time")
plt.xlabel("Quarter")
plt.ylabel("Transactions per Quarter")
plt.grid(True)
plt.show()

## Agent Dataframew
# Convert month to quarter
agent_df["transaction_date"] = pd.to_datetime(agent_df["transaction_date"], format="mixed")
agent_df["quarter"] = agent_df["transaction_date"].dt.to_period("Q").dt.to_timestamp()
# Filter for HDB transactions
agent_df = agent_df[(agent_df["property_type"] == "HDB") & (agent_df["transaction_type"] == "RESALE")]
# Split into Seller and Buyer agents
seller_df = agent_df[agent_df["represented"] == "SELLER"]
buyer_df = agent_df[agent_df["represented"] == "BUYER"]
# Count monthly sales
sales_seller = seller_df.groupby("transaction_date").size()
sales_buyer = buyer_df.groupby("transaction_date").size()
# Aggregate no. of sales per quarter
sales_seller_q = seller_df.groupby("quarter").size()
sales_buyer_q = buyer_df.groupby("quarter").size()

## Graph 2 - HDB Resale Quantity vs Agent Trasactions over Time
plt.plot(sales_seller_q.index, sales_seller_q.values, marker="o", label="Agent Transactions (Seller)")
plt.plot(sales_buyer_q.index, sales_buyer_q.values, marker="o",label="Agent Transactions (Buyer)")

plt.plot(sales_per_quarter.index, sales_per_quarter.values, marker="o", label="HDB Resale Transactions")

plt.title("HDB Resales vs Agent Transactions over Time")
plt.xlabel("Quarter")
plt.ylabel("Transactions per Quarter")
plt.grid(True)
plt.legend()
plt.show()


## Graph 3 - Percentage of Transactions not handled by Agents
# Compute % Gaps
gap_buyer_pct = (sales_per_quarter - sales_buyer_q.reindex(sales_per_quarter.index, fill_value=0)) / sales_per_quarter * 100
gap_seller_pct = (sales_per_quarter - sales_seller_q.reindex(sales_per_quarter.index, fill_value=0)) / sales_per_quarter * 100

x = np.arange(len(sales_per_quarter))

# Best Fit Lines
trendline_buyer = np.polyval(np.polyfit(x, gap_buyer_pct.values, 1), x)
trendline_seller = np.polyval(np.polyfit(x, gap_seller_pct.values, 1), x)

# Plot Graph
plt.plot(gap_buyer_pct.index, gap_buyer_pct.values, marker="o", linewidth=2, label="% of Transactions Unrepresented (Buyer)")
plt.plot(gap_buyer_pct.index, trendline_buyer, linestyle="--", color="blue")

plt.plot(gap_seller_pct.index, gap_seller_pct.values, marker="o", linewidth=2, label="% of Transactions Unrepresented (Seller)")
plt.plot(gap_seller_pct.index, trendline_seller, linestyle="--", color="orange")

plt.title("% of Transactions not Represented by Agents (Buyers vs Sellers)")
plt.xlabel("Quarter")
plt.ylabel("Gap as % of Total Transactions")
plt.grid(True)
plt.legend()
plt.show()
