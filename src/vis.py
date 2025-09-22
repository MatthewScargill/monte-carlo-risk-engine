import pandas as pd
import matplotlib.pyplot as plt
from calibration import *

# currently only works for single stock / change to multi csv in long format
def csv_graph(directory="data/AAPL_SIM.csv", type="returns"):

    # Load the CSV file (replace with your file path)
    df = pd.read_csv(directory, parse_dates=["Date"])

    # Convert close prices to returns
    returns = to_returns(df["Adj Close"])

    # Initiate plot 
    plt.figure(figsize=(12,6))

    if type == "returns":
        plt.plot(df["Date"].iloc[1:], returns["Adj Close"], label="Log Returns", linewidth=1.5, color="green")
        plt.ylabel("Close Returns")

    elif type == "prices":
        plt.plot(df["Date"], df["Adj Close"], label="Close Price", linewidth=2, color="blue")
        plt.ylabel("Close Price")

    else:
        raise ValueError("type must be 'prices' or 'returns'")
    
    # Unchanging labels, title, and factors
    plt.title("Stock Evolution")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

csv_graph(directory="data/T3STstock.csv")




# comparative and illustrative visualisations

# 1) comparing performace with different additions against hsitorical data (add backtesting later)
# 2) showing all the simulated paths (and colour coding them for risk)
# 3) future projections for various stocks (maybe have them fade into red as the time steps get longer and longer to show possible inaccuracy)
