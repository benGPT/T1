import yfinance as yf

def get_fundamental_data(symbol):
    """
    Fetch fundamental data for the given symbol using yfinance.
    """
    stock = yf.Ticker(symbol)
    
    # Get key statistics
    info = stock.info
    
    # Get financial statements
    income_stmt = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow
    
    # Calculate some key ratios
    if 'totalAssets' in info and 'totalLiab' in info:
        debt_to_equity = info['totalLiab'] / (info['totalAssets'] - info['totalLiab'])
    else:
        debt_to_equity = None
    
    if 'trailingPE' in info:
        pe_ratio = info['trailingPE']
    else:
        pe_ratio = None
    
    if 'priceToBook' in info:
        price_to_book = info['priceToBook']
    else:
        price_to_book = None
    
    return {
        "Market Cap": info.get('marketCap'),
        "P/E Ratio": pe_ratio,
        "Price to Book Ratio": price_to_book,
        "Debt to Equity Ratio": debt_to_equity,
        "Revenue (TTM)": info.get('totalRevenue'),
        "Gross Profit (TTM)": info.get('grossProfits'),
        "Free Cash Flow (TTM)": info.get('freeCashflow'),
        "Dividend Yield": info.get('dividendYield'),
        "52 Week High": info.get('fiftyTwoWeekHigh'),
        "52 Week Low": info.get('fiftyTwoWeekLow'),
    }

