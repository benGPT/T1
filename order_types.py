import ccxt

def place_advanced_order(exchange, symbol, side, amount, price):
    """
    Place an advanced order with various order types.
    """
    order_type = 'market'  # You can change this to 'limit', 'stop', etc.
    
    if order_type == 'market':
        return exchange.create_market_order(symbol, side, amount)
    elif order_type == 'limit':
        return exchange.create_limit_order(symbol, side, amount, price)
    elif order_type == 'stop':
        return exchange.create_stop_order(symbol, side, amount, price, {'stopPrice': price * 0.95})
    else:
        raise ValueError(f"Unsupported order type: {order_type}")

