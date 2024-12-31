import hashlib
import json
import os

USER_DB_FILE = "user_db.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_user_db():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_db(user_db):
    with open(USER_DB_FILE, 'w') as f:
        json.dump(user_db, f)

def authenticate_user(username, password):
    user_db = load_user_db()
    if username in user_db and user_db[username]['password'] == hash_password(password):
        return user_db[username]
    return None

def create_user(username, password):
    user_db = load_user_db()
    if username in user_db:
        return False
    user_db[username] = {
        'username': username,
        'password': hash_password(password),
        'portfolio': {'holdings': {}, 'transactions': [], 'performance': {'dates': [], 'values': []}}
    }
    save_user_db(user_db)
    return True

def get_user_portfolio(username):
    user_db = load_user_db()
    return user_db[username]['portfolio']

def update_user_portfolio(username, portfolio):
    user_db = load_user_db()
    user_db[username]['portfolio'] = portfolio
    save_user_db(user_db)

