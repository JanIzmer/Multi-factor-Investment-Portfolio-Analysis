from sqlalchemy import text
from database.src.connection import setup_engine

engine = setup_engine()

def create_schema():
    with engine.begin() as conn:

        # -----------------------------
        # 1. Assets
        # -----------------------------
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS assets (
            asset_id INT AUTO_INCREMENT PRIMARY KEY,
            ticker VARCHAR(10) UNIQUE,
            name VARCHAR(100),
            sector VARCHAR(50)
        );
        """))

        # -----------------------------
        # 2. Historical prices
        # -----------------------------
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS historical_prices (
            price_id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE NOT NULL,
            asset_id INT NOT NULL,
            close DECIMAL(15,2),
            volume BIGINT,
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
            UNIQUE KEY uq_asset_date_price (asset_id, date)
        );
        """))

        # -----------------------------
        # 3. Returns table
        # -----------------------------
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS returns (
            return_id INT AUTO_INCREMENT PRIMARY KEY,
            asset_id INT NOT NULL,
            date DATE NOT NULL,
            `return` DOUBLE,
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
            UNIQUE KEY uq_asset_date_return (asset_id, date)
        );
        """))

        # -----------------------------
        # 4. Factors table
        # -----------------------------
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS factors (
            factor_id INT AUTO_INCREMENT PRIMARY KEY,
            asset_id INT NOT NULL,
            date DATE NOT NULL,
            beta DOUBLE,
            volatility DOUBLE,
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
            UNIQUE KEY uq_asset_date_factor (asset_id, date)
        );
        """))

        # -----------------------------
        # 5. Portfolio definitions
        # -----------------------------
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS portfolio_weights (
            portfolio_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """))

        # -----------------------------
        # 6. Portfolio rows
        # -----------------------------
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS portfolio_weight_rows (
            id INT AUTO_INCREMENT PRIMARY KEY,
            portfolio_id INT NOT NULL,
            asset_id INT NOT NULL,
            weight DOUBLE,
            FOREIGN KEY (portfolio_id) REFERENCES portfolio_weights(portfolio_id),
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
        );
        """))

        print("Schema created successfully.")

if __name__ == "__main__":
    create_schema()
