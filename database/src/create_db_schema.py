from typing import Text

from sqlalchemy import text
from database.src.connection import get_engine, setup_engine

engine = setup_engine()

def create_schema():
    with engine.begin() as conn:
        
        create_table_assets = """
        CREATE TABLE IF NOT EXISTS assets (
            asset_id INT AUTO_INCREMENT PRIMARY KEY,
            ticker VARCHAR(10),
            name VARCHAR(100),
            sector VARCHAR(50)
        )
        """
        conn.execute(text(create_table_assets))

        
        create_table_historical_prices = """
        CREATE TABLE IF NOT EXISTS historical_prices (
            price_id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE NOT NULL,
            asset_id INT,
            close DECIMAL(15,2),
            volume BIGINT,
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
        )
        """
        conn.execute(text(create_table_historical_prices))

        
        create_table_portfolio_weights = """
        CREATE TABLE IF NOT EXISTS portfolio_weights (
            portfolio_id INT AUTO_INCREMENT PRIMARY KEY,
            asset_id INT,
            weight DECIMAL(5,4),
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
        )
        """
        conn.execute(text(create_table_portfolio_weights))

        
        create_table_factors = """
        CREATE TABLE IF NOT EXISTS factors (
            factor_id INT AUTO_INCREMENT PRIMARY KEY,
            asset_id INT,
            date DATE,
            beta DECIMAL(10,4),
            sector_factor DECIMAL(10,4),
            FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
        )
        """
        conn.execute(text(create_table_factors))

        print('Schema created')

if __name__ == "__main__":
    create_schema()