-- Bootstrap script: creates the database and an application user.
-- Change 'your_password' before running, and mirror it in your .env file.
CREATE DATABASE investment_portfolio;
CREATE USER 'portfolio_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON investment_portfolio.* TO 'portfolio_user'@'localhost';
FLUSH PRIVILEGES;
