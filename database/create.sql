CREATE DATABASE investment_portfolio;
CREATE USER 'portfolio_user'@'localhost' IDENTIFIED BY '';
GRANT ALL PRIVILEGES ON investment_portfolio.* TO 'portfolio_user'@'localhost';
FLUSH PRIVILEGES;