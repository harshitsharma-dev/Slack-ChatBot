-- Seed data for Slack AI Data Bot
-- This file contains sample data to populate the database for testing

-- Insert sample users
INSERT INTO users (name, email, phone, address, city, country) VALUES
('John Doe', 'john.doe@email.com', '+1-555-0101', '123 Main St', 'New York', 'USA'),
('Jane Smith', 'jane.smith@email.com', '+1-555-0102', '456 Oak Ave', 'Los Angeles', 'USA'),
('Bob Johnson', 'bob.johnson@email.com', '+1-555-0103', '789 Pine Rd', 'Chicago', 'USA'),
('Alice Brown', 'alice.brown@email.com', '+1-555-0104', '321 Elm St', 'Houston', 'USA'),
('Charlie Wilson', 'charlie.wilson@email.com', '+1-555-0105', '654 Maple Dr', 'Phoenix', 'USA'),
('Diana Davis', 'diana.davis@email.com', '+1-555-0106', '987 Cedar Ln', 'Philadelphia', 'USA'),
('Eve Miller', 'eve.miller@email.com', '+1-555-0107', '147 Birch Ct', 'San Antonio', 'USA'),
('Frank Garcia', 'frank.garcia@email.com', '+1-555-0108', '258 Spruce Way', 'San Diego', 'USA'),
('Grace Lee', 'grace.lee@email.com', '+1-555-0109', '369 Willow St', 'Dallas', 'USA'),
('Henry Taylor', 'henry.taylor@email.com', '+1-555-0110', '741 Ash Blvd', 'San Jose', 'USA');

-- Insert sample products
INSERT INTO products (name, description, price, category, stock_quantity) VALUES
-- Electronics
('iPhone 15 Pro', 'Latest Apple smartphone with advanced features', 999.99, 'Electronics', 50),
('Samsung Galaxy S24', 'Premium Android smartphone', 899.99, 'Electronics', 45),
('MacBook Pro 16"', 'High-performance laptop for professionals', 2499.99, 'Electronics', 25),
('Dell XPS 13', 'Ultrabook with excellent build quality', 1299.99, 'Electronics', 30),
('iPad Air', 'Versatile tablet for work and entertainment', 599.99, 'Electronics', 40),
('AirPods Pro', 'Wireless earbuds with noise cancellation', 249.99, 'Electronics', 100),
('Sony WH-1000XM5', 'Premium noise-canceling headphones', 399.99, 'Electronics', 35),
('Nintendo Switch', 'Popular gaming console', 299.99, 'Electronics', 60),

-- Clothing
('Nike Air Max 270', 'Comfortable running shoes', 150.00, 'Clothing', 80),
('Adidas Ultraboost 22', 'High-performance running shoes', 180.00, 'Clothing', 70),
('Levi\'s 501 Jeans', 'Classic straight-fit jeans', 89.99, 'Clothing', 120),
('North Face Jacket', 'Weather-resistant outdoor jacket', 199.99, 'Clothing', 45),
('Nike Dri-FIT T-Shirt', 'Moisture-wicking athletic shirt', 29.99, 'Clothing', 150),
('Adidas Hoodie', 'Comfortable cotton blend hoodie', 79.99, 'Clothing', 90),

-- Home & Garden
('Dyson V15 Vacuum', 'Powerful cordless vacuum cleaner', 749.99, 'Home & Garden', 20),
('KitchenAid Stand Mixer', 'Professional-grade stand mixer', 379.99, 'Home & Garden', 25),
('Instant Pot Duo 7-in-1', 'Multi-functional pressure cooker', 99.99, 'Home & Garden', 55),
('Philips Air Fryer', 'Healthy cooking with hot air circulation', 149.99, 'Home & Garden', 40),
('Roomba i7+', 'Self-emptying robot vacuum', 599.99, 'Home & Garden', 15),

-- Books
('The Psychology of Money', 'Financial wisdom and behavioral insights', 16.99, 'Books', 200),
('Atomic Habits', 'Guide to building good habits', 18.99, 'Books', 180),
('Sapiens', 'A brief history of humankind', 19.99, 'Books', 150),
('The Lean Startup', 'Entrepreneurship and innovation guide', 17.99, 'Books', 120),
('Clean Code', 'Programming best practices', 42.99, 'Books', 80),

-- Sports
('Yoga Mat Premium', 'Non-slip exercise mat', 39.99, 'Sports', 100),
('Dumbbells Set 20lb', 'Adjustable weight dumbbells', 89.99, 'Sports', 30),
('Tennis Racket Pro', 'Professional tennis racket', 199.99, 'Sports', 25),
('Basketball Official', 'Official size basketball', 29.99, 'Sports', 60),
('Resistance Bands Set', 'Complete workout band set', 24.99, 'Sports', 75);

-- Insert sample orders (spread across different dates)
INSERT INTO orders (user_id, product_id, quantity, unit_price, total_amount, status, order_date) VALUES
-- Recent orders (last 30 days)
(1, 1, 1, 999.99, 999.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '5 days'),
(2, 9, 2, 150.00, 300.00, 'delivered', CURRENT_TIMESTAMP - INTERVAL '7 days'),
(3, 15, 1, 749.99, 749.99, 'shipped', CURRENT_TIMESTAMP - INTERVAL '3 days'),
(4, 11, 3, 89.99, 269.97, 'delivered', CURRENT_TIMESTAMP - INTERVAL '10 days'),
(5, 6, 2, 249.99, 499.98, 'delivered', CURRENT_TIMESTAMP - INTERVAL '12 days'),
(6, 16, 1, 379.99, 379.99, 'confirmed', CURRENT_TIMESTAMP - INTERVAL '2 days'),
(7, 20, 5, 16.99, 84.95, 'delivered', CURRENT_TIMESTAMP - INTERVAL '8 days'),
(8, 3, 1, 2499.99, 2499.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '15 days'),
(9, 13, 2, 29.99, 59.98, 'delivered', CURRENT_TIMESTAMP - INTERVAL '6 days'),
(10, 18, 1, 149.99, 149.99, 'shipped', CURRENT_TIMESTAMP - INTERVAL '4 days'),

-- Orders from 1-3 months ago
(1, 21, 3, 18.99, 56.97, 'delivered', CURRENT_TIMESTAMP - INTERVAL '45 days'),
(2, 4, 1, 1299.99, 1299.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '52 days'),
(3, 7, 1, 399.99, 399.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '38 days'),
(4, 24, 2, 29.99, 59.98, 'delivered', CURRENT_TIMESTAMP - INTERVAL '41 days'),
(5, 12, 1, 199.99, 199.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '67 days'),
(6, 17, 1, 99.99, 99.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '73 days'),
(7, 2, 1, 899.99, 899.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '55 days'),
(8, 25, 4, 39.99, 159.96, 'delivered', CURRENT_TIMESTAMP - INTERVAL '48 days'),
(9, 14, 2, 79.99, 159.98, 'delivered', CURRENT_TIMESTAMP - INTERVAL '61 days'),
(10, 5, 1, 599.99, 599.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '44 days'),

-- Older orders (3-6 months ago)
(1, 8, 1, 299.99, 299.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '95 days'),
(2, 19, 1, 599.99, 599.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '112 days'),
(3, 22, 2, 19.99, 39.98, 'delivered', CURRENT_TIMESTAMP - INTERVAL '128 days'),
(4, 10, 1, 180.00, 180.00, 'delivered', CURRENT_TIMESTAMP - INTERVAL '134 days'),
(5, 23, 1, 17.99, 17.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '156 days'),
(6, 26, 3, 89.99, 269.97, 'delivered', CURRENT_TIMESTAMP - INTERVAL '143 days'),
(7, 27, 1, 199.99, 199.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '167 days'),
(8, 28, 2, 29.99, 59.98, 'delivered', CURRENT_TIMESTAMP - INTERVAL '121 days'),
(9, 29, 1, 24.99, 24.99, 'delivered', CURRENT_TIMESTAMP - INTERVAL '178 days'),
(10, 1, 1, 999.99, 999.99, 'cancelled', CURRENT_TIMESTAMP - INTERVAL '189 days'),

-- Some pending and confirmed orders
(1, 15, 1, 749.99, 749.99, 'pending', CURRENT_TIMESTAMP - INTERVAL '1 day'),
(3, 6, 1, 249.99, 249.99, 'confirmed', CURRENT_TIMESTAMP - INTERVAL '2 days'),
(5, 20, 2, 16.99, 33.98, 'pending', CURRENT_TIMESTAMP),
(7, 3, 1, 2499.99, 2499.99, 'confirmed', CURRENT_TIMESTAMP - INTERVAL '1 day'),
(9, 16, 1, 379.99, 379.99, 'pending', CURRENT_TIMESTAMP);

-- Insert some sample query logs (simulating bot interactions)
INSERT INTO query_logs (user_question, generated_sql, sql_result, bot_response, slack_user_id, slack_channel_id, execution_time_ms, success) VALUES
('How many orders were placed today?', 
 'SELECT COUNT(*) FROM orders WHERE DATE(order_date) = CURRENT_DATE', 
 '2', 
 'There were 2 orders placed today.', 
 'U1234567890', 
 'C1234567890', 
 150, 
 true),

('What are the top 3 selling products?', 
 'SELECT p.name, SUM(o.quantity) as total_sold FROM orders o JOIN products p ON o.product_id = p.id GROUP BY p.id, p.name ORDER BY total_sold DESC LIMIT 3', 
 'iPhone 15 Pro: 2, Nike Air Max 270: 2, Levi''s 501 Jeans: 3', 
 'The top 3 selling products are:\n1. Levi''s 501 Jeans (3 units sold)\n2. iPhone 15 Pro (2 units sold)\n3. Nike Air Max 270 (2 units sold)', 
 'U1234567890', 
 'C1234567890', 
 245, 
 true),

('Show me total revenue for Electronics category', 
 'SELECT SUM(o.total_amount) FROM orders o JOIN products p ON o.product_id = p.id WHERE p.category = ''Electronics'' AND o.status != ''cancelled''', 
 '8899.89', 
 'The total revenue for Electronics category is $8,899.89', 
 'U0987654321', 
 'C1234567890', 
 180, 
 true);

-- Update some timestamps to make data more realistic
UPDATE orders SET 
    shipped_date = order_date + INTERVAL '2 days',
    delivered_date = order_date + INTERVAL '5 days'
WHERE status = 'delivered';

UPDATE orders SET 
    shipped_date = order_date + INTERVAL '2 days'
WHERE status = 'shipped';