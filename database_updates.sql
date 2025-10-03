-- Database updates for multi-source search functionality

-- Table to store PDF document links and metadata
CREATE TABLE pdf_documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    description TEXT,
    content_hash VARCHAR(64),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to cache Slack messages for search
CREATE TABLE slack_messages (
    id SERIAL PRIMARY KEY,
    message_ts VARCHAR(50) NOT NULL UNIQUE,
    user_id VARCHAR(50) NOT NULL,
    username VARCHAR(100),
    channel_id VARCHAR(50) NOT NULL,
    channel_name VARCHAR(100),
    text TEXT NOT NULL,
    thread_ts VARCHAR(50),
    message_type VARCHAR(20) DEFAULT 'message',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for slack_messages table
CREATE INDEX idx_slack_messages_text ON slack_messages USING gin(to_tsvector('english', text));
CREATE INDEX idx_slack_messages_user ON slack_messages (user_id);
CREATE INDEX idx_slack_messages_channel ON slack_messages (channel_id);

-- Update query_logs table to track source type
ALTER TABLE query_logs ADD COLUMN source_type VARCHAR(20) DEFAULT 'sql';
ALTER TABLE query_logs ADD COLUMN source_data TEXT;

-- Insert sample PDF document
INSERT INTO pdf_documents (title, url, description) VALUES 
('Project Requirements', 'https://example.com/project-requirements.pdf', 'Main project requirements document');