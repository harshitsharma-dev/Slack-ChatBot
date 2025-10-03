import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, send_file

# Load environment variables
load_dotenv()
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from datetime import datetime
import requests
import io
import PyPDF2

# Event deduplication to prevent duplicate processing
processed_events = set()

# PDF content cache
pdf_content_cache = {}

# Slack message cache
slack_message_cache = []

# Enhanced source routing keywords
SQL_KEYWORDS = ['users', 'products', 'orders', 'database', 'table', 'count', 'sum', 'total', 'revenue', 'sales', 'how many', 'show me', 'list all']
SLACK_KEYWORDS = ['said', 'mentioned', 'discussed', 'requirements', 'meeting', 'conversation', 'chat', 'message', 'who is', 'what did', 'told', 'talked about', 'summarize this', 'this pdf', 'this document']
PDF_KEYWORDS = ['document', 'pdf', 'specification', 'manual', 'guide', 'documentation', 'procedure', 'policy', 'summarize', 'summary']

import logging

app = Flask(__name__, static_folder='build', static_url_path='')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://localhost/slack_ai_bot')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

db = SQLAlchemy(app)
CORS(app, origins="*")

# Initialize clients
slack_client = WebClient(token=os.getenv('SLACK_BOT_TOKEN'))
signature_verifier = SignatureVerifier(os.getenv('SLACK_SIGNING_SECRET'))

# Initialize Gemini LLM (lazy loading)
gemini_llm = None

def get_gemini_llm():
    global gemini_llm
    print(f"DEBUG: get_gemini_llm called, current gemini_llm: {gemini_llm}")
    
    if gemini_llm is None:
        try:
            print("DEBUG: Initializing new Gemini LLM...")
            # Use environment variable directly in the call
            api_key = os.getenv('GOOGLE_API_KEY')
            print(f"DEBUG: Retrieved API key from env: {bool(api_key)}")
            
            if not api_key:
                print("ERROR: Google API key not found in environment")
                return None
            
            print("DEBUG: Creating ChatGoogleGenerativeAI instance...")
            
            # Use proper LangChain ChatGoogleGenerativeAI with latest model
            gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )
            print(f"DEBUG: Successfully created LLM: {type(gemini_llm)}")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize Gemini: {str(e)}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            import traceback
            print(f"ERROR: Full traceback: {traceback.format_exc()}")
            return None
    else:
        print("DEBUG: Using existing gemini_llm instance")
        
    return gemini_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class QueryLog(db.Model):
    __tablename__ = 'query_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_question = db.Column(db.Text, nullable=False)
    generated_sql = db.Column(db.Text)
    bot_response = db.Column(db.Text)
    slack_user_id = db.Column(db.String(50))
    slack_channel_id = db.Column(db.String(50))
    success = db.Column(db.Boolean, default=True)
    source_type = db.Column(db.String(20), default='sql')
    source_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Product(db.Model):
    __tablename__ = 'products'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    stock_quantity = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Order(db.Model):
    __tablename__ = 'orders'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    total_amount = db.Column(db.Numeric(10, 2), nullable=False)
    status = db.Column(db.String(20), default='pending')
    order_date = db.Column(db.DateTime, default=datetime.utcnow)

class PDFDocument(db.Model):
    __tablename__ = 'pdf_documents'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    url = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text)
    content_hash = db.Column(db.String(64))
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SlackMessage(db.Model):
    __tablename__ = 'slack_messages'
    id = db.Column(db.Integer, primary_key=True)
    message_ts = db.Column(db.String(50), nullable=False, unique=True)
    user_id = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(100))
    channel_id = db.Column(db.String(50), nullable=False)
    channel_name = db.Column(db.String(100))
    text = db.Column(db.Text, nullable=False)
    thread_ts = db.Column(db.String(50))
    message_type = db.Column(db.String(20), default='message')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

def clean_sql(sql_text):
    """Clean SQL by removing markdown formatting and extra whitespace"""
    if not sql_text:
        return None
    
    # Remove markdown code blocks
    sql_text = sql_text.replace('```sql', '').replace('```', '')
    
    # Remove extra whitespace and newlines
    sql_text = sql_text.strip()
    
    # Remove any leading/trailing quotes
    sql_text = sql_text.strip('"\'')
    
    return sql_text

def generate_sql_query(question):
    """Generate SQL query from natural language using Gemini"""
    print(f"\n=== DEBUG: Starting SQL generation for question: '{question}' ===")
    
    schema_info = """
    Database Schema:
    - users: id, name, email, created_at
    - products: id, name, price, category, stock_quantity, created_at  
    - orders: id, user_id, product_id, quantity, total_amount, status, order_date
    """
    
    prompt = f"""
    {schema_info}
    
    Convert this question to a SQL query: {question}
    
    Return only the SQL query, no explanation.
    """
    
    try:
        # Check if API key is configured
        api_key = os.getenv('GOOGLE_API_KEY')
        print(f"DEBUG: API key present: {bool(api_key)}")
        print(f"DEBUG: API key length: {len(api_key) if api_key else 0}")
        
        if not api_key:
            print("ERROR: Google API key not found in environment")
            return None
            
        print(f"DEBUG: Attempting to get Gemini LLM...")
        llm = get_gemini_llm()
        print(f"DEBUG: LLM object: {llm}")
        print(f"DEBUG: LLM type: {type(llm)}")
        
        if not llm:
            print("ERROR: Failed to get Gemini LLM - llm is None")
            return None
        
        full_prompt = f"You are a SQL expert. Generate only valid SQL queries based on the given schema.\n\n{prompt}"
        print(f"DEBUG: Full prompt length: {len(full_prompt)}")
        print(f"DEBUG: Creating HumanMessage...")
        
        # Use LangChain message format
        print(f"DEBUG: Creating messages for LangChain...")
        
        messages = [
            ("system", "You are a SQL expert. Generate only valid SQL queries based on the given schema."),
            ("human", prompt)
        ]
        
        print(f"DEBUG: Invoking LLM with messages...")
        response = llm.invoke(messages)
        print(f"DEBUG: Response received: {type(response)}")
        print(f"DEBUG: Response object: {response}")
        
        # Extract content from response
        if response and hasattr(response, 'content'):
            raw_sql = response.content.strip()
            print(f"DEBUG: Raw SQL from Gemini: '{raw_sql}'")
            
            # Clean SQL by removing markdown formatting
            sql = clean_sql(raw_sql)
            print(f"DEBUG: Cleaned SQL: '{sql}'")
            return sql
        else:
            print(f"DEBUG: No content in response")
            return None
            
    except Exception as e:
        print(f"ERROR: Exception in generate_sql_query: {str(e)}")
        print(f"ERROR: Exception type: {type(e).__name__}")
        import traceback
        print(f"ERROR: Full traceback: {traceback.format_exc()}")
        return None

def validate_sql_query(sql):
    """Validate SQL query for safety"""
    if not sql:
        return False, "Empty query"
    
    sql_upper = sql.upper().strip()
    
    # Block dangerous operations
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False, f"Dangerous operation '{keyword}' not allowed"
    
    # Only allow SELECT statements
    if not sql_upper.startswith('SELECT'):
        return False, "Only SELECT queries are allowed"
    
    # Block system tables
    system_tables = ['pg_', 'information_schema', 'sys']
    for table in system_tables:
        if table in sql.lower():
            return False, f"Access to system tables not allowed"
    
    return True, "Valid"

def execute_query(sql):
    """Execute SQL query safely with validation"""
    try:
        from sqlalchemy import text
        import time
        
        # Validate query first
        is_valid, validation_msg = validate_sql_query(sql)
        if not is_valid:
            logger.error(f"Query validation failed: {validation_msg}")
            return None
        
        # Add LIMIT if not present
        sql_upper = sql.upper().strip()
        if sql_upper.startswith('SELECT') and 'LIMIT' not in sql_upper:
            # Remove semicolon if present, add LIMIT, then add semicolon back
            sql = sql.rstrip(';').strip() + ' LIMIT 100'
        
        # Measure execution time
        start_time = time.time()
        result = db.session.execute(text(sql))
        execution_time = time.time() - start_time
        
        print(f"Query executed in {execution_time:.3f}s")
        
        if execution_time > 30:
            logger.warning(f"Query took too long: {execution_time:.3f}s")
        
        if sql.strip().upper().startswith('SELECT'):
            rows = result.fetchall()
            print(f"Returned {len(rows)} rows")
            # Convert Row objects to lists and handle datetime objects
            converted_rows = []
            for row in rows:
                converted_row = []
                for item in row:
                    if hasattr(item, 'isoformat'):  # datetime object
                        converted_row.append(item.isoformat())
                    else:
                        converted_row.append(item)
                converted_rows.append(converted_row)
            return converted_rows
        return result.rowcount
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return None

def determine_query_source(question):
    """Legacy function - kept for backward compatibility"""
    # This is now handled by analyze_question_for_sources
    return 'comprehensive'

def process_multi_source_query(question, channel_id):
    """Process query from multiple sources with intelligent chaining"""
    try:
        print(f"\n🔍 Processing query: '{question}'")
        
        # Step 1: Intelligent source analysis
        source_strategy = analyze_question_for_sources(question)
        print(f"📊 Strategy: {source_strategy}")
        
        # Step 2: Execute search based on strategy
        if source_strategy == 'SLACK_ONLY':
            return search_slack_messages(question, channel_id)
        elif source_strategy == 'SQL_ONLY':
            return search_sql_database(question)
        elif source_strategy == 'PDF_ONLY':
            return search_pdf_documents(question)
        elif source_strategy == 'SLACK_PDF_COMBO':
            return search_slack_with_pdfs(question, channel_id)
        elif source_strategy == 'MULTI_SOURCE':
            return execute_multi_source_search(question, channel_id, ['slack', 'sql'])
        else:  # COMPREHENSIVE
            return execute_comprehensive_search(question, channel_id)
            
    except Exception as e:
        print(f"Error in multi-source query: {e}")
        return 'error', "I encountered an error processing your question.", None

def analyze_question_for_sources(question):
    """Analyze question to determine optimal search strategy using AI"""
    try:
        llm = get_gemini_llm()
        if not llm:
            return 'SQL_ONLY'
        
        analysis_prompt = f"""
        Analyze this user question and determine the best search approach:
        
        QUESTION: "{question}"
        
        AVAILABLE DATA SOURCES:
        1. SLACK_MESSAGES - Recent conversations, discussions, shared files, team communications
        2. SQL_DATABASE - Structured data: users, products, orders, sales analytics
        3. PDF_DOCUMENTS - Documentation, specifications, manuals, reports
        
        ANALYSIS CRITERIA:
        - If asking about people, conversations, "who said", "what was discussed" → SLACK_ONLY
        - If asking about data analysis, counts, sales, revenue, "how many" → SQL_ONLY  
        - If asking about documents, PDFs, specifications, "summarize PDF" → PDF_ONLY
        - If asking about recent files/PDFs shared in Slack → SLACK_PDF_COMBO
        - If question needs multiple sources → MULTI_SOURCE
        - If unclear or general → COMPREHENSIVE
        
        RESPOND WITH ONLY ONE OF:
        SLACK_ONLY | SQL_ONLY | PDF_ONLY | SLACK_PDF_COMBO | MULTI_SOURCE | COMPREHENSIVE
        """
        
        response = llm.invoke([("human", analysis_prompt)])
        strategy = response.content.strip().upper()
        
        valid_strategies = ['SLACK_ONLY', 'SQL_ONLY', 'PDF_ONLY', 'SLACK_PDF_COMBO', 'MULTI_SOURCE', 'COMPREHENSIVE']
        
        if strategy in valid_strategies:
            return strategy
        else:
            print(f"⚠️ Invalid strategy '{strategy}', defaulting to COMPREHENSIVE")
            return 'COMPREHENSIVE'
            
    except Exception as e:
        print(f"Error in question analysis: {e}")
        return 'COMPREHENSIVE'

def get_sources_for_strategy(strategy):
    """Get source list for strategy"""
    if strategy == 'slack_only':
        return ['slack']
    elif strategy == 'sql_only':
        return ['sql']
    elif strategy == 'pdf_only':
        return ['pdf']
    elif strategy == 'multi_source':
        return ['slack', 'sql']
    else:
        return ['slack', 'sql', 'pdf']

def execute_multi_source_search(question, channel_id, sources):
    """Execute search across specified sources and combine results"""
    results = {}
    
    # Search each source
    for source in sources:
        if source == 'slack':
            source_type, response, data = search_slack_messages(question, channel_id)
            if response and "couldn't find" not in response.lower():
                results['slack'] = {'response': response, 'data': data}
        elif source == 'sql':
            source_type, response, data = search_sql_database(question)
            if response and "no data found" not in response.lower():
                results['sql'] = {'response': response, 'data': data}
        elif source == 'pdf':
            source_type, response, data = search_pdf_documents(question)
            if response and "couldn't find" not in response.lower():
                results['pdf'] = {'response': response, 'data': data}
    
    # Combine results intelligently
    return combine_multi_source_results(question, results)

def execute_comprehensive_search(question, channel_id):
    """Search all sources and provide best answer"""
    return execute_multi_source_search(question, channel_id, ['slack', 'sql', 'pdf'])

def search_slack_with_pdfs(question, channel_id):
    """Search Slack messages with focus on PDF processing"""
    try:
        # Get Slack messages with PDFs
        slack_result = search_slack_messages(question, channel_id)
        
        # The result should already include PDF processing from find_relevant_slack_info
        return slack_result
        
    except Exception as e:
        print(f"Error in Slack PDF search: {e}")
        return 'slack', "Error searching Slack messages with PDFs.", None

def extract_pdf_files_from_slack_data(slack_data):
    """Extract PDF files from Slack search data"""
    try:
        # This would extract PDF info from the slack data
        # For now, return empty list
        return []
    except:
        return []

def combine_multi_source_results(question, results):
    """Combine results from multiple sources into coherent response with improved synthesis"""
    if not results:
        return 'multi', "I couldn't find relevant information from any source.", None
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() 
                         if isinstance(v, tuple) and len(v) >= 2 and v[1] and "couldn't find" not in v[1].lower()}
    
    if not successful_results:
        return 'multi', "I couldn't find relevant information from any source.", None
    
    if len(successful_results) == 1:
        # Single source result
        source = list(successful_results.keys())[0]
        result = successful_results[source]
        return result[0], result[1], result[2] if len(result) > 2 else None
    
    # Multiple sources - intelligent synthesis
    try:
        llm = get_gemini_llm()
        if llm:
            # Prepare context from all sources
            context_parts = []
            for source, result in successful_results.items():
                if len(result) >= 2:
                    context_parts.append(f"From {source.upper()}: {result[1]}")
            
            combined_context = "\n\n".join(context_parts)
            
            synthesis_prompt = f"""
            MULTI-SOURCE SYNTHESIS TASK
            
            ORIGINAL QUESTION: "{question}"
            
            INFORMATION FROM MULTIPLE SOURCES:
            {combined_context}
            
            TASK: Provide a comprehensive answer that synthesizes information from all sources.
            
            REQUIREMENTS:
            1. Create a coherent, unified response
            2. If sources complement each other, combine the information
            3. If sources contradict, mention both perspectives
            4. Prioritize the most relevant and reliable information
            5. Keep the response concise and helpful
            6. Don't mention "sources" explicitly - just provide the answer
            
            SYNTHESIZED ANSWER:
            """
            
            response = llm.invoke([("human", synthesis_prompt)])
            synthesized_answer = response.content.strip()
            
            return 'multi', synthesized_answer, str(successful_results)
        else:
            # Fallback: concatenate results
            combined_response = "\n\n".join([f"**{source.title()}**: {result[1]}" for source, result in successful_results.items() if len(result) >= 2])
            return 'multi', combined_response, str(successful_results)
            
    except Exception as e:
        print(f"Error combining results: {e}")
        # Fallback: return first successful result
        first_source = list(successful_results.keys())[0]
        first_result = successful_results[first_source]
        return first_result[0], first_result[1], first_result[2] if len(first_result) > 2 else None

def search_slack_messages(question, channel_id):
    """Search Slack messages for relevant information with enhanced search"""
    try:
        print(f"Searching Slack messages for: {question}")
        
        # Fetch recent messages from channel
        messages = fetch_slack_messages(channel_id, limit=100)  # Increased limit
        
        if not messages:
            print("No messages found in channel")
            return 'slack', "No recent messages found in this channel.", None
        
        print(f"Found {len(messages)} messages to search")
        
        # Use AI to find relevant messages
        relevant_info = find_relevant_slack_info(question, messages)
        
        if relevant_info:
            print(f"Found relevant info: {relevant_info[:100]}...")
            return 'slack', relevant_info, str(messages[:10])
        else:
            print("No relevant information found in messages")
            return 'slack', "I couldn't find relevant information in recent Slack messages.", None
            
    except Exception as e:
        print(f"Slack search error: {e}")
        return 'slack', "Error searching Slack messages.", None

def search_pdf_documents(question):
    """Search PDF documents for relevant information"""
    try:
        # Get PDF documents from database
        pdfs = PDFDocument.query.all()
        
        if not pdfs:
            return 'pdf', "No PDF documents available.", None
        
        # Search through PDF content
        for pdf in pdfs:
            content = get_pdf_content(pdf.url)
            if content:
                relevant_info = find_relevant_pdf_info(question, content, pdf.title)
                if relevant_info:
                    return 'pdf', relevant_info, pdf.url
        
        return 'pdf', "I couldn't find relevant information in the PDF documents.", None
        
    except Exception as e:
        print(f"PDF search error: {e}")
        return 'pdf', "Error searching PDF documents.", None

def search_sql_database(question):
    """Search SQL database using existing functionality"""
    try:
        sql = generate_sql_query(question)
        if not sql:
            return 'sql', "I couldn't understand your database question.", None
        
        result = execute_query(sql)
        if result:
            response = format_sql_response(result, question)
            return 'sql', response, sql
        else:
            return 'sql', "No data found for your query.", sql
            
    except Exception as e:
        print(f"SQL search error: {e}")
        return 'sql', "Error querying database.", None

def format_sql_response(result, question):
    """Format SQL query result into human-readable response"""
    if not result:
        return "I couldn't find any data for your question."
    
    if isinstance(result, int):
        return f"Result: {result}"
    
    if len(result) == 1 and len(result[0]) == 1:
        return f"The answer is: {result[0][0]}"
    
    response = "Here's what I found:\n"
    for row in result[:5]:  # Limit to 5 rows
        response += f"• {' | '.join(str(col) for col in row)}\n"
    
    if len(result) > 5:
        response += f"... and {len(result) - 5} more results"
    
    return response

def fetch_slack_messages(channel_id, limit=100):
    """Fetch recent messages from Slack channel with file attachments"""
    try:
        print(f"Fetching messages from channel: {channel_id}")
        
        response = slack_client.conversations_history(
            channel=channel_id,
            limit=limit
        )
        
        print(f"Slack API response: {response.get('ok', False)}")
        
        messages = []
        if response.get('ok') and 'messages' in response:
            for msg in response['messages']:
                # Include all message types except bot messages from our bot
                if (msg.get('type') == 'message' and 
                    msg.get('user') != 'U09HPU23ABH' and  # Skip our bot's messages
                    not (msg.get('bot_id') == 'B09J592CGN6')):  # Skip our bot's messages
                    
                    username = get_username(msg.get('user', ''))
                    text = msg.get('text', '')
                    files = msg.get('files', [])
                    
                    # Process file attachments
                    pdf_files = []
                    for file in files:
                        if file.get('mimetype') == 'application/pdf' or file.get('name', '').lower().endswith('.pdf'):
                            pdf_files.append({
                                'name': file.get('name', 'Unknown PDF'),
                                'url': file.get('url_private_download', ''),
                                'size': file.get('size', 0)
                            })
                    
                    # Include message if it has text or PDF files
                    if text.strip() or pdf_files:
                        message_data = {
                            'text': text,
                            'user': msg.get('user', ''),
                            'ts': msg.get('ts', ''),
                            'username': username,
                            'pdf_files': pdf_files
                        }
                        messages.append(message_data)
        
        print(f"Processed {len(messages)} messages")
        return messages
        
    except Exception as e:
        print(f"Error fetching Slack messages: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return []

def get_username(user_id):
    """Get username from user ID"""
    try:
        response = slack_client.users_info(user=user_id)
        return response['user']['name']
    except:
        return user_id

def find_relevant_slack_info(question, messages):
    """Use AI to find relevant information in Slack messages including PDF files"""
    try:
        if not messages:
            return None
        
        # Check if question is asking about PDFs specifically
        is_pdf_question = any(word in question.lower() for word in ['pdf', 'document', 'file', 'summarize', 'summary'])
        
        # Look for recent PDF files in messages
        recent_pdfs = []
        for msg in messages[:10]:  # Check recent messages for PDFs
            if msg.get('pdf_files'):
                for pdf in msg['pdf_files']:
                    recent_pdfs.append({
                        'name': pdf['name'],
                        'url': pdf['url'],
                        'username': msg['username'],
                        'timestamp': msg['ts']
                    })
        
        # If asking about PDF and we found PDFs, process them
        if is_pdf_question and recent_pdfs:
            return process_recent_pdfs(question, recent_pdfs)
        
        # Prepare enhanced context from messages
        context_parts = []
        for msg in messages[:30]:
            username = msg.get('username', 'Unknown')
            text = msg.get('text', '')
            
            # Include PDF file information in context
            if msg.get('pdf_files'):
                pdf_info = ", ".join([f"PDF: {pdf['name']}" for pdf in msg['pdf_files']])
                text += f" [Shared files: {pdf_info}]"
            
            if text.strip():
                context_parts.append(f"[{username}]: {text}")
        
        context = "\n".join(context_parts)
        
        enhanced_prompt = f"""
        You are analyzing Slack conversation history to answer a question.
        
        SLACK MESSAGES:
        {context}
        
        QUESTION: {question}
        
        INSTRUCTIONS:
        1. Look for any mentions, discussions, or information related to the question
        2. Pay attention to names, topics, context clues, and shared files
        3. If you find relevant information, provide a clear, helpful answer
        4. If no relevant information exists, respond with "No relevant information found"
        5. Include specific details and context from the messages when possible
        6. If PDFs or documents are mentioned, include that information
        
        ANSWER:
        """
        
        llm = get_gemini_llm()
        if llm:
            response = llm.invoke([("human", enhanced_prompt)])
            answer = response.content.strip()
            
            if "no relevant information found" in answer.lower() or "information not found" in answer.lower():
                return None
            return answer
        
        return None
        
    except Exception as e:
        print(f"Error finding relevant Slack info: {e}")
        return None

def process_recent_pdfs(question, recent_pdfs):
    """Process recent PDF files from Slack messages"""
    try:
        print(f"Processing {len(recent_pdfs)} recent PDFs")
        
        pdf_summaries = []
        for pdf in recent_pdfs[:3]:  # Process up to 3 most recent PDFs
            print(f"Processing PDF: {pdf['name']}")
            
            # Download and extract PDF content
            content = download_slack_pdf(pdf['url'])
            if content:
                # Summarize PDF content
                summary = summarize_pdf_content(content, pdf['name'], question)
                if summary:
                    pdf_summaries.append(f"**{pdf['name']}** (shared by {pdf['username']}): {summary}")
        
        if pdf_summaries:
            return "\n\n".join(pdf_summaries)
        else:
            return "I found PDF files in recent messages but couldn't process them."
            
    except Exception as e:
        print(f"Error processing recent PDFs: {e}")
        return "Error processing PDF files."

def download_slack_pdf(url):
    """Download PDF content from Slack private URL with improved error handling"""
    try:
        # Check cache first
        if url in pdf_content_cache:
            print(f"📋 Using cached PDF content")
            return pdf_content_cache[url]
        
        headers = {
            'Authorization': f'Bearer {os.getenv("SLACK_BOT_TOKEN")}',
            'User-Agent': 'SlackBot/1.0'
        }
        
        print(f"⬇️ Downloading PDF from: {url[:50]}...")
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            print(f"✅ Downloaded {len(response.content)} bytes")
            
            # Extract text using PyPDF2
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                    continue
            
            if text_content.strip():
                # Cache the content
                pdf_content_cache[url] = text_content
                print(f"✅ Extracted {len(text_content)} characters")
                return text_content
            else:
                print(f"⚠️ No text content extracted")
                return None
        else:
            print(f"❌ Download failed: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ PDF download/extraction error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def summarize_pdf_content(content, filename, question):
    """Summarize PDF content based on the specific question"""
    try:
        # Limit content to avoid token limits (keep first 6000 chars)
        content_snippet = content[:6000] if len(content) > 6000 else content
        
        summary_prompt = f"""
        DOCUMENT ANALYSIS TASK
        
        DOCUMENT: "{filename}"
        QUESTION: "{question}"
        
        DOCUMENT CONTENT:
        {content_snippet}
        
        INSTRUCTIONS:
        1. If the question asks for a summary, provide a comprehensive overview of the document
        2. If the question is specific, answer it using information from the document
        3. Focus on the most relevant information related to the question
        4. Keep the response concise but informative (2-3 paragraphs max)
        5. If the document doesn't contain relevant information, say so clearly
        
        RESPONSE:
        """
        
        llm = get_gemini_llm()
        if llm:
            response = llm.invoke([("human", summary_prompt)])
            return response.content.strip()
        
        return None
        
    except Exception as e:
        print(f"❌ Error summarizing PDF: {e}")
        return None

def get_pdf_content(url):
    """Extract text content from PDF URL"""
    try:
        # Check cache first
        if url in pdf_content_cache:
            return pdf_content_cache[url]
        
        # Download PDF
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        
        # Extract text using PyPDF2
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Cache the content
        pdf_content_cache[url] = text
        return text
        
    except Exception as e:
        print(f"Error extracting PDF content: {e}")
        return None

def find_relevant_pdf_info(question, content, title):
    """Use AI to find relevant information in PDF content"""
    try:
        # Limit content to avoid token limits
        content_snippet = content[:3000] if len(content) > 3000 else content
        
        prompt = f"""
        Based on this document "{title}":
        {content_snippet}
        
        Answer this question: {question}
        
        If the information is not available in the document, say "Information not found in document".
        """
        
        llm = get_gemini_llm()
        if llm:
            response = llm.invoke([("human", prompt)])
            answer = response.content.strip()
            
            if "information not found" in answer.lower():
                return None
            return f"From {title}: {answer}"
        
        return None
        
    except Exception as e:
        print(f"Error finding relevant PDF info: {e}")
        return None

@app.route('/slack/events', methods=['POST'])
def slack_events():
    """Handle Slack events"""
    print("\n" + "="*60)
    print("SLACK EVENTS ENDPOINT HIT!")
    print("="*60)
    
    try:
        print(f"Request method: {request.method}")
        print(f"Request headers: {dict(request.headers)}")
        print(f"Request data: {request.get_data()}")
        
        # Check signature
        request_data = request.get_data()
        print(f"Signature verification...")
        
        if not signature_verifier.is_valid_request(request_data, request.headers):
            print("ERROR: Invalid Slack signature!")
            return "Invalid signature", 403
        
        print("✓ Signature verified successfully")
        
        data = request.json
        print(f"Parsed JSON data: {data}")
        print(f"Event type: {data.get('type')}")
        
        # Handle URL verification
        if data.get('type') == 'url_verification':
            challenge = data.get('challenge')
            print(f"URL verification challenge: {challenge}")
            return challenge
        
        # Handle app mentions
        if data.get('type') == 'event_callback':
            event = data.get('event', {})
            print(f"Event callback received: {event}")
            print(f"Event type: {event.get('type')}")
            
            # Skip bot messages to prevent infinite loops
            if event.get('bot_id') or event.get('user') == 'U09HPU23ABH':
                print("Skipping bot message to prevent infinite loop")
                return "OK", 200
            
            # Event deduplication - use timestamp + user + text as unique key
            event_key = f"{event.get('ts')}_{event.get('user')}_{event.get('text', '')[:50]}"
            if event_key in processed_events:
                print(f"Skipping duplicate event: {event_key}")
                return "OK", 200
            
            # Only process app_mention events (ignore regular message events)
            if event.get('type') == 'app_mention':
                print("Processing app mention...")
                processed_events.add(event_key)
                # Clean old events (keep only last 100)
                if len(processed_events) > 100:
                    processed_events.clear()
                handle_mention(event)
            else:
                print(f"Ignoring event type: {event.get('type')}")
        else:
            print(f"Unhandled request type: {data.get('type')}")
        
        print("Returning OK response")
        return "OK", 200
        
    except Exception as e:
        print(f"ERROR in slack_events: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return "Error", 500

def handle_mention(event):
    """Process bot mention and respond"""
    print("\n" + "="*50)
    print("HANDLE MENTION CALLED!")
    print("="*50)
    
    try:
        print(f"Event data: {event}")
        
        # Skip if this is a bot message
        if event.get('bot_id') or event.get('subtype') == 'bot_message':
            print("Skipping bot message in handle_mention")
            return
        
        user_id = event.get('user')
        channel_id = event.get('channel')
        raw_text = event.get('text', '')
        
        print(f"User ID: {user_id}")
        print(f"Channel ID: {channel_id}")
        print(f"Raw text: '{raw_text}'")
        
        # Skip if user is the bot itself
        if user_id == 'U09HPU23ABH':
            print("Skipping message from bot user")
            return
        
        # Use hardcoded bot user ID to avoid API calls
        bot_user_id = 'U09HPU23ABH'
        print(f"Bot user ID: {bot_user_id}")
        
        # Clean the text by removing bot mention
        text = raw_text.replace(f'<@{bot_user_id}>', '').strip()
        print(f"Cleaned text: '{text}'")
        
        if not text or len(text) < 3:
            print("No meaningful text after cleaning, returning")
            return
        
        print(f"Processing multi-source query: '{text}'")
        # Determine source and process query
        source_type, response, source_data = process_multi_source_query(text, channel_id)
        print(f"Source: {source_type}, Response: {response[:100]}...")
        
        print(f"Final response: '{response}'")
        
        # Log the interaction
        try:
            log = QueryLog(
                user_question=text,
                bot_response=response,
                slack_user_id=user_id,
                slack_channel_id=channel_id,
                success=response != "I couldn't find relevant information.",
                source_type=source_type,
                source_data=source_data
            )
            db.session.add(log)
            db.session.commit()
            print("✓ Logged to database")
        except Exception as e:
            print(f"Database logging error: {e}")
        
        # Send response to Slack
        try:
            print(f"Sending message to Slack channel {channel_id}")
            slack_response = slack_client.chat_postMessage(
                channel=channel_id,
                text=response
            )
            print(f"Slack response: {slack_response}")
            print("✓ Message sent to Slack successfully")
        except Exception as e:
            print(f"Error sending to Slack: {e}")
        
    except Exception as e:
        print(f"ERROR in handle_mention: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

@app.route('/api/queries')
def get_queries():
    """Get recent queries"""
    queries = QueryLog.query.order_by(QueryLog.created_at.desc()).limit(50).all()
    return jsonify([{
        'id': q.id,
        'question': q.user_question,
        'response': q.bot_response,
        'success': q.success,
        'created_at': q.created_at.isoformat()
    } for q in queries])

@app.route('/api/query', methods=['POST'])
def test_query():
    """Test multi-source query endpoint"""
    try:
        data = request.json
        question = data.get('question') if data else None
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Process multi-source query
        source_type, response, source_data = process_multi_source_query(question, 'test_channel')
        
        return jsonify({
            "question": question,
            "source_type": source_type,
            "response": response,
            "success": response != "I couldn't find relevant information.",
            "source_data": source_data,
            "raw_data": [] if source_type != 'sql' else source_data
        })
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/pdf-documents', methods=['GET', 'POST'])
def manage_pdf_documents():
    """Manage PDF documents"""
    if request.method == 'GET':
        pdfs = PDFDocument.query.all()
        return jsonify([{
            'id': pdf.id,
            'title': pdf.title,
            'url': pdf.url,
            'description': pdf.description,
            'created_at': pdf.created_at.isoformat()
        } for pdf in pdfs])
    
    elif request.method == 'POST':
        data = request.json
        pdf = PDFDocument(
            title=data.get('title'),
            url=data.get('url'),
            description=data.get('description', '')
        )
        db.session.add(pdf)
        db.session.commit()
        return jsonify({'success': True, 'id': pdf.id})

@app.route('/api/slack-messages/<channel_id>')
def get_slack_messages(channel_id):
    """Get recent Slack messages for a channel"""
    try:
        messages = fetch_slack_messages(channel_id)
        return jsonify({
            'success': True,
            'messages': messages[:10],
            'count': len(messages)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get bot statistics"""
    total_queries = QueryLog.query.count()
    successful_queries = QueryLog.query.filter_by(success=True).count()
    
    return jsonify({
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "success_rate": (successful_queries / total_queries * 100) if total_queries > 0 else 0
    })

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve React app for all non-API routes"""
    # Skip API and Slack routes
    if path.startswith('api/') or path.startswith('slack/'):
        return None
    
    # Check if file exists in build directory
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # Serve index.html for all other routes (React Router)
        return send_from_directory(app.static_folder, 'index.html')

# Serve static files explicitly
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files from build/static"""
    return send_from_directory(os.path.join(app.static_folder, 'static'), filename)

@app.route('/api/test-gemini')
def test_gemini():
    """Test Gemini API connectivity"""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return jsonify({"error": "Google API key not configured"}), 400
            
        llm = get_gemini_llm()
        if not llm:
            return jsonify({"error": "Failed to initialize Gemini LLM"}), 500
        
        messages = [("human", "Say hello")]
        response = llm.invoke(messages)
        response_text = response.content if response and hasattr(response, 'content') else "No response"
        
        return jsonify({
            "success": True,
            "response": response_text if response_text else "No response",
            "api_key_present": bool(api_key)
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/api/export-csv', methods=['POST'])
def export_csv():
    """Export query result as CSV"""
    try:
        data = request.json
        result = data.get('raw_data', [])
        question = data.get('question', '')
        
        if not result:
            return jsonify({'error': 'No data to export'}), 400
        
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        if result and len(result) > 0:
            # Infer column headers based on common query patterns
            num_cols = len(result[0])
            headers = []
            
            # Smart header inference based on question and data
            question_lower = question.lower()
            if 'user' in question_lower and 'count' in question_lower:
                headers = ['user_name', 'count'] if num_cols == 2 else [f'column_{i+1}' for i in range(num_cols)]
            elif 'product' in question_lower and 'price' in question_lower:
                headers = ['product_name', 'price'] if num_cols == 2 else [f'column_{i+1}' for i in range(num_cols)]
            elif 'order' in question_lower and 'total' in question_lower:
                headers = ['order_info', 'total_amount'] if num_cols == 2 else [f'column_{i+1}' for i in range(num_cols)]
            elif 'category' in question_lower:
                headers = ['category', 'value'] if num_cols == 2 else [f'column_{i+1}' for i in range(num_cols)]
            elif 'date' in question_lower or 'time' in question_lower:
                headers = ['date', 'count'] if num_cols == 2 else [f'column_{i+1}' for i in range(num_cols)]
            else:
                # Generic headers
                if num_cols == 1:
                    headers = ['result']
                elif num_cols == 2:
                    headers = ['name', 'value']
                else:
                    headers = [f'column_{i+1}' for i in range(num_cols)]
            
            # Write headers
            writer.writerow(headers)
            
            # Write data rows
            for row in result:
                # Handle different data types properly
                clean_row = []
                for item in row:
                    if item is None:
                        clean_row.append('')
                    elif isinstance(item, (int, float)):
                        clean_row.append(str(item))
                    else:
                        clean_row.append(str(item))
                writer.writerow(clean_row)
        
        csv_content = output.getvalue()
        output.close()
        
        return jsonify({
            'success': True,
            'csv_content': csv_content,
            'filename': f'query_result_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-chart', methods=['POST'])
def generate_chart():
    """Generate intelligent charts based on data patterns"""
    try:
        data = request.json
        result = data.get('raw_data', [])
        question = data.get('question', '')
        
        print(f"Chart generation request: {len(result) if result else 0} rows")
        
        if not result or len(result) == 0:
            return jsonify({'error': 'No data for chart'}), 400
        
        # Generate multiple charts
        charts = generate_multiple_charts(result, question)
        
        return jsonify({
            'success': True,
            'charts': charts
        })
        
    except Exception as e:
        print(f"Chart generation error: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

def generate_multiple_charts(result, question):
    """Generate multiple chart visualizations for the data"""
    charts = []
    
    if not result or len(result) == 0:
        return [{'type': 'summary', 'html': create_data_summary([], question), 'title': 'Data Summary'}]
    
    # Single value - KPI card
    if len(result) == 1 and len(result[0]) == 1:
        value = result[0][0]
        return [{'type': 'kpi', 'html': create_kpi_card(value, question), 'title': 'Key Metric'}]
    
    # Analyze data structure
    try:
        data_analysis = analyze_data_structure(result, question)
        
        # Generate different chart types based on data
        if data_analysis['cols'] >= 2 and len(data_analysis['labels']) > 0:
            labels = data_analysis['labels']
            values = data_analysis['values']
            
            # Ensure we have valid data
            if len(labels) > 0 and len(values) > 0:
                # Bar Chart
                charts.append({
                    'type': 'bar',
                    'html': create_smart_bar_chart(data_analysis, question),
                    'title': 'Bar Chart Analysis'
                })
                
                # Pie Chart (if suitable)
                if len(labels) <= 8 and all(v > 0 for v in values if isinstance(v, (int, float))):
                    charts.append({
                        'type': 'pie',
                        'html': create_pie_chart(data_analysis, question),
                        'title': 'Distribution Breakdown'
                    })
                
                # Ranking Chart
                charts.append({
                    'type': 'ranking',
                    'html': create_ranking_chart(labels, values, question),
                    'title': 'Ranking Analysis'
                })
                
                # Revenue Chart (if currency data)
                if data_analysis.get('has_currency') or any(word in question.lower() for word in ['revenue', 'sales', 'price', 'total', 'amount']):
                    charts.append({
                        'type': 'revenue',
                        'html': create_revenue_chart(labels, values, question),
                        'title': 'Financial Analysis'
                    })
        
        # Add summary chart
        charts.append({
            'type': 'summary',
            'html': create_data_summary(result, question),
            'title': 'Data Overview'
        })
        
    except Exception as e:
        print(f"Error in chart generation: {e}")
        charts = [{'type': 'error', 'html': f'<div style="padding: 20px; color: red;">Chart generation failed: {str(e)}</div>', 'title': 'Error'}]
    
    return charts if charts else [{'type': 'summary', 'html': create_data_summary(result, question), 'title': 'Data Summary'}]

def analyze_data_structure(result, question):
    """Analyze data structure, types, and patterns"""
    analysis = {
        'rows': len(result),
        'cols': len(result[0]) if result else 0,
        'data_types': [],
        'labels': [],
        'values': [],
        'has_dates': False,
        'has_currency': False,
        'has_percentages': False,
        'value_ranges': [],
        'is_categorical': False,
        'is_numerical': False,
        'is_temporal': False
    }
    
    if not result:
        return analysis
    
    try:
        # Extract labels and values for 2-column data
        if analysis['cols'] == 2:
            analysis['labels'] = [str(row[0])[:25] for row in result[:20]]
            analysis['values'] = extract_numeric_values([row[1] for row in result[:20]])
            
            # Check data types
            first_col = [row[0] for row in result]
            second_col = [row[1] for row in result]
            
            analysis['is_categorical'] = is_categorical_data(first_col)
            analysis['is_temporal'] = is_temporal_data(first_col)
            analysis['has_dates'] = analysis['is_temporal']
            analysis['has_currency'] = is_currency_data(second_col)
            analysis['has_percentages'] = is_percentage_data(second_col)
            analysis['is_numerical'] = is_numeric_data(second_col)
        
        # Single column data
        elif analysis['cols'] == 1:
            analysis['labels'] = [f"Row {i+1}" for i in range(len(result[:20]))]
            analysis['values'] = extract_numeric_values([row[0] for row in result[:20]])
        
        # Multi-column analysis
        elif analysis['cols'] > 2:
            analysis['labels'] = [str(row[0])[:25] for row in result[:15]]
            analysis['values'] = extract_numeric_values([row[1] for row in result[:15]])
    
    except Exception as e:
        print(f"Error in data analysis: {e}")
        # Fallback to basic analysis
        analysis['labels'] = [f"Item {i+1}" for i in range(min(len(result), 10))]
        analysis['values'] = [1] * min(len(result), 10)
    
    return analysis

def analyze_column(col_data):
    """Analyze individual column characteristics"""
    return {
        'type': detect_data_type(col_data),
        'unique_count': len(set(str(x) for x in col_data)),
        'null_count': sum(1 for x in col_data if x is None),
        'sample_values': col_data[:5]
    }

def detect_data_type(col_data):
    """Detect the primary data type of a column"""
    if not col_data:
        return 'empty'
    
    # Check for dates
    if is_temporal_data(col_data):
        return 'datetime'
    
    # Check for numbers
    if is_numeric_data(col_data):
        if is_currency_data(col_data):
            return 'currency'
        elif is_percentage_data(col_data):
            return 'percentage'
        else:
            return 'numeric'
    
    # Check for categories
    unique_ratio = len(set(str(x) for x in col_data)) / len(col_data)
    if unique_ratio < 0.5:  # Less than 50% unique values
        return 'categorical'
    
    return 'text'

def is_temporal_data(data):
    """Check if data contains dates/times"""
    import re
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO datetime
    ]
    
    date_count = 0
    for item in data[:10]:  # Check first 10 items
        item_str = str(item)
        if any(re.search(pattern, item_str) for pattern in date_patterns):
            date_count += 1
    
    return date_count > len(data[:10]) * 0.7  # 70% threshold

def is_numeric_data(data):
    """Check if data is primarily numeric"""
    numeric_count = 0
    for item in data:
        try:
            float(str(item).replace('$', '').replace('%', '').replace(',', ''))
            numeric_count += 1
        except:
            pass
    return numeric_count > len(data) * 0.8  # 80% threshold

def is_currency_data(data):
    """Check if data represents currency"""
    currency_indicators = ['$', 'USD', 'EUR', '€', '£']
    currency_count = 0
    
    for item in data[:10]:
        item_str = str(item)
        if any(indicator in item_str for indicator in currency_indicators) or \
           (is_numeric_data([item]) and float(str(item).replace('$', '').replace(',', '')) > 10):
            currency_count += 1
    
    return currency_count > len(data[:10]) * 0.5

def is_percentage_data(data):
    """Check if data represents percentages"""
    return any('%' in str(item) for item in data[:10])

def is_categorical_data(data):
    """Check if data is categorical"""
    unique_count = len(set(str(x) for x in data))
    return unique_count <= min(len(data) * 0.5, 20)  # Max 20 categories or 50% unique

def extract_numeric_values(data):
    """Extract numeric values from mixed data"""
    values = []
    for item in data:
        try:
            if item is None:
                values.append(0)
                continue
            # Clean and convert to float
            clean_val = str(item).replace('$', '').replace('%', '').replace(',', '').strip()
            if clean_val == '' or clean_val.lower() in ['none', 'null', 'n/a']:
                values.append(0)
            else:
                values.append(float(clean_val))
        except (ValueError, TypeError):
            values.append(0)
    return values

def determine_optimal_chart_type(analysis, question):
    """Determine the best chart type based on data analysis"""
    q_lower = question.lower()
    
    # Single metric
    if analysis['rows'] == 1:
        return 'kpi'
    
    # Time series data
    if analysis['has_dates'] or analysis['is_temporal']:
        return 'time_series'
    
    # Large number of categories - use pie chart
    if analysis['is_categorical'] and analysis['rows'] <= 8:
        return 'pie'
    
    # Revenue/financial data
    if analysis['has_currency'] or any(word in q_lower for word in ['revenue', 'sales', 'price', 'cost', 'profit']):
        return 'revenue'
    
    # Ranking/comparison data
    if any(word in q_lower for word in ['top', 'best', 'highest', 'most', 'least', 'bottom']):
        return 'ranking'
    
    # Distribution data
    if any(word in q_lower for word in ['distribution', 'spread', 'range']):
        return 'histogram'
    
    # Multi-series data (more than 2 columns)
    if analysis['cols'] > 2:
        return 'multi_series'
    
    # Correlation data (two numeric columns)
    if analysis['cols'] == 2 and analysis['is_numerical']:
        return 'scatter'
    
    # Category analysis
    if analysis['is_categorical']:
        return 'category'
    
    # Default to smart bar chart
    return 'bar'

def create_kpi_card(value, question):
    """Create KPI card for single values"""
    try:
        # Try to format as currency if it's a large number
        num_val = float(str(value).replace('$', '').replace(',', ''))
        if num_val > 100:
            formatted_value = f"${num_val:,.2f}"
        else:
            formatted_value = str(value)
    except:
        formatted_value = str(value)
    
    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; text-align: center; max-width: 400px; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
        <h3 style="margin: 0 0 10px 0; font-size: 16px; opacity: 0.9;">{question[:60]}{'...' if len(question) > 60 else ''}</h3>
        <div style="font-size: 48px; font-weight: bold; margin: 20px 0;">{formatted_value}</div>
        <div style="font-size: 14px; opacity: 0.8;">📊 Query Result</div>
    </div>
    """

def create_revenue_chart(labels, values, question):
    """Create revenue/financial chart"""
    max_val = max(values) if values else 1
    chart_bars = ""
    
    for i, (label, value) in enumerate(zip(labels, values)):
        percentage = (value / max_val) * 100 if max_val > 0 else 0
        formatted_value = f"${value:,.2f}" if value > 100 else f"{value}"
        color = f"hsl({120 - (i * 15) % 120}, 70%, 50%)"
        
        chart_bars += f"""
        <div style="margin: 10px 0; padding: 8px; background: #f8f9fa; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <span style="font-weight: 500; color: #333;">{label}</span>
                <span style="font-weight: bold; color: {color};">{formatted_value}</span>
            </div>
            <div style="background: #e9ecef; height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: {color}; height: 100%; width: {percentage}%; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 600px;">
        <h3 style="margin: 0 0 20px 0; color: #333; display: flex; align-items: center;">
            <span style="margin-right: 10px;">💰</span>{question[:50]}...
        </h3>
        {chart_bars}
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef; font-size: 12px; color: #666; text-align: center;">
            Revenue Analysis • Total: ${sum(values):,.2f}
        </div>
    </div>
    """

def create_count_chart(labels, values, question):
    """Create count/quantity chart"""
    max_val = max(values) if values else 1
    chart_bars = ""
    
    for i, (label, value) in enumerate(zip(labels, values)):
        percentage = (value / max_val) * 100 if max_val > 0 else 0
        color = f"hsl({200 + (i * 20) % 160}, 70%, 50%)"
        
        chart_bars += f"""
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <div style="width: 120px; font-size: 14px; color: #333; margin-right: 10px;">{label}</div>
            <div style="flex: 1; background: #f1f3f4; height: 24px; border-radius: 12px; overflow: hidden; position: relative;">
                <div style="background: {color}; height: 100%; width: {percentage}%; transition: width 0.3s ease;"></div>
                <span style="position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: 12px; font-weight: bold; color: #333;">{int(value)}</span>
            </div>
        </div>
        """
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 600px;">
        <h3 style="margin: 0 0 20px 0; color: #333; display: flex; align-items: center;">
            <span style="margin-right: 10px;">📊</span>{question[:50]}...
        </h3>
        {chart_bars}
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef; font-size: 12px; color: #666; text-align: center;">
            Count Analysis • Total: {sum(values):,.0f} items
        </div>
    </div>
    """

def create_category_chart(labels, values, question):
    """Create category distribution chart"""
    total = sum(values) if values else 1
    chart_segments = ""
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    for i, (label, value) in enumerate(zip(labels, values)):
        percentage = (value / total) * 100 if total > 0 else 0
        color = colors[i % len(colors)]
        
        chart_segments += f"""
        <div style="display: flex; align-items: center; margin: 10px 0; padding: 8px; background: #f8f9fa; border-radius: 8px;">
            <div style="width: 16px; height: 16px; background: {color}; border-radius: 50%; margin-right: 12px;"></div>
            <div style="flex: 1;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 500; color: #333;">{label}</span>
                    <span style="font-size: 12px; color: #666;">{percentage:.1f}%</span>
                </div>
                <div style="font-size: 14px; color: {color}; font-weight: bold;">{value}</div>
            </div>
        </div>
        """
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 600px;">
        <h3 style="margin: 0 0 20px 0; color: #333; display: flex; align-items: center;">
            <span style="margin-right: 10px;">🎯</span>{question[:50]}...
        </h3>
        {chart_segments}
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef; font-size: 12px; color: #666; text-align: center;">
            Category Distribution • {len(labels)} categories
        </div>
    </div>
    """

def create_ranking_chart(labels, values, question):
    """Create ranking/leaderboard chart"""
    # Sort by values descending
    sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    chart_items = ""
    
    medals = ['🥇', '🥈', '🥉']
    
    for i, (label, value) in enumerate(sorted_data):
        rank = i + 1
        medal = medals[i] if i < 3 else f"#{rank}"
        formatted_value = f"${value:,.2f}" if value > 100 else f"{value}"
        
        bg_color = "#fff3cd" if i < 3 else "#f8f9fa"
        
        chart_items += f"""
        <div style="display: flex; align-items: center; margin: 8px 0; padding: 12px; background: {bg_color}; border-radius: 8px; border-left: 4px solid {'#ffc107' if i < 3 else '#dee2e6'};">
            <div style="font-size: 20px; margin-right: 15px; min-width: 40px;">{medal}</div>
            <div style="flex: 1;">
                <div style="font-weight: 600; color: #333; margin-bottom: 2px;">{label}</div>
                <div style="font-size: 14px; color: #666;">Value: {formatted_value}</div>
            </div>
        </div>
        """
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 600px;">
        <h3 style="margin: 0 0 20px 0; color: #333; display: flex; align-items: center;">
            <span style="margin-right: 10px;">🏆</span>{question[:50]}...
        </h3>
        {chart_items}
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef; font-size: 12px; color: #666; text-align: center;">
            Ranking Analysis • Top {len(sorted_data)} results
        </div>
    </div>
    """

def create_smart_bar_chart(analysis, question):
    """Create intelligent bar chart based on data analysis"""
    labels = analysis.get('labels', [])
    values = analysis.get('values', [])
    
    # Ensure we have data
    if not labels or not values:
        return create_data_summary([], question)
    
    if analysis.get('has_currency'):
        return create_revenue_chart(labels, values, question)
    elif any(word in question.lower() for word in ['count', 'number', 'quantity']):
        return create_count_chart(labels, values, question)
    else:
        return create_category_chart(labels, values, question)

def create_time_series_chart(analysis, question):
    """Create time series line chart"""
    labels = analysis['labels']
    values = analysis['values']
    
    chart_points = ""
    for i, (label, value) in enumerate(zip(labels, values)):
        color = f"hsl({200 + i * 10}, 70%, 50%)"
        chart_points += f"""
        <div style="display: flex; align-items: center; margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 6px;">
            <div style="width: 100px; font-size: 12px; color: #666;">{label}</div>
            <div style="flex: 1; margin: 0 10px; position: relative; height: 20px; background: #e9ecef; border-radius: 10px;">
                <div style="position: absolute; left: 0; top: 0; height: 100%; background: {color}; border-radius: 10px; width: {min(value/max(values)*100, 100) if values else 0}%;"></div>
            </div>
            <div style="width: 80px; text-align: right; font-weight: bold; color: {color};">{value}</div>
        </div>
        """
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 700px;">
        <h3 style="margin: 0 0 20px 0; color: #333; display: flex; align-items: center;">
            <span style="margin-right: 10px;">📈</span>{question[:50]}...
        </h3>
        {chart_points}
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef; font-size: 12px; color: #666; text-align: center;">
            Time Series Analysis • {len(labels)} data points
        </div>
    </div>
    """

def create_pie_chart(analysis, question):
    """Create pie chart for categorical data"""
    labels = analysis['labels']
    values = analysis['values']
    total = sum(values) if values else 1
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    # Create pie segments
    segments = ""
    cumulative = 0
    
    for i, (label, value) in enumerate(zip(labels, values)):
        percentage = (value / total) * 100 if total > 0 else 0
        color = colors[i % len(colors)]
        
        segments += f"""
        <div style="display: flex; align-items: center; margin: 8px 0; padding: 10px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {color};">
            <div style="width: 20px; height: 20px; background: {color}; border-radius: 50%; margin-right: 12px;"></div>
            <div style="flex: 1;">
                <div style="font-weight: 600; color: #333; margin-bottom: 2px;">{label}</div>
                <div style="font-size: 12px; color: #666;">{value} ({percentage:.1f}%)</div>
            </div>
            <div style="font-size: 18px; font-weight: bold; color: {color};">{percentage:.0f}%</div>
        </div>
        """
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 600px;">
        <h3 style="margin: 0 0 20px 0; color: #333; display: flex; align-items: center;">
            <span style="margin-right: 10px;">🥧</span>{question[:50]}...
        </h3>
        {segments}
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef; font-size: 12px; color: #666; text-align: center;">
            Distribution Analysis • Total: {sum(values):,.0f}
        </div>
    </div>
    """

def create_scatter_plot(analysis, question):
    """Create scatter plot for correlation data"""
    labels = analysis['labels']
    values = analysis['values']
    
    scatter_points = """
    <div style="position: relative; width: 400px; height: 300px; background: #f8f9fa; border-radius: 8px; margin: 20px 0;">
    """
    
    for i, (label, value) in enumerate(zip(labels[:15], values[:15])):
        x_pos = (i / len(labels)) * 350 + 25
        y_pos = 250 - (value / max(values)) * 200 if values else 150
        
        scatter_points += f"""
        <div style="position: absolute; left: {x_pos}px; top: {y_pos}px; width: 8px; height: 8px; background: #4ECDC4; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);" title="{label}: {value}"></div>
        """
    
    scatter_points += "</div>"
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 500px;">
        <h3 style="margin: 0 0 20px 0; color: #333; display: flex; align-items: center;">
            <span style="margin-right: 10px;">📊</span>{question[:50]}...
        </h3>
        {scatter_points}
        <div style="margin-top: 15px; font-size: 12px; color: #666; text-align: center;">
            Scatter Plot Analysis • {len(labels)} data points
        </div>
    </div>
    """

def create_histogram(analysis, question):
    """Create histogram for distribution analysis"""
    values = analysis['values']
    
    # Create bins
    if not values:
        return create_data_summary([], question)
    
    min_val, max_val = min(values), max(values)
    bin_count = min(10, len(set(values)))
    bin_size = (max_val - min_val) / bin_count if bin_count > 0 else 1
    
    bins = {}
    for value in values:
        bin_idx = int((value - min_val) / bin_size) if bin_size > 0 else 0
        bin_idx = min(bin_idx, bin_count - 1)
        bin_range = f"{min_val + bin_idx * bin_size:.1f}-{min_val + (bin_idx + 1) * bin_size:.1f}"
        bins[bin_range] = bins.get(bin_range, 0) + 1
    
    max_count = max(bins.values()) if bins else 1
    
    histogram_bars = ""
    for bin_range, count in bins.items():
        bar_height = (count / max_count) * 100
        
        histogram_bars += f"""
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <div style="width: 100px; font-size: 12px; color: #666; margin-right: 10px;">{bin_range}</div>
            <div style="flex: 1; background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #4ECDC4, #45B7D1); height: 100%; width: {bar_height}%; transition: width 0.3s ease;"></div>
            </div>
            <div style="width: 40px; text-align: right; font-size: 12px; font-weight: bold; color: #333; margin-left: 10px;">{count}</div>
        </div>
        """
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 600px;">
        <h3 style="margin: 0 0 20px 0; color: #333; display: flex; align-items: center;">
            <span style="margin-right: 10px;">📊</span>{question[:50]}...
        </h3>
        {histogram_bars}
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef; font-size: 12px; color: #666; text-align: center;">
            Distribution Histogram • Range: {min_val:.1f} - {max_val:.1f}
        </div>
    </div>
    """

def create_multi_series_chart(analysis, question):
    """Create multi-series chart for complex data"""
    labels = analysis['labels']
    
    # Handle multiple value series
    if isinstance(analysis['values'][0], list):
        series_data = analysis['values']
    else:
        series_data = [analysis['values']]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    chart_rows = ""
    for i, label in enumerate(labels[:10]):
        chart_rows += f'<div style="display: flex; align-items: center; margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 6px;">'
        chart_rows += f'<div style="width: 120px; font-size: 14px; color: #333; margin-right: 10px;">{label}</div>'
        
        for series_idx, series in enumerate(series_data[:4]):
            if i < len(series):
                value = series[i]
                color = colors[series_idx % len(colors)]
                chart_rows += f'<div style="margin: 0 5px; padding: 4px 8px; background: {color}; color: white; border-radius: 4px; font-size: 12px; font-weight: bold;">{value}</div>'
        
        chart_rows += '</div>'
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 700px;">
        <h3 style="margin: 0 0 20px 0; color: #333; display: flex; align-items: center;">
            <span style="margin-right: 10px;">📈</span>{question[:50]}...
        </h3>
        {chart_rows}
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef; font-size: 12px; color: #666; text-align: center;">
            Multi-Series Analysis • {len(series_data)} data series
        </div>
    </div>
    """

def create_heatmap(analysis, question):
    """Create heatmap for matrix data"""
    return create_multi_series_chart(analysis, question)  # Fallback to multi-series

def create_data_summary(result, question):
    """Create summary for complex data"""
    row_count = len(result) if result else 0
    col_count = len(result[0]) if result and len(result) > 0 else 0
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 600px;">
        <h3 style="margin: 0 0 15px 0; color: #333; display: flex; align-items: center;">
            <span style="margin-right: 10px;">📋</span>{question[:50]}{'...' if len(question) > 50 else ''}
        </h3>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 24px; font-weight: bold; color: #333; margin-bottom: 5px;">{row_count}</div>
            <div style="color: #666;">rows × {col_count} columns</div>
        </div>
        <div style="margin-top: 15px; font-size: 12px; color: #666; text-align: center;">
            Data Summary • View table above for details
        </div>
    </div>
    """

def create_data_heatmap(result, question):
    """Data heatmap visualization"""
    colors = ['#FF6B6B', '#FFEAA7', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    heatmap_cells = ""
    for i, row in enumerate(result[:8]):
        heatmap_cells += '<div style="display: flex; margin: 2px 0;">'
        for j, cell in enumerate(row[:6]):
            try:
                intensity = float(str(cell).replace('$', '').replace(',', '')) % 5
                color = colors[int(intensity)]
            except:
                color = '#e9ecef'
            
            heatmap_cells += f'<div style="width: 40px; height: 30px; background: {color}; margin: 1px; border-radius: 3px; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold;" title="{cell}">{str(cell)[:3]}</div>'
        heatmap_cells += '</div>'
    
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 600px;">
        <h4 style="margin: 0 0 20px 0; color: #333; text-align: center;">🔥 {question[:40]}...</h4>
        <div style="display: inline-block; padding: 10px; background: #f8f9fa; border-radius: 8px;">
            {heatmap_cells}
        </div>
    </div>
    """

def create_radar_chart(result, question):
    """Radar/spider chart"""
    return f"""
    <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; max-width: 500px;">
        <h4 style="margin: 0 0 20px 0; color: #333; text-align: center;">🕸️ {question[:40]}...</h4>
        <div style="text-align: center; padding: 20px;">
            <div style="width: 200px; height: 200px; border: 2px solid #e9ecef; border-radius: 50%; margin: 0 auto; position: relative; background: radial-gradient(circle, #f8f9fa 0%, #e9ecef 100%);">
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 14px; font-weight: bold; color: #333;">Multi-Axis</div>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 12px; color: #666; text-align: center;">Radar Analysis • {len(result)} dimensions</div>
    </div>
    """

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING FLASK APPLICATION")
    print("="*60)
    
    try:
        with app.app_context():
            db.create_all()
        print("✅ Database connected successfully")
    except Exception as e:
        print(f"⚠️ Database connection failed: {str(e)[:100]}...")
        print("Starting server without database (API endpoints will work)")
    
    print(f"\n🚀 Server starting on http://localhost:5000")
    print(f"📡 Slack webhook URL: https://mallie-buirdly-fussily.ngrok-free.dev/slack/events")
    print(f"🧪 Test endpoint: http://localhost:5000/api/query")
    print(f"❤️ Health check: http://localhost:5000/api/health")
    print("\n" + "="*60)
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)