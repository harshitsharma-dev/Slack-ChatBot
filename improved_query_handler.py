"""
Improved Multi-Source Query Handler with Better Prompt Chaining and PDF Processing
"""

import os
import requests
import io
import PyPDF2
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

class ImprovedQueryHandler:
    def __init__(self, slack_client, db, gemini_llm):
        self.slack_client = slack_client
        self.db = db
        self.gemini_llm = gemini_llm
        self.pdf_cache = {}
        
    def process_query(self, question, channel_id):
        """Main entry point for processing multi-source queries"""
        print(f"\n🔍 Processing query: '{question}'")
        
        # Step 1: Intelligent source analysis
        source_strategy = self.analyze_query_intent(question)
        print(f"📊 Strategy: {source_strategy}")
        
        # Step 2: Execute search based on strategy
        results = self.execute_search_strategy(question, channel_id, source_strategy)
        
        # Step 3: Synthesize final answer
        final_answer = self.synthesize_answer(question, results, source_strategy)
        
        return final_answer
    
    def analyze_query_intent(self, question):
        """Analyze query to determine optimal search strategy with better prompting"""
        try:
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
            
            response = self.gemini_llm.invoke([("human", analysis_prompt)])
            strategy = response.content.strip().upper()
            
            valid_strategies = ['SLACK_ONLY', 'SQL_ONLY', 'PDF_ONLY', 'SLACK_PDF_COMBO', 'MULTI_SOURCE', 'COMPREHENSIVE']
            
            if strategy in valid_strategies:
                return strategy
            else:
                print(f"⚠️ Invalid strategy '{strategy}', defaulting to COMPREHENSIVE")
                return 'COMPREHENSIVE'
                
        except Exception as e:
            print(f"❌ Error in query analysis: {e}")
            return 'COMPREHENSIVE'
    
    def execute_search_strategy(self, question, channel_id, strategy):
        """Execute search based on determined strategy"""
        results = {}
        
        if strategy == 'SLACK_ONLY':
            results['slack'] = self.search_slack_messages(question, channel_id)
            
        elif strategy == 'SQL_ONLY':
            results['sql'] = self.search_database(question)
            
        elif strategy == 'PDF_ONLY':
            results['pdf'] = self.search_pdf_documents(question)
            
        elif strategy == 'SLACK_PDF_COMBO':
            # First check Slack for recent PDFs, then process them
            slack_result = self.search_slack_messages(question, channel_id)
            pdf_files = self.extract_pdf_files_from_slack(slack_result)
            
            if pdf_files:
                results['slack_pdfs'] = self.process_slack_pdfs(question, pdf_files)
            else:
                results['slack'] = slack_result
                
        elif strategy == 'MULTI_SOURCE':
            results['slack'] = self.search_slack_messages(question, channel_id)
            results['sql'] = self.search_database(question)
            
        else:  # COMPREHENSIVE
            results['slack'] = self.search_slack_messages(question, channel_id)
            results['sql'] = self.search_database(question)
            results['pdf'] = self.search_pdf_documents(question)
        
        return results
    
    def search_slack_messages(self, question, channel_id):
        """Enhanced Slack message search with better PDF detection"""
        try:
            print(f"🔍 Searching Slack messages in channel {channel_id}")
            
            # Fetch messages with file attachments
            response = self.slack_client.conversations_history(
                channel=channel_id,
                limit=100
            )
            
            if not response.get('ok'):
                return {'success': False, 'error': 'Failed to fetch Slack messages'}
            
            messages = []
            pdf_files = []
            
            for msg in response.get('messages', []):
                # Skip bot messages
                if msg.get('bot_id') or msg.get('user') == 'U09HPU23ABH':
                    continue
                
                username = self.get_username(msg.get('user', ''))
                text = msg.get('text', '')
                timestamp = msg.get('ts', '')
                
                # Extract PDF files
                files = msg.get('files', [])
                msg_pdfs = []
                
                for file in files:
                    if (file.get('mimetype') == 'application/pdf' or 
                        file.get('name', '').lower().endswith('.pdf')):
                        
                        pdf_info = {
                            'name': file.get('name', 'Unknown PDF'),
                            'url': file.get('url_private_download', ''),
                            'size': file.get('size', 0),
                            'username': username,
                            'timestamp': timestamp
                        }
                        msg_pdfs.append(pdf_info)
                        pdf_files.append(pdf_info)
                
                if text.strip() or msg_pdfs:
                    messages.append({
                        'text': text,
                        'username': username,
                        'timestamp': timestamp,
                        'pdf_files': msg_pdfs
                    })
            
            # Use AI to find relevant information
            relevant_info = self.find_relevant_slack_content(question, messages)
            
            return {
                'success': True,
                'content': relevant_info,
                'messages': messages[:10],  # Keep sample for debugging
                'pdf_files': pdf_files
            }
            
        except Exception as e:
            print(f"❌ Slack search error: {e}")
            return {'success': False, 'error': str(e)}
    
    def find_relevant_slack_content(self, question, messages):
        """Use AI to extract relevant information from Slack messages"""
        try:
            if not messages:
                return "No recent messages found."
            
            # Check if question is about PDFs
            is_pdf_question = any(word in question.lower() 
                                for word in ['pdf', 'document', 'file', 'summarize', 'summary'])
            
            # Prepare context from messages
            context_parts = []
            recent_pdfs = []
            
            for msg in messages[:30]:  # Limit context size
                username = msg.get('username', 'Unknown')
                text = msg.get('text', '')
                
                # Collect PDF information
                if msg.get('pdf_files'):
                    for pdf in msg['pdf_files']:
                        recent_pdfs.append(pdf)
                        text += f" [Shared PDF: {pdf['name']}]"
                
                if text.strip():
                    context_parts.append(f"[{username}]: {text}")
            
            # If asking about PDFs and we found them, process them
            if is_pdf_question and recent_pdfs:
                return self.process_recent_pdfs(question, recent_pdfs[:3])
            
            # Otherwise, search through message content
            context = "\\n".join(context_parts)
            
            search_prompt = f"""
            SLACK CONVERSATION ANALYSIS
            
            QUESTION: {question}
            
            RECENT MESSAGES:
            {context}
            
            TASK: Find information relevant to the question from the Slack messages.
            
            INSTRUCTIONS:
            1. Look for direct mentions, discussions, or context related to the question
            2. Pay attention to names, topics, decisions, and shared files
            3. If you find relevant information, provide a clear, helpful answer
            4. If no relevant information exists, respond with "No relevant information found"
            5. Include specific details and context when available
            
            RESPONSE:
            """
            
            response = self.gemini_llm.invoke([("human", search_prompt)])
            answer = response.content.strip()
            
            if "no relevant information found" in answer.lower():
                return None
            
            return answer
            
        except Exception as e:
            print(f"❌ Error analyzing Slack content: {e}")
            return None
    
    def process_recent_pdfs(self, question, pdf_files):
        """Process recent PDF files from Slack with improved error handling"""
        try:
            print(f"📄 Processing {len(pdf_files)} PDF files")
            
            pdf_summaries = []
            
            for pdf in pdf_files:
                print(f"📄 Processing: {pdf['name']}")
                
                # Download and extract content
                content = self.download_and_extract_pdf(pdf['url'])
                
                if content:
                    # Summarize based on question
                    summary = self.summarize_pdf_content(content, pdf['name'], question)
                    if summary:
                        pdf_summaries.append({
                            'name': pdf['name'],
                            'username': pdf['username'],
                            'summary': summary
                        })
                else:
                    print(f"⚠️ Failed to extract content from {pdf['name']}")
            
            if pdf_summaries:
                # Format response
                response_parts = []
                for pdf_summary in pdf_summaries:
                    response_parts.append(
                        f"**{pdf_summary['name']}** (shared by {pdf_summary['username']}):\\n"
                        f"{pdf_summary['summary']}"
                    )
                
                return "\\n\\n".join(response_parts)
            else:
                return "Found PDF files but couldn't process their content."
                
        except Exception as e:
            print(f"❌ Error processing PDFs: {e}")
            return "Error processing PDF files."
    
    def download_and_extract_pdf(self, url):
        """Download PDF from Slack and extract text content"""
        try:
            # Check cache first
            if url in self.pdf_cache:
                print(f"📋 Using cached PDF content")
                return self.pdf_cache[url]
            
            # Download PDF
            headers = {
                'Authorization': f'Bearer {os.getenv("SLACK_BOT_TOKEN")}',
                'User-Agent': 'SlackBot/1.0'
            }
            
            print(f"⬇️ Downloading PDF from Slack...")
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ Download failed: HTTP {response.status_code}")
                return None
            
            print(f"✅ Downloaded {len(response.content)} bytes")
            
            # Extract text using PyPDF2
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"\\n--- Page {page_num + 1} ---\\n{page_text}\\n"
                except Exception as e:
                    print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                    continue
            
            if text_content.strip():
                # Cache the content
                self.pdf_cache[url] = text_content
                print(f"✅ Extracted {len(text_content)} characters")
                return text_content
            else:
                print(f"⚠️ No text content extracted")
                return None
                
        except Exception as e:
            print(f"❌ PDF download/extraction error: {e}")
            return None
    
    def summarize_pdf_content(self, content, filename, question):
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
            
            response = self.gemini_llm.invoke([("human", summary_prompt)])
            return response.content.strip()
            
        except Exception as e:
            print(f"❌ Error summarizing PDF: {e}")
            return None
    
    def search_database(self, question):
        """Search SQL database with improved query generation"""
        try:
            print(f"🗄️ Searching database for: {question}")
            
            # Generate SQL query
            sql_query = self.generate_sql_query(question)
            
            if not sql_query:
                return {'success': False, 'error': 'Could not generate SQL query'}
            
            # Execute query
            result = self.execute_sql_query(sql_query)
            
            if result is not None:
                # Format response
                formatted_response = self.format_sql_response(result, question)
                return {
                    'success': True,
                    'content': formatted_response,
                    'sql': sql_query,
                    'raw_data': result
                }
            else:
                return {'success': False, 'error': 'Query execution failed'}
                
        except Exception as e:
            print(f"❌ Database search error: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_sql_query(self, question):
        """Generate SQL query with improved prompting"""
        try:
            schema_prompt = f"""
            DATABASE SCHEMA:
            
            Table: users
            - id (INTEGER, PRIMARY KEY)
            - name (VARCHAR(100))
            - email (VARCHAR(100), UNIQUE)
            - created_at (TIMESTAMP)
            
            Table: products  
            - id (INTEGER, PRIMARY KEY)
            - name (VARCHAR(200))
            - price (DECIMAL(10,2))
            - category (VARCHAR(50))
            - stock_quantity (INTEGER)
            - created_at (TIMESTAMP)
            
            Table: orders
            - id (INTEGER, PRIMARY KEY)
            - user_id (INTEGER, FOREIGN KEY → users.id)
            - product_id (INTEGER, FOREIGN KEY → products.id)
            - quantity (INTEGER)
            - total_amount (DECIMAL(10,2))
            - status (VARCHAR(20))
            - order_date (TIMESTAMP)
            
            QUESTION: {question}
            
            TASK: Generate a SQL SELECT query to answer the question.
            
            REQUIREMENTS:
            - Use only SELECT statements
            - Include appropriate JOINs when needed
            - Add LIMIT clause if not specified (default: 100)
            - Use proper aggregation functions (COUNT, SUM, AVG) when appropriate
            - Format dates properly
            
            RESPOND WITH ONLY THE SQL QUERY:
            """
            
            response = self.gemini_llm.invoke([("human", schema_prompt)])
            sql = response.content.strip()
            
            # Clean the SQL
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            return sql
            
        except Exception as e:
            print(f"❌ SQL generation error: {e}")
            return None
    
    def execute_sql_query(self, sql):
        """Execute SQL query safely"""
        try:
            from sqlalchemy import text
            
            # Validate query
            if not self.validate_sql_query(sql):
                print(f"❌ SQL validation failed")
                return None
            
            # Execute query
            result = self.db.session.execute(text(sql))
            
            if sql.strip().upper().startswith('SELECT'):
                rows = result.fetchall()
                # Convert to list of lists
                return [[item for item in row] for row in rows]
            
            return result.rowcount
            
        except Exception as e:
            print(f"❌ SQL execution error: {e}")
            return None
    
    def validate_sql_query(self, sql):
        """Validate SQL query for safety"""
        if not sql:
            return False
        
        sql_upper = sql.upper().strip()
        
        # Only allow SELECT
        if not sql_upper.startswith('SELECT'):
            return False
        
        # Block dangerous keywords
        dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous:
            if keyword in sql_upper:
                return False
        
        return True
    
    def format_sql_response(self, result, question):
        """Format SQL results into human-readable response"""
        if not result:
            return "No data found for your query."
        
        if isinstance(result, int):
            return f"Result: {result}"
        
        if len(result) == 1 and len(result[0]) == 1:
            return f"The answer is: {result[0][0]}"
        
        # Format multiple rows
        response = "Here's what I found:\\n"
        for i, row in enumerate(result[:10]):  # Limit to 10 rows
            response += f"{i+1}. {' | '.join(str(col) for col in row)}\\n"
        
        if len(result) > 10:
            response += f"... and {len(result) - 10} more results"
        
        return response
    
    def search_pdf_documents(self, question):
        """Search stored PDF documents"""
        try:
            print(f"📄 Searching PDF documents")
            
            # This would query the PDFDocument table
            # For now, return placeholder
            return {
                'success': False,
                'content': 'PDF document search not implemented yet'
            }
            
        except Exception as e:
            print(f"❌ PDF search error: {e}")
            return {'success': False, 'error': str(e)}
    
    def synthesize_answer(self, question, results, strategy):
        """Synthesize final answer from multiple sources"""
        try:
            print(f"🔄 Synthesizing answer from {len(results)} sources")
            
            # Filter successful results
            successful_results = {k: v for k, v in results.items() 
                                if v.get('success', False) and v.get('content')}
            
            if not successful_results:
                return "I couldn't find relevant information to answer your question."
            
            # Single source
            if len(successful_results) == 1:
                source, result = list(successful_results.items())[0]
                return f"{result['content']}"
            
            # Multiple sources - synthesize
            context_parts = []
            for source, result in successful_results.items():
                context_parts.append(f"From {source.upper()}: {result['content']}")
            
            synthesis_prompt = f"""
            MULTI-SOURCE SYNTHESIS TASK
            
            ORIGINAL QUESTION: "{question}"
            
            INFORMATION FROM MULTIPLE SOURCES:
            {chr(10).join(context_parts)}
            
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
            
            response = self.gemini_llm.invoke([("human", synthesis_prompt)])
            return response.content.strip()
            
        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            # Fallback: return first successful result
            for result in results.values():
                if result.get('success') and result.get('content'):
                    return result['content']
            
            return "I encountered an error processing your question."
    
    def get_username(self, user_id):
        """Get username from user ID"""
        try:
            response = self.slack_client.users_info(user=user_id)
            if response.get('ok'):
                return response['user']['name']
        except:
            pass
        return user_id
    
    def extract_pdf_files_from_slack(self, slack_result):
        """Extract PDF files from Slack search result"""
        if slack_result.get('success') and slack_result.get('pdf_files'):
            return slack_result['pdf_files']
        return []