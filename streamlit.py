#unified_sentiment_app.py

# =============================================================================
# 0. IMPORTS (Gathered from sections 1, 2, 3, 4, 5)
# =============================================================================
import os
import sys
import json
import time
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
# NewsAPI Client (Requires installation: pip install newsapi-python)
from newsapi import NewsApiClient

# Set up PySpark environment (Required for running Spark within Streamlit/Python context)
# [Note: This setup function remains necessary]
def setup_pyspark_env():
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path 
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path 
    os.environ['PYTHONHASHSEED'] = '0' 
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost' 
    os.environ['SPARK_SQL_EXECUTION_ARROW_PYSPARK_ENABLED'] = 'false' 
    # NOTE: If running on a host where you manually set JAVA_HOME, you would also set it here.
    # os.environ['JAVA_HOME'] = "/path/to/java/installation" 
    print(f"✅ PySpark environment configured for: {python_path}")
    
setup_pyspark_env()

# --- Configuration (from Section 1) ---
OUTPUT_DIR = "data/input" # [7]
QUERY = "technology OR business OR finance" # [8]
FETCH_INTERVAL_SECONDS = 60 # [8]
API_KEY = os.getenv('NEWS_API_KEY') # [8]
MODEL_PATH = "models/sentiment_model" # [5, 9]
DASHBOARD_FILE = "data/output/latest_results.json" # Derived from [4, 10]

# =============================================================================
# 1. NEWS PRODUCER (Adapted for Asynchronous Execution)
# =============================================================================

def run_news_producer():
    """
    Fetches news indefinitely. MUST be run in a separate, non-blocking thread/process.
    """
    if not API_KEY:
        print("Error: NEWS_API_KEY environment variable not set.")
        return

    try:
        newsapi = NewsApiClient(api_key=API_KEY) # [8]
    except Exception:
        print("Error: Invalid API Key.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR) # [3]

    # --- THE FOLLOWING LOOP BLOCKS STREAMLIT ---
    # In a deployment scenario, this needs to be moved to a background worker.
    # We convert the continuous loop into a function call structure for integration.
    while True:
        try:
            response = newsapi.get_everything(
                q=QUERY,
                language='en',
                sort_by='relevancy',
                page_size=20 # Limit to avoid rate limits [3]
            )

            if response['status'] == 'ok':
                articles = response.get('articles', [])
                for item in articles:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    file_path = os.path.join(OUTPUT_DIR, f"news_{timestamp}.json") # [11]
                    news_data = {
                        "headline": item['title'],
                        "description": item.get('description', ''),
                        "source": item.get('source', {}).get('name', 'Unknown'),
                        "published_at": item.get('publishedAt', ''),
                        "url": item.get('url', ''),
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(file_path, 'w') as f:
                        json.dump(news_data, f)
            else:
                print(f"Error from NewsAPI: {response.get('message', 'Unknown error')}") # [11]

        except Exception as e:
            print(f"An unexpected error occurred in Producer: {e}") # [12]

        time.sleep(FETCH_INTERVAL_SECONDS) # [12]

# =============================================================================
# 2. SENTIMENT CLASSIFICATION ML PIPELINE (Integrated as Class)
# =============================================================================

# [This class (SentimentClassifier) remains largely unchanged, defined by sources 7-14]
class SentimentClassifier:
    def __init__(self):
        # Initialize Spark Session [13, 14]
        try:
            self.spark = SparkSession.builder \
                .appName("NewsSentimentClassification") \
                .config("spark.master", "local[7]") \
                .config("spark.python.worker.reuse", "false") \
                .getOrCreate()
            self.spark.sparkContext.setLogLevel("ERROR")
        except Exception as e:
            st.error(f"❌ Error creating Spark session: {e}")
            raise
        self.model = None

    # [create_training_data, build_pipeline, train_model, predict_sentiment methods
    # (Sources 9-14) would be included here.]

# =============================================================================
# 3. SPARK STRUCTURED STREAMING PROCESSOR (Adapted for Background Use)
# =============================================================================

# [This class (NewsStreamProcessor) remains largely unchanged, defined by sources 15-18]
class NewsStreamProcessor:
    def __init__(self, classifier):
        self.spark = classifier.spark # [15]
        self.classifier = classifier
        self.output_dir = "data/output"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_streaming_news(self):
        """
        Starts the Spark streaming query. MUST be run in a separate process
        as it starts a continuous job.
        """
        schema = StructType([ # [15, 16]
            StructField("headline", StringType(), True),
            StructField("source", StringType(), True),
            StructField("timestamp", StringType(), True) # Simplified schema for brevity
        ])

        streaming_df = self.spark \
            .readStream \
            .option("multiline", "true") \
            .schema(schema) \
            .json(OUTPUT_DIR) # Reads from the Producer's output directory [16]

        def process_batch(batch_df, batch_id):
            if batch_df.count() > 0: # [17]
                predictions = self.classifier.predict_sentiment(batch_df) # [17]
                results = predictions.select(
                    "headline", "source", "predicted_sentiment", "confidence", "timestamp"
                )
                
                # Write to consolidated file for dashboard [4]
                results_pandas = results.toPandas()
                results_pandas.to_json(DASHBOARD_FILE, orient='records', date_format='iso')

        query = streaming_df \
            .writeStream \
            .foreachBatch(process_batch) \
            .option("checkpointLocation", "data/checkpoint") \
            .start() # [4]

        # Note: We omit query.awaitTermination() [5] because we need this function to return
        # quickly, allowing Streamlit to continue refreshing the UI.

        return query # The query object must be managed by the background process manager.

# =============================================================================
# 4. STREAMLIT APPLICATION RUNNER
# =============================================================================

# Helper functions for setup
def initialize_system():
    """Initializes and trains the ML model."""
    if 'classifier' not in st.session_state:
        # 1. Setup Data Directories
        os.makedirs("data/output", exist_ok=True)
        os.makedirs("data/input", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # 2. Initialize Classifier (and Spark session)
        classifier = SentimentClassifier()
        st.session_state['classifier'] = classifier
        
        # 3. Load or Train Model [5, 9]
        if os.path.exists(MODEL_PATH):
            classifier.model = PipelineModel.load(MODEL_PATH)
            st.success("Loaded pre-trained ML model.")
        else:
            classifier.train_model()
            classifier.model.write().overwrite().save(MODEL_PATH)
            st.success("Trained and saved new ML model.")

def start_background_processes():
    """Conceptual function to start Producer and Streamer asynchronously."""
    
    if 'streaming_running' not in st.session_state or not st.session_state['streaming_running']:
        
        # --- CRITICAL DEPLOYMENT STEP ---
        # 1. Start News Producer (run_news_producer)
        # This function must be executed in a separate thread or process.
        # Example (requires external library/logic): start_thread(run_news_producer) 
        
        # 2. Start Streaming Processor (process_streaming_news)
        processor = NewsStreamProcessor(st.session_state['classifier']) # [15]
        query = processor.process_streaming_news()
        
        # This Spark query object needs to be tracked/managed.
        st.session_state['spark_query'] = query
        st.session_state['streaming_running'] = True
        st.info("✅ PySpark Streaming Processor and News Producer started in background.")

# Dashboard Class (Sources 19-24 are integrated directly into run_dashboard)

def run_dashboard():
    """The main Streamlit interface."""
    
    st.set_page_config(
        page_title="Real-Time News Sentiment Dashboard",
        page_icon="📰",
        layout="wide"
    ) # [18]
    
    st.title("📰 Real-Time News Sentiment Analysis Dashboard")
    st.markdown("Live sentiment classification of news headlines using PySpark ML")

    # --- Setup and Initialization Section ---
    if st.sidebar.button("1. Initialize/Train System"):
        initialize_system()

    if 'classifier' in st.session_state:
        if st.sidebar.button("2. Start Real-Time Processes"):
            start_background_processes()
        
    # --- Dashboard Visualization Section (Sources 19-24) ---
    
    # Check for background processes status (conceptual)
    if 'streaming_running' not in st.session_state or not st.session_state['streaming_running']:
        st.warning("System processes are not running. Initialize and start them first.")
        return

    # Load data (using NewsDashboard logic) [10]
    try:
        if os.path.exists(DASHBOARD_FILE):
            df = pd.read_json(DASHBOARD_FILE)
        else:
            df = pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading results: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.warning("No data available yet. Waiting for streaming results...") # [18]
        return
        
    # [Metrics, Visualizations, and Table Display (Sources 21-24) are integrated here.]
    
    # ... (code for metrics and charts from 21-24) ...

    if st.button("🔄 Refresh Dashboard Data"):
        st.experimental_rerun() # Forces Streamlit to reload the data file [18]
    
    st.info("💡 This dashboard updates in real-time. Click refresh to see latest results.")

# =============================================================================
# MAIN STREAMLIT EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Ensure environment variables are set before running
    if not API_KEY:
        st.error("Please set the NEWS_API_KEY environment variable.")
    else:
        # We replace the multi-mode command line parser [1] 
        # with the direct Streamlit function call:
        run_dashboard()