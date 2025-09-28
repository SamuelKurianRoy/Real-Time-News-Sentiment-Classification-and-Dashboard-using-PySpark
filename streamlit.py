import os
import sys
import json
import time
import threading
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Imports for PySpark components
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline, PipelineModel # [1], [2], [3]
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF # [4], [2]
from pyspark.ml.classification import LogisticRegression # [1], [5]
from newsapi import NewsApiClient # [6], [5]

# =============================================================================
# --- Configuration & Environment Setup ---
# =============================================================================

OUTPUT_DIR = "data/input" # [6], [5]
MODEL_PATH = "models/sentiment_model" # [5]
DASHBOARD_FILE = "data/output/latest_results.json" # [5]

QUERY = "technology OR business OR finance" # [7], [5]
FETCH_INTERVAL_SECONDS = 60 # [7], [8]
API_KEY = os.getenv('NEWS_API_KEY') # [7], [8]

# Set up PySpark environment variables [9], [8]
def setup_pyspark_env():
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path # [4], [8]
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path # [4], [8]
    # FIX: Commented out PYTHONHASHSEED assignment to resolve deployment IndexErrors
    # os.environ['PYTHONHASHSEED'] = '0' # [4]
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost' # [4], [8]
    os.environ['SPARK_SQL_EXECUTION_ARROW_PYSPARK_ENABLED'] = 'false' # [4], [8]
    print(f"✅ PySpark environment configured for: {python_path}") # [4], [10]
    
setup_pyspark_env() # [4], [10]


# =============================================================================
# 1. NEWS PRODUCER (Adapted for Threading)
# =============================================================================

def fetch_and_write_news(): # [7], [10]
    """
    Fetches news using NewsAPI and writes each headline
    to a separate JSON file in the output directory. (Continuous loop)
    """
    if not API_KEY: # [10]
        print("Error: NEWS_API_KEY environment variable not set.")
        return
        
    print("Starting news producer with NewsAPI...") # [11]
    try:
        newsapi = NewsApiClient(api_key=API_KEY) # [7], [11]
    except Exception:
        print("Error: Invalid API Key. Please check your API_KEY environment variable.") # [11]
        return

    if not os.path.exists(OUTPUT_DIR): # [11]
        os.makedirs(OUTPUT_DIR) # [12], [11]
        print(f"Created directory: {OUTPUT_DIR}") # [12], [11]

    while True: # [12], [11]
        try:
            response = newsapi.get_everything( # [12], [11]
                q=QUERY,
                language='en',
                sort_by='relevancy', # [13]
                page_size=20 
            )

            if response['status'] == 'ok': # [12], [13]
                articles = response.get('articles', [])
                if articles: # [12], [13]
                    for item in articles:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') # [12], [13]
                        file_path = os.path.join(OUTPUT_DIR, f"news_{timestamp}.json") # [14], [13]
                        
                        news_data = {
                            "headline": item['title'], # [14], [15]
                            "description": item.get('description', ''),
                            "source": item.get('source', {}).get('name', 'Unknown'), # [14], [13]
                            "published_at": item.get('publishedAt', ''),
                            "url": item.get('url', ''), # [14], [15]
                            "timestamp": datetime.now().isoformat()
                        }
                        with open(file_path, 'w') as f:
                            json.dump(news_data, f) # [14], [15]
                else:
                    print("No news articles found in this fetch.") # [12]
            else:
                print(f"Error from NewsAPI: {response.get('message', 'Unknown error')}") # [14], [15]

        except Exception as e:
            print(f"An unexpected error occurred: {e}") # [9], [15]

        time.sleep(FETCH_INTERVAL_SECONDS) # [9], [16]


# =============================================================================
# 2. SENTIMENT CLASSIFICATION ML PIPELINE
# =============================================================================

class SentimentClassifier: # [1], [16]
    def __init__(self):
        try:
            self.spark = SparkSession.builder \
                .appName("NewsSentimentClassification") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .config("spark.python.worker.reuse", "false")\
                .master("local") \
                .getOrCreate()
            
            self.spark.sparkContext.setLogLevel("ERROR") # [17], [18]
            print("✅ Spark session created successfully")
        except Exception as e:
            print(f"❌ Error creating Spark session: {e}")
            raise
        self.model = None
        self.pipeline = None

    def create_training_data(self): # [17], [18]
        """Create synthetic training data for sentiment classification."""
        positive_headlines = [
            "Company reports record profits this quarter", # [19], [20]
            "New breakthrough in renewable energy technology",
            "Stock market reaches new all-time high",
            "Unemployment rate drops to lowest level",
            "Innovation drives economic growth",
            "Successful product launch exceeds expectations",
            "Technology advances improve healthcare outcomes",
            "Strong earnings drive share price up",
            "Investment in green energy creates jobs",
            "Consumer confidence reaches decade high" # [19], [20]
        ]
        negative_headlines = [
            "Major data breach affects millions of users", # [21], [22]
            "Company announces massive layoffs",
            "Stock prices plummet amid recession fears",
            "Economic downturn impacts global markets",
            "Cybersecurity threat shuts down operations",
            "Inflation reaches concerning levels",
            "Supply chain disruptions cause delays",
            "Environmental disaster affects local business",
            "Trade war escalates between major economies",
            "Corporate scandal leads to investigations" # [21], [22]
        ]

        training_data = []
        for headline in positive_headlines:
            training_data.append((headline, "positive", 1)) # [23], [24]
        
        for headline in negative_headlines:
            training_data.append((headline, "negative", 0)) # [23], [24]

        schema = StructType([
            StructField("headline", StringType(), True),
            StructField("sentiment_label", StringType(), True),
            StructField("label", IntegerType(), True)
        ])
        return self.spark.createDataFrame(training_data, schema) # [23], [24]

    def build_pipeline(self): # [23], [24]
        """Build ML pipeline for sentiment classification"""
        tokenizer = Tokenizer(inputCol="headline", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words") # [25], [26]
        cv = CountVectorizer(inputCol="filtered_words", outputCol="raw_features",
                             vocabSize=1000, minDF=2.0) # [25], [26]
        idf = IDF(inputCol="raw_features", outputCol="features") # [25], [26]
        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20) # [25], [26]
        self.pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lr]) # [25], [26]
        return self.pipeline

    def train_model(self): # [27], [28]
        """Train the sentiment classification model"""
        print("Creating training data...") # [27], [28]
        training_df = self.create_training_data() # [27], [28]
        print("Building ML pipeline...") # [27], [28]
        pipeline = self.build_pipeline()
        print("Training model...")
        self.model = pipeline.fit(training_df) # [27], [28]
        print("Model training completed!")
        return self.model

    def predict_sentiment(self, df): # [29], [30]
        """Predict sentiment for given DataFrame"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        predictions = self.model.transform(df) # [29], [30]

        # FIX: Explicitly cast the Vector UDT to Array<Double> type for safe extraction.
        # This resolves the AnalysisException: Cannot cast UDT(...) to "ARRAY" [31, 32].
        predictions = predictions.withColumn(
            "probability_array",
            col("probability").cast("array<double>") # <-- CRITICAL FIX APPLIED
        ) # [30]

        # Add readable sentiment labels [29], [30]
        predictions = predictions.withColumn(
            "predicted_sentiment",
            when(col("prediction") == 1, "positive").otherwise("negative") # [29], [30]
        ).withColumn(
            "confidence",
            # Access elements from the newly cast Array column [29], [33]
            greatest(col("probability_array").getItem(0), col("probability_array").getItem(1))
        )
        
        # Drop the temporary column and return [33]
        return predictions.drop("probability_array")


# =============================================================================
# 3. SPARK STRUCTURED STREAMING PROCESSOR
# =============================================================================

class NewsStreamProcessor: # [29], [33]
    def __init__(self, classifier):
        self.spark = classifier.spark # [34], [35]
        self.classifier = classifier
        self.output_dir = "data/output"

        if not os.path.exists(self.output_dir): # [34], [35]
            os.makedirs(self.output_dir)

    def process_streaming_news(self): # [36], [35]
        """Process streaming news data and classify sentiment in real-time"""
        print("Starting streaming news processing...") # [36], [35]

        schema = StructType([
            StructField("headline", StringType(), True),
            StructField("description", StringType(), True), # [34], [37]
            StructField("source", StringType(), True), # [36], [37]
            StructField("published_at", StringType(), True),
            StructField("url", StringType(), True),
            StructField("timestamp", StringType(), True) # [36], [37]
        ])

        streaming_df = self.spark \
            .readStream \
            .option("multiline", "true") \
            .schema(schema) \
            .json(OUTPUT_DIR) # [36], [37]

        streaming_df = streaming_df.withColumn("processing_time", current_timestamp()) # [36], [37]

        def process_batch(batch_df, batch_id):
            print(f"Processing batch {batch_id} with {batch_df.count()} records...") # [38], [37]
            if batch_df.count() > 0: # [38], [37]
                predictions = self.classifier.predict_sentiment(batch_df) # [38], [37]
                
                results = predictions.select(
                    "headline", "source", "predicted_sentiment", # [38], [39]
                    "confidence", "timestamp", "processing_time" 
                )
                
                # Write to consolidated file for dashboard
                dashboard_file = DASHBOARD_FILE
                results_pandas = results.toPandas() # [40], [39]
                results_pandas.to_json(dashboard_file, orient='records', date_format='iso') # [40], [39]
                
        query = streaming_df \
            .writeStream \
            .foreachBatch(process_batch) \
            .option("checkpointLocation", "data/checkpoint") \
            .start() # [40], [41]

        return query # [40], [41]


# =============================================================================
# 4. STREAMLIT DASHBOARD & RUNNER
# =============================================================================

class NewsDashboard: # [42], [41]
    def __init__(self):
        self.results_file = DASHBOARD_FILE # [42], [41]

    def load_results(self): # [42], [41]
        """Load latest sentiment analysis results"""
        try:
            if os.path.exists(self.results_file): # [42], [43]
                return pd.read_json(self.results_file)
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading results: {e}") # [42], [43]
            return pd.DataFrame()

    def create_dashboard(self, df, stream_running): # [44], [43]
        """Create Streamlit dashboard interface"""
        st.set_page_config(
            page_title="Real-Time News Sentiment Dashboard",
            page_icon="📰",
            layout="wide"
        ) # [44], [43]
        
        st.title("📰 Real-Time News Sentiment Analysis Dashboard") # [44], [43]
        st.markdown("Live sentiment classification of news headlines using PySpark ML") # [44], [45]

        if not stream_running: # [45]
            st.warning("System processes are inactive. Please start the real-time processes.")
        
        if df.empty: # [46], [45]
            st.warning("No data available yet. Waiting for streaming results...") 
            return

        # Metrics [46], [45]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Headlines", len(df))
        with col2:
            positive_count = len(df[df['predicted_sentiment'] == 'positive'])
            st.metric("Positive", positive_count)
        with col3:
            negative_count = len(df[df['predicted_sentiment'] == 'negative']) # [46], [47]
            st.metric("Negative", negative_count)
        with col4:
            avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}") # [46], [47]

        # Visualizations [48], [47]
        col1, col2 = st.columns(2)
        with col1:
            sentiment_counts = df['predicted_sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={'positive': 'green', 'negative': 'red'}
            )
            st.plotly_chart(fig_pie, use_container_width=True) # [48], [47]
        
        with col2: # [48], [49]
            if 'source' in df.columns:
                source_sentiment = df.groupby(['source', 'predicted_sentiment']).size().unstack(fill_value=0) # [50], [49]
                fig_bar = px.bar(
                    source_sentiment.reset_index(),
                    x='source',
                    y=['positive', 'negative'],
                    title="Sentiment by News Source",
                    color_discrete_map={'positive': 'green', 'negative': 'red'}
                )
                st.plotly_chart(fig_bar, use_container_width=True) # [50], [49]

        # Recent headlines table [50], [51]
        st.subheader("Recent Headlines")
        display_df = df[['headline', 'source', 'predicted_sentiment', 'confidence']].copy()
        display_df = display_df.sort_values('confidence', ascending=False).head(10) # [52], [51]
        
        def style_sentiment(val): # [52], [51]
            color = 'green' if val == 'positive' else 'red'
            return f'color: {color}; font-weight: bold'

        styled_df = display_df.style.applymap(
            style_sentiment, subset=['predicted_sentiment']
        ).format({'confidence': '{:.3f}'}) # [52], [51]
        
        st.dataframe(styled_df, use_container_width=True)
        st.info("💡 This dashboard updates in real-time. Click refresh to see latest results.") # [52], [51]


def initialize_system(): # [53]
    """Initializes environment, classifier, and trains/loads the ML model."""
    
    os.makedirs("data/output", exist_ok=True) # [53]
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("models", exist_ok=True) # [53]
    
    if 'classifier' not in st.session_state: # [53]
        try:
            classifier = SentimentClassifier() # [53]
            st.session_state['classifier'] = classifier
            
            if os.path.exists(MODEL_PATH): # [53]
                classifier.model = PipelineModel.load(MODEL_PATH) # [54], [3]
                st.success("Loaded pre-trained ML model.") # [54], [3]
            else:
                st.info("No model found. Starting training process...") # [54]
                classifier.train_model() # [54], [55]
                classifier.model.write().overwrite().save(MODEL_PATH) # [54], [55]
                st.success(f"Trained and saved new ML model to {MODEL_PATH}.") # [54]
        except Exception as e:
            st.error(f"Initialization Error: {e}") # [54]
            st.session_state['init_failed'] = True

def start_background_processes(): # [56]
    """Starts News Producer via Threading and initiates PySpark Streaming."""
    
    if st.session_state.get('streaming_running', False): # [56]
        st.info("Background processes are already running.")
        return

    if 'classifier' not in st.session_state: # [56]
        st.error("Cannot start streamer: Model Classifier has not been initialized. Please click 'Initialize/Train System'.")
        return

    st.info("Attempting to start continuous processes...") # [56]
    
    # 1. Start News Producer in a separate thread (Threading required for non-blocking execution)
    producer_thread = threading.Thread(target=fetch_and_write_news, daemon=True) # [56]
    producer_thread.start()
    st.session_state['producer_thread'] = producer_thread # [57]
    
    # 2. Start Streaming Processor
    processor = NewsStreamProcessor(st.session_state['classifier']) # [57], [34]
    
    try:
        query = processor.process_streaming_news() # [57], [40]
        
        st.session_state['spark_query'] = query
        st.session_state['streaming_running'] = True # [57]
        
        st.success("✅ PySpark Streaming Processor and News Producer started in background.") # [57]
        st.warning(f"Data may take up to {FETCH_INTERVAL_SECONDS} seconds to appear on the dashboard.") # [57]
        
    except Exception as e:
         st.error(f"❌ Failed to start Streaming Processor: {e}") # [58]
         st.session_state['streaming_running'] = False


def run_unified_app(): # [58]
    """The main Streamlit interface function."""
    
    dashboard = NewsDashboard() # [58], [59]

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("System Controls")
        
        if st.button("1. Initialize/Train System"): # [58]
            initialize_system()
        
        if 'classifier' in st.session_state and not st.session_state.get('init_failed', False): # [58]
            if st.button("2. Start Real-Time Processes"):
                start_background_processes()
                
        # Status indicators [58], [60]
        st.markdown("---")
        st.subheader("Status")
        if 'classifier' in st.session_state:
            st.success("Model Initialized") # [60]
        else:
            st.error("Model Not Initialized")
            
        if st.session_state.get('streaming_running', False):
            st.success("Streaming Active") # [60]
        else:
            st.error("Streaming Inactive")

    # --- Dashboard Display ---
    
    df = dashboard.load_results() # [60], [42]
    dashboard.create_dashboard(df, st.session_state.get('streaming_running', False)) # [60], [44]


if __name__ == "__main__": # [60]
    if not API_KEY: # [61]
        st.error("Please set the NEWS_API_KEY environment variable (see SETUP INSTRUCTIONS).")
    else:
        run_unified_app() # [61]