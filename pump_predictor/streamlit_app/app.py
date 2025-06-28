"""
Streamlit dashboard for pump maintenance prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pump_predictor.utils.logger import get_logger
from pump_predictor.config import API_CONFIG
import joblib

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="🔧 Pump Maintenance Predictor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .high-risk {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .needs-maintenance {
        background-color: #ffcdd2;
        color: #d32f2f;
    }
    .no-maintenance {
        background-color: #c8e6c9;
        color: #388e3c;
    }
</style>
""", unsafe_allow_html=True)

class DashboardAPI:
    """Handle API communication"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or f"http://localhost:{API_CONFIG['port']}"
    
    def check_api_health(self):
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.status_code == 200 else None
        except requests.RequestException:
            return False, None
    
    def make_prediction(self, data: dict):
        """Make single prediction"""
        try:
            response = requests.post(f"{self.base_url}/predict", json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
        except requests.RequestException as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def batch_predict(self, data_list: list):
        """Make batch prediction"""
        try:
            payload = {"data": data_list, "include_details": True}
            response = requests.post(f"{self.base_url}/predict/batch", json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
        except requests.RequestException as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model/info", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
        except requests.RequestException as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def get_feature_importance(self):
        """Get feature importance"""
        try:
            response = requests.get(f"{self.base_url}/model/feature-importance", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
        except requests.RequestException as e:
            return {"error": f"Connection error: {str(e)}"}

# Initialize API client
api_client = DashboardAPI()

def load_local_model():
    """Load model locally if API is not available"""
    try:
        model_path = Path("models/best_model.joblib")
        if model_path.exists():
            return joblib.load(model_path)
        return None
    except Exception as e:
        logger.error(f"Error loading local model: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_model_info():
    """Get model info with caching"""
    return api_client.get_model_info()

@st.cache_data(ttl=300)
def get_cached_feature_importance():
    """Get feature importance with caching"""
    return api_client.get_feature_importance()

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🔧 Pump Maintenance Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Check API status
    api_healthy, health_data = api_client.check_api_health()
    
    if api_healthy:
        st.sidebar.success("✅ API Connected")
        if health_data:
            st.sidebar.metric("API Uptime", f"{health_data.get('uptime', 0):.0f}s")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.warning("Using local mode (limited functionality)")
    
    # Navigation
    page = st.sidebar.selectbox(
        "📄 Select Page",
        ["🏠 Home", "🔮 Single Prediction", "📊 Batch Analysis", "📈 Model Info", "⚙️ Settings"]
    )
    
    if page == "🏠 Home":
        show_home_page(api_healthy, health_data)
    elif page == "🔮 Single Prediction":
        show_prediction_page(api_healthy)
    elif page == "📊 Batch Analysis":
        show_batch_analysis_page(api_healthy)
    elif page == "📈 Model Info":
        show_model_info_page(api_healthy)
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page(api_healthy: bool, health_data: dict):
    """Display home page with overview"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Prediction Accuracy</h3>
            <p>Our ML models achieve high accuracy in predicting pump maintenance needs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time Analysis</h3>
            <p>Get instant predictions based on current sensor readings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Comprehensive Reports</h3>
            <p>Detailed analysis and visualization of maintenance patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    if api_healthy and health_data:
        st.subheader("📊 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Status", "🟢 Online" if api_healthy else "🔴 Offline")
        
        with col2:
            st.metric("Model Status", "✅ Loaded" if health_data.get('model_loaded') else "❌ Not Loaded")
        
        with col3:
            st.metric("Version", health_data.get('version', 'N/A'))
        
        with col4:
            uptime_hours = health_data.get('uptime', 0) / 3600
            st.metric("Uptime", f"{uptime_hours:.1f}h")
    
    # Feature overview
    st.markdown("---")
    st.subheader("🔧 Features")
    
    features = [
        {"name": "Single Prediction", "desc": "Get maintenance prediction for individual pump", "icon": "🔮"},
        {"name": "Batch Analysis", "desc": "Analyze multiple pumps simultaneously", "icon": "📊"},
        {"name": "Model Insights", "desc": "Understand how predictions are made", "icon": "🧠"},
        {"name": "Real-time Monitoring", "desc": "Continuous monitoring and alerts", "icon": "⏱️"}
    ]
    
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            **{feature['icon']} {feature['name']}**  
            {feature['desc']}
            """)

def show_prediction_page(api_healthy: bool):
    """Display single prediction page"""
    
    st.subheader("🔮 Single Pump Prediction")
    st.write("Enter sensor readings to get maintenance prediction for a single pump.")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.number_input(
                "🌡️ Temperature (°C)", 
                min_value=-50.0, 
                max_value=200.0, 
                value=75.0,
                help="Current temperature reading from the pump sensor"
            )
            
            pressure = st.number_input(
                "📊 Pressure (PSI)", 
                min_value=0.0, 
                max_value=1000.0, 
                value=150.0,
                help="Current pressure reading from the pump sensor"
            )
        
        with col2:
            vibration = st.number_input(
                "📳 Vibration (Hz)", 
                min_value=0.0, 
                max_value=100.0, 
                value=2.5,
                help="Current vibration reading from the pump sensor"
            )
            
            flow_rate = st.number_input(
                "💧 Flow Rate (L/min)", 
                min_value=0.0, 
                max_value=1000.0, 
                value=250.0,
                help="Current flow rate reading from the pump sensor"
            )
        
        submitted = st.form_submit_button("🔍 Predict Maintenance Need", use_container_width=True)
    
    if submitted:
        # Prepare data
        data = {
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration,
            "flow_rate": flow_rate
        }
        
        if api_healthy:
            # Use API
            with st.spinner("Making prediction..."):
                result = api_client.make_prediction(data)
            
            if "error" not in result:
                display_prediction_result(result)
            else:
                st.error(f"Prediction failed: {result['error']}")
        else:
            # Use local model
            local_model = load_local_model()
            if local_model:
                try:
                    features = np.array([temperature, pressure, vibration, flow_rate]).reshape(1, -1)
                    prediction = local_model.predict(features)[0]
                    
                    # Create result in API format
                    result = {
                        "needs_maintenance": bool(prediction),
                        "confidence": 0.75,  # Default confidence
                        "risk_level": "MEDIUM" if prediction else "LOW",
                        "model_type": "LocalModel",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    display_prediction_result(result)
                    
                except Exception as e:
                    st.error(f"Local prediction failed: {str(e)}")
            else:
                st.error("No model available for prediction")
    
    # Show example values
    with st.expander("📋 Example Sensor Values"):
        st.markdown("""
        **Normal Operation:**
        - Temperature: 70-80°C
        - Pressure: 100-200 PSI
        - Vibration: 1-3 Hz
        - Flow Rate: 200-300 L/min
        
        **Maintenance Required:**
        - Temperature: >90°C or <60°C
        - Pressure: >250 PSI or <80 PSI
        - Vibration: >5 Hz
        - Flow Rate: <150 L/min or >400 L/min
        """)

def display_prediction_result(result: dict):
    """Display prediction result with styling"""
    
    needs_maintenance = result.get("needs_maintenance", False)
    confidence = result.get("confidence", 0.0)
    risk_level = result.get("risk_level", "UNKNOWN")
    
    # Main result
    if needs_maintenance:
        st.markdown(
            '<div class="prediction-result needs-maintenance">⚠️ MAINTENANCE REQUIRED</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="prediction-result no-maintenance">✅ NORMAL OPERATION</div>',
            unsafe_allow_html=True
        )
    
    # Details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col2:
        risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk_level, "⚪")
        st.metric("Risk Level", f"{risk_color} {risk_level}")
    
    with col3:
        st.metric("Model", result.get("model_type", "Unknown"))
    
    # Timestamp
    if "timestamp" in result:
        st.caption(f"Prediction made at: {result['timestamp']}")

def show_batch_analysis_page(api_healthy: bool):
    """Display batch analysis page"""
    
    st.subheader("📊 Batch Pump Analysis")
    st.write("Upload CSV file or enter multiple pump readings for batch analysis.")
    
    # Upload option
    uploaded_file = st.file_uploader(
        "📁 Upload CSV File",
        type=['csv'],
        help="CSV file should have columns: temperature, pressure, vibration, flow_rate"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} pump records")
            
            # Validate columns
            required_columns = ["temperature", "pressure", "vibration", "flow_rate"]
            if all(col in df.columns for col in required_columns):
                
                # Show preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head())
                
                if st.button("🔍 Analyze All Pumps", use_container_width=True):
                    analyze_batch_data(df, api_healthy)
            else:
                st.error(f"❌ CSV must contain columns: {', '.join(required_columns)}")
                st.write("Current columns:", list(df.columns))
                
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
    
    # Manual input option
    st.markdown("---")
    st.subheader("✏️ Manual Entry")
    
    # Initialize session state for batch data
    if 'batch_data' not in st.session_state:
        st.session_state.batch_data = []
    
    with st.form("add_pump_form"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp = st.number_input("Temperature", value=75.0)
        with col2:
            press = st.number_input("Pressure", value=150.0)
        with col3:
            vib = st.number_input("Vibration", value=2.5)
        with col4:
            flow = st.number_input("Flow Rate", value=250.0)
        
        if st.form_submit_button("➕ Add Pump"):
            pump_data = {
                "temperature": temp,
                "pressure": press,
                "vibration": vib,
                "flow_rate": flow
            }
            st.session_state.batch_data.append(pump_data)
            st.success(f"Added pump #{len(st.session_state.batch_data)}")
    
    # Show current batch
    if st.session_state.batch_data:
        st.write(f"📊 Current batch: {len(st.session_state.batch_data)} pumps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Analyze Batch"):
                df_batch = pd.DataFrame(st.session_state.batch_data)
                analyze_batch_data(df_batch, api_healthy)
        
        with col2:
            if st.button("🗑️ Clear Batch"):
                st.session_state.batch_data = []
                st.experimental_rerun()
        
        with col3:
            if st.button("📋 Show Data"):
                st.dataframe(pd.DataFrame(st.session_state.batch_data))

def analyze_batch_data(df: pd.DataFrame, api_healthy: bool):
    """Analyze batch data and display results"""
    
    with st.spinner("Analyzing pumps..."):
        if api_healthy:
            # Use API for batch prediction
            data_list = df.to_dict('records')
            result = api_client.batch_predict(data_list)
            
            if "error" not in result:
                display_batch_results(result, df)
            else:
                st.error(f"Batch analysis failed: {result['error']}")
        else:
            st.warning("API not available. Batch analysis requires API connection.")

def display_batch_results(result: dict, df: pd.DataFrame):
    """Display batch analysis results"""
    
    predictions = result.get("predictions", [])
    summary = result.get("summary", {})
    
    # Summary metrics
    st.subheader("📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pumps", result.get("total_count", 0))
    
    with col2:
        st.metric("Need Maintenance", result.get("maintenance_needed_count", 0))
    
    with col3:
        maintenance_rate = result.get("maintenance_needed_count", 0) / result.get("total_count", 1)
        st.metric("Maintenance Rate", f"{maintenance_rate:.1%}")
    
    with col4:
        avg_confidence = summary.get("average_confidence", 0)
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Risk distribution
    risk_dist = summary.get("risk_distribution", {})
    if risk_dist:
        st.subheader("🎯 Risk Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(risk_dist.keys()),
                y=list(risk_dist.values()),
                marker_color=['#f44336', '#ff9800', '#4caf50']
            )
        ])
        
        fig.update_layout(
            title="Risk Level Distribution",
            xaxis_title="Risk Level",
            yaxis_title="Number of Pumps"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Create results dataframe
    results_df = df.copy()
    results_df['needs_maintenance'] = [p.get('needs_maintenance', False) for p in predictions]
    results_df['confidence'] = [p.get('confidence', 0) for p in predictions]
    results_df['risk_level'] = [p.get('risk_level', 'UNKNOWN') for p in predictions]
    
    # Color coding
    def highlight_risk(row):
        if row['risk_level'] == 'HIGH':
            return ['background-color: #ffcdd2'] * len(row)
        elif row['risk_level'] == 'MEDIUM':
            return ['background-color: #ffe0b2'] * len(row)
        elif row['risk_level'] == 'LOW':
            return ['background-color: #c8e6c9'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = results_df.style.apply(highlight_risk, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name=f"pump_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def show_model_info_page(api_healthy: bool):
    """Display model information page"""
    
    st.subheader("📈 Model Information & Performance")
    
    if api_healthy:
        # Get model info
        model_info = get_cached_model_info()
        feature_importance = get_cached_feature_importance()
        
        if "error" not in model_info:
            # Model details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 Model Details")
                st.write(f"**Type:** {model_info.get('model_type', 'Unknown')}")
                st.write(f"**Version:** {model_info.get('version', 'Unknown')}")
                st.write(f"**Features:** {len(model_info.get('features', []))}")
                if model_info.get('training_date'):
                    st.write(f"**Trained:** {model_info.get('training_date')}")
                if model_info.get('model_size'):
                    st.write(f"**Size:** {model_info.get('model_size')}")
            
            with col2:
                st.markdown("### 📊 Performance Metrics")
                metrics = model_info.get('performance_metrics', {})
                if metrics:
                    for metric, value in metrics.items():
                        st.metric(metric.capitalize(), f"{value:.4f}")
                else:
                    st.write("No performance metrics available")
            
            # Feature importance
            if "error" not in feature_importance:
                st.markdown("### 🎯 Feature Importance")
                
                features = feature_importance.get('features', {})
                if features:
                    # Create bar chart
                    feature_names = list(features.keys())
                    importance_values = list(features.values())
                    
                    fig = go.Figure([
                        go.Bar(
                            x=importance_values,
                            y=feature_names,
                            orientation='h',
                            marker_color='lightblue'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Feature Importance",
                        xaxis_title="Importance Score",
                        yaxis_title="Features"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature descriptions
                    with st.expander("📝 Feature Descriptions"):
                        descriptions = {
                            "temperature": "Operating temperature of the pump",
                            "pressure": "Internal pressure measurement",
                            "vibration": "Vibration frequency indicating mechanical condition",
                            "flow_rate": "Fluid flow rate through the pump"
                        }
                        
                        for feature in feature_names:
                            desc = descriptions.get(feature, "No description available")
                            importance = features[feature]
                            st.write(f"**{feature.capitalize()}** ({importance:.3f}): {desc}")
            
            # Model features
            st.markdown("### 🔧 Model Features")
            features_list = model_info.get('features', [])
            if features_list:
                cols = st.columns(min(4, len(features_list)))
                for i, feature in enumerate(features_list):
                    with cols[i % len(cols)]:
                        st.write(f"• {feature}")
        
        else:
            st.error("Could not load model information")
    
    else:
        st.warning("Model information requires API connection")

def show_settings_page():
    """Display settings page"""
    
    st.subheader("⚙️ Settings & Configuration")
    
    # API Settings
    st.markdown("### 🌐 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_host = st.text_input("API Host", value="localhost")
        api_port = st.number_input("API Port", value=8000, min_value=1000, max_value=65535)
    
    with col2:
        if st.button("🔄 Test Connection"):
            test_client = DashboardAPI(f"http://{api_host}:{api_port}")
            healthy, data = test_client.check_api_health()
            
            if healthy:
                st.success("✅ Connection successful!")
                if data:
                    st.json(data)
            else:
                st.error("❌ Connection failed!")
    
    # Display Settings
    st.markdown("### 🎨 Display Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
    
    with col2:
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 60)
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        st.markdown("### 📊 Chart Settings")
        chart_height = st.slider("Default Chart Height", 300, 800, 400)
        show_gridlines = st.checkbox("Show Gridlines", value=True)
        
        st.markdown("### 🔒 Security Settings")
        enable_logging = st.checkbox("Enable Request Logging", value=True)
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # About
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Pump Maintenance Predictor Dashboard**  
    Version 1.0.0  
    Built with Streamlit and FastAPI  
    
    This dashboard provides an intuitive interface for predicting pump maintenance needs
    using advanced machine learning models.
    """)
    
    # System info
    with st.expander("🖥️ System Information"):
        st.write(f"Python Version: {sys.version}")
        st.write(f"Streamlit Version: {st.__version__}")
        st.write(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()