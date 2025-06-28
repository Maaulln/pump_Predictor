"""
Streamlit Dashboard for Pump Maintenance Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pump_predictor.utils.logger import get_logger
from pump_predictor.config import API_CONFIG

logger = get_logger(__name__)

# Configure Streamlit page
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
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .prediction-high {
        color: #dc3545;
        font-weight: bold;
    }
    .prediction-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    .prediction-low {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class PumpDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.api_base_url = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}"
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'api_status' not in st.session_state:
            st.session_state.api_status = 'unknown'
    
    def check_api_status(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                st.session_state.api_status = 'healthy'
                return True
            else:
                st.session_state.api_status = 'unhealthy'
                return False
        except:
            st.session_state.api_status = 'offline'
            return False
    
    def display_header(self):
        """Display main header"""
        st.markdown('<h1 class="main-header">🔧 Pump Maintenance Predictor</h1>', 
                   unsafe_allow_html=True)
        
        # API Status
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if self.check_api_status():
                st.success("🟢 API Online")
            else:
                st.error("🔴 API Offline")
        
        with col2:
            if st.button("🔄 Refresh Status"):
                st.rerun()
    
    def display_sidebar(self):
        """Display sidebar controls"""
        st.sidebar.title("🎛️ Control Panel")
        
        # Navigation
        page = st.sidebar.radio(
            "Navigate to:",
            ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", 
             "📈 Analytics", "⚙️ Model Info", "📋 History"]
        )
        
        return page.split(" ", 1)[1]  # Remove emoji
    
    def single_prediction_page(self):
        """Single prediction interface"""
        st.header("🔮 Single Pump Prediction")
        
        if st.session_state.api_status != 'healthy':
            st.error("⚠️ API is not available. Please check the API service.")
            return
        
        # Input form
        with st.form("single_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.number_input(
                    "Temperature (°C)", 
                    min_value=-50.0, max_value=200.0, 
                    value=75.0, step=0.1
                )
                vibration = st.number_input(
                    "Vibration (Hz)", 
                    min_value=0.0, max_value=100.0, 
                    value=2.5, step=0.1
                )
            
            with col2:
                pressure = st.number_input(
                    "Pressure (PSI)", 
                    min_value=0.0, max_value=1000.0, 
                    value=150.0, step=1.0
                )
                flow_rate = st.number_input(
                    "Flow Rate (L/min)", 
                    min_value=0.0, max_value=1000.0, 
                    value=250.0, step=1.0
                )
            
            # Prediction type
            explain_prediction = st.checkbox("🔍 Get Detailed Explanation", value=True)
            
            submitted = st.form_submit_button("🔮 Predict Maintenance Need")
        
        if submitted:
            self.make_single_prediction(temperature, pressure, vibration, flow_rate, explain_prediction)
    
    def make_single_prediction(self, temperature, pressure, vibration, flow_rate, explain=True):
        """Make a single prediction"""
        data = {
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration,
            "flow_rate": flow_rate
        }
        
        try:
            with st.spinner("🔄 Making prediction..."):
                if explain:
                    response = requests.post(f"{self.api_base_url}/predict/explain", json=data)
                else:
                    response = requests.post(f"{self.api_base_url}/predict", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    self.display_prediction_result(result, explain)
                    
                    # Add to history
                    history_item = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'input': data,
                        'result': result['prediction'] if explain else result,
                        'type': 'single'
                    }
                    st.session_state.prediction_history.append(history_item)
                    
                else:
                    st.error(f"❌ Prediction failed: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    def display_prediction_result(self, result, is_explained=False):
        """Display prediction result"""
        if is_explained:
            prediction = result['prediction']
            explanation = result.get('explanation_text', '')
            contributions = result.get('feature_contributions', {})
        else:
            prediction = result
            explanation = None
            contributions = {}
        
        # Main result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction['needs_maintenance']:
                st.markdown(f'<div class="prediction-high">⚠️ MAINTENANCE REQUIRED</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-low">✅ NORMAL OPERATION</div>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "Confidence", 
                f"{prediction['confidence']:.1%}",
                delta=None
            )
        
        with col3:
            risk_color = {
                'HIGH': 'prediction-high',
                'MEDIUM': 'prediction-medium',
                'LOW': 'prediction-low'
            }
            st.markdown(f'<div class="{risk_color[prediction["risk_level"]]}">Risk Level: {prediction["risk_level"]}</div>', 
                       unsafe_allow_html=True)
        
        # Detailed explanation
        if explanation:
            st.subheader("🔍 Detailed Explanation")
            st.write(explanation)
            
            if contributions:
                self.plot_feature_contributions(contributions)
    
    def plot_feature_contributions(self, contributions):
        """Plot feature contributions"""
        if not contributions or 'explanation' in contributions:
            return
        
        features = list(contributions.keys())
        values = list(contributions.values())
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=['red' if v > 0 else 'blue' for v in values]
        ))
        
        fig.update_layout(
            title="Feature Contributions to Prediction",
            xaxis_title="Contribution Score",
            yaxis_title="Features",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def batch_prediction_page(self):
        """Batch prediction interface"""
        st.header("📊 Batch Pump Prediction")
        
        if st.session_state.api_status != 'healthy':
            st.error("⚠️ API is not available. Please check the API service.")
            return
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload a CSV file with columns: temperature, pressure, vibration, flow_rate"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_columns = ['temperature', 'pressure', 'vibration', 'flow_rate']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"❌ Missing required columns. Expected: {required_columns}")
                    return
                
                st.success(f"✅ File uploaded successfully! {len(df)} records found.")
                
                # Show preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(10))
                
                # Batch size control
                batch_size = min(len(df), 1000)  # API limit
                if len(df) > 1000:
                    st.warning(f"⚠️ File contains {len(df)} records. Only first 1000 will be processed.")
                
                if st.button("🚀 Run Batch Prediction"):
                    self.run_batch_prediction(df.head(batch_size))
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        
        # Manual data entry
        st.subheader("✏️ Manual Data Entry")
        self.manual_batch_entry()
    
    def manual_batch_entry(self):
        """Manual batch data entry"""
        if 'batch_data' not in st.session_state:
            st.session_state.batch_data = []
        
        with st.form("manual_batch_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                temp = st.number_input("Temperature", value=75.0, key="batch_temp")
            with col2:
                press = st.number_input("Pressure", value=150.0, key="batch_pressure")
            with col3:
                vib = st.number_input("Vibration", value=2.5, key="batch_vib")
            with col4:
                flow = st.number_input("Flow Rate", value=250.0, key="batch_flow")
            
            if st.form_submit_button("➕ Add to Batch"):
                st.session_state.batch_data.append({
                    'temperature': temp,
                    'pressure': press,
                    'vibration': vib,
                    'flow_rate': flow
                })
                st.success("✅ Data added to batch!")
        
        # Show current batch
        if st.session_state.batch_data:
            st.subheader(f"📋 Current Batch ({len(st.session_state.batch_data)} items)")
            batch_df = pd.DataFrame(st.session_state.batch_data)
            st.dataframe(batch_df)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Predict Batch"):
                    self.run_batch_prediction(batch_df)
            with col2:
                if st.button("🗑️ Clear Batch"):
                    st.session_state.batch_data = []
                    st.rerun()
    
    def run_batch_prediction(self, df):
        """Run batch prediction"""
        try:
            # Convert dataframe to API format
            data_list = df.to_dict('records')
            batch_request = {
                "data": data_list,
                "include_details": True
            }
            
            with st.spinner("🔄 Processing batch prediction..."):
                response = requests.post(f"{self.api_base_url}/predict/batch", json=batch_request)
                
                if response.status_code == 200:
                    result = response.json()
                    self.display_batch_results(result, df)
                    
                    # Add to history
                    history_item = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'input': f"Batch of {len(data_list)} records",
                        'result': result,
                        'type': 'batch'
                    }
                    st.session_state.prediction_history.append(history_item)
                    
                else:
                    st.error(f"❌ Batch prediction failed: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    def display_batch_results(self, result, original_df):
        """Display batch prediction results"""
        st.subheader("📊 Batch Prediction Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pumps", result['total_count'])
        with col2:
            st.metric("Maintenance Needed", result['maintenance_needed_count'])
        with col3:
            maintenance_rate = result['maintenance_needed_count'] / result['total_count']
            st.metric("Maintenance Rate", f"{maintenance_rate:.1%}")
        with col4:
            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
        
        # Risk distribution
        risk_summary = result['summary']['risk_distribution']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution pie chart
            fig = px.pie(
                values=list(risk_summary.values()),
                names=list(risk_summary.keys()),
                title="Risk Level Distribution",
                color_discrete_map={
                    'HIGH': '#dc3545',
                    'MEDIUM': '#fd7e14',
                    'LOW': '#28a745'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Maintenance distribution
            maintenance_data = {
                'Maintenance Needed': result['maintenance_needed_count'],
                'Normal Operation': result['total_count'] - result['maintenance_needed_count']
            }
            
            fig = px.pie(
                values=list(maintenance_data.values()),
                names=list(maintenance_data.keys()),
                title="Maintenance Distribution",
                color_discrete_map={
                    'Maintenance Needed': '#dc3545',
                    'Normal Operation': '#28a745'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Create results dataframe
        results_df = original_df.copy()
        predictions = result['predictions']
        
        results_df['Maintenance_Needed'] = [p['needs_maintenance'] for p in predictions]
        results_df['Confidence'] = [p['confidence'] for p in predictions]
        results_df['Risk_Level'] = [p['risk_level'] for p in predictions]
        
        # Color coding
        def highlight_risk(row):
            if row['Risk_Level'] == 'HIGH':
                return ['background-color: #ffebee'] * len(row)
            elif row['Risk_Level'] == 'MEDIUM':
                return ['background-color: #fff3e0'] * len(row)
            else:
                return ['background-color: #e8f5e8'] * len(row)
        
        styled_df = results_df.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f"pump_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def analytics_page(self):
        """Analytics dashboard"""
        st.header("📈 Analytics Dashboard")
        
        if not st.session_state.prediction_history:
            st.info("📊 No prediction history available. Make some predictions first!")
            return
        
        # Filter controls
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_type = st.selectbox(
                "Prediction Type", 
                ["All", "Single", "Batch"]
            )
        
        with col2:
            days_back = st.selectbox(
                "Time Range", 
                [1, 7, 30, 90],
                format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
            )
        
        # Filter history
        filtered_history = self.filter_history(prediction_type.lower(), days_back)
        
        if not filtered_history:
            st.info("📊 No data found for selected filters.")
            return
        
        # Analytics visualizations
        self.display_analytics(filtered_history)
    
    def filter_history(self, pred_type, days_back):
        """Filter prediction history"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        filtered = []
        for item in st.session_state.prediction_history:
            # Check date
            item_date = datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S")
            if item_date < cutoff_date:
                continue
            
            # Check type
            if pred_type != 'all' and item['type'] != pred_type:
                continue
            
            filtered.append(item)
        
        return filtered
    
    def display_analytics(self, history):
        """Display analytics visualizations"""
        # Prediction trends over time
        timestamps = [datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S") for item in history]
        
        # Count predictions by day
        daily_counts = {}
        maintenance_counts = {}
        
        for i, item in enumerate(history):
            date = timestamps[i].date()
            daily_counts[date] = daily_counts.get(date, 0) + 1
            
            # Count maintenance predictions
            if item['type'] == 'single':
                needs_maintenance = item['result']['needs_maintenance']
            else:  # batch
                needs_maintenance = item['result']['maintenance_needed_count'] > 0
            
            if needs_maintenance:
                maintenance_counts[date] = maintenance_counts.get(date, 0) + 1
        
        # Create time series plot
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]
        maintenance = [maintenance_counts.get(date, 0) for date in dates]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=counts, name="Total Predictions",
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=maintenance, name="Maintenance Predictions",
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Prediction Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Predictions"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def model_info_page(self):
        """Model information page"""
        st.header("⚙️ Model Information")
        
        if st.session_state.api_status != 'healthy':
            st.error("⚠️ API is not available. Please check the API service.")
            return
        
        try:
            # Get model info
            response = requests.get(f"{self.api_base_url}/model/info")
            if response.status_code == 200:
                model_info = response.json()
                self.display_model_info(model_info)
            else:
                st.error("❌ Failed to fetch model information")
            
            # Get feature importance
            response = requests.get(f"{self.api_base_url}/model/feature-importance")
            if response.status_code == 200:
                feature_importance = response.json()
                self.display_feature_importance(feature_importance)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    def display_model_info(self, model_info):
        """Display model information"""
        st.subheader("🤖 Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", model_info.get('model_type', 'Unknown'))
            st.metric("Version", model_info.get('version', 'Unknown'))
            st.metric("Features Count", len(model_info.get('features', [])))
        
        with col2:
            if 'training_date' in model_info and model_info['training_date']:
                training_date = model_info['training_date']
                st.metric("Training Date", training_date)
            
            if 'model_size' in model_info:
                st.metric("Model Size", model_info['model_size'])
        
        # Performance metrics
        if 'performance_metrics' in model_info and model_info['performance_metrics']:
            st.subheader("📊 Performance Metrics")
            metrics = model_info['performance_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
        
        # Features list
        if 'features' in model_info:
            st.subheader("🏷️ Model Features")
            features_df = pd.DataFrame(model_info['features'], columns=['Feature Name'])
            st.dataframe(features_df, use_container_width=True)
    
    def display_feature_importance(self, feature_importance):
        """Display feature importance"""
        st.subheader("🎯 Feature Importance")
        
        if 'features' in feature_importance:
            features = feature_importance['features']
            
            # Create bar chart
            fig = go.Figure(go.Bar(
                x=list(features.values()),
                y=list(features.keys()),
                orientation='h',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Feature Importance Scores",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            importance_df = pd.DataFrame(
                list(features.items()), 
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df, use_container_width=True)
    
    def history_page(self):
        """Prediction history page"""
        st.header("📋 Prediction History")
        
        if not st.session_state.prediction_history:
            st.info("📋 No prediction history available.")
            return
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        
        with col2:
            show_details = st.checkbox("Show Details", value=False)
        
        with col3:
            items_per_page = st.selectbox("Items per page", [10, 25, 50], index=0)
        
        # Display history
        history = st.session_state.prediction_history[::-1]  # Most recent first
        
        for i, item in enumerate(history[:items_per_page]):
            with st.expander(f"🕐 {item['timestamp']} - {item['type'].title()} Prediction"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📥 Input")
                    if item['type'] == 'single':
                        st.json(item['input'])
                    else:
                        st.write(item['input'])
                
                with col2:
                    st.subheader("📤 Result")
                    if show_details:
                        st.json(item['result'])
                    else:
                        if item['type'] == 'single':
                            result = item['result']
                            st.write(f"**Maintenance Needed:** {result['needs_maintenance']}")
                            st.write(f"**Confidence:** {result['confidence']:.1%}")
                            st.write(f"**Risk Level:** {result['risk_level']}")
                        else:
                            result = item['result']
                            st.write(f"**Total Pumps:** {result['total_count']}")
                            st.write(f"**Maintenance Needed:** {result['maintenance_needed_count']}")
    
    def run(self):
        """Main dashboard runner"""
        self.display_header()
        
        # Sidebar navigation
        page = self.display_sidebar()
        
        # Route to appropriate page
        if page == "Home":
            self.home_page()
        elif page == "Single Prediction":
            self.single_prediction_page()
        elif page == "Batch Prediction":
            self.batch_prediction_page()
        elif page == "Analytics":
            self.analytics_page()
        elif page == "Model Info":
            self.model_info_page()
        elif page == "History":
            self.history_page()
    
    def home_page(self):
        """Home page with overview"""
        st.header("🏠 Dashboard Overview")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predictions = len(st.session_state.prediction_history)
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            single_predictions = sum(1 for item in st.session_state.prediction_history if item['type'] == 'single')
            st.metric("Single Predictions", single_predictions)
        
        with col3:
            batch_predictions = sum(1 for item in st.session_state.prediction_history if item['type'] == 'batch')
            st.metric("Batch Predictions", batch_predictions)
        
        with col4:
            api_status = "🟢 Online" if st.session_state.api_status == 'healthy' else "🔴 Offline"
            st.metric("API Status", api_status)
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔮 Make Single Prediction", use_container_width=True):
                st.session_state.page = "Single Prediction"
                st.rerun()
        
        with col2:
            if st.button("📊 Batch Prediction", use_container_width=True):
                st.session_state.page = "Batch Prediction"
                st.rerun()
        
        with col3:
            if st.button("📈 View Analytics", use_container_width=True):
                st.session_state.page = "Analytics"
                st.rerun()
        
        # Recent activity
        if st.session_state.prediction_history:
            st.subheader("📋 Recent Activity")
            recent_items = st.session_state.prediction_history[-5:][::-1]
            
            for item in recent_items:
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**{item['timestamp']}**")
                    
                    with col2:
                        st.write(f"{item['type'].title()} Prediction")
                    
                    with col3:
                        if item['type'] == 'single':
                            status = "⚠️" if item['result']['needs_maintenance'] else "✅"
                        else:
                            status = "📊"
                        st.write(status)

# Main execution
if __name__ == "__main__":
    dashboard = PumpDashboard()
    dashboard.run()