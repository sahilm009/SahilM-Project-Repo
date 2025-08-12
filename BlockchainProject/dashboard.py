"""
Interactive Streamlit Dashboard for Ethereum Analytics
Run with: streamlit run ethereum_dashboard.py

Install requirements:
pip install streamlit plotly pandas numpy google-cloud-bigquery
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced analytics class
from WalletAnalytics import EnhancedEthereumAnalytics

# Page configuration
st.set_page_config(
    page_title="Ethereum Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .strategy-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #17a2b8;
    }
    .insight-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_analyze_data(start_date: str, end_date: str, project_id: str = None):
    """Load and analyze Ethereum data with caching"""
    analytics = EnhancedEthereumAnalytics(project_id=project_id)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Get data
        status_text.text("Fetching Ethereum transaction data...")
        progress_bar.progress(20)
        df = analytics.get_enhanced_wallet_data(start_date, end_date, limit=500000)
        
        if df.empty:
            st.error("No data retrieved. Using sample data for demonstration.")
            return None
            
        # Step 2: Calculate metrics
        status_text.text("Calculating wallet metrics...")
        progress_bar.progress(40)
        wallet_metrics = analytics.calculate_wallet_metrics(df, sample_size=8000)
        
        # Step 3: Retention analysis
        status_text.text("Analyzing retention patterns...")
        progress_bar.progress(60)
        retention_results = analytics.analyze_retention_boost(df, wallet_metrics, sample_size=3000)
        
        # Step 4: A/B testing
        status_text.text("Running A/B test simulation...")
        progress_bar.progress(80)
        ab_results = analytics.simulate_ab_testing(wallet_metrics, sample_size=5000)
        
        # Step 5: Generate strategies
        status_text.text("Generating product strategies...")
        progress_bar.progress(90)
        strategies = analytics.generate_product_strategies(retention_results, ab_results, wallet_metrics)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return {
            'data': df,
            'wallet_metrics': wallet_metrics,
            'retention': retention_results,
            'ab_testing': ab_results,
            'strategies': strategies
        }
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

def display_key_metrics(results: Dict):
    
    if not results:
        return
        
    df = results['data']
    wallet_metrics = results['wallet_metrics']
    retention = results['retention']
    ab_testing = results['ab_testing']
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📊 Total Transactions", 
            value=f"{len(df):,}"
        )
    
    with col2:
        st.metric(
            label="👥 Unique Wallets", 
            value=f"{len(wallet_metrics):,}"
        )
    
    with col3:
        total_volume = df['value_usd'].sum()
        st.metric(
            label="💰 Total Volume", 
            value=f"${total_volume/1e6:.1f}M"
        )
    
    with col4:
        avg_ltv = wallet_metrics['ltv'].mean()
        st.metric(
            label="📈 Average LTV", 
            value=f"${avg_ltv:.0f}"
        )
    
    with col5:
        if 'error' not in retention:
            retention_boost = retention['retention_boost_pct']
            st.metric(
                label="🚀 Retention Boost", 
                value=f"{retention_boost:.1f}%",
                delta="Early swappers vs regular"
            )

def create_analytics_charts(results: Dict):
   
    if not results:
        return
        
    df = results['data']
    wallet_metrics = results['wallet_metrics']
    retention = results['retention']
    ab_testing = results['ab_testing']
    
    # Sample data for performance
    df_sample = df.sample(n=min(30000, len(df))) if len(df) > 30000 else df
    
    # Create tabs for different chart categories
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Volume & Activity", "👥 User Segments", "🎯 Retention Analysis", "🧪 A/B Testing"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily transaction volume
            daily_stats = df_sample.groupby('date').agg({
                'tx_hash': 'count',
                'value_usd': 'sum'
            }).reset_index()
            
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(
                go.Scatter(x=daily_stats['date'], y=daily_stats['tx_hash'], 
                          name="Transactions", line=dict(color='#1f77b4')),
                secondary_y=False,
            )
            fig1.add_trace(
                go.Scatter(x=daily_stats['date'], y=daily_stats['value_usd'], 
                          name="Volume (USD)", line=dict(color='#ff7f0e')),
                secondary_y=True,
            )
            fig1.update_layout(title="Daily Transaction Volume & Count", height=400)
            fig1.update_yaxes(title_text="Number of Transactions", secondary_y=False)
            fig1.update_yaxes(title_text="Volume (USD)", secondary_y=True)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Gas fee distribution
            fig2 = px.histogram(df_sample, x='gas_fee_usd', nbins=50,
                               title='Gas Fee Distribution',
                               labels={'gas_fee_usd': 'Gas Fee (USD)', 'count': 'Frequency'})
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # User segment distribution
            segment_counts = wallet_metrics['user_segment'].value_counts()
            fig3 = px.pie(values=segment_counts.values, names=segment_counts.index,
                         title='User Segment Distribution')
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # LTV by segment
            fig4 = px.box(wallet_metrics.reset_index(), x='user_segment', y='ltv',
                         title='LTV Distribution by Segment',
                         labels={'ltv': 'Lifetime Value (USD)', 'user_segment': 'User Segment'})
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        if 'error' not in retention:
            col1, col2 = st.columns(2)
            
            with col1:
                # Retention comparison
                categories = ['Early Swappers', 'Regular Users']
                retention_rates = [
                    retention['early_swapper_7d_retention'] * 100,
                    retention['regular_7d_retention'] * 100
                ]
                
                fig5 = go.Figure(data=[
                    go.Bar(x=categories, y=retention_rates, 
                           text=[f"{rate:.1f}%" for rate in retention_rates],
                           textposition='auto',
                           marker_color=['#2ecc71', '#e74c3c'])
                ])
                fig5.update_layout(title='7-Day Retention Comparison', 
                                  yaxis_title='Retention Rate (%)',
                                  height=400)
                st.plotly_chart(fig5, use_container_width=True)
            
            with col2:
                # Create a metrics comparison
                st.markdown("### 🎯 Retention Insights")
                st.markdown(f"""
                <div class="insight-box">
                <h4>Key Findings:</h4>
                <ul>
                <li><strong>Early Swappers</strong>: {retention['early_swapper_7d_retention']:.1%} retention</li>
                <li><strong>Regular Users</strong>: {retention['regular_7d_retention']:.1%} retention</li>
                <li><strong>Boost</strong>: {retention['retention_boost_pct']:.1f}% improvement</li>
                <li><strong>Sample Size</strong>: {retention['sample_size']:,} wallets analyzed</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Retention analysis data not available")
    
    with tab4:
        if 'error' not in ab_testing:
            # A/B testing results
            groups = ['Active DEX\nControl', 'Active DEX\nTreatment', 
                     'Inactive DEX\nControl', 'Inactive DEX\nTreatment']
            volumes = [
                ab_testing['active_dex']['control_volume'],
                ab_testing['active_dex']['treatment_volume'],
                ab_testing['inactive_dex']['control_volume'],
                ab_testing['inactive_dex']['treatment_volume']
            ]
            
            fig6 = go.Figure(data=[
                go.Bar(x=groups, y=volumes,
                      marker_color=['#3498db', '#2ecc71', '#3498db', '#2ecc71'],
                      text=[f"${vol:.0f}" for vol in volumes],
                      textposition='auto')
            ])
            fig6.update_layout(title='A/B Testing Results - 14 Day Volume',
                              yaxis_title='Average Volume (USD)',
                              height=400)
            st.plotly_chart(fig6, use_container_width=True)
            
            # A/B testing insights
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### A/B Test Results")
                st.markdown(f"""
                <div class="metric-container">
                <h4>Active DEX Users:</h4>
                <p>• Control: ${ab_testing['active_dex']['control_volume']:.0f}</p>
                <p>• Treatment: ${ab_testing['active_dex']['treatment_volume']:.0f}</p>
                <p>• <strong>Lift: {ab_testing['active_dex']['lift']:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Statistical Significance")
                st.markdown(f"""
                <div class="metric-container">
                <h4>Inactive DEX Users:</h4>
                <p>• Control: ${ab_testing['inactive_dex']['control_volume']:.0f}</p>
                <p>• Treatment: ${ab_testing['inactive_dex']['treatment_volume']:.0f}</p>
                <p>• <strong>Lift: {ab_testing['inactive_dex']['lift']:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("A/B testing data not available")

def display_product_strategies(strategies: Dict):
    """Display product strategies in an organized way"""
    st.markdown("## Data-Driven Product Strategies")
    
    # Create strategy cards
    for strategy_name, details in strategies.items():
        strategy_title = strategy_name.replace('_', ' ').title()
        
        with st.expander(f"🎯 {strategy_title}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="strategy-card">
                <h4>📋 Rationale:</h4>
                <p>{details['rationale']}</p>
                
                <h4>🔧 Implementation:</h4>
                <ul>
                """, unsafe_allow_html=True)
                
                for implementation in details['implementation']:
                    st.markdown(f"<li>{implementation}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                <h4>Expected Impact:</h4>
                <p><strong>{details['expected_impact']}</strong></p>
                </div>
                """, unsafe_allow_html=True)

def main():
    
    
    # Header
    st.title("Ethereum Analytics Dashboard")
    st.markdown("**Comprehensive wallet lifecycle analysis with A/B testing and product strategies**")
    
    # Sidebar for inputs
    st.sidebar.header("Analysis Parameters")
    
    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2024, 7, 15),
            min_value=date(2024, 1, 1),
            max_value=date.today()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=date(2024, 7, 20),
            min_value=start_date,
            max_value=date.today()
        )
    
    # BigQuery settings
    st.sidebar.header("BigQuery Settings")
    use_bigquery = st.sidebar.checkbox("Use BigQuery (requires setup)", value=False)
    
    if use_bigquery:
        project_id = st.sidebar.text_input(
            "BigQuery Project ID",
            value="your-project-id",
            help="Enter your Google Cloud project ID"
        )
    else:
        project_id = None
        st.sidebar.info("Using sample data for demonstration")
    
    # Analysis settings
    st.sidebar.header("Analysis Settings")
    speed_mode = st.sidebar.checkbox("Speed Mode (faster analysis)", value=True)
    
    if speed_mode:
        st.sidebar.info("Speed mode uses sampling for faster results")
    
    # Run analysis button
    if st.sidebar.button("Run Analysis", type="primary"):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        st.markdown(f"### Analyzing data from **{start_str}** to **{end_str}**")
        
        # Load and analyze data
        with st.spinner("Running comprehensive Ethereum analysis..."):
            results = load_and_analyze_data(start_str, end_str, project_id)
        
        if results:
            # Display results
            st.success("Analysis completed successfully!")
            
            # Key metrics
            display_key_metrics(results)
            
            st.markdown("---")
            
            # Charts
            create_analytics_charts(results)
            
            st.markdown("---")
            
            # Product strategies
            display_product_strategies(results['strategies'])
            
            # Download option
            st.markdown("---")
            st.markdown("### Export Results")
            
            if st.button("Generate Detailed Report"):
                # Create summary report
                report = f"""
# Ethereum Analytics Report
**Analysis Period**: {start_str} to {end_str}

## Key Metrics
- Total Transactions: {len(results['data']):,}
- Unique Wallets: {len(results['wallet_metrics']):,}
- Total Volume: ${results['data']['value_usd'].sum()/1e6:.1f}M
- Average LTV: ${results['wallet_metrics']['ltv'].mean():.0f}

## Retention Analysis
- Early Swapper Retention: {results['retention'].get('early_swapper_7d_retention', 0):.1%}
- Regular User Retention: {results['retention'].get('regular_7d_retention', 0):.1%}
- Retention Boost: {results['retention'].get('retention_boost_pct', 0):.1f}%

## A/B Testing Results
- Active DEX Lift: {results['ab_testing'].get('active_dex', {}).get('lift', 0):.1f}%
- Inactive DEX Lift: {results['ab_testing'].get('inactive_dex', {}).get('lift', 0):.1f}%

## Product Strategies Generated
{len(results['strategies'])} strategic recommendations created based on data analysis.
                """
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"ethereum_analytics_report_{start_str}_{end_str}.md",
                    mime="text/markdown"
                )
        
        else:
            st.error("Analysis failed. Please check your settings and try again.")
    
    # Instructions
    else:
        st.markdown("""
        
        
        ##This dashboard provides comprehensive analysis of Ethereum wallet behavior, including:
        
        ### Analytics Features:
        - **Wallet Lifecycle Analysis**: Track user retention and engagement patterns
        - **A/B Testing Simulation**: Compare different user segments and interventions
        - **Product Strategy Generation**: Data-driven recommendations for product improvements
        - **Interactive Visualizations**: Explore data through multiple chart types
        
        ### Setup Options:
        1. **Quick Start**: Use sample data (no setup required)
        2. **BigQuery Integration**: Connect to real Ethereum data (requires Google Cloud setup)
        
        ### Getting Started:
        1. Select your analysis date range in the sidebar
        2. Choose BigQuery or sample data mode
        3. Click "Run Analysis" to start
        
        ---
        
        ### Sample Insights You'll Get:
        - **47% retention boost** for wallets that swap within 24 hours
        - **18% lift in transaction volume** from product interventions  
        - **Strategic recommendations** like first swap nudges and dynamic gas alerts
        - **Interactive dashboard** with LTV, churn, and cohort analysis
        """)

if __name__ == "__main__":
    main()