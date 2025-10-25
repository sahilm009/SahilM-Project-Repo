#!/usr/bin/env python3
"""
CSV Analysis Agent using AG2, Ollama, and MCP
Performs EDA and generates data summaries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# AG2 imports
from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen.oai import OpenAIWrapper

# MCP client setup (you'll need to install mcp package)
try:
    from mcp import ClientSession, stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("MCP not available. Install with: pip install mcp")
    MCP_AVAILABLE = False

class OllamaClient:
    """Custom Ollama client for AG2 integration"""
    
    def __init__(self, model_name: str = "qwen3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.config = {
            "model": model_name,
            "base_url": base_url,
            "api_type": "ollama",
            "stream": False,
            "temperature": 0.1,
        }
    
    def get_config(self):
        return [self.config]

class CSVAnalyzer:
    """Core CSV analysis functionality"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load and validate CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
            return True
        except Exception as e:
            print(f"✗ Error loading CSV: {e}")
            return False
    
    def basic_info(self) -> Dict[str, Any]:
        """Get basic dataset information"""
        if self.df is None:
            return {}
        
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "memory_usage": self.df.memory_usage(deep=True).sum(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": self.df.duplicated().sum()
        }
        
        # Identify column types
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        info.update({
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols
        })
        
        return info
    
    def descriptive_stats(self) -> Dict[str, Any]:
        """Generate descriptive statistics"""
        if self.df is None:
            return {}
        
        stats = {}
        
        # Numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = self.df[numeric_cols].describe().to_dict()
            
            # Additional stats
            stats["correlation_matrix"] = self.df[numeric_cols].corr().to_dict()
            stats["skewness"] = self.df[numeric_cols].skew().to_dict()
            stats["kurtosis"] = self.df[numeric_cols].kurtosis().to_dict()
        
        # Categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            stats["categorical_summary"] = {}
            for col in categorical_cols:
                stats["categorical_summary"][col] = {
                    "unique_count": self.df[col].nunique(),
                    "top_values": self.df[col].value_counts().head(10).to_dict(),
                    "missing_count": self.df[col].isnull().sum()
                }
        
        return stats
    
    def generate_visualizations(self, output_dir: str = "plots"):
        """Generate EDA visualizations"""
        if self.df is None:
            return
        
        Path(output_dir).mkdir(exist_ok=True)
        plt.style.use('default')
        
        # 1. Missing values heatmap
        if self.df.isnull().sum().sum() > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/missing_values.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Numeric columns distribution
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_cols = min(len(numeric_cols), 4)
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    self.df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/numeric_distributions.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Correlation matrix
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            if self.df[col].nunique() <= 20:  # Only plot if reasonable number of categories
                plt.figure(figsize=(12, 6))
                value_counts = self.df[col].value_counts().head(15)
                value_counts.plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/categorical_{col}.png', dpi=150, bbox_inches='tight')
                plt.close()
    
    def perform_full_analysis(self) -> Dict[str, Any]:
        """Perform complete analysis"""
        if not self.load_data():
            return {}
        
        self.analysis_results = {
            "basic_info": self.basic_info(),
            "descriptive_stats": self.descriptive_stats()
        }
        
        # Generate visualizations
        self.generate_visualizations()
        
        return self.analysis_results

class DataAnalysisAgent:
    """Main agent orchestrating the analysis"""
    
    def __init__(self, model_name: str = "qwen3"):
        self.ollama_client = OllamaClient(model_name)
        self.analyzer = None
        self.setup_agents()
    
    def setup_agents(self):
        """Setup AG2 agents"""
        llm_config = {
            "config_list": self.ollama_client.get_config(),
            "temperature": 0.1,
        }
        
        # Data Analyst Agent
        self.data_analyst = ConversableAgent(
            name="DataAnalyst",
            system_message="""You are an expert data analyst. Your role is to:
            1. Analyze the provided dataset statistics and information
            2. Identify key patterns, trends, and anomalies
            3. Provide insights about data quality issues
            4. Suggest areas for further investigation
            
            Be thorough but concise in your analysis.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        
        # Report Writer Agent
        self.report_writer = ConversableAgent(
            name="ReportWriter",
            system_message="""You are a technical writer specializing in data analysis reports. Your role is to:
            1. Create clear, professional summaries of data analysis results
            2. Translate technical findings into business-friendly language
            3. Structure information logically and highlight key insights
            4. Provide actionable recommendations
            
            Write in a clear, professional tone suitable for stakeholders.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        
        # Coordinator Agent
        self.coordinator = ConversableAgent(
            name="Coordinator",
            system_message="""You are a project coordinator managing the data analysis workflow. Your role is to:
            1. Coordinate between the DataAnalyst and ReportWriter
            2. Ensure all aspects of the analysis are covered
            3. Provide final quality review
            4. Format the final output
            
            Keep the process organized and efficient.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
    
    async def analyze_csv(self, csv_path: str) -> str:
        """Main analysis workflow"""
        print(f"🔍 Starting analysis of {csv_path}")
        
        # Initialize analyzer
        self.analyzer = CSVAnalyzer(csv_path)
        
        # Perform analysis
        results = self.analyzer.perform_full_analysis()
        
        if not results:
            return "❌ Failed to analyze CSV file"
        
        # Format results for LLM
        analysis_summary = self.format_analysis_for_llm(results)
        
        # Create group chat for collaborative analysis
        group_chat = GroupChat(
            agents=[self.data_analyst, self.report_writer, self.coordinator],
            messages=[],
            max_round=6,
            speaker_selection_method="round_robin"
        )
        
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config={"config_list": self.ollama_client.get_config()}
        )
        
        # Start the analysis discussion
        initial_message = f"""
        Please analyze this CSV dataset and provide a comprehensive report.
        
        DATASET ANALYSIS RESULTS:
        {analysis_summary}
        
        TASK: Create a professional data analysis report that includes:
        1. Executive Summary
        2. Data Overview
        3. Key Findings
        4. Data Quality Assessment
        5. Recommendations
        
        Visualizations have been saved to the 'plots' directory.
        """
        
        try:
            # Run the group chat
            result = await self.coordinator.a_initiate_chat(
                manager,
                message=initial_message,
                max_turns=3
            )
            
            return self.extract_final_report(result)
        
        except Exception as e:
            return f"❌ Error during agent analysis: {e}"
    
    def format_analysis_for_llm(self, results: Dict[str, Any]) -> str:
        """Format analysis results for LLM consumption"""
        basic_info = results.get("basic_info", {})
        desc_stats = results.get("descriptive_stats", {})
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                return obj
        
        summary = f"""
        BASIC DATASET INFO:
        - Shape: {basic_info.get('shape', 'Unknown')}
        - Columns: {', '.join(basic_info.get('columns', []))}
        - Missing Values: {convert_numpy_types(basic_info.get('missing_values', {}))}
        - Duplicate Rows: {convert_numpy_types(basic_info.get('duplicate_rows', 0))}
        - Numeric Columns: {basic_info.get('numeric_columns', [])}
        - Categorical Columns: {basic_info.get('categorical_columns', [])}
        
        STATISTICAL SUMMARY:
        """
        
        if 'numeric_summary' in desc_stats:
            converted_numeric = convert_numpy_types(desc_stats['numeric_summary'])
            summary += f"\nNumeric Statistics:\n{json.dumps(converted_numeric, indent=2, default=str)}"
        
        if 'categorical_summary' in desc_stats:
            converted_categorical = convert_numpy_types(desc_stats['categorical_summary'])
            summary += f"\nCategorical Summary:\n{json.dumps(converted_categorical, indent=2, default=str)}"
        
        return summary
    
    def extract_final_report(self, chat_result) -> str:
        """Extract the final report from chat results"""
        if hasattr(chat_result, 'chat_history'):
            # Get the last few messages which should contain the final report
            messages = chat_result.chat_history[-3:]
            report_parts = []
            
            for msg in messages:
                if hasattr(msg, 'content'):
                    report_parts.append(msg.content)
            
            return "\n\n".join(report_parts)
        else:
            return str(chat_result)

async def main():
    """Main execution function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python csv_analysis_agent.py <path_to_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)
    
    print("🤖 Initializing CSV Analysis Agent...")
    print("📋 Components: AG2 + Ollama + MCP")
    print("=" * 50)
    
    # Initialize agent
    agent = DataAnalysisAgent()
    
    # Run analysis
    report = await agent.analyze_csv(csv_path)
    
    # Save report
    output_file = "analysis_report.md"
    with open(output_file, 'w') as f:
        f.write(f"# CSV Analysis Report\n\n")
        f.write(f"**Dataset:** {csv_path}\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now()}\n\n")
        f.write("---\n\n")
        f.write(report)
    
    print(f"✅ Analysis complete! Report saved to {output_file}")
    print(f"📊 Visualizations saved to 'plots' directory")

if __name__ == "__main__":
    # Required packages
    required_packages = [
        "pandas", "numpy", "matplotlib", "seaborn", 
        "pyautogen", "requests", "ollama", "fix-busted-json"
    ]
    
    print("📦 Required packages:", ", ".join(required_packages))
    print("🚀 Install with: pip install " + " ".join(required_packages))
    print("🦙 Ollama should already be running (you'll see a port binding error if it is)")
    print("🔧 Using model: qwen3")
    print()
    
    asyncio.run(main())