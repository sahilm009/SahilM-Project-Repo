#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import warnings
import inspect
warnings.filterwarnings('ignore')

# AG2 imports
from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen.oai import OpenAIWrapper

# MCP-style tool registry
class MCPToolRegistry:
    
    
    def __init__(self):
        self.tools = {}
        self.current_dataset = None
    
    def register_tool(self, name: str, description: str, parameters: Dict):
        """Register a tool with MCP-style metadata"""
        def decorator(func):
            self.tools[name] = {
                "function": func,
                "description": description,
                "parameters": parameters,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters
                    }
                }
            }
            return func
        return decorator
    
    def get_available_tools(self) -> List[Dict]:
        """Get list of available tools for agents"""
        return [tool["schema"] for tool in self.tools.values()]
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool call"""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}
        
        try:
            result = await self.tools[tool_name]["function"](**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
    
    def set_dataset(self, df: pd.DataFrame):
        """Set the current dataset for all tools"""
        self.current_dataset = df

# Initialize tool registry
tool_registry = MCPToolRegistry()

# Register MCP-style data analysis tools
@tool_registry.register_tool(
    name="load_csv",
    description="Load a CSV file and get basic information",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the CSV file"}
        },
        "required": ["file_path"]
    }
)
async def load_csv(file_path: str) -> Dict[str, Any]:
    """Load CSV and return basic info"""
    try:
        df = pd.read_csv(file_path)
        tool_registry.set_dataset(df)
        
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample_data": df.head(3).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
    except Exception as e:
        return {"error": f"Failed to load CSV: {str(e)}"}

@tool_registry.register_tool(
    name="check_missing_values",
    description="Analyze missing values in the dataset",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    }
)
async def check_missing_values() -> Dict[str, Any]:
    """Check for missing values"""
    if tool_registry.current_dataset is None:
        return {"error": "No dataset loaded"}
    
    df = tool_registry.current_dataset
    missing_info = df.isnull().sum()
    missing_percent = (missing_info / len(df)) * 100
    
    return {
        "missing_counts": missing_info.to_dict(),
        "missing_percentages": missing_percent.to_dict(),
        "total_missing": missing_info.sum(),
        "columns_with_missing": missing_info[missing_info > 0].index.tolist()
    }

@tool_registry.register_tool(
    name="analyze_numeric_columns",
    description="Get descriptive statistics for numeric columns",
    parameters={
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific columns to analyze (optional)"
            }
        },
        "required": []
    }
)
async def analyze_numeric_columns(columns: List[str] = None) -> Dict[str, Any]:
    """Analyze numeric columns"""
    if tool_registry.current_dataset is None:
        return {"error": "No dataset loaded"}
    
    df = tool_registry.current_dataset
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if columns:
        numeric_cols = [col for col in columns if col in numeric_cols]
    
    if not numeric_cols:
        return {"error": "No numeric columns found"}
    
    stats = df[numeric_cols].describe().to_dict()
    
    # Additional statistics
    additional_stats = {}
    for col in numeric_cols:
        additional_stats[col] = {
            "skewness": df[col].skew(),
            "kurtosis": df[col].kurtosis(),
            "outliers_iqr": len(detect_outliers_iqr(df[col])),
            "zero_values": (df[col] == 0).sum()
        }
    
    return {
        "descriptive_stats": stats,
        "additional_stats": additional_stats,
        "numeric_columns": numeric_cols
    }

def detect_outliers_iqr(series: pd.Series) -> List[int]:
    """Helper function to detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)].index.tolist()

@tool_registry.register_tool(
    name="analyze_categorical_columns",
    description="Analyze categorical columns and their distributions",
    parameters={
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific columns to analyze (optional)"
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top values to return per column",
                "default": 10
            }
        },
        "required": []
    }
)
async def analyze_categorical_columns(columns: List[str] = None, top_n: int = 10) -> Dict[str, Any]:
    """Analyze categorical columns"""
    if tool_registry.current_dataset is None:
        return {"error": "No dataset loaded"}
    
    df = tool_registry.current_dataset
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if columns:
        cat_cols = [col for col in columns if col in cat_cols]
    
    if not cat_cols:
        return {"error": "No categorical columns found"}
    
    analysis = {}
    for col in cat_cols:
        value_counts = df[col].value_counts()
        analysis[col] = {
            "unique_count": df[col].nunique(),
            "top_values": value_counts.head(top_n).to_dict(),
            "missing_count": df[col].isnull().sum(),
            "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
            "frequency_of_most_common": value_counts.iloc[0] if len(value_counts) > 0 else 0
        }
    
    return {
        "categorical_analysis": analysis,
        "categorical_columns": cat_cols
    }

@tool_registry.register_tool(
    name="calculate_correlations",
    description="Calculate correlation matrix for numeric columns",
    parameters={
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["pearson", "spearman", "kendall"],
                "description": "Correlation method",
                "default": "pearson"
            },
            "threshold": {
                "type": "number",
                "description": "Minimum correlation threshold to report",
                "default": 0.1
            }
        },
        "required": []
    }
)
async def calculate_correlations(method: str = "pearson", threshold: float = 0.1) -> Dict[str, Any]:
    """Calculate correlation matrix"""
    if tool_registry.current_dataset is None:
        return {"error": "No dataset loaded"}
    
    df = tool_registry.current_dataset
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns for correlation"}
    
    corr_matrix = df[numeric_cols].corr(method=method)
    
    # Find high correlations
    high_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_correlations.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": corr_val
                })
    
    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "high_correlations": high_correlations,
        "method": method
    }

@tool_registry.register_tool(
    name="detect_outliers",
    description="Detect outliers in numeric columns using IQR method",
    parameters={
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific columns to analyze (optional)"
            }
        },
        "required": []
    }
)
async def detect_outliers(columns: List[str] = None) -> Dict[str, Any]:
    """Detect outliers using IQR method"""
    if tool_registry.current_dataset is None:
        return {"error": "No dataset loaded"}
    
    df = tool_registry.current_dataset
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if columns:
        numeric_cols = [col for col in columns if col in numeric_cols]
    
    outlier_analysis = {}
    for col in numeric_cols:
        outlier_indices = detect_outliers_iqr(df[col])
        outlier_analysis[col] = {
            "outlier_count": len(outlier_indices),
            "outlier_percentage": (len(outlier_indices) / len(df)) * 100,
            "outlier_values": df.loc[outlier_indices, col].tolist()[:10]  # First 10 outliers
        }
    
    return {
        "outlier_analysis": outlier_analysis,
        "total_outliers": sum(info["outlier_count"] for info in outlier_analysis.values())
    }

@tool_registry.register_tool(
    name="create_visualization",
    description="Create a visualization for the data",
    parameters={
        "type": "object",
        "properties": {
            "plot_type": {
                "type": "string",
                "enum": ["histogram", "boxplot", "scatter", "correlation_heatmap", "missing_heatmap", "bar"],
                "description": "Type of plot to create"
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns to include in the plot"
            },
            "title": {
                "type": "string",
                "description": "Title for the plot"
            }
        },
        "required": ["plot_type"]
    }
)
async def create_visualization(plot_type: str, columns: List[str] = None, title: str = None) -> Dict[str, Any]:
    """Create visualizations"""
    if tool_registry.current_dataset is None:
        return {"error": "No dataset loaded"}
    
    df = tool_registry.current_dataset
    
    # Create plots directory
    Path("Documents/Intro Agent Project/plots").mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    try:
        if plot_type == "histogram" and columns:
            for col in columns:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    plt.hist(df[col].dropna(), alpha=0.7, label=col, bins=30)
            plt.legend()
            plt.title(title or f"Histogram of {', '.join(columns)}")
            
        elif plot_type == "boxplot" and columns:
            numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
            if numeric_cols:
                df[numeric_cols].boxplot()
                plt.title(title or f"Boxplot of {', '.join(numeric_cols)}")
                
        elif plot_type == "correlation_heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
                plt.title(title or "Correlation Heatmap")
                
        elif plot_type == "missing_heatmap":
            if df.isnull().sum().sum() > 0:
                sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
                plt.title(title or "Missing Values Heatmap")
            else:
                return {"message": "No missing values to visualize"}
                
        elif plot_type == "scatter" and columns and len(columns) >= 2:
            x_col, y_col = columns[0], columns[1]
            if x_col in df.columns and y_col in df.columns:
                plt.scatter(df[x_col], df[y_col], alpha=0.6)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(title or f"Scatter plot: {x_col} vs {y_col}")
                
        elif plot_type == "bar" and columns:
            col = columns[0]
            if col in df.columns:
                value_counts = df[col].value_counts().head(15)
                value_counts.plot(kind='bar')
                plt.title(title or f"Distribution of {col}")
                plt.xticks(rotation=45)
        
        filename = f"plots/{plot_type}_{len(list(Path('plots').glob('*.png')))}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {"message": f"Visualization saved to {filename}", "filename": filename}
        
    except Exception as e:
        plt.close()
        return {"error": f"Failed to create visualization: {str(e)}"}

class AgenticAnalyst(ConversableAgent):
    """Agentic data analyst that can call MCP tools"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tool_registry = tool_registry
        
    async def call_tool(self, tool_name: str, **kwargs):
        """Call an MCP tool"""
        result = await self.tool_registry.call_tool(tool_name, **kwargs)
        return result

class OllamaClient:
    
    
    def __init__(self, model_name: str = "qwen2.5", base_url: str = "http://localhost:11434"):
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

class AgenticCSVAnalyzer:
    """Main orchestrator for agentic CSV analysis"""
    
    def __init__(self, model_name: str = "qwen2.5"):
        self.ollama_client = OllamaClient(model_name)
        self.setup_agents()
        
    def setup_agents(self):
        """Setup specialized agentic agents"""
        llm_config = {
            "config_list": self.ollama_client.get_config(),
            "temperature": 0.1,
        }
        
        # Data Explorer Agent - decides what tools to use
        self.explorer = AgenticAnalyst(
            name="DataExplorer",
            system_message="""You are a data exploration specialist. Your role is to:
            1. Start by loading the CSV file using the load_csv tool
            2. Based on initial findings, decide which analysis tools to use next
            3. Use tools like check_missing_values, analyze_numeric_columns, analyze_categorical_columns
            4. Make decisions about what to investigate based on what you discover
            5. Call tools strategically - you don't have to use all tools, only what's needed based on the data, however, create_visualization must be called no matter what
            
            Available tools: load_csv, check_missing_values, analyze_numeric_columns, analyze_categorical_columns, calculate_correlations, detect_outliers, create_visualization
            
            Start with load_csv, then explore based on what you find. Be methodical and explain your reasoning for each tool call.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        
        # Pattern Detective Agent - looks for deeper insights
        self.detective = AgenticAnalyst(
            name="PatternDetective", 
            system_message="""You are a pattern detection specialist. Your role is to:
            1. Review the exploration results from DataExplorer
            2. Look for interesting patterns, correlations, and anomalies
            3. Use correlation analysis and outlier detection tools when needed
            4. Create targeted visualizations to investigate specific findings
            5. Focus on statistical significance and business insights
            
            You can call the same tools as DataExplorer, but focus on deeper analysis based on initial findings.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        
        # Report Synthesizer Agent - creates final insights
        self.synthesizer = ConversableAgent(
            name="ReportSynthesizer",
            system_message="""You are a business intelligence synthesizer. Your role is to:
            1. Review all tool results and agent findings
            2. Synthesize insights into actionable business intelligence
            3. Create a structured report with key findings and recommendations
            4. Translate technical findings into business language
            5. Prioritize the most important insights
            
            Create a comprehensive but concise final report.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
    
    async def analyze_csv(self, csv_path: str) -> str:
        """Run the agentic analysis workflow"""
        print(f"Starting analysis of {csv_path}")
        
        # Create group chat for collaboration
        agents = [self.explorer, self.detective, self.synthesizer]
        
        # Create a simple message exchange system
        messages = []
        
        # Phase 1: Data Explorer analyzes the file
        print("Phase 1: Data exploration...")
        
        exploration_prompt = f"""
        I need you to analyze the CSV file: {csv_path}
        
        Start by using the load_csv tool to load the file and understand its structure.
        Based on what you find, decide which other analysis tools to use. Create visualizations to help understand the data! Use create_visualization tool with appropriate plot types.
        
        Remember to call tools using this format:
        TOOL_CALL: tool_name(parameter1=value1, parameter2=value2)
        
        Be strategic about which tools to use based on your discoveries.
        """
        
        explorer_response = await self.run_agent_with_tools(self.explorer, exploration_prompt)
        messages.append(f"DataExplorer: {explorer_response}")
        
        # Phase 2: Pattern Detective investigates deeper
        print("Phase 2: Pattern detection...")
        
        detective_prompt = f"""
        Based on the initial exploration findings:
        {explorer_response}
        
        Please investigate deeper patterns, correlations, and anomalies.
        Use correlation analysis, outlier detection, and create visualizations as needed.
        Focus on finding interesting insights that the initial exploration might have missed.
        """
        
        detective_response = await self.run_agent_with_tools(self.detective, detective_prompt)
        messages.append(f"PatternDetective: {detective_response}")
        
        # Phase 3: Synthesize final report
        print("Phase 3: Report synthesis...")
        
        synthesis_prompt = f"""
        Please create a comprehensive analysis report based on all findings:
        
        EXPLORATION RESULTS:
        {explorer_response}
        
        PATTERN ANALYSIS:
        {detective_response}
        
        Create a structured business intelligence report with:
        1. Executive Summary
        2. Key Data Characteristics  
        3. Important Patterns and Insights
        4. Data Quality Assessment
        5. Recommendations for Further Analysis
        """
        
        final_report = await self.run_agent_with_tools(self.synthesizer, synthesis_prompt)
        
        return final_report
    
    async def run_agent_with_tools(self, agent: AgenticAnalyst, prompt: str) -> str:
        """Run an agent and handle tool calls"""
        try:
            # Generate initial response
            response = await agent.a_generate_reply(
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Look for tool calls in the response
            if isinstance(response, dict) and "content" in response:
                content = response["content"]
            else:
                content = str(response)
            
            # Process tool calls if found
            tool_results = []
            lines = content.split('\n')
            
            for line in lines:
                if line.strip().startswith('TOOL_CALL:'):
                    tool_call = line.replace('TOOL_CALL:', '').strip()
                    result = await self.execute_tool_call(tool_call)
                    tool_results.append(f"Tool result: {result}")
            
            # Combine original response with tool results
            if tool_results:
                full_response = content + "\n\nTool Results:\n" + "\n".join(tool_results)
            else:
                full_response = content
                
            return full_response
            
        except Exception as e:
            return f"Error running agent: {str(e)}"
    
    async def execute_tool_call(self, tool_call_str: str) -> str:
        """Parse and execute a tool call string"""
        try:
            # Simple parsing for tool calls like: tool_name(param1=value1, param2=value2)
            if '(' in tool_call_str and ')' in tool_call_str:
                tool_name = tool_call_str.split('(')[0].strip()
                params_str = tool_call_str.split('(', 1)[1].rsplit(')', 1)[0]
                
                # Parse parameters
                params = {}
                if params_str.strip():
                    for param in params_str.split(','):
                        if '=' in param:
                            key, value = param.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            # Try to convert to appropriate type
                            try:
                                if value.lower() in ['true', 'false']:
                                    value = value.lower() == 'true'
                                elif value.replace('.', '').isdigit():
                                    value = float(value) if '.' in value else int(value)
                                elif value.startswith('[') and value.endswith(']'):
                                    # Simple list parsing
                                    value = [v.strip().strip('"\'') for v in value[1:-1].split(',')]
                            except:
                                pass  # Keep as string if conversion fails
                            params[key] = value
                
                result = await tool_registry.call_tool(tool_name, **params)
                return json.dumps(result, indent=2, default=str)
            else:
                # Simple tool call without parameters
                tool_name = tool_call_str.strip()
                result = await tool_registry.call_tool(tool_name)
                return json.dumps(result, indent=2, default=str)
                
        except Exception as e:
            return f"Error executing tool call '{tool_call_str}': {str(e)}"

async def main():
    """Main execution function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python agentic_csv_analyzer.py <path_to_csv>")
        print("\nThis is a truly agentic CSV analyzer where:")
        print("- Agents dynamically decide which tools to use")
        print("- MCP-style tools are called based on discoveries")
        print("- Analysis adapts to the specific dataset characteristics")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)
    
    print("Initializing...")
    print("Available Tools:")
    for tool_name, tool_info in tool_registry.tools.items():
        print(f"   - {tool_name}: {tool_info['description']}")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = AgenticCSVAnalyzer()
    
    # Run agentic analysis
    report = await analyzer.analyze_csv(csv_path)
    
    # Save report
    output_file = "agentic_analysis_report.md"
    with open(output_file, 'w') as f:
        f.write(f"# Agentic CSV Analysis Report\n\n")
        f.write(f"**Dataset:** {csv_path}\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now()}\n\n")
        f.write("---\n\n")
        f.write(report)
    
    print(f"\nAnalysis complete")
    print(f"Report saved to: {output_file}")
    print(f"Visualizations saved to: plots/ directory")
    

if __name__ == "__main__":
    # Required packages
    required_packages = [
        "pandas", "numpy", "matplotlib", "seaborn", 
        "pyautogen", "requests", "ollama"
    ]
    

    print("qwen2.5")
    print("Running...")
    print()
    
    asyncio.run(main())