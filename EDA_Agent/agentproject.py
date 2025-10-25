#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import subprocess
import sys

# Data Explorer (same as before but simplified)
class DataExplorer:
    def __init__(self):
        self.current_df = None
        self.file_path = None
    
    def load_file(self, file_path: str) -> str:
        """Load dataset from file"""
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"
        
        try:
            if path.suffix.lower() == '.csv':
                self.current_df = pd.read_csv(file_path)
            elif path.suffix.lower() in ['.json', '.jsonl']:
                self.current_df = pd.read_json(file_path, lines=path.suffix.lower()=='.jsonl')
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                self.current_df = pd.read_excel(file_path)
            elif path.suffix.lower() == '.parquet':
                self.current_df = pd.read_parquet(file_path)
            else:
                self.current_df = pd.read_csv(file_path)
            
            self.file_path = file_path
            return f"✅ Loaded {path.name}: {self.current_df.shape[0]} rows × {self.current_df.shape[1]} columns"
        except Exception as e:
            return f"❌ Error loading file: {str(e)}"
    
    def overview(self) -> str:
        """Get dataset overview"""
        if self.current_df is None:
            return "❌ No dataset loaded"
        
        df = self.current_df
        missing_summary = df.isnull().sum()
        missing_cols = [col for col in missing_summary.index if missing_summary[col] > 0]
        
        overview = f"""📊 Dataset Overview:
• File: {Path(self.file_path).name if self.file_path else 'Unknown'}
• Shape: {df.shape[0]} rows × {df.shape[1]} columns
• Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
• Columns: {', '.join(df.columns.tolist())}
• Data Types: {dict(df.dtypes.value_counts())} 
• Missing Data: {len(missing_cols)} columns have missing values
"""
        
        if missing_cols:
            overview += f"• Columns with missing data: {', '.join(missing_cols[:5])}"
            if len(missing_cols) > 5:
                overview += f" and {len(missing_cols) - 5} more"
        
        return overview
    
    def stats(self, column: Optional[str] = None) -> str:
        """Get statistics"""
        if self.current_df is None:
            return "❌ No dataset loaded"
        
        df = self.current_df
        
        if column:
            if column not in df.columns:
                return f"❌ Column '{column}' not found"
            
            col_data = df[column]
            if pd.api.types.is_numeric_dtype(col_data):
                stats = col_data.describe()
                missing = col_data.isnull().sum()
                return f"""📈 Statistics for '{column}':
• Count: {stats['count']:.0f} | Missing: {missing}
• Mean: {stats['mean']:.2f} | Std: {stats['std']:.2f}
• Min: {stats['min']:.2f} | Max: {stats['max']:.2f}
• Q1: {stats['25%']:.2f} | Median: {stats['50%']:.2f} | Q3: {stats['75%']:.2f}"""
            else:
                unique_vals = col_data.nunique()
                missing = col_data.isnull().sum()
                top_vals = col_data.value_counts().head(3)
                return f"""📝 Summary for '{column}' (categorical):
• Unique values: {unique_vals} | Missing: {missing}
• Top values: {', '.join([f'{k}({v})' for k, v in top_vals.items()])}"""
        
        # All numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return "❌ No numeric columns found"
        
        stats_summary = f"📊 Numeric Columns Summary ({len(numeric_df.columns)} columns):\n"
        for col in numeric_df.columns:
            data = numeric_df[col]
            stats_summary += f"• {col}: mean={data.mean():.2f}, std={data.std():.2f}, missing={data.isnull().sum()}\n"
        
        return stats_summary.strip()
    
    def head(self, n: int = 5) -> str:
        """Show first n rows"""
        if self.current_df is None:
            return "❌ No dataset loaded"
        return f"📋 First {n} rows:\n{self.current_df.head(n).to_string()}"
    
    def search(self, query: str, column: Optional[str] = None) -> str:
        """Search for values"""
        if self.current_df is None:
            return "❌ No dataset loaded"
        
        df = self.current_df
        
        if column:
            if column not in df.columns:
                return f"❌ Column '{column}' not found"
            mask = df[column].astype(str).str.contains(str(query), case=False, na=False)
            results = df[mask]
        else:
            string_cols = df.select_dtypes(include=['object']).columns
            mask = pd.Series([False] * len(df))
            for col in string_cols:
                mask |= df[col].astype(str).str.contains(str(query), case=False, na=False)
            results = df[mask]
        
        if len(results) == 0:
            return f"🔍 No matches found for '{query}'"
        
        return f"🔍 Found {len(results)} matches for '{query}':\n{results.head(10).to_string()}"
    
    def correlations(self, threshold: float = 0.5) -> str:
        """Find correlations"""
        if self.current_df is None:
            return "❌ No dataset loaded"
        
        numeric_df = self.current_df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return "❌ Need at least 2 numeric columns for correlation"
        
        corr_matrix = numeric_df.corr()
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if not high_corr:
            return f"📊 No correlations above {threshold} found"
        
        result = f"📊 High correlations (≥{threshold}):\n"
        for col1, col2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            result += f"• {col1} ↔ {col2}: {corr:.3f}\n"
        
        return result.strip()


# Simple function-based tool for AG2/AutoGen
class SimpleTool:
    """Wrapper for functions to be used as tools"""
    def __init__(self, func: Callable, name: str, description: str):
        self.func = func
        self.name = name
        self.description = description
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# AG2 Integration with multiple import attempts
AG2_AVAILABLE = False
AG2_IMPORT_ERROR = None

# Try different import paths for AG2/AutoGen
try:
    # Try AG2 first
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.models import OllamaModel
    from autogen_agentchat.messages import TextMessage
    AG2_AVAILABLE = True
    AG2_VERSION = "AG2"
except ImportError as e1:
    try:
        # Try Microsoft AutoGen
        from autogen import AssistantAgent, GroupChat, GroupChatManager
        from autogen.oai import OllamaClient
        AG2_AVAILABLE = True
        AG2_VERSION = "AutoGen"
    except ImportError as e2:
        try:
            # Try older autogen-agentchat
            import autogen
            AG2_AVAILABLE = True
            AG2_VERSION = "AutoGen_Legacy"
        except ImportError as e3:
            AG2_AVAILABLE = False
            AG2_IMPORT_ERROR = f"AG2: {e1}\nAutoGen: {e2}\nLegacy: {e3}"


class DataAnalysisTeam:
    """AG2/AutoGen team for data analysis with Ollama"""
    
    def __init__(self, model_name: str = "llama3.2"):
        if not AG2_AVAILABLE:
            raise ImportError(f"AG2/AutoGen not available. Errors:\n{AG2_IMPORT_ERROR}")
        
        self.explorer = DataExplorer()
        self.model_name = model_name
        self.version = AG2_VERSION
        
        # Check if Ollama is running
        self._check_ollama()
        
        # Create tools from explorer methods
        self.tools = self._create_tools()
        
        # Initialize based on available version
        if self.version == "AG2":
            self._init_ag2()
        elif self.version == "AutoGen":
            self._init_autogen()
        else:
            self._init_legacy()
    
    def _check_ollama(self):
        """Check if Ollama is running"""
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise ConnectionError("Ollama not accessible")
        except Exception:
            print("⚠️  Warning: Ollama might not be running. Start it with: ollama serve")
            print(f"⚠️  Make sure model '{self.model_name}' is available: ollama pull {self.model_name}")
    
    def _create_tools(self) -> List[SimpleTool]:
        """Create tools from explorer methods"""
        tools = []
        
        # Load dataset tool
        def load_dataset(file_path: str) -> str:
            """Load a dataset from file (CSV, JSON, Excel, Parquet, TSV)"""
            return self.explorer.load_file(file_path)
        
        tools.append(SimpleTool(load_dataset, "load_dataset", "Load dataset from file"))
        
        # Overview tool  
        def get_overview() -> str:
            """Get overview of the currently loaded dataset"""
            return self.explorer.overview()
        
        tools.append(SimpleTool(get_overview, "get_overview", "Get dataset overview"))
        
        # Statistics tool
        def get_statistics(column: Optional[str] = None) -> str:
            """Get statistical summary for numeric columns or specific column"""
            return self.explorer.stats(column)
        
        tools.append(SimpleTool(get_statistics, "get_statistics", "Get statistical analysis"))
        
        # Head tool
        def show_data(n: int = 5) -> str:
            """Show first n rows of the dataset"""
            return self.explorer.head(n)
        
        tools.append(SimpleTool(show_data, "show_data", "Preview dataset rows"))
        
        # Search tool
        def search_data(query: str, column: Optional[str] = None) -> str:
            """Search for specific values in the dataset"""
            return self.explorer.search(query, column)
        
        tools.append(SimpleTool(search_data, "search_data", "Search dataset"))
        
        # Correlation tool
        def find_correlations(threshold: float = 0.5) -> str:
            """Find correlations between numeric columns above threshold"""
            return self.explorer.correlations(threshold)
        
        tools.append(SimpleTool(find_correlations, "find_correlations", "Find correlations"))
        
        return tools
    
    def _init_ag2(self):
        """Initialize AG2 version"""
        from autogen_agentchat.models import OllamaModel
        
        # Create Ollama model
        self.model = OllamaModel(
            model=self.model_name,
            base_url="http://localhost:11434"
        )
        
        # Convert tools to AG2 format
        ag2_tools = []
        for tool in self.tools:
            ag2_tools.append(tool.func)  # AG2 can work with plain functions
        
        # Create agents
        self.data_analyst = AssistantAgent(
            name="DataAnalyst",
            model_client=self.model,
            tools=ag2_tools,
            system_message="""You are a data analyst expert. You help users explore and analyze datasets.
            
Your capabilities:
- Load datasets from files (CSV, JSON, Excel, Parquet)
- Provide dataset overviews and summaries
- Generate statistical analyses
- Search for specific data
- Find correlations between variables
- Show data previews

Always use the available tools to explore data. Be thorough but concise in your analysis.
When presenting results, highlight key insights and patterns you discover."""
        )
        
        self.consultant = AssistantAgent(
            name="DataConsultant", 
            model_client=self.model,
            system_message="""You are a data science consultant. You provide strategic insights and recommendations based on data analysis.

Your role:
- Interpret analysis results from the DataAnalyst
- Suggest follow-up questions and deeper analysis
- Provide business insights and recommendations
- Help prioritize which aspects of the data are most important
- Suggest data quality improvements or additional data needs

You don't directly use tools - instead you guide the analysis and interpret results."""
        )
        
        # Create team
        from autogen_agentchat.teams import RoundRobinGroupChat
        self.team = RoundRobinGroupChat([self.data_analyst, self.consultant])
    
    def _init_autogen(self):
        """Initialize Microsoft AutoGen version"""
        print("⚠️  AutoGen version detected - basic implementation")
        # Basic implementation for AutoGen
        self.team = None
    
    def _init_legacy(self):
        """Initialize legacy AutoGen version"""
        print("⚠️  Legacy AutoGen version detected - basic implementation")
        self.team = None
    
    async def analyze(self, user_request: str, max_turns: int = 10) -> str:
        """Run data analysis based on user request"""
        if self.version != "AG2" or self.team is None:
            return self._simple_analyze(user_request)
        
        try:
            # AG2 stream conversation
            result_messages = []
            
            async for message in self.team.run_stream(
                task=user_request,
                max_turns=max_turns
            ):
                print(f"\n--- {message.source} ---")
                print(message.content)
                result_messages.append(message)
            
            return "Analysis completed. Check the conversation above for detailed results."
            
        except Exception as e:
            return f"❌ Error during analysis: {str(e)}"
    
    def _simple_analyze(self, request: str) -> str:
        """Fallback analysis without full AG2"""
        print(f"🤖 Analyzing with {self.version}: {request}")
        
        # Simple keyword-based analysis
        request_lower = request.lower()
        results = []
        
        if "load" in request_lower:
            # Try to extract file path
            words = request.split()
            for word in words:
                if word.endswith(('.csv', '.json', '.xlsx', '.parquet')):
                    results.append(self.explorer.load_file(word))
                    break
            if not results:
                results.append("Please specify a file path to load")
        
        if "overview" in request_lower or "summary" in request_lower:
            results.append(self.explorer.overview())
        
        if "stats" in request_lower or "statistics" in request_lower:
            results.append(self.explorer.stats())
        
        if "correlation" in request_lower:
            results.append(self.explorer.correlations())
        
        if "head" in request_lower or "preview" in request_lower:
            results.append(self.explorer.head())
        
        if not results:
            results.append("Try commands like: load file.csv, overview, stats, correlations, head")
        
        return "\n\n".join(results)


# Simple fallback without AG2
class SimpleDataAnalyst:
    """Fallback analyst without AG2"""
    
    def __init__(self):
        self.explorer = DataExplorer()
    
    def analyze(self, request: str) -> str:
        """Simple analysis without LLM"""
        request_lower = request.lower()
        
        if "load" in request_lower and ("csv" in request_lower or "file" in request_lower):
            # Try to extract file path
            words = request.split()
            for word in words:
                if word.endswith(('.csv', '.json', '.xlsx', '.parquet')):
                    return self.explorer.load_file(word)
            return "Please specify a file path to load"
        
        elif "overview" in request_lower or "summary" in request_lower:
            return self.explorer.overview()
        
        elif "stats" in request_lower or "statistics" in request_lower:
            return self.explorer.stats()
        
        elif "head" in request_lower or "preview" in request_lower:
            return self.explorer.head()
        
        elif "correlation" in request_lower:
            return self.explorer.correlations()
        
        elif "search" in request_lower:
            # Extract search term (very basic)
            parts = request.split()
            if len(parts) > 1:
                query = parts[-1]  # Use last word as query
                return self.explorer.search(query)
            return "Please specify what to search for"
        
        else:
            return """Available commands:
• load [file.csv] - Load dataset
• overview - Dataset summary  
• stats - Statistical analysis
• head - Show first rows
• correlations - Find correlations
• search [term] - Search data"""


def main():
    print("🚀 AG2/AutoGen + Ollama Data Analysis Setup")
    
    # Check for AG2/AutoGen
    if AG2_AVAILABLE:
        print(f"✅ {AG2_VERSION} available")
        
        # Ask for model
        model = input("Enter Ollama model name (default: llama3.2): ").strip()
        if not model:
            model = "llama3.2"
        
        try:
            analyst = DataAnalysisTeam(model_name=model)
            print(f"✅ Created {AG2_VERSION} team with {model}")
            
            print("\n📋 Usage examples:")
            print("• Load data: 'Load the sales_data.csv file and give me an overview'")
            print("• Analyze: 'What are the key patterns in this dataset?'")
            print("• Explore: 'Find correlations and show me the most interesting insights'")
            
            while True:
                try:
                    user_input = input("\n🔍 What would you like to analyze? (or 'quit'): ").strip()
                    
                    if user_input.lower() == 'quit':
                        break
                    
                    if user_input:
                        print("\n" + "="*50)
                        if AG2_VERSION == "AG2":
                            result = asyncio.run(analyst.analyze(user_input))
                        else:
                            result = analyst._simple_analyze(user_input)
                        print("\n" + "="*50)
                        print(f"📊 Summary: {result}")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
        except Exception as e:
            print(f"❌ {AG2_VERSION} setup failed: {e}")
            print("🔄 Falling back to simple mode...")
            simple_mode()
    
    else:
        print(f"❌ AG2/AutoGen not found. Install with:")
        print("  pip install autogen-agentchat  # for AG2")
        print("  pip install pyautogen         # for AutoGen")
        print(f"❌ Import errors:\n{AG2_IMPORT_ERROR}")
        print("🔄 Running in simple mode...")
        simple_mode()


def simple_mode():
    """Simple mode without AG2"""
    analyst = SimpleDataAnalyst()
    
    print("\n📋 Simple commands:")
    print("• load file.csv")
    print("• overview") 
    print("• stats")
    print("• head")
    print("• correlations")
    print("• search term")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input:
                result = analyst.analyze(user_input)
                print(result)
                
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()