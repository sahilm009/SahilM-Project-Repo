"""
Multi-Agent System for GPU Runtime Orchestration
Three agents: Planner (LLM), Executor (Benchmark), Evaluator (Analysis)
"""

import json
import requests
from typing import Dict, Any, Optional
from benchmark import run_benchmark


class PlannerAgent:
    """
    Uses Ollama LLM to plan GPU compute tasks
    Generates task specifications for the Executor
    """
    
    def __init__(self, model="llama3.2", ollama_url="http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/chat"
    
    def plan_task(self, context: str = "") -> Dict[str, Any]:
        """
        Ask LLM to suggest a GPU benchmark task
        Returns task specification
        """
        prompt = f"""You are a GPU computing expert. Suggest ONE specific matrix multiplication benchmark task.

Context: {context if context else "Initial benchmark run"}

Respond in JSON format ONLY with these fields:
{{
    "task_type": "matrix_multiplication",
    "size": <integer between 1024 and 8192>,
    "iterations": <integer between 2 and 5>,
    "reasoning": "<brief explanation>"
}}

Example: {{"task_type": "matrix_multiplication", "size": 2048, "iterations": 3, "reasoning": "Standard size for baseline measurement"}}

Your JSON:"""

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                content = response.json()["message"]["content"]
                
                # Extract JSON from response
                content = content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                task = json.loads(content.strip())
                
                # Validate and set defaults
                task.setdefault("task_type", "matrix_multiplication")
                task.setdefault("size", 2048)
                task.setdefault("iterations", 3)
                task.setdefault("reasoning", "Planned by LLM")
                
                # Ensure values are in valid ranges
                task["size"] = max(1024, min(8192, int(task["size"])))
                task["iterations"] = max(2, min(5, int(task["iterations"])))
                
                print(f"\n Planner Agent Decision:")
                print(f"   Task: {task['task_type']}")
                print(f"   Size: {task['size']}x{task['size']}")
                print(f"   Iterations: {task['iterations']}")
                print(f"   Reasoning: {task['reasoning']}\n")
                
                return task
            else:
                print(f"  Ollama API error: {response.status_code}")
                return self._fallback_plan()
                
        except requests.exceptions.ConnectionError:
            print("  Cannot connect to Ollama. Is it running? (ollama serve)")
            print("   Using fallback planning...")
            return self._fallback_plan()
        except Exception as e:
            print(f"  Planner error: {e}")
            return self._fallback_plan()
    
    def _fallback_plan(self) -> Dict[str, Any]:
        """Fallback plan if Ollama is unavailable"""
        return {
            "task_type": "matrix_multiplication",
            "size": 2048,
            "iterations": 3,
            "reasoning": "Fallback plan (Ollama unavailable)"
        }


class ExecutorAgent:
    """
    Executes GPU benchmarks based on Planner's specifications
    """
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the planned benchmark task
        Returns benchmark results
        """
        print(f"  Executor Agent: Running benchmark...")
        
        if task["task_type"] == "matrix_multiplication":
            results = run_benchmark(
                size=task["size"],
                iterations=task["iterations"]
            )
            results["reasoning"] = task.get("reasoning", "")
            return results
        else:
            raise ValueError(f"Unknown task type: {task['task_type']}")


class EvaluatorAgent:
    """
    Analyzes benchmark results and provides insights
    """
    
    def evaluate_results(self, results: Dict[str, Any]) -> str:
        """
        Analyze results and generate summary report
        """
        print(f"\n Evaluator Agent: Analyzing results...\n")
        
        summary = []
        summary.append("="*60)
        summary.append("PERFORMANCE EVALUATION REPORT")
        summary.append("="*60)
        summary.append("")
        
        # Task info
        summary.append(f"Task: {results['size']}x{results['size']} matrix multiplication")
        summary.append(f"Iterations: {results['iterations']}")
        if results.get('reasoning'):
            summary.append(f"Planning: {results['reasoning']}")
        summary.append("")
        
        # Results
        summary.append("Results:")
        summary.append(f"  CPU Time:  {results['cpu_time_ms']:>8.2f} ms")
        
        if results['gpu_time_ms']:
            summary.append(f"  GPU Time:  {results['gpu_time_ms']:>8.2f} ms ({results.get('gpu_backend', 'Unknown')})")
            summary.append(f"  Speedup:   {results['speedup']:>8.2f}x")
            summary.append("")
            
            # Analysis
            speedup = results['speedup']
            if speedup > 10:
                analysis = " Excellent GPU utilization! High parallelism achieved."
            elif speedup > 5:
                analysis = " Good GPU performance. Consider larger matrices for more speedup."
            elif speedup > 2:
                analysis = " Moderate speedup. GPU overhead visible. Try larger workloads."
            else:
                analysis = "  Limited speedup. CPU competitive at this size."
            
            summary.append(f"Analysis: {analysis}")
            summary.append("")
            
            # Recommendations
            summary.append("Recommendations:")
            if speedup < 5:
                summary.append("  • Increase matrix size for better GPU utilization")
                summary.append("  • Consider batch processing multiple smaller matrices")
            else:
                summary.append("  • Explore kernel fusion opportunities")
                summary.append("  • Profile with torch.profiler for optimization insights")
            
        else:
            summary.append("  GPU Time:  N/A (GPU not available)")
            summary.append("")
            summary.append("Analysis: GPU benchmarking unavailable")
        
        summary.append("")
        summary.append("="*60)
        
        report = "\n".join(summary)
        print(report)
        
        return report


if __name__ == "__main__":
    # Test agents individually
    print("Testing Agent System...\n")
    
    # Test Planner
    planner = PlannerAgent()
    task = planner.plan_task()
    print(f"Planned task: {task}\n")
    
    # Test Executor
    executor = ExecutorAgent()
    results = executor.execute_task(task)
    print(f"Results: {results}\n")
    
    # Test Evaluator
    evaluator = EvaluatorAgent()
    report = evaluator.evaluate_results(results)