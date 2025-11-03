"""
GPU-Accelerated Multi-Agent Runtime
Orchestrates Planner → Executor → Evaluator workflow
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
from agents import PlannerAgent, ExecutorAgent, EvaluatorAgent


class GPUAgentRuntime:
    """
    Main orchestrator for multi-agent GPU benchmark system
    Coordinates planning, execution, and evaluation phases
    """
    
    def __init__(self, results_dir="results"):
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.evaluator = EvaluatorAgent()
        self.results_dir = results_dir
        self.history = []
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
    
    def run_cycle(self, context: str = "") -> Dict[str, Any]:
        """
        Execute one complete agent cycle:
        Planner → Executor → Evaluator
        """
        print("\n" + " "*30)
        print("Starting Agent Cycle")
        print(" "*30 + "\n")
        
        # Phase 1: Planning
        task = self.planner.plan_task(context)
        
        # Phase 2: Execution
        results = self.executor.execute_task(task)
        
        # Phase 3: Evaluation
        report = self.evaluator.evaluate_results(results)
        
        # Store cycle results
        cycle_data = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "results": results,
            "report": report
        }
        
        self.history.append(cycle_data)
        
        return cycle_data
    
    def run_multi_cycle(self, num_cycles=3):
        """
        Run multiple agent cycles with evolving context
        Each cycle can inform the next
        """
        print("\n" + "="*60)
        print(f"GPU AGENT RUNTIME - Multi-Cycle Execution ({num_cycles} cycles)")
        print("="*60 + "\n")
        
        for i in range(num_cycles):
            print(f"\n{'='*60}")
            print(f"CYCLE {i+1}/{num_cycles}")
            print(f"{'='*60}\n")
            
            # Build context from previous cycles
            if i == 0:
                context = "First benchmark - establish baseline"
            else:
                prev_speedup = self.history[-1]["results"].get("speedup")
                prev_size = self.history[-1]["results"]["size"]
                
                if prev_speedup and prev_speedup < 5:
                    context = f"Previous: {prev_size}x{prev_size} matrix, {prev_speedup:.1f}x speedup. Try larger size for better GPU utilization."
                elif prev_speedup and prev_speedup > 10:
                    context = f"Previous: {prev_size}x{prev_size} matrix, {prev_speedup:.1f}x speedup. Excellent! Try different size to explore performance."
                else:
                    context = f"Previous: {prev_size}x{prev_size} matrix, {prev_speedup:.1f}x speedup. Good performance."
            
            # Run cycle
            self.run_cycle(context)
            
            # Short pause between cycles
            import time
            time.sleep(1)
        
        # Save all results
        self.save_results()
        
        # Generate summary
        self.print_summary()
    
    def save_results(self):
        """Save benchmark history to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"benchmark_log_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n Results saved to: {filepath}")
    
    def print_summary(self):
        """Print overall summary of all cycles"""
        print("\n" + "="*60)
        print("RUNTIME SUMMARY")
        print("="*60 + "\n")
        
        print(f"Total Cycles: {len(self.history)}")
        print(f"Results Directory: {self.results_dir}/")
        print("")
        
        # Summary table
        print(f"{'Cycle':<8} {'Size':<10} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
        print("-" * 60)
        
        for i, cycle in enumerate(self.history, 1):
            res = cycle["results"]
            size = res["size"]
            cpu_time = res["cpu_time_ms"]
            gpu_time = res.get("gpu_time_ms", "N/A")
            speedup = res.get("speedup", "N/A")
            
            if isinstance(gpu_time, (int, float)):
                gpu_time_str = f"{gpu_time:.2f}"
            else:
                gpu_time_str = str(gpu_time)
            
            if isinstance(speedup, (int, float)):
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = str(speedup)
            
            print(f"{i:<8} {size}x{size:<4} {cpu_time:<12.2f} {gpu_time_str:<12} {speedup_str:<10}")
        
        print("")
        
        # Best performance
        valid_results = [c["results"] for c in self.history if c["results"].get("speedup")]
        if valid_results:
            best = max(valid_results, key=lambda x: x["speedup"])
            print(f" Best Performance: {best['speedup']:.2f}x speedup at {best['size']}x{best['size']} matrix")
        
        print("\n" + "="*60 + "\n")


def main():
    """Main entry point"""
    # Create runtime
    runtime = GPUAgentRuntime()
    
    # Run multiple cycles
    runtime.run_multi_cycle(num_cycles=3)
    
  


if __name__ == "__main__":
    main()