import torch
import json


def main(args):
    with open(args.file_path, "r") as f:
        trace_data = json.load(f)
    top_list  = trace_data['traceEvents']
    
    # Extract all pytorch ops
    pytorch_ops = [event for event in top_list if event.get("name") == ""]
    print(pytorch_ops)
    breakpoint()


    # Top ops by wall time
    # top_wall = events.key_averages().table(
    #     sort_by=args.type,
    #     row_limit=args.limit
    # )
    # print(top_wall)

   
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the profiler trace file"
    )
    parser.add_argument(
        "--type", type=str, default="self_cpu_time_total", help="Type of profiling to display (default: wall_time)"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Number of top entries to display (default: 10)"
    )
    
    args = parser.parse_args()
    main(args)
