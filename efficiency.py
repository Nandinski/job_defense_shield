import json
import gzip
import base64
import pandas as pd


def get_stats_dict(ss64):
    """Convert the base64-encoded summary statistics to JSON."""
    if (not ss64) or pd.isna(ss64) or ss64 == "JS1:Short" or ss64 == "JS1:None":
        return {}
    return json.loads(gzip.decompress(base64.b64decode(ss64[4:])))


def cpu_efficiency(ss, elapsedraw, jobid, cluster, single=False, precision=1, verbose=True):
    """Return a (CPU time used, CPU time allocated, error code)-tuple for a given job.
       If single=True then return a (CPU time used / CPU time allocated, error code)-tuple.
       The error code is needed since the summary statistics (ss) may be malformed."""
    if 'nodes' not in ss:
        if verbose:
            msg = "Warning: nodes not in ss for cpu_efficiency."
            print(msg, jobid, cluster, flush=True)
        error_code = 3
        return (-1, error_code) if single else (-1, -1, error_code)
    total = 0
    total_used = 0
    error_code = 0
    for node in ss['nodes']:
        try:
            used  = ss['nodes'][node]['total_time']
            cores = ss['nodes'][node]['cpus']
        except:
            if verbose:
                msg = "Warning: JSON probably missing keys in cpu_efficiency."
                print(msg, jobid, cluster, flush=True)
            error_code = 1
            return (-1, error_code) if single else (-1, -1, error_code)
        else:
            alloc = elapsedraw * cores  # equal to cputimeraw
            total += alloc
            total_used += used
    if total_used > total:
        error_code = 2
        if verbose:
            msg = "Warning: total_used > total in cpu_efficiency:"
            print(msg, jobid, cluster, total_used, total, flush=True)
    if single:
        return (round(100 * total_used / total, precision), error_code)
    return (total_used, total, error_code)


def gpu_efficiency(ss, elapsedraw, jobid, cluster, single=False, precision=1, verbose=True):
    """Return a (GPU time used, GPU time allocated, error code)-tuple for a given job.
       If single=True then return a (GPU time used / GPU time allocated, error code)-tuple.
       The error code is needed since the summary statistics (ss) may be malformed."""
    if 'nodes' not in ss:
        if verbose:
            msg = "Warning: nodes not in ss for gpu_efficiency."
            print(msg, jobid, cluster, flush=True)
        error_code = 3
        return (-1, error_code) if single else (-1, -1, error_code)
    total = 0
    total_used = 0
    error_code = 0
    for node in ss['nodes']:
        try:
            gpus = list(ss['nodes'][node]['gpu_utilization'].keys())
        except:
            if verbose:
                msg = "Warning: probably missing keys in gpu_efficiency."
                print(msg, jobid, cluster, flush=True)
            error_code = 1
            return (-1, error_code) if single else (-1, -1, error_code)
        else:
            for gpu in gpus:
                util = ss['nodes'][node]['gpu_utilization'][gpu]
                total      += elapsedraw
                total_used += elapsedraw * (float(util) / 100)
    if total_used > total:
        error_code = 2
        if verbose:
            msg = "Warning: total_used > total in gpu_efficiency."
            print(msg, jobid, cluster, total_used, total, flush=True)
    if single:
        return (round(100 * total_used / total, precision), error_code)
    return (total_used, total, error_code)


def cpu_memory_usage(ss, jobid, cluster, precision=0, verbose=True):
    if 'nodes' not in ss:
        if verbose:
            msg = "Warning: nodes not in ss for cpu_memory_usage."
            print(msg, jobid, cluster, flush=True)
        error_code = 2
        return (-1, -1, error_code) 
    total = 0
    total_used = 0
    for node in ss['nodes']:
        try:
            used  = ss['nodes'][node]['used_memory']
            alloc = ss['nodes'][node]['total_memory']
        except:
            if verbose:
                msg = "Warning: used_memory or total_memory not in ss for cpu_memory_usage."
                print(msg, jobid, cluster, flush=True)
            error_code = 1
            return (-1, -1, error_code)
        else:
            total += alloc
            total_used += used
    if total_used > total:
        print("CPU memory usage > 100%:", jobid, cluster, total_used, total, flush=True)
    error_code = 0
    fac = 1024**3
    return (round(total_used / fac, precision), round(total / fac, precision), error_code)
    

def gpu_memory_usage_eff_tuples(ss, jobid, cluster, precision=1, verbose=True):
    """Return a list of tuples for each GPU of the job. Each tuple contains the
       memory used, memory allocated, and GPU utilization. An error code is
       added at the end."""
    if 'nodes' not in ss:
        if verbose:
            msg = "Warning: nodes not in ss for gpu_memory_usage_eff_tuples."
            print(msg, jobid, cluster, flush=True)
        error_code = 2
        return ([], error_code)
    all_gpus = []
    for node in ss['nodes']:
        try:
            used  = ss['nodes'][node]['gpu_used_memory']
            alloc = ss['nodes'][node]['gpu_total_memory']
            util  = ss['nodes'][node]['gpu_utilization']
        except:
            if verbose:
                msg = "Warning: missing key in ss[nodes][node] for gpu_memory_usage_eff_tuples."
                print(msg, jobid, cluster, flush=True)
            error_code = 1
            return ([], error_code)
        else:
            assert sorted(list(used.keys())) == sorted(list(alloc.keys())), "keys do not match"
            for g in used.keys():
                all_gpus.append((round(used[g] / 1024**3, precision),
                                 round(alloc[g] / 1024**3, precision),
                                 float(util[g])))
                if used[g] > alloc[g]:
                    if verbose:
                        print("GPU memory > 100%:", jobid, cluster, used[g], alloc[g], flush=True)
                if util[g] > 100 or util[g] < 0:
                    if verbose:
                        print("GPU util erroneous:", jobid, cluster, util[g], flush=True)
    error_code = 0
    return (all_gpus, error_code)


def max_cpu_memory_used_per_node(ss, jobid, cluster, precision=0, verbose=True):
    """Return the maximum of the used memory per node. The error code is needed
       since the summary statistics (ss) may be malformed."""
    if 'nodes' not in ss:
        if verbose:
            msg = "Warning: nodes not in ss for max_cpu_memory_used_per_node."
            print(msg, jobid, cluster, flush=True)
        error_code = 2
        return (-1, error_code)
    total = 0
    total_used = 0
    error_code = 0
    mem_per_node = []
    for node in ss['nodes']:
        try:
            used  = ss['nodes'][node]['used_memory']
            alloc = ss['nodes'][node]['total_memory']
        except:
            if verbose:
                msg = "Warning: used_memory or total_memory not in ss for max_cpu_memory_used_per_node."
                print(msg, jobid, cluster, flush=True)
            error_code = 1
            return (-1, error_code)
        else:
	    mem_per_node.append(used)
	    if used > alloc:
                if verbose:
                    msg = "Warning: CPU memory used > 100% in max_cpu_memory_used_per_node."
                    print(msg, jobid, cluster, total_used, total, flush=True)
                error_code = 3
    return (round(max(mem_per_node) / 1024**3, precision), error_code)


def num_gpus_with_zero_util(ss):
    """Return the number of GPUs with zero utilization. The error code is needed
       since the summary statistics (ss) may be malformed."""
    if 'nodes' not in ss:
        if verbose:
            msg = "Warning: nodes not in ss for num_gpus_with_zero_util."
            print(msg, jobid, cluster, flush=True)
        error_code = 2
        return (-1, error_code) 
    ct = 0
    for node in ss['nodes']:
        try:
            gpus = list(ss['nodes'][node]['gpu_utilization'].keys())
        except:
            if verbose:
                msg = f"gpu_utilization not found: node is {node} for max_cpu_memory_used_per_node."
                print(msg, jobid, cluster, flush=True)
            error_code = 1
            return (-1, error_code)
        else:
            for gpu in gpus:
                util = ss['nodes'][node]['gpu_utilization'][gpu]
                if float(util) == 0:
                    ct += 1
    error_code = 0
    return (ct, error_code)


def cpu_nodes_with_zero_util(ss, verbose=True):
    """Return the number of nodes with zero CPU utilization. The error code is needed
       since the summary statistics (ss) may be malformed."""
    if 'nodes' not in ss:
        if verbose:
            msg = "Warning: nodes not in ss for cpu_nodes_with_zero_util."
            print(msg, jobid, cluster, flush=True)
        error_code = 2
        return (-1, error_code)
    ct = 0
    for node in ss['nodes']:
        try:
            cpu_time = ss['nodes'][node]['total_time']
        except:
            if verbose:
                msg = f"total_time not found for node {node} in cpu_nodes_with_zero_util."
                print(msg)
            return (-1, error_code)
        else:
            if float(cpu_time) == 0:
                ct += 1
    error_code = 0
    return (ct, error_code)
