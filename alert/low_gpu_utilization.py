import os
import subprocess
import textwrap
import pickle
from time import sleep
from base import Alert
from utils import SECONDS_PER_MINUTE
from utils import SECONDS_PER_HOUR
from utils import MINUTES_PER_HOUR
from utils import get_first_name
from utils import get_email 
from utils import send_email_from_cmd
from efficiency import gpus_with_low_util

class LowGpuUtilization(Alert):
    
    """Send warnings and cancel jobs with low GPU utilization. Interactive jobs
       with only 1 GPU and under a configurable run time limit are ignored."""

    jobstatsMod = None
    
    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        if LowGpuUtilization.jobstatsMod == None:
            import importlib.machinery
            import importlib.util
            import sys
            jobstats_dir = '/data/secure/SLURM/utilities/jobstats'
            loader = importlib.machinery.SourceFileLoader('jobstats', os.path.join(jobstats_dir, 'jobstats.py'))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, jobstats_dir)
            loader.exec_module(mod)
            sys.path.remove(jobstats_dir)
            LowGpuUtilization.jobstatsMod = mod
            
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)
        
    @staticmethod
    def get_stats_for_running_job(jobid, cluster):
        """Get the job statistics for running jobs by calling jobstats"""
            
        stats = LowGpuUtilization.jobstatsMod.JobStats(jobid=jobid, cluster=cluster, prom_server='http://mcmgt01:9090')
        return eval(stats.report_job_json(False))

    def _filter_and_add_new_fields(self):
        lower = self.first_warning_minutes * SECONDS_PER_MINUTE
        self.jb = self.df[(self.df.state == "RUNNING") &
                          (self.df.gpus > 0) &
                          self.df.cluster.isin(self.clusters) &
                          self.df.partition.isin(self.partitions) &
                          (self.df.elapsedraw >= lower) &
                          (self.df["limit-minutes"] > self.cancel_minutes) &
                          (~self.df.netid.isin(self.excluded_users))].copy()
        self.jb.rename(columns={"netid":"NetID"}, inplace=True)
        
        if not self.jb.empty:
            # interactive jobs
            self.jb["interactive"] = self.jb["submitline"].apply(lambda x: True if not x.startswith("sbatch") else False)
            msk = self.jb["interactive"] & (self.jb.gpus == 1) & (self.jb["limit-minutes"] <= self.max_interactive_hours * MINUTES_PER_HOUR)
            self.jb = self.jb[~msk]

            self.jb["jobstats"] = self.jb.apply(lambda row: 
                                                       LowGpuUtilization.get_stats_for_running_job(row["jobid"], row["cluster"]), axis='columns')
            
            self.jb["eval_time"] = self.jb.apply(lambda row: row["elapsedraw"] / SECONDS_PER_HOUR - self.max_interactive_hours if row["interactive"] \
                                                        else row["elapsedraw"] / SECONDS_PER_HOUR, axis='columns')      
            # Step size is configured in jobstats in hours
            steps = [1,2]
            self.jb["GPU-lu-tpl"] = self.jb.apply(lambda row: 
                                                  gpus_with_low_util(row["jobstats"], row["eval_time"], row["jobid"], row["cluster"], low_util=self.low_gpu_util, steps=steps), axis='columns')
            self.jb["error-code"] = self.jb["GPU-lu-tpl"].apply(lambda tpl: tpl[1])
            self.jb = self.jb[self.jb["error-code"] == 0]
            self.jb["GPU-lu"] = self.jb["GPU-lu-tpl"].apply(lambda tpl: tpl[0])
            def add_util_step_cols(row):
                low_gpu_util_steps = row["GPU-lu"]
                row["Max-Low-GPU-step"] = len(low_gpu_util_steps)
                for step, lu_step in low_gpu_util_steps.items():
                    gpus_low_util = []
                    for node in lu_step:
                        gpus_util = [f"{gpu[0]}:{gpu[1]:>2}%" for gpu in lu_step[node]]
                        gpus_low_util.append( node + "[" + ", ".join(gpus_util)  + "]" )
                    row[f"Low-GPU-{step}h"] = ",".join(gpus_low_util)
                return row
            self.jb = self.jb.apply(add_util_step_cols, axis='columns') 
            self.jb = self.jb[self.jb["Max-Low-GPU-step"] > 0]
            
            self.jb = self.jb[["jobid", "jobname", "NetID", "cluster", "partition", "gpus", "elapsedraw", "Max-Low-GPU-step"] + [f"Low-GPU-{step}h" for step in reversed(steps)]]
            renamings = {"jobid":"JobID", "jobname":"JobName", "netid":"NetID", "cluster":"Cluster", "partition":"Partition", "gpus":"GPUs-Allocated"}
            self.jb.rename(columns=renamings, inplace=True)

    def send_emails_to_users(self):
        email_cols = []

        for user in self.jb.NetID.unique():
            user_mail = get_email(user)
            #################
            #### warning ####
            #################
            usr = self.jb[(self.jb["Max-Low-GPU-step"] == 1) &
                          (self.jb.NetID == user)].copy()
            print("Warning jobs:")
            print(usr)
            if not usr.empty:
                s = f"{get_first_name(user)},\n\n"
                s += f'You have running GPU job(s) that appear to have GPUs with less than {self.low_gpu_util}% GPU utilization in the last hour:'
                s += "\n\n"

                usr["State"] = "WARNING"
                usr = usr[["JobID", "JobName", "Partition", "State", "Low-GPU-1h"]]

                usr_str = usr.to_string(index=False, justify="center")
                s += "\n".join([5 * " " + row for row in usr_str.split("\n")])
                s += "\n"

                s += textwrap.dedent(f"""
                Jobs will be AUTOMATICALLY CANCELED if they have GPUs with low utilization for two consecutive hours.
                For more information see the ROCS testbed's <a href="https://sands.kaust.edu.sa/internal/rocs-testbed/slurm-environment/#automatic-job-termination">Termination Policies</a>.
                """)

                s += 'You can check your job\'s resource utilization on Grafana by opening the link from the command:'
                s += "\n\n"
                s += f"     $ jobstats {usr.JobID.values[0]} -g"
                s += "\n"
                
                s += textwrap.dedent("""
                Perhaps the GPUs being used are too powerful for the code that is running. Consider using an older GPU.
                See Princeton's <a href="https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#Low-GPU-Utilization--Potential-Solutions">GPU Computing</a> webpage for tips on how to improve GPU utilization.
                """)
                
                s += "\n"
                s += f'If you\'re no longer using the GPUs, consider canceling the job(s) above using the "scancel" command:'
                s += "\n\n"
                s += f"     $ scancel {usr.JobID.values[0]}"
                s += "\n\n"

                send_email_from_cmd(s, user_mail, subject=f"{self.subject} - WARNING", share_w=self.admin_emails)
                print(s)

                # append the new violations to the log file
                vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
                Alert.update_violation_log(usr, vfile)

            ###############
            # cancel jobs #
            ###############
            usr = self.jb[(self.jb["Max-Low-GPU-step"] == 2) &
                          (self.jb.NetID == user)].copy()
            print("Canceled jobs:")
            print(usr)
            if not usr.empty:
                s = f"{get_first_name(user)},\n\n"
                s += f'These GPU jobs were canceled because they had GPUs with less than {self.low_gpu_util}% GPU utilization in the last 2 hours.'
                s += "\n\n"
 
                usr["State"] = "CANCELLED"
                usr["Hours"] = usr.elapsedraw.apply(lambda x: round(x / SECONDS_PER_HOUR, 1))
                usr = usr[["JobID", "JobName", "Partition", "State", "Low-GPU-2h", "Low-GPU-1h"]]

                usr_str = usr.to_string(index=False, justify="center")
                s += "\n".join([5 * " " + row for row in usr_str.split("\n")])
                s += "\n\n"

                s += 'You can check your job\'s resource utilization on Grafana by opening the link from the command:'
                s += "\n\n"
                s += f"     $ jobstats {usr.JobID.values[0]} -g"
                s += "\n"
                s += textwrap.dedent("""
                For more information on automatic job termination see the ROCS testbed's <a href="https://sands.kaust.edu.sa/internal/rocs-testbed/slurm-environment/#automatic-job-termination">Termination Policies</a>.

                Perhaps the GPUs being used are too powerful for the code that is running. Consider using an older GPU.
                See Princeton's <a href="https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#Low-GPU-Utilization--Potential-Solutions">GPU Computing</a> webpage for tips on how to improve GPU utilization.

                Consider replying to this automatic email to receive assistance. We'd be happy to help.
                """)

                # append the new violations to the log file
                vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
                Alert.update_violation_log(usr, vfile)
                
                send_email_from_cmd(s, user_mail, subject=f"{self.subject} - CANCELED", share_w=self.admin_emails)
                print(s)
                
                for jobid in usr.JobID.tolist():
                    cmd = f"scancel {jobid}"
                    # _ = subprocess.run(cmd,
                    #                    stdout=subprocess.PIPE,
                    #                    shell=True,
                    #                    timeout=10,
                    #                    text=True,
                    #                    check=True)
                    cfile = f"{self.vpath}/cancelations.txt"
                    with open(cfile, "a") as fp:
                        fp.write(f"{jobid},{user}\n")
                
