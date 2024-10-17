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
from utils import send_email
from efficiency import gpus_with_low_util

class LowGpuUtilization(Alert):

    """Send warnings and cancel jobs with low GPU utilization. Interactive jobs
       with only 1 GPU and under a configurable run time limit are ignored."""

    def __init__(self, df, days_between_emails, violation, vpath, subject, **kwargs):
        super().__init__(df, days_between_emails, violation, vpath, subject, **kwargs)

    @staticmethod
    def get_stats_for_running_job(jobid, cluster):
        """Get the job statistics for running jobs by calling jobstats"""
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.SourceFileLoader('jobstats', '/data/secure/SLURM/utilities/bin/jobstats')
        spec = importlib.util.spec_from_loader('jobstats', loader)
        mymodule = importlib.util.module_from_spec(spec)
        loader.exec_module(mymodule)
        stats = mymodule.JobStats(jobid=jobid, cluster=cluster, prom_server="http://mcmgt01:9090")
        sleep(0.5)
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
            self.jb["jobstats"] = self.jb.apply(lambda row:
                                                LowGpuUtilization.get_stats_for_running_job(row["jobid"],
                                                                                            row["cluster"]),
                                                                                            axis='columns')
            # interactive jobs
            self.jb["interactive"] = self.jb["submitline"].apply(lambda x: True if not x.startswith("sbatch") else False)
            msk = self.jb["interactive"] & (self.jb.gpus == 1) & (self.jb["limit-minutes"] <= self.max_interactive_hours * MINUTES_PER_HOUR)
            self.jb = self.jb[~msk]
            
            # Step size is configured in jobstats in hours
            steps = [1,2]
            self.jb["GPU-lu-tpl"] = self.jb.jobstats.apply(gpus_with_low_util, low_util=self.low_gpu_util, steps=steps), axis='columns')
            self.jb["error-code"] = self.jb["GPU-lu-tpl"].apply(lambda tpl: tpl[1])
            self.jb = self.jb[self.jb["error-code"] == 0]
            self.jb["GPU-lu"] = self.jb["GPU-lu-tpl"].apply(lambda tpl: tpl[0])
            def add_util_step_cols(row):
                low_gpu_util_step = row["GPU-lu"]
                row["Max-Low-GPU-step"] = len(low_gpu_util_step)
                for step in low_gpu_util_step.keys():
                    gpus_low_util = [node + "[" + ",".join(low_gpu_util_step[node])  + "]" for node in low_gpu_util_step]
                    row[f"Low-GPU-{step}-step"] = ",".join(gpus_low_util)
                return row
            self.jb = self.jb.apply(add_util_step_cols) 
            self.jb = self.jb[self.jb["Max-Low-GPU-step"] > 0]
            
            self.jb = self.jb[["jobid", "NetID", "cluster", "partition", "gpus", "elapsedraw", "Max-Low-GPU-step"] + [f"Low-GPU-{step}-step" for step in steps]]
            renamings = {"gpus":"GPUs-Allocated", "jobid":"JobID", "cluster":"Cluster", "partition":"Partition"}
            self.jb.rename(columns=renamings, inplace=True)

    def send_emails_to_users(self):
        for user in self.jb.NetID.unique():
            #################
            #### warning ####
            #################
            usr = self.jb[(self.jb["Max-Low-GPU-step""] == 1) &
                          (self.jb.NetID == user)].copy()
            print(usr)
            if not usr.empty:
                s = f"{get_first_name(user)},\n\n"
                text = (
                f'You have running GPU job(s) that appear to have less than {self.low_gpu_util}% GPU utilization in the last hour:'
                )
                s += "\n".join(textwrap.wrap(text, width=80))
                s += "\n\n"

                usr["Hours"] = usr.elapsedraw.apply(lambda x: round(x / SECONDS_PER_HOUR, 1))
                usr.drop(columns=["NetID", "elapsedraw"], inplace=True)

                usr_str = usr.to_string(index=False, justify="center")
                s += "\n".join([5 * " " + row for row in usr_str.split("\n")])
                s += "\n"

                s += textwrap.dedent(f"""
                Your jobs will be AUTOMATICALLY CANCELLED if they are found to have low GPU utilization for the last 2 hours.
                For more information see the <a href="https://sands.kaust.edu.sa/internal/rocs-testbed/slurm-environment/#automatic-job-termination">Termination Policies</a>.
                """)

                s += "\n"
                text = (
                f'Please consider cancelling the job(s) listed above by using the "scancel" command:'
                )
                s += "\n".join(textwrap.wrap(text, width=80))
                s += "\n\n"
                s += f"     $ scancel {usr.JobID.values[0]}"
                s += "\n\n"

                zero = 'Check your job\'s resource utilization in Grafana to see the hourly GPU utilization. The Grafana link can be obtained by running the commmand:'
                s += "\n".join(textwrap.wrap(zero, width=80))
                s += "\n\n"
                s += f"     $ jobstats {usr.JobID.values[0]} -g"
                s += "\n"

                s += textwrap.dedent("""
                See <a href="https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#zero-util">GPU Computing</a> webpage for three common reasons for encountering zero GPU
                utilization.

                """)

                send_email(s, f"{user}@kaust.edu.sa", subject=f"{self.subject}", sender="ubuntu@localhost")
                print(s)

                # append the new violations to the log file
                vfile = f"{self.vpath}/{self.violation}/{user}.email.csv"
                usr.drop(columns=["GPU-Util"], inplace=True)
                Alert.update_violation_log(usr, vfile)

            ###############
            # cancel jobs #
            ###############
            usr = self.jb[(self.jb["Max-Low-GPU-step""] == 2) &
                          (self.jb.NetID == user)].copy()
            print(usr)
            if not usr.empty:
                s = f"{get_first_name(user)},\n\n"
                text = (
                f'The jobs below have been cancelled because they had GPUs utilizing less than {self.low_gpu_util}% GPU utilization for the last 2 hours.'
                )
                s += "\n".join(textwrap.wrap(text, width=80))
                s += "\n\n"
 
                usr["State"] = "CANCELLED"
                usr["Hours"] = usr.elapsedraw.apply(lambda x: round(x / SECONDS_PER_HOUR, 1))
                # usr = usr[["JobID", "Cluster", "Partition", "State", "GPUs-Allocated", "GPU-Util", "Hours"]]

                usr_str = usr.to_string(index=False, justify="center")
                s += "\n".join([5 * " " + row for row in usr_str.split("\n")])
                s += "\n"
                s += textwrap.dedent("""
                For more information about job cancellations see <a href="https://sands.kaust.edu.sa/internal/rocs-testbed/slurm-environment/#automatic-job-termination">Termination Policies</a>.

                See <a href="https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#zero-util">GPU Computing</a> webpage for three common reasons for encountering zero GPU
                utilization.

                Consider replying to this automatic email to receive assistance. Let us know if we can be of help.
                """)
                
                send_email(s, f"{user}@kaust.edu.sa", subject=f"{self.subject}", sender="ubuntu@localhost")
                for email in admin_self.emails: 
                    send_email(s, f"{email}", subject=f"{self.subject}", sender="ubuntu@localhost")
                print(s)

                for jobid in usr.JobID.tolist():
                    cmd = f"scancel {jobid}"
                    print(f"Fake: {cmd}")
                    # _ = subprocess.run(cmd,
                    #                    stdout=subprocess.PIPE,
                    #                    shell=True,
                    #                    timeout=10,
                    #                    text=True,
                    #                    check=True)
                    with open("/var/spool/slurm/job_defense_shield/cancelled.txt", "a") as fp:
                        fp.write(f"{jobid},{user}\n")
