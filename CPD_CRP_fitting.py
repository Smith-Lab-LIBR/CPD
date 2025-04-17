
import sys, os, re, subprocess

subject_list_path = sys.argv[1]
results_crp_fitting = sys.argv[2]

if not os.path.exists(results_crp_fitting):
    os.makedirs(results_crp_fitting)
    print(f"Created results directory {results_crp_fitting}")

if not os.path.exists(f"{results_crp_fitting}/logs"):
    os.makedirs(f"{results_crp_fitting}/logs")
    print(f"Created results_CRP-logs directory {results_crp_fitting}/logs")

subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'id' not in line:
            subjects.append(line.strip())

ssub_path = '/media/labs/rsmith/lab-members/rhodson/CPD/CPD_CRP_bash.ssub'

for subject in subjects:
    print(f"SUBMITTED SUBJECT [{subject}]")
    stdout_name = f"{results_crp_fitting}/logs/{subject}-%J.stdout"
    stderr_name = f"{results_crp_fitting}/logs/{subject}-%J.stderr"

    jobname = 'CPD_fits_crp'
    os.environ['seed'] = str(subject)
    os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {results_crp_fitting}")
    print(f"SUBMITTED JOB [{jobname}]")


    ###python3 CPD_CRP_fitting.py "/media/labs/rsmith/lab-members/nli/CPD_updated/T442_list.csv" results_crp_fitting

    
