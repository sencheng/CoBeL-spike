import subprocess
import argparse
import signal
import shutil
import psutil
import time
import os


def signal_handler(sig, frame):
    global run_simulation
    global simulation_process

    print("[INFO] Exiting the simulation")

    run_simulation = False
    os.killpg(os.getpgid(simulation_process.pid), signal.SIGKILL)


run_simulation = True
signal.signal(signal.SIGINT, signal_handler)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('startSeed', type=int, help='start seed')
parser.add_argument('endSeed', type=int, help='end seed')
parser.add_argument('--thresh', type=float, default=80.0, help='threshold value of the maximum cpu usage (default: 80.0)')
parser.add_argument('--checkTime', type=float, default=5.0, help='determines how long the cpu usage should be checked. Larger values correspond to more precise values')
parser.add_argument('--driveCheck', type=str, help='Specify the drive that should be checked. If given with threshDrive the program checks in-between the simulations if the given drive is above threshDrive. Stops the simulation if the given drive is above driveCheck')
parser.add_argument('--threshDrive', type=float, help='Specifies the minimum amount storge left on driveCheck in percent')
args = parser.parse_args()

if args.driveCheck is None and args.threshDrive is not None:
    print("\033[93m[INFO] threshDrive is given, while driveCheck is not given. Specify both parameters for the drive check. Going to continue without driveCheck\033[0m")
    time.sleep(5.0)

if args.threshDrive is None and args.driveCheck is not None:
    print("\033[93m[INFO] driveCheck is given, while threshDrive is not given. Specify both parameters for the drive check. Going to continue without driveCheck\033[0m")
    time.sleep(5.0)

checkDriveSet = False
if args.driveCheck is not None and args.threshDrive is not None:
    print(f"[INFO] driveChek {args.driveCheck} and threshDrive {args.threshDrive} given")
    checkDriveSet = True

print(f"[INFO] Run the simulation from {args.startSeed} to {args.endSeed}")
for seed in range(args.startSeed, args.endSeed + 1):
    if not run_simulation:
        break

    if checkDriveSet:
        total, used, _ = shutil.disk_usage(args.driveCheck)
        percentage_used = (used / total) * 100
        if percentage_used > args.threshDrive:
            print(f"\033[91m[INFO] The drive {args.driveCheck} is with {percentage_used:.2f}% above {args.threshDrive}%. Going to quit simulation \033[0m")
            exit()

    print(f"[INFO] Check the mean CPU usage for the next {args.checkTime} seconds")
    cpu_usage = psutil.cpu_percent(interval=args.checkTime)
    print(f"[INFO] The mean CPU usage is {cpu_usage}" + (f" Halt further simulating, while CPU usage is above {args.thresh}" if cpu_usage > args.thresh else ""))

    while True:
        cpu_usage = psutil.cpu_percent(interval=args.checkTime)
        if cpu_usage > args.thresh:
            print(f"[INFO] CPU usage is still above threshhold of {args.thresh}% with {cpu_usage}%. Going to further halt the simulation.")
        else:
            print(f"[INFO] Continue simulation")
            break

    print(f"[INFO] Run simulation with seed {seed}")

    # Start the simulation subprocess
    simulation_process = subprocess.Popen(
        f"./run_simulation.sh {seed} {seed}",
        shell=True,
        text=True,
        preexec_fn=os.setsid
    )

    # Wait for the subprocess to finish
    while True:
        status = simulation_process.poll()
        if status is not None:
            break

print("[INFO] Program exited.")
