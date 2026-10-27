import os
import shutil
import time
import subprocess
import zipfile

CGN_DIR = '/workspace/contact_graspnet'

PREDICTIONS_PATH = (
    CGN_DIR + '/results/predictions_input.npz'
)

DONE_PATH = '/shared/cgn/cgn_done'
results_reported = False

SHARED_PREDICTIONS_PATH = '/shared/cgn/grasp_predictions.npz'

REQUEST_PATH = '/shared/cgn/run_cgn.request'

# Signal indicating the worker can accept another inference
READY_PATH = '/shared/cgn/cgn_ready'

process = None

print('CGN worker started. Waiting for inference requests...', flush=True)

def start_inference():
    command = [
        'python',
        'contact_graspnet/inference.py',
        '--np_path=/shared/cgn/input.npz',
        '--ckpt_dir=checkpoints/scene_test_2048_bs3_hor_sigma_001',
        '--local_regions',
        '--filter_grasps'
    ]

    return subprocess.Popen(
        command,
        cwd=CGN_DIR
    )

def predictions_ready():

    if not os.path.exists(PREDICTIONS_PATH):
        return False

    try:
        with zipfile.ZipFile(PREDICTIONS_PATH, 'r') as file:
            return file.testzip() is None

    except (zipfile.BadZipFile, OSError):
        return False

while True:

    # Worker is ready only when the previous inference process has exited
    if process is None or process.poll() is not None:
        if not os.path.exists(READY_PATH):
            with open(READY_PATH, 'w') as f:
                f.write('ready')
    else:
        if os.path.exists(READY_PATH):
            os.remove(READY_PATH)

    # Task 1: Start inference when requested
    if os.path.exists(REQUEST_PATH):

        # Only start new inference if the previous process has exited
        if process is None or process.poll() is not None:

            os.remove(REQUEST_PATH)

            # Clear previous inference results and completion signal
            for path in (PREDICTIONS_PATH, SHARED_PREDICTIONS_PATH, DONE_PATH):
                if os.path.exists(path):
                    os.remove(path)

            print('Starting Contact-GraspNet inference...', flush=True)

            results_reported = False

            # Worker is no longer available for another inference
            if os.path.exists(READY_PATH):
                os.remove(READY_PATH)

            process = start_inference()

    # Task 2: Check whether predictions are ready
    if process is not None and not results_reported:

        if predictions_ready():

            shutil.copyfile(
                PREDICTIONS_PATH,
                SHARED_PREDICTIONS_PATH
            )

            with open(DONE_PATH, 'w') as f:
                f.write('success')

            results_reported = True

            print('Grasp predictions ready!', flush=True)

    time.sleep(0.2)