import os
import subprocess
import sys
import venv
import shutil
import tempfile


def create_temp_virtualenv():
    """Create a temporary virtual environment and return its path."""
    temp_dir = tempfile.mkdtemp()
    env_dir = os.path.join(temp_dir, "temp_env_PMF")
    print(f"Creating temporary virtual environment at {env_dir}...")
    venv.create(env_dir, with_pip=True)
    return temp_dir, env_dir


def install_requirements(env_python, requirements_file):
    """Install required packages into the temporary virtual environment."""
    print("Installing requirements in the temporary virtual environment...")
    try:
        subprocess.check_call([env_python, "-m", "pip", "install", "-r", requirements_file])
        print("Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False


def install_package(env_python, package_path):
    """Install the current package into the temporary virtual environment."""
    print(f"Installing the package located at {package_path} into the temporary virtual environment...")
    try:
        subprocess.check_call([env_python, "-m", "pip", "install", package_path])
        print("Package installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing the package: {e}")
        return False

def install_kernel_and_register(env_python, kernel_name):
    """Install Jupyter and register the virtual environment as a kernel."""
    print("Installing Jupyter and registering the virtual environment as a kernel...")
    try:
        # Install Jupyter and ipykernel
        subprocess.check_call([env_python, "-m", "pip", "install", "jupyter", "ipykernel"])
        # Register the virtual environment as a Jupyter kernel
        subprocess.check_call([
            env_python, "-m", "ipykernel", "install", "--user",
            "--name", kernel_name,
            "--display-name", f"Python ({kernel_name})"
        ])
        print("Jupyter kernel registered successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing Jupyter or registering the kernel: {e}")
        return False
    
def run_tests(env_python):
    """Run all tests with coverage in the temporary virtual environment."""
    print("Running tests...")
    try:
        cmd = [env_python, "-m", "pytest", "--cov=PMF_toolkits", "--cov-report=term-missing"]
        subprocess.check_call(cmd)
        print("Tests ran successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return False


def use_virtualenv_message(env_dir):
    """Print instructions for how to use the temporary virtual environment."""
    print("\nThe temporary virtual environment has been set up successfully!")
    if os.name == "nt":
        print(f"To activate the environment, run:\n    {os.path.join(env_dir, 'Scripts', 'activate')}")
    else:
        print(f"To activate the environment, run:\n    source {os.path.join(env_dir, 'bin', 'activate')}")
    print("Once activated, you can use the package and run Python commands in this environment.")
    print("Remember to deactivate the environment when you're done by running:\n    deactivate")


def cleanup_temp_virtualenv(temp_dir):
    """Clean up the temporary virtual environment."""
    print(f"Cleaning up temporary virtual environment at {temp_dir}...")
    try:
        shutil.rmtree(temp_dir)
        print("Temporary virtual environment cleaned up successfully.")
    except Exception as e:
        print(f"Failed to clean up temporary virtual environment: {e}")


if __name__ == "__main__":
    success = True
    temp_dir = None
    kernel_name = "temp_env_PMF"  # You can change this to any name you'd like

    try:
        # 1. Create a temporary virtual environment
        temp_dir, env_dir = create_temp_virtualenv()
        env_python = os.path.join(env_dir, "Scripts", "python") if os.name == "nt" else os.path.join(env_dir, "bin", "python")

        # 2. Install requirements in the temporary environment
        if not install_requirements(env_python, "requirements-dev.txt"):
            success = False

        # 3. Install the package into the temporary environment
        package_path = "."  # Assuming the package is in the current directory
        if not install_package(env_python, package_path):
            success = False

        # 3. Install Jupyter and register the kernel
        if not install_kernel_and_register(env_python, kernel_name):
            success = False

        # 4. Run tests
        if not run_tests(env_python):
            success = False

        # 5. Allow user to use the virtual environment
        if success:
            use_virtualenv_message(env_dir)

    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        success = False

    finally:
        # 6. Optional: Keep the virtual environment for usage
        if not success and temp_dir:
            cleanup_temp_virtualenv(temp_dir)