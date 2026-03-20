import subprocess


def test_shift_cli_help():
    """Verify that the shift CLI runs and displays the help menu."""
    result = subprocess.run(["drift", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    assert "Usage" in result.stdout or "help" in result.stdout.lower()
