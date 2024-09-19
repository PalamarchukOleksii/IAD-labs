Here's the updated README with the deactivation instructions integrated:

---

# **Instructions on How to Use the Project**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PalamarchukOleksii/IAD-labs.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd IAD-labs
   ```

3. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

4. **Activate the virtual environment:**

   - On **Windows**:
     ```bash
     .\.venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

5. **Install the required dependencies:**
   ```bash
   pip install -r lab1/requirements.txt
   ```

6. **Navigate to the specific project directory to run Python files:**
   ```bash
   cd lab1
   ```

7. **Run the Python script with the active virtual environment:**
   ```bash
   python gen1.py
   ```
   or
   ```bash
   python gen2.py
   ```
   *(Replace `1` or `2` with the specific version you want to run)*

8. **To run Jupyter notebooks with the active virtual environment, use the command:**
   ```bash
   jupyter notebook
   ```

9. **Deactivate the virtual environment when done:**

   - On **Windows, macOS, and Linux**:
     ```bash
     deactivate
     ```

---
