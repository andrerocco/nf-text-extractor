## Table of contents

1. [Prerequisites](#prerequisites)
2. [Setup instructions](#setup-instructions)

## Prerequisites

- Python 3.10 or higher installed on your system
- Git installed

## Setup instructions

1. Clone the Repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
    pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python app.py
   ```

If you added a new dependency to the project, make sure to update the `requirements.txt` with the latest version of the installed packages. You can do this by running: `pip freeze > requirements.txt`. To exit the virtual environment, you can run `deactivate`.
