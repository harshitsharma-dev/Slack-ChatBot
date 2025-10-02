#!/usr/bin/env python3
"""
Setup script for Slack AI Data Bot
"""
import os
import subprocess
import sys

def run_command(command, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return False

def setup_backend():
    """Setup Python backend"""
    print("\n🐍 Setting up Python backend...")
    
    # Create virtual environment
    if not os.path.exists('venv'):
        if not run_command('python -m venv venv'):
            return False
    
    # Install requirements
    pip_cmd = 'venv\\Scripts\\pip' if os.name == 'nt' else 'venv/bin/pip'
    if not run_command(f'{pip_cmd} install -r requirements.txt'):
        return False
    
    return True

def setup_frontend():
    """Setup React frontend"""
    print("\n⚛️ Setting up React frontend...")
    
    # Install npm dependencies
    if not run_command('npm install'):
        return False
    
    return True

def create_env_file():
    """Create .env file from example"""
    if not os.path.exists('.env'):
        print("\n📝 Creating .env file...")
        try:
            with open('.env.example', 'r') as example:
                content = example.read()
            with open('.env', 'w') as env_file:
                env_file.write(content)
            print("✓ Created .env file - please update with your credentials")
        except Exception as e:
            print(f"✗ Failed to create .env file: {e}")
            return False
    else:
        print("✓ .env file already exists")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Slack AI Data Bot...")
    
    success = True
    
    # Create .env file
    success &= create_env_file()
    
    # Setup backend
    success &= setup_backend()
    
    # Setup frontend
    success &= setup_frontend()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Update .env file with your credentials")
        print("2. Set up your PostgreSQL database")
        print("3. Run: python app.py (backend)")
        print("4. Run: npm start (frontend)")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()