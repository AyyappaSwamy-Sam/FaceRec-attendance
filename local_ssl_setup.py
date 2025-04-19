#!/usr/bin/env python3
import os
import subprocess
import sys
import socket
import platform

def create_self_signed_cert(output_dir):
    """Create a self-signed SSL certificate for local development"""
    
    print("Creating self-signed SSL certificate...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cert_file = os.path.join(output_dir, "cert.pem")
    key_file = os.path.join(output_dir, "key.pem")
    
    # Check if certificate already exists
    if os.path.exists(cert_file) and os.path.exists(key_file):
        answer = input(f"Certificate files already exist at {output_dir}. Regenerate? (y/n): ")
        if answer.lower() != 'y':
            print(f"Using existing certificate files at {output_dir}")
            return cert_file, key_file
    
    # Get hostname for certificate
    hostname = socket.gethostname()
    localhost = "localhost"
    
    # Use OpenSSL to generate a self-signed certificate
    openssl_cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", key_file,
        "-out", cert_file, "-days", "365", "-nodes", "-subj",
        f"/CN={hostname}/CN={localhost}/CN=127.0.0.1", "-addext", "subjectAltName=DNS:localhost,DNS:127.0.0.1,DNS:" + hostname
    ]
    
    try:
        subprocess.run(openssl_cmd, check=True)
        print(f"✅ SSL certificate created successfully at {output_dir}")
        # Set appropriate permissions
        os.chmod(key_file, 0o600)
        return cert_file, key_file
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating SSL certificate: {e}")
        sys.exit(1)

def update_hosts_file(domain_name):
    """Add domain to /etc/hosts file"""
    hosts_file = r"C:\Windows\System32\drivers\etc\hosts" if platform.system() == "Windows" else "/etc/hosts"
    entry = f"127.0.0.1 {domain_name}"
    
    # Check if entry already exists
    with open(hosts_file, 'r') as f:
        if entry in f.read():
            print(f"✅ Domain {domain_name} already exists in hosts file")
            return True
    
    print(f"\nTo use a custom domain name locally, you need to add it to your hosts file.")
    print(f"Run the following command as administrator/root:")
    
    if platform.system() == "Windows":
        print(f'\nAdd this line to {hosts_file}:')
        print(f"{entry}")
        print("\nYou can do this by running Notepad as administrator and editing the file.")
    else:
        print(f'\nRun this command:')
        print(f'echo "{entry}" | sudo tee -a {hosts_file}')
    
    return True

def update_flask_app(app_file, cert_file, key_file):
    """Update Flask app to use SSL certificate"""
    if not os.path.exists(app_file):
        print(f"❌ Flask app not found at {app_file}")
        return False
    
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Check if app.run is present at the end of the file
    if "if __name__ == '__main__':" in content and "app.run" in content:
        # Check if SSL context is already configured
        if "ssl_context" in content:
            content = content.replace("ssl_context", "# ssl_context")
        
        # Replace app.run line
        if "app.run" in content:
            # Find the line with app.run
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "app.run" in line and "if __name__ == '__main__':" in "".join(lines[max(0,i-5):i]):
                    # Replace or add ssl_context parameter
                    if ")" in line:
                        ssl_param = f", ssl_context=('{cert_file}', '{key_file}')"
                        lines[i] = line.replace(")", ssl_param + ")")
                    else:
                        lines[i] = line + f", ssl_context=('{cert_file}', '{key_file}')"
            
            content = '\n'.join(lines)
        else:
            # Add app.run at the end
            content += f"\n\nif __name__ == '__main__':\n    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('{cert_file}', '{key_file}'))\n"
    else:
        # Add app.run at the end
        content += f"\n\nif __name__ == '__main__':\n    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('{cert_file}', '{key_file}'))\n"
    
    # Backup the original file
    backup_file = f"{app_file}.bak"
    with open(backup_file, 'w') as f:
        f.write(content)
    
    # Write the modified content
    with open(app_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Flask app updated to use SSL certificate")
    print(f"✅ Original file backed up to {backup_file}")
    return True

def main():
    domain_name = "loggedin.test"  # local domain for testing
    app_file = "app.py"  # Default Flask app filename
    cert_dir = os.path.join(os.getcwd(), "ssl")
    
    print("\n=== Setting up SSL for Local Development ===\n")
    
    # Check if OpenSSL is installed
    try:
        subprocess.run(["openssl", "version"], check=True, stdout=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ OpenSSL is not installed or not in PATH")
        print("Please install OpenSSL and try again")
        sys.exit(1)
    
    # Find app.py in current directory
    if not os.path.exists(app_file):
        app_files = [f for f in os.listdir() if f.endswith('.py')]
        if not app_files:
            print("❌ No Python files found in current directory")
            app_file = input("Enter the path to your Flask app file: ")
        else:
            print("Available Python files:")
            for i, f in enumerate(app_files):
                print(f"{i+1}. {f}")
            choice = int(input("Select your Flask app file (number): "))
            app_file = app_files[choice-1]
    
    # Create self-signed certificate
    cert_file, key_file = create_self_signed_cert(cert_dir)
    
    # Update hosts file with instructions
    update_hosts_file(domain_name)
    
    # Update Flask app
    update_flask_app(app_file, cert_file, key_file)
    
    print("\n=== SSL Setup Complete ===")
    print(f"You can now access your app at: https://{domain_name}:5000")
    print("⚠️ Your browser will show a warning about the self-signed certificate.")
    print("This is normal for local development. Click 'Advanced' and 'Proceed' to continue.")
    print("\nTo run your app:")
    print(f"python {app_file}")

if __name__ == "__main__":
    main()