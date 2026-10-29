"""
ASDS — Create Admin User Script

Usage:
    Interactive mode:
        python create_admin.py

    Non-interactive / CLI mode:
        python create_admin.py --username admin --email admin@asds.local --password mysecretpass --name "System Administrator"
"""

import sys
import os
import argparse
import getpass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app import create_app
from app.models import db, User


def parse_args():
    parser = argparse.ArgumentParser(description="Create or update an ASDS administrator account.")
    parser.add_argument("-u", "--username", help="Admin username")
    parser.add_argument("-e", "--email", help="Admin email address")
    parser.add_argument("-p", "--password", help="Admin password (plain text; omitted will prompt securely)")
    parser.add_argument("-n", "--name", help="Admin full display name", default="System Administrator")
    return parser.parse_args()


def prompt_for_details(args):
    print("=" * 55)
    print("        ASDS — Administrator Account Setup")
    print("=" * 55)

    username = (args.username or "").strip()
    while not username:
        username = input("Enter admin username: ").strip()
        if not username:
            print("Error: Username cannot be empty. Please try again.")

    email = (args.email or "").strip()
    while not email:
        email = input("Enter admin email: ").strip()
        if not email or "@" not in email:
            print("Error: Please enter a valid email address.")
            email = ""

    name = (args.name or "").strip() or "System Administrator"
    if not args.username:  # only prompt for name in interactive mode
        user_name = input(f"Enter full name [{name}]: ").strip()
        if user_name:
            name = user_name

    password = args.password
    if not password:
        while True:
            password = getpass.getpass("Enter password: ")
            if not password:
                print("Error: Password cannot be empty.")
                continue
            if len(password) < 6:
                print("Warning: Password is shorter than 6 characters.")
            confirm = getpass.getpass("Confirm password: ")
            if password != confirm:
                print("Error: Passwords do not match. Please try again.\n")
                continue
            break

    return username, email, name, password


def create_or_update_admin(username, email, name, password):
    env = os.environ.get('FLASK_ENV', 'development')
    config_name = 'development' if env == 'development' else 'production'
    app = create_app(config_name)

    with app.app_context():
        # Ensure database tables exist
        db.create_all()

        # Check existing username
        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()

        if existing_user and existing_email and existing_user.id != existing_email.id:
            print(f"\n[ERROR] Conflict: Username '{username}' and email '{email}' belong to different users.")
            return False

        target_user = existing_user or existing_email

        if target_user:
            if target_user.role == 'admin':
                print(f"\n[!] Admin user '{target_user.username}' already exists.")
                target_user.set_password(password)
                if name:
                    target_user.name = name
                if email:
                    target_user.email = email
                db.session.commit()
                print(f"[+] Password and details for admin '{target_user.username}' have been updated successfully.")
            else:
                print(f"\n[!] User '{target_user.username}' exists with role '{target_user.role}'. Promoting to 'admin'...")
                target_user.role = 'admin'
                target_user.set_password(password)
                if name:
                    target_user.name = name
                db.session.commit()
                print(f"[+] User '{target_user.username}' has been promoted to 'admin' with updated credentials.")
        else:
            new_admin = User(
                username=username,
                email=email,
                name=name,
                role='admin'
            )
            new_admin.set_password(password)
            db.session.add(new_admin)
            db.session.commit()
            print(f"\n[+] Administrator '{username}' created successfully!")

        print("\n" + "-" * 55)
        print("Login URL:  http://127.0.0.1:5000/admin/login")
        print(f"Username:   {username}")
        print("Role:       admin")
        print("-" * 55)
        return True


def main():
    args = parse_args()
    if args.username and args.email and args.password:
        # Full CLI mode
        username = args.username.strip()
        email = args.email.strip()
        name = args.name.strip()
        password = args.password
    else:
        # Interactive mode
        username, email, name, password = prompt_for_details(args)

    success = create_or_update_admin(username, email, name, password)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

