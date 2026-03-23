#!/usr/bin/env python3
"""
Google Calendar OAuth Setup for my-brain-bot.

Run this ONCE on your local machine to authorize the bot with your Google Calendar.
It will create a token file that the bot uses for ongoing access.

SETUP STEPS:
1. Go to https://console.cloud.google.com/
2. Create a new project (or use existing)
3. Enable the Google Calendar API:
   - Go to APIs & Services > Library
   - Search "Google Calendar API" and enable it
4. Create OAuth 2.0 credentials:
   - Go to APIs & Services > Credentials
   - Click "Create Credentials" > "OAuth client ID"
   - Application type: "Desktop app"
   - Download the JSON file
5. Save it as vault/google_credentials.json
6. Run this script: python setup_gcal.py
7. It will open a browser for you to authorize
8. The token will be saved to vault/google_token.json
9. Deploy to Railway — the token persists and auto-refreshes
"""

import os
import sys
from pathlib import Path

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    print("Missing dependencies. Run:")
    print("  pip install google-api-python-client google-auth-oauthlib")
    sys.exit(1)

VAULT = Path(__file__).parent / "vault"
CREDS_FILE = VAULT / "google_credentials.json"
TOKEN_FILE = VAULT / "google_token.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def main():
    if not CREDS_FILE.exists():
        print(f"\nERROR: {CREDS_FILE} not found!")
        print("\nSteps:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Enable Google Calendar API")
        print("3. Create OAuth 2.0 Desktop credentials")
        print("4. Download the JSON and save it as:")
        print(f"   {CREDS_FILE}")
        sys.exit(1)

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("Opening browser for Google Calendar authorization...")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json())
        print(f"\nToken saved to {TOKEN_FILE}")

    print("\nGoogle Calendar is connected!")
    print("The bot will now be able to read and create calendar events.")
    print("\nMake sure to include google_token.json in your deployment.")


if __name__ == "__main__":
    main()
