import argparse
import json
import logging
import os
import sys

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def setup_logging():
    """Configure logging for debugging or verbose output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Send messages (and optional artifacts) to Slack, with optional threading."
    )

    parser.add_argument(
        "--token",
        help="Slack Bot token (xoxb-...). If not provided, uses env var SLACK_BOT_TOKEN.",
    )
    parser.add_argument(
        "--channel",
        help="Slack channel (e.g., #general) or channel ID. If not provided, uses env var SLACK_CHANNEL.",
    )
    parser.add_argument("--message_file", "-m", help="Path to a .txt file containing the text to post to Slack.")
    parser.add_argument(
        "--artifact",
        action="append",
        help="File path(s) for artifacts to upload (e.g., logs, zips). Can be specified multiple times.",
    )
    parser.add_argument(
        "--thread_ts",
        help="Thread timestamp to post as a reply in a thread. If not provided, posts a new message.",
    )
    parser.add_argument(
        "--pin", action="store_true", help="Whether to pin the posted message."
    )
    parser.add_argument(
        "--status",
        choices=["success", "failure", "warning", "info"],
        help="Optional status for color-coded attachments (success, failure, warning, info).",
    )

    args = parser.parse_args()

    # Fallback to environment variables
    if not args.token:
        args.token = os.environ.get("SLACK_BOT_TOKEN")
        print(args.token)
    if not args.channel:
        args.channel = os.environ.get("SLACK_CHANNEL")

    if not args.token:
        logging.error(
            "Error: Slack Bot Token is required (--token or SLACK_BOT_TOKEN)."
        )
        sys.exit(2)
    if not args.channel:
        logging.error("Error: Slack channel is required (--channel or SLACK_CHANNEL).")
        sys.exit(2)

    return args


def format_message_with_status(message, status):
    """Format message based on status with appropriate block formatting."""
    status_headers = {
        "success": "🎉 *SUCCESS*",
        "failure": "🚨 *ERROR*",
        "warning": "⚠️ *WARNING*",
        "info": "ℹ️ *INFO*",
    }

    header = status_headers.get(status, "")

    if not message:
        status_messages = {
            "success": "Operation completed successfully",
            "failure": "Operation failed",
            "warning": "Operation completed with warnings",
            "info": "Operation information",
        }
        message = status_messages.get(status, "")

    # Format with block quote for multi-line messages
    formatted_msg = message.replace("\n", "\n>")
    return f"{header}\n>{formatted_msg}"


def build_status_attachment(status):
    """Build a single color-coded Slack attachment based on status."""
    color_map = {
        "success": "#36a64f",  # green
        "failure": "#ff0000",  # red
        "warning": "#ffae42",  # orange
        "info": "#439FE0",  # blue
    }
    return {
        "color": color_map.get(status, "#439FE0"),
    }


def load_json_file(filepath):
    """Load JSON content from a file."""
    if not filepath:
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Could not load JSON from {filepath}: {e}")
        sys.exit(2)


def get_or_open_dm_channel(client, user_id):
    """Open a DM channel with a user or retrieve the channel ID if it already exists."""
    try:
        response = client.conversations_open(users=[user_id])
        return response["channel"]["id"]
    except SlackApiError as e:
        logging.error(f"Error opening DM channel: {e.response['error']}")
        sys.exit(1)


def resolve_channel(client, channel):
    """
    Resolve channel to a valid Slack channel ID. Handles user IDs (to open DMs).
    """
    if channel.startswith("U"):  # User ID
        logging.info(f"Resolving DM channel for user {channel}")
        return get_or_open_dm_channel(client, channel)
    print(channel)
    return channel  # Return as-is for normal channel IDs


def post_message(client, channel, text, attachments=None, thread_ts=None, pin=False):
    """
    Post a message to a Slack channel or DM.
    """
    resolved_channel = resolve_channel(client, channel)
    try:
        response = client.chat_postMessage(
            channel=resolved_channel,
            text=text,
            attachments=attachments,
            thread_ts=thread_ts,
        )
        msg_ts = response["ts"]

        if pin:
            client.pins_add(channel=resolved_channel, timestamp=msg_ts)

        return msg_ts
    except SlackApiError as e:
        logging.error(f"Error posting message: {e.response['error']}")
        sys.exit(1)


def upload_files(client, channel, file_paths, thread_ts=None):
    """
    Upload files to Slack channel, optionally in a thread (specified by thread_ts).
    Returns a list of timestamps for each uploaded file message.
    """
    timestamps = []
    missing_files = []
    for path in file_paths:
        if not os.path.exists(path):
            logging.error(f"File not found: {path}")
            missing_files.append(path)
            continue

        try:
            logging.info(f"Uploading file: {path}")
            resp = client.files_upload_v2(
                channel=channel,
                file=path,
                initial_comment=f"Uploaded file: {os.path.basename(path)}",
                thread_ts=thread_ts,
            )
            timestamps.append(resp.get("ts", ""))
        except SlackApiError as e:
            logging.error(f"Error uploading file '{path}': {e.response['error']}")
            sys.exit(1)

    if missing_files:
        error_msg = "The following files were not found and could not be uploaded:\n"
        for file in missing_files:
            error_msg += f"- {file}\n"
        client.chat_postMessage(
            channel=channel,
            text=format_message_with_status(error_msg, "failure"),
            thread_ts=thread_ts,
            attachments=[build_status_attachment("failure")],
        )

    return timestamps


def main():
    setup_logging()
    args = parse_arguments()

    client = WebClient(token=args.token)

    # Resolve channel ID or DM ID
    args.channel = resolve_channel(client, args.channel)

    attachments = []
    # If status is specified, add a color-coded attachment
    if args.status:
        attachments.append(build_status_attachment(args.status))

    # If we have no attachments at all, set to None so Slack doesn't ignore the field
    attachments = attachments if attachments else None

    # Read message from file
    if args.message_file:
        try:
            with open(args.message_file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Could not read message file {args.message_file}: {e}")
            text = "Either the message file is not found or the file is not readable."
    else:
        text = ""

    # Format message based on status
    text = text.replace("\\n", "\n") if text else ""
    text = format_message_with_status(text, args.status if args.status else "info")

    logging.info(
        f"Posting message to channel {args.channel}. Thread TS: {args.thread_ts}"
    )
    msg_ts = post_message(
        client=client,
        channel=args.channel,
        text=text,
        attachments=attachments,
        thread_ts=args.thread_ts,
        pin=args.pin,
    )

    # Upload artifacts if any
    if args.artifact:
        logging.info("Uploading artifacts...")
        upload_files(
            client=client,
            channel=args.channel,
            file_paths=args.artifact,
            thread_ts=msg_ts if args.thread_ts else msg_ts,
        )

    # Print the timestamp so it can be captured in GitHub Actions, if desired
    print(msg_ts)
    logging.info("Slack notification complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()

# Sample command to run this script:
"""
python3 src/synthefy_pkg/scripts/send_slack_notification.py \
    --message_file <path_to_message_file> \
    --artifact <path_to_artifact_file> \
    --thread_ts <optional_thread_timestamp>
"""
